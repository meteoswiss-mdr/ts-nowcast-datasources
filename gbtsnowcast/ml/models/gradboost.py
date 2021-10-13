import gc
from itertools import chain, combinations
import os
import pickle
import re

import dask
import lightgbm as lgb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .base import PersistenceModel, RegressionModel, split_validation
from .base import create_mean_features, poh_features, trt_features


class ResidualGBModel:
    def __init__(self, preprocessing='persistence', persistence_var_name=None,
        persistence_data=None):

        self.persistence_data = persistence_data
        if persistence_data is None:
            if preprocessing == "persistence":
                self.preproc_model = PersistenceModel(persistence_var_name)
            elif preprocessing == "linear":
                self.preproc_model = RegressionModel(reg_type="linear")

    def fit(self, past_feat_train, future_feat_train, 
        past_feat_valid, future_feat_valid, params=None, *kwargs):

        if self.persistence_data is None:
            self.preproc_model.fit(past_feat_train, future_feat_train)
            pred_feat_preproc_train = self.preproc_model.predict(past_feat_train)
            pred_feat_preproc_valid = self.preproc_model.predict(past_feat_valid)
        else:
            pred_feat_preproc_train = self.persistence_data["train"]
            pred_feat_preproc_valid = self.persistence_data["valid"]

        diff_train = future_feat_train - pred_feat_preproc_train
        diff_valid = future_feat_valid - pred_feat_preproc_valid
        self.bias = np.nanmean(diff_train)
        
        if not past_feat_train.empty:
            self.gb_model = fit_gradboost_regression(
                past_feat_train, diff_train - self.bias,
                past_feat_valid, diff_valid - self.bias,
                additional_params=params
            )
        
    def predict(self, past_features, dataset=None):
        if self.persistence_data is None:
            pred_feat_preproc = self.preproc_model.predict(past_features)
        else:
            assert(dataset is not None)
            pred_feat_preproc = self.persistence_data[dataset]

        result = pred_feat_preproc + self.bias
        if not past_features.empty:
            result += self.gb_model.predict(past_features)
        return result


def add_gb_features(past_features, future_features):
    for dataset in ["train", "valid", "test"]:
        past_features[dataset] = create_mean_features(past_features[dataset])
        past_features[dataset] = trt_features(past_features[dataset], insert=True)
        past_features[dataset] = poh_features(past_features[dataset], insert=True)
        future_features[dataset] = trt_features(future_features[dataset], insert=True)
        future_features[dataset] = poh_features(future_features[dataset], insert=True)
    return (past_features, future_features)


def fit_gradboost_regression(past_feat_train, future_feat_train, past_feat_valid, future_feat_valid, 
    additional_params=None, **kwargs):

    param = {
        "objective": "mae",        
        "max_depth": 6,
        "learning_rate": 0.1,
        "reg_alpha": 0.1,
        "seed": 2345,
        "metric": ["l1", "l2"],
        "first_metric_only": True,
        "force_col_wise": True,
        "num_leaves": 48,
        "histogram_pool_size": 16000,
        "path_smooth": 10.0,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,        
    }

    if additional_params is not None:
        for p in additional_params:
            param[p] = additional_params[p]

    past_feat_train = past_feat_train.rename(
        columns=lambda x : re.sub('[^A-Za-z0-9_]+', '', x)
    )
    past_feat_valid = past_feat_valid.rename(
        columns=lambda x : re.sub('[^A-Za-z0-9_]+', '', x)
    )

    train_data = lgb.Dataset(past_feat_train.values.astype(np.float32).copy(),
        label=future_feat_train.values.astype(np.float32).copy())
    valid_data = lgb.Dataset(past_feat_valid.values.astype(np.float32).copy(),
        label=future_feat_valid.values.astype(np.float32).copy())
    train_data.raw_data = None
    valid_data.raw_data = None
    del past_feat_train, past_feat_valid
    gc.collect()
    
    bst = lgb.train(param, train_data, num_boost_round=100000, 
        valid_sets=[train_data, valid_data], early_stopping_rounds=20, **kwargs)
    gc.collect()

    return bst


def fit_regression_multiple(
    past_feat_train, future_feat_train,
    past_feat_valid, future_feat_valid,
    target_feature_name,
    hyperparams=None
    ):

    target_feature_names = list(future_feat_train.keys())
    target_feature_names = [t for t in target_feature_names 
        if t.startswith(target_feature_name)]
    
    def fit_one(future_feat_train, future_feat_valid):
        gbmodel = ResidualGBModel(persistence_var_name=target_feature_name)
        gbmodel.fit(past_feat_train, future_feat_train, 
            past_feat_valid, future_feat_valid, params=hyperparams)
        return gbmodel

    models = {name: fit_one(future_feat_train[name], future_feat_valid[name]) 
        for name in target_feature_names}

    return models

        
def eval_model(model, x, y_true):
    y_pred = model.predict(x)
    metrics = {}
    metrics['rmse'] = np.sqrt(np.nanmean((y_pred-y_true)**2))
    metrics['mae'] = np.nanmean(abs(y_pred-y_true))
    return metrics


def total_feat_importance(gbmodel, feature_names, importance_type='gain'):
    importance = gbmodel.feature_importance(
        importance_type=importance_type)

    total = {}
    for (k,imp) in enumerate(importance):
        feat_name = feature_names[k]
        parts = feat_name.split("::")
        source = parts[0]
        feature = parts[1]
        if source == "composites":
            if feature.startswith("ABIC"):
                source = "goesabi"
            elif feature == "upslope_flow_radar":
                source = "aster"
        elif source in ("trt", "poh"):
            source = "nexrad"
        root = "::".join((source, feature))
        
        if root not in total:
            total[root] = 0
        total[root] += imp

    total = [(v,k) for (k,v) in total.items()]
    total.sort(reverse=True)
    return total


def source_importance(total_importance):
    source_imp = {}
    for (g,f) in total_importance:
        source = f.split("::")[0]
        if source not in source_imp:
            source_imp[source] = 0.0
        source_imp[source] += g
    return sorted(
        ((g,s) for (s,g) in source_imp.items()),
        reverse=True
    )


def evaluate_regression(models, past_features, future_features):
    def leadtime_from_model(model_name):
        return int(model_name.split("::")[-1])

    (X_train, X_valid, Y_train, Y_valid) = split_validation(
        past_features, future_features)

    var_name = "::".join(list(models.keys())[0].split("::")[:-1])
    pm = PersistenceModel(var_name)
    models_by_leadtime = {leadtime_from_model(n): models[n] for n in models}

    leadtime = []
    persistence_rmse = []
    persistence_mae = []
    model_rmse = []
    model_mae = []

    for lt in range(min(models_by_leadtime), max(models_by_leadtime)+1):
        print(lt)
        pm_pred = pm.predict(X_valid)
        model_pred = pm_pred + models_by_leadtime[lt].predict(X_valid)
        true_value = Y_valid[var_name+"::{}".format(lt)].values

        pm_l1 = np.nanmean(abs(pm_pred-true_value))
        pm_l2 = np.sqrt(np.nanmean((pm_pred-true_value)**2))
        model_l1 = np.nanmean(abs(model_pred-true_value))
        model_l2 = np.sqrt(np.nanmean((model_pred-true_value)**2))

        leadtime.append(lt)
        persistence_rmse.append(pm_l2)
        persistence_mae.append(pm_l1)
        model_rmse.append(model_l2)
        model_mae.append(model_l1)

    return (leadtime, persistence_rmse, persistence_mae, model_rmse, model_mae)


def fit_gradboost_binary(past_feat_train, future_feat_train, past_feat_valid, future_feat_valid, 
    additional_params=None, weight_data=False, **kwargs):

    param = {
        "seed": 2345,
        "objective": "binary",        
        "metric": ["binary", "binary_error"],
        "first_metric_only": True,
        "force_col_wise": True,
        "max_depth": 5,
        "learning_rate": 0.05,
        "lambda_l1": 0.1,
        "num_leaves": 48,
        "histogram_pool_size": 16000,
        "min_gain_to_split": 2.0,
        "bagging_fraction": 0.75,
        "bagging_freq": 5
    }

    if additional_params is not None:
        for p in additional_params:
            param[p] = additional_params[p]

    past_feat_train = past_feat_train.rename(
        columns=lambda x : re.sub('[^A-Za-z0-9_]+', '', x)
    )
    past_feat_valid = past_feat_valid.rename(
        columns=lambda x : re.sub('[^A-Za-z0-9_]+', '', x)
    )

    if weight_data:
        y = future_feat_train.astype(int)
        w = compute_class_weight('balanced', classes=[0,1], y=y)
        weights = np.full_like(future_feat_train, w[0])
        weights[future_feat_train] = w[1]
    else:
        weights = None

    train_data = lgb.Dataset(past_feat_train.values.astype(np.float32).copy(),
        label=future_feat_train.values.astype(bool).copy(), weight=weights)
    valid_data = lgb.Dataset(past_feat_valid.values.astype(np.float32).copy(),
        label=future_feat_valid.values.astype(bool).copy())
    train_data.raw_data = None
    valid_data.raw_data = None
    del past_feat_train, past_feat_valid
    gc.collect()
    
    bst = lgb.train(param, train_data, num_boost_round=100000, 
        valid_sets=[train_data, valid_data], early_stopping_rounds=20, **kwargs)
    gc.collect()

    return bst


def exclusion_study(past_features, future_feature,
    objective="regression_residual", persistence_var_name=None,
    always_include=(), hyperparams=None):

    feat_sources = {f: f.split("::")[0] for f in past_features["train"].keys()}
    for f in list(feat_sources.keys()):
        source = feat_sources[f]
        
        # replace sources of composite features
        if source in ["trt", "poh"]:
            feat_sources[f] = "nexrad"
        elif source == "composites":
            feat = f.split("::")[1]
            if feat.startswith("ABI"):
                feat_sources[f] = "goesabi"
            elif feat.startswith("upslope_flow_radar"):
                feat_sources[f] = "aster"

    sources = sorted(set(feat_sources.values()))

    # compute all subsets of sources
    source_subsets = chain.from_iterable(
        combinations(sources, r) for r in range(1,len(sources)+1)
    )
    if objective == "regression_residual":
        source_subsets = chain(((),), source_subsets)

    if persistence_var_name is not None:
        persistence_data = {
            dataset: 
            past_features[dataset][persistence_var_name+"::1"]
            for dataset in past_features
        }
    predict_kwargs = lambda dataset: {}

    errors_by_subset = {}
    # evaluate model and compute errors for all subsets
    for subset in source_subsets:        
        if not set(always_include).issubset(set(subset)):
            continue

        # select features in subset
        keys = [f for f in feat_sources if (feat_sources[f] in subset)]

        if objective == "regression_residual":
            model = ResidualGBModel(persistence_data=persistence_data)
            model.fit(past_features["train"][keys], future_feature["train"],
                past_features["valid"][keys], future_feature["valid"],
                params=hyperparams)
            predict_kwargs = lambda dataset: {"dataset": dataset}
        elif objective == "regression":
            model = fit_gradboost_regression(
                past_features["train"][keys], future_feature["train"],
                past_features["valid"][keys], future_feature["valid"],
                additional_params=hyperparams
            )
        elif objective == "binary":
            model = fit_gradboost_binary(
                past_features["train"][keys], future_feature["train"],
                past_features["valid"][keys], future_feature["valid"],
                additional_params=hyperparams
            )
        
        y_pred = {
            dataset: 
            model.predict(past_features[dataset][keys], **predict_kwargs(dataset))
            for dataset in past_features.keys()
        }
        y = future_feature

        errors_by_subset[subset] = {}
        if objective.startswith("regression"):
            mae = {dataset: np.nanmean(
                    abs(y_pred[dataset]-y[dataset])
                )
                for dataset in y_pred.keys()}
            rmse = {dataset: np.sqrt(np.nanmean((y_pred[dataset]-y[dataset])**2))
                for dataset in y_pred.keys()}
            errors_by_subset[subset]["mae"] = mae
            errors_by_subset[subset]["rmse"] = rmse

        elif objective == "binary":
            cross_entropy = {dataset: -np.nanmean(
                    y[dataset]*np.log(y_pred[dataset]) + 
                    (1-y[dataset])*np.log(1-y_pred[dataset])
                )
                for dataset in y_pred.keys()}
            binary = {dataset: np.nanmean(
                    abs(y_pred[dataset].round() != y[dataset])
                )
                for dataset in y_pred.keys()
            }
            errors_by_subset[subset]["cross_entropy"] = cross_entropy
            errors_by_subset[subset]["binary"] = binary

    return errors_by_subset
