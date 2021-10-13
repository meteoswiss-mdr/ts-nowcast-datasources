from datetime import datetime, timedelta
import itertools

import netCDF4
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


def valid_maxz(maxz, threshold=0.7):
    num_nan = np.isnan(maxz).sum(axis=1)
    valid = (num_nan/maxz.shape[1] < threshold)
    return valid


def prepare_scalar_dataset(features_fn, valid_frac=0.1, test_frac=0.1, random_seed=1234):
    with netCDF4.Dataset(features_fn, 'r') as ds:
        maxz = ds["past_features/nexrad::MAXZ::nbagg-mean-25-circular"][:]
        valid = valid_maxz(maxz)

        # load time data and group by day
        time = np.array(ds.variables["time"][:], copy=False)
        n_total = len(time)
        epoch = datetime(1970,1,1)
        time = [epoch+timedelta(seconds=int(t)) for t in time]    
        indices_by_day = {}
        for (i,t) in enumerate(time):
            date = t.date()
            if valid[i]:
                if date not in indices_by_day:
                    indices_by_day[date] = []
                indices_by_day[date].append(i)

        rng = np.random.RandomState(seed=random_seed)

        # select at least valid_frac fraction of data for validation
        # using entire days
        valid_indices = []
        while len(valid_indices) / n_total < valid_frac:
            day = rng.choice(list(indices_by_day), 1)[0]
            valid_indices += indices_by_day[day]
            del indices_by_day[day]
            
        # then select test_frac for testing
        test_indices = []
        while len(test_indices) / n_total < test_frac:
            day = rng.choice(list(indices_by_day), 1)[0]
            test_indices += indices_by_day[day]
            del indices_by_day[day]

        # the rest of the data go to the training set
        train_indices = list(
            itertools.chain.from_iterable(indices_by_day.values())
        )

        indices = {
            "train": np.array(train_indices),
            "valid": np.array(valid_indices),
            "test": np.array(test_indices)
        }
        for dataset in ["train", "valid", "test"]:
            rng.shuffle(indices[dataset])

        print("Training N={}, validation N={}, test N={}".format(
            len(train_indices), len(valid_indices), len(test_indices)
        ))

        # load data from file
        past_features = {"train": {}, "valid": {}, "test": {}}
        future_features = {"train": {}, "valid": {}, "test": {}}

        for (features, group) in [
            (past_features, "past_features"),
            (future_features, "future_features")
        ]:

            for feat in sorted(ds[group].variables.keys()):
                (data_source, var, feat_type) = feat.split("::")
                if feat_type.startswith("nbagg"):
                    print(group+"/"+feat)
                    n = ds[group][feat].shape[1]
                    data = np.array(ds[group][feat][:], copy=False)
                    for dataset in indices:
                        for k in range(n):
                            time_ind = n-k if group == "past_features" else k
                            scalar_name = feat + "::{}".format(time_ind)
                            features[dataset][scalar_name] = data[indices[dataset],k]

    for dataset in ["train", "valid", "test"]:
        past_features[dataset] = pd.DataFrame.from_dict(past_features[dataset])
        future_features[dataset] = pd.DataFrame.from_dict(future_features[dataset])

    return (past_features, future_features)


def split_validation(X_train, Y_train, valid_frac=0.1):
    return train_test_split(X_train, Y_train, test_size=valid_frac,
        random_state=1234)


def create_mean_features(features, subset="-mean-"):
    processed = set()
    diff_feat = {}
    
    for key in list(features.keys()):
        parts = key.split("::")
        if not subset in key:
            continue
        root = "::".join(parts[:-1])
        if root in processed:
            continue

        values = []
        i = 1
        while True:
            k = root + "::{}".format(i)
            if k not in features:
                break
            values.append(features[k])
            i += 1
        
        values = np.vstack(values)
        mean = np.nanmean(values, axis=0)
        diff_feat[root+"::timemean"] = mean

    feat = features.copy()
    for k in diff_feat:
        feat[k] = diff_feat[k]
    return feat


def create_diff_features(features, subset="-mean-"):
    processed = set()
    diff_feat = {}
    
    for key in list(features.keys()):
        parts = key.split("::")
        if not subset in key:
            continue
        root = "::".join(parts[:-1])
        if root in processed:
            continue

        values = []
        i = 1
        while True:
            k = root + "::{}".format(i)
            if k not in features:
                break
            values.append(features[k])
            i += 1
        
        values = np.vstack(values)
        diff = np.diff(values, axis=0)
        for i in range(diff.shape[0]):
            diff_feat[root+"::diff{}".format(i+1)] = diff[i,:]

    feat = features.copy()
    for k in diff_feat:
        feat[k] = diff_feat[k]
    return feat


def get_timestep(var):
    return var.split("::")[-1]


def trt_features(features, insert=False):
    maxz_feat = [k for k in features.keys() if 
        "nexrad::MAXZ::nbagg-max" in k]
    vil_feat = [k for k in features.keys() if 
        "nexrad::VIL::nbagg-max" in k]
    et45_feat = [k for k in features.keys() if 
        "nexrad::ECHOTOP-45::nbagg-median" in k]
    area57_feat = [k for k in features.keys() if 
        "nexrad::MAXZ::nbagg-numgt_57" in k]

    maxz_feat = {get_timestep(k): features[k] for k in maxz_feat}
    vil_feat = {get_timestep(k): features[k] for k in vil_feat}
    et45_feat = {get_timestep(k): features[k] for k in et45_feat}
    area57_feat = {get_timestep(k): features[k] for k in area57_feat}

    keys_in_all = set(maxz_feat) & set(vil_feat) & \
        set(et45_feat) & set(area57_feat)

    trt_feat = {}
    for k in keys_in_all:
        maxz = maxz_feat[k]
        vil = vil_feat[k]
        et45 = et45_feat[k]
        area57 = area57_feat[k]

        trt_feat[k] = (4/7) * (
            2 * np.clip(vil/65.0, 0, 1) +
            2 * np.clip(et45/10.0, 0, 1) +
            np.clip((maxz-4500.0)/(6500.0-4500.0), 0, 1) +
            2 * np.clip(area57/40.0, 0, 1)
        )
        trt_feat[k][~np.isfinite(trt_feat[k])] = 0.0

    if insert:
        features = features.copy()
        for k in keys_in_all:
            features["trt::trt::nbagg-mean-25-circular::{}".format(k)] = trt_feat[k]
        return features
    else:
        return trt_feat


def poh_features(features, insert=False):
    et45_feat = [k for k in features.keys() if 
        "nexrad::ECHOTOP-45::nbagg-median" in k]
    deg0l_feat = [k for k in features.keys() if 
        "ecmwf::deg0l::nbagg-median" in k]

    et45_feat = {get_timestep(k): features[k] for k in et45_feat}
    deg0l_feat = {get_timestep(k): features[k] for k in deg0l_feat}
    keys_in_all = set(et45_feat) & set(deg0l_feat)

    poh_feat = {}
    for k in keys_in_all:
        et45 = et45_feat[k]
        deg0l = deg0l_feat[k]

        poh_feat[k] = np.clip(((et45-deg0l)-1600)/(6000-1600), 0, 1)
        poh_feat[k][~np.isfinite(poh_feat[k])] = 0

    if insert:
        features = features.copy()
        for k in keys_in_all:
            features["poh::poh::nbagg-mean-25-circular::{}".format(k)] = poh_feat[k]
        return features
    else:
        return poh_feat


class PersistenceModel:
    def __init__(self, var_name):
        self.var_name = var_name

    def fit(self, past_features, future_feature):
        pass

    def predict(self, past_features):
        latest_features = past_features[self.var_name+"::1"]
        
        return latest_features


class RegressionModel:
    def __init__(self, reg_type="linear"):
        self.reg_type = reg_type

    def fit(self, past_features, future_feature):
        self.past_imputer = SimpleImputer()
        self.past_imputer.fit(past_features)
        past_features = self.past_imputer.transform(past_features)
        if type(future_feature) == pd.core.series.Series:
            future_feature = future_feature.to_frame()
        future_feature = SimpleImputer().fit_transform(future_feature)
        
        if self.reg_type == "linear":
            self.model = LinearRegression().fit(past_features, future_feature)
        elif self.reg_type == "logistic":
            self.model = LogisticRegression().fit(past_features, future_feature)

    def predict(self, past_features):
        past_features = self.past_imputer.transform(past_features)
        return self.model.predict(past_features).flatten()
