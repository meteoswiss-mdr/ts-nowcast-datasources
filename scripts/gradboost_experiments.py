import argparse
from datetime import datetime
import os
import pickle
import string

from matplotlib import gridspec, pyplot as plt
import numpy as np
import pandas as pd

from gbtsnowcast.ml.models import base, gradboost
from gbtsnowcast.visualization import plots


def load_data(nc_file=None, pickle_file=None):
    if pickle_file:
        with open(pickle_file, 'rb') as f:
            (past_features, future_features) = pickle.load(f)
    else:
        (past_features, future_features) = \
            base.prepare_scalar_dataset(nc_file)

    (past_features, future_features) = gradboost.add_gb_features(
        past_features, future_features)

    return (past_features, future_features)


def reduce_in_time_range(features, var_name, index_rng,
    test_func=np.isfinite, reduce_func=np.any):
    """Perform reduction over a range of times 
    (used to create target variables).
    """
    
    variables = [features[var_name+"::"+str(i)] for i in range(*index_rng)]
    variables = np.vstack(variables)
    test_result = test_func(variables)
    test_reduced = reduce_func(test_result, axis=0)
    return pd.Series(test_reduced)


def target_variables(features, maxz_start_index=0):
    """Create prediction targets.
    """

    maxz = {
        i: {
            dataset:
                features[dataset][
                    "nexrad::MAXZ::nbagg-mean-25-circular::{}".format(i)
                ]
            for dataset in ["train", "valid", "test"]    
        }        
        for i in range(maxz_start_index,13)
    }
    
    lightning_present = {}
    echo45_exists = {}
    echo45_height = {}

    for time_range in [(1,7), (7,13)]:

        lightning_present[time_range] = {
            dataset: 
            reduce_in_time_range(
                features[dataset],
                "goesglm::event_density::nbagg-any-25-circular",
                time_range,
                test_func=lambda x: x.astype(bool)
            )        
            for dataset in ["train", "valid", "test"]
        }

        echo45_exists[time_range] = {
            dataset: 
            reduce_in_time_range(
                features[dataset],
                "nexrad::ECHOTOP-45::nbagg-mean-25-circular",
                time_range
            )        
            for dataset in ["train", "valid", "test"]
        }

        echo45_keys = [
            "nexrad::ECHOTOP-45::nbagg-mean-25-circular::{}".format(i) 
            for i in range(*time_range)
        ]
        echo45_datasets = {
            dataset:
            features[dataset][echo45_keys][echo45_exists[time_range][dataset]]
            for dataset in ["train", "valid", "test"]
        }
        echo45_height[time_range] = {
            dataset:
            reduce_in_time_range(
                echo45_datasets[dataset],
                "nexrad::ECHOTOP-45::nbagg-mean-25-circular",
                time_range,
                test_func=lambda x: x,
                reduce_func=np.nanmean
            )
            for dataset in ["train", "valid", "test"]
        }

    return (maxz, lightning_present, echo45_exists, echo45_height)


hyperparams_maxz = {} # the defaults have been tuned for this
hyperparams_lightning = {} # the defaults have been tuned for this
hyperparams_echo45_exists = {
    "max_bin": 64
}
hyperparams_echo45_height = {
    "path_smooth": 100.0,
    "max_depth": 5,
    "min_gain_to_split": 10.0
}


def train_models(past_features, past_feat_echo45, future_features,
        lightning_present, echo45_exists, echo45_height):

    models_maxz = gradboost.fit_regression_multiple(
        past_features["train"], future_features["train"],
        past_features["valid"], future_features["valid"],
        "nexrad::MAXZ::nbagg-mean-25-circular",
        hyperparams=hyperparams_maxz
    )

    models_lightning = {}
    models_echo45_exists = {}
    models_echo45_height = {}

    for time_range in [(1,7), (7,13)]:
        models_lightning[time_range] = gradboost.fit_gradboost_binary(
            past_features["train"], lightning_present[time_range]["train"],
            past_features["valid"], lightning_present[time_range]["valid"],
            additional_params=hyperparams_lightning
        )
        
        models_echo45_exists[time_range] = gradboost.fit_gradboost_binary(
            past_features["train"], echo45_exists[time_range]["train"],
            past_features["valid"], echo45_exists[time_range]["valid"],
            additional_params=hyperparams_echo45_exists
        )

        models_echo45_height[time_range] = gradboost.fit_gradboost_regression(
            past_feat_echo45[time_range]["train"], echo45_height[time_range]["train"],
            past_feat_echo45[time_range]["valid"], echo45_height[time_range]["valid"],
            additional_params=hyperparams_echo45_height
        )

    return (models_maxz, models_lightning, 
        models_echo45_exists, models_echo45_height)

def sum_importances(importance):
    total_importance = {}
    for imp in importance:
        for (gain, variable) in imp:
            if variable not in total_importance:
                total_importance[variable] = 0.0
            total_importance[variable] += gain
    
    return sorted(
        ((v,k) for (k,v) in total_importance.items()),
        reverse=True
    )


def feature_importance(models_maxz, models_lightning,
    models_echo45_exists, models_echo45_height, feature_names):

    models_maxz_keys = sorted(
        models_maxz.keys(),
        key=lambda k: int(k.split("::")[-1])
    )
    importance_maxz = [
        gradboost.total_feat_importance(models_maxz[k].gb_model, feature_names)
        for k in models_maxz_keys
    ]
    total_importance_maxz = sum_importances(importance_maxz)
    source_importance_maxz = [
        gradboost.source_importance(imp) for imp in importance_maxz
    ]
    
    importance_lightning = [
        gradboost.total_feat_importance(model, feature_names)
        for (_, model) in models_lightning.items()
    ]
    total_importance_lightning = sum_importances(importance_lightning)
    source_importance_lightning = [
        gradboost.source_importance(imp) for imp in importance_lightning
    ]

    importance_echo45_exists = [
        gradboost.total_feat_importance(model, feature_names)
        for (_, model) in models_echo45_exists.items()
    ]
    total_importance_echo45_exists = sum_importances(importance_echo45_exists)
    source_importance_echo45_exists = [
        gradboost.source_importance(imp) for imp in importance_echo45_exists
    ]

    importance_echo45_height = [
        gradboost.total_feat_importance(model, feature_names)
        for (_, model) in models_echo45_height.items()
    ]
    total_importance_echo45_height = sum_importances(importance_echo45_height)
    source_importance_echo45_height = [
        gradboost.source_importance(imp) for imp in importance_echo45_height
    ]

    return (
        total_importance_maxz, source_importance_maxz,
        total_importance_lightning, source_importance_lightning,
        total_importance_echo45_exists, source_importance_echo45_exists,
        total_importance_echo45_height, source_importance_echo45_height,
    )


def combination_errors(past_features, past_feat_echo45, maxz,
    lightning_present, echo45_exists, echo45_height,
    lightning_period=(7,13), echo45_period=(1,7)
    ):

    combination_errors_maxz = gradboost.exclusion_study(
        past_features, maxz[12], objective="regression_residual",
        persistence_var_name='nexrad::MAXZ::nbagg-mean-25-circular',
        hyperparams=hyperparams_maxz
    )

    combination_errors_lightning = gradboost.exclusion_study(
        past_features, lightning_present[lightning_period], objective="binary",
        hyperparams=hyperparams_lightning
    )
    # climatological errors
    combination_errors_lightning[()] = {
        "binary": {
            ds: lightning_present[lightning_period][ds].mean()
            for ds in ["train", "valid", "test"]
        },
        "cross_entropy": {
            ds: np.nan for ds in ["train", "valid", "test"]
        }
    }

    combination_errors_echo45_exists = gradboost.exclusion_study(
        past_features, echo45_exists[(1,7)], objective="binary",
        hyperparams=hyperparams_echo45_exists
    )
    # climatological errors
    combination_errors_echo45_exists[()] = {
        "binary": {
            ds: echo45_exists[echo45_period][ds].mean()
            for ds in ["train", "valid", "test"]
        },
        "cross_entropy": {
            ds: np.nan for ds in ["train", "valid", "test"]
        }
    }

    combination_errors_echo45_height = gradboost.exclusion_study(
        past_feat_echo45[(1,7)], echo45_height[(1,7)], objective="regression",
        hyperparams=hyperparams_echo45_height
    )

    return (
        combination_errors_maxz, combination_errors_lightning,
        combination_errors_echo45_exists, combination_errors_echo45_height
    )


def importance_plots(
        total_importance_maxz, source_importance_maxz,
        total_importance_lightning, source_importance_lightning,
        total_importance_echo45_exists, source_importance_echo45_exists,
        total_importance_echo45_height, source_importance_echo45_height
    ):

    figsize = (12,12)
    fig1 = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, hspace=0.15, wspace=0.1)

    axes_maxz = [fig1.add_subplot(gs[:3,0]), fig1.add_subplot(gs[3,0])]
    plots.plot_gradboost_importance(
        total_importance_maxz, source_importance_maxz,
        5*(1+np.arange(len(source_importance_maxz))),
        source_style="lines",
        title="Maximum reflectivity",
        fig=fig1,
        axes=axes_maxz
    )

    axes_lightning = [fig1.add_subplot(gs[:3,1]), fig1.add_subplot(gs[3,1])]
    plots.plot_gradboost_importance(
        total_importance_lightning, source_importance_lightning,
        np.array([0, 30, 60]),
        source_style="bars",
        title="Lightning occurrence",
        legend=False,
        fig=fig1,
        axes=axes_lightning
    )

    fig2 = plt.figure(figsize=figsize)

    axes_echo45_exists = [fig2.add_subplot(gs[:3,0]), fig2.add_subplot(gs[3,0])]
    plots.plot_gradboost_importance(
        total_importance_echo45_exists, source_importance_echo45_exists,
        np.array([0, 30, 60]),
        source_style="bars",
        title="45 dBZ echo top presence",
        legend=False,
        fig=fig2,
        axes=axes_echo45_exists
    )

    axes_echo45_height = [fig2.add_subplot(gs[:3,1]), fig2.add_subplot(gs[3,1])]
    plots.plot_gradboost_importance(
        total_importance_echo45_height, source_importance_echo45_height,
        np.array([0, 30, 60]),
        source_style="bars",
        title="45 dBZ echo top height",
        legend=False,
        fig=fig2,
        axes=axes_echo45_height
    )

    def label(ax, x, y, label):
        ax.text(
            x, y, "({})".format(label),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox={"facecolor": (1,1,1,0.5), "edgecolor": (1,1,1,0)}
        )
    fig1_axes = [axes_maxz, axes_lightning]
    fig2_axes = [axes_echo45_exists, axes_echo45_height]
    for axes in [fig1_axes, fig2_axes]:
        for (i, (top_ax, bottom_ax)) in enumerate(axes):
            top_label = string.ascii_lowercase[2*i]
            label(top_ax, 0.98, 0.985, top_label)
            bottom_label = string.ascii_lowercase[2*i+1]
            label(bottom_ax, 0.98, 0.96, bottom_label) 

    return (fig1, fig2)


def combination_plots(
        combination_errors_maxz, combination_errors_lightning,
        combination_errors_echo45_exists, combination_errors_echo45_height,
        dataset='test'
    ):
    
    figsize = (12.5,7.5)
    fig1 = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, hspace=0.15, wspace=0.1)

    axes_maxz = [fig1.add_subplot(gs[0,0]), fig1.add_subplot(gs[0,1])]
    plots.exclusion_plot(
        combination_errors_maxz,
        fig=fig1, axes=axes_maxz,
        variable_name="Maximum reflectivity",
        significant_digits=4,
        dataset=dataset
    )

    axes_lightning = [fig1.add_subplot(gs[0,2]), fig1.add_subplot(gs[0,3])]
    plots.exclusion_plot(
        combination_errors_lightning,
        fig=fig1, axes=axes_lightning,
        variable_name="Lightning occ.",
        subplot_index=2,
        dataset=dataset
    )

    fig2 = plt.figure(figsize=figsize)

    axes_echo45_exists = [fig2.add_subplot(gs[0,0]), fig2.add_subplot(gs[0,1])]
    plots.exclusion_plot(
        combination_errors_echo45_exists,
        fig=fig2, axes=axes_echo45_exists,
        variable_name="45 dBZ ET presence", 
        dataset=dataset      
    )

    axes_echo45_height = [fig2.add_subplot(gs[0,2]), fig2.add_subplot(gs[0,3])]
    plots.exclusion_plot(
        combination_errors_echo45_height,
        fig=fig2, axes=axes_echo45_height,
        variable_name="45 dBZ ET height",
        subplot_index=2,
        significant_digits=4,
        dataset=dataset
    )

    return (fig1, fig2)


# def track_plot(cfg_fn, time=datetime(2020,8,25,19,20)):
#     cfg = config.load_config(cfg_fn)
#     (_, readers) = ds_config.prepare_readers(cfg)
#     tracks = stormtrack.StormTracks(readers["nexrad"], storm_track_var="MAXZ",
#         min_valid_fraction=0.05, recenter_radius=0)
#
#     fig = plt.figure(figsize=(8.25,4.5))
#     ax = fig.add_axes([0,0,0.97,1], label="borders")
#     m = area_def2basemap(readers["nexrad"].grid_projection.area,
#         ax=ax, resolution='i')
#     m.drawcoastlines(linewidth=1.5)
#     m.drawcountries(linewidth=1)
#     m.drawstates(color=(0.5,0.5,0.5))
#
#     cax = fig.add_axes([0.91,0,0.03,1], label="colorbar")
#     ax = fig.add_axes([0,0,0.97,1], label="image", facecolor=(0,0,0,0))
#     plots.storm_tracks(
#         readers["nexrad"].motion_vectors,
#         {time: tracks(time)},
#         time,
#         cax=cax
#     )
#
#     return fig


def confusion_plot(
    models_lightning, models_echo45_exists,
    past_features,
    lightning_present, echo45_exists,
    dataset="test"
):
    models_data = [
        (models_lightning[(1,7)], lightning_present[(1,7)][dataset], "(a) Lightning occ.\n0-30 min"),
        (models_lightning[(7,13)], lightning_present[(7,13)][dataset], "(b) Lightning occ.\n30-60 min"),
        (models_echo45_exists[(1,7)], echo45_exists[(1,7)][dataset], "(c) 45 dBZ ET occ.\n0-30 min"),
        (models_echo45_exists[(7,13)], echo45_exists[(7,13)][dataset], "(d) 45 dBZ ET occ.\n30-60 min"),
    ]
    fig = plt.figure(figsize=(4,6))
    gs = gridspec.GridSpec(18,4,hspace=0.15,wspace=0.15)
    cbar_ax = fig.add_subplot(gs[17,:])
    for (k,(model, y_true, title)) in enumerate(models_data):
        y_pred = model.predict(past_features[dataset])
        i = k // 2
        j = k % 2
        ax = fig.add_subplot(gs[8*i:8*(i+1),2*j:2*(j+1)])
        plots.confusion_matrix(y_true, y_pred, axes=ax,
            cbar_ax=(cbar_ax if i==0 else None),
            xlabel=(i==1),
            ylabel=(j==0)
        )
        ax.set_title(title, fontsize=10)
    
    return fig


def all_plots(
    maxz,
    lightning_present,
    echo45_exists,
    models_maxz, 
    models_lightning,
    models_echo45_exists,
    past_features,
    importances,
    combination_metrics,
    fig_dir=None
    ):

    if fig_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(script_dir,"../figures/")

    maxz_test = {i: maxz[i]["test"] for i in maxz}
    fig = plots.metrics_by_time(models_maxz,
        ["mae", "rmse"], past_features["test"], maxz_test)
    fig.savefig(os.path.join(fig_dir, "maxz_errors.pdf"), bbox_inches='tight')
    plt.close('all')

    fig = confusion_plot(models_lightning, models_echo45_exists,
        past_features, lightning_present, echo45_exists)

    (fig1, fig2) = importance_plots(*importances)
    fig1.savefig(os.path.join(fig_dir, "importance_maxz_lightning.pdf"),
        bbox_inches='tight')
    fig2.savefig(os.path.join(fig_dir, "importance_echo45.pdf"),
        bbox_inches='tight')
    plt.close('all')

    (fig1, fig2) = combination_plots(*combination_metrics)
    fig1.savefig(os.path.join(fig_dir, "exclusion_maxz_lightning.pdf"),
        bbox_inches='tight')
    fig2.savefig(os.path.join(fig_dir, "exclusion_echo45.pdf"),
        bbox_inches='tight')
    plt.close('all')


def cross_entropy(y, y_pred):
    return -np.nanmean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))


def print_errors(
        maxz, lightning_present, echo45_exists, echo45_height,
        models_maxz, models_lightning, models_echo45_exists, models_echo45_height,
        past_features, past_feat_echo45,
        dataset='test'
    ):

    for i in range(13):
        y = maxz[i][dataset]
        model_key = "nexrad::MAXZ::nbagg-mean-25-circular::{}".format(i)
        y_pred = models_maxz[model_key].predict(past_features[dataset])
        diff = y_pred-y
        mae = np.nanmean(abs(diff))
        rmse = np.sqrt(np.nanmean(diff**2))
        print("MAXZ step={} MAE={:.4g} RMSE={:.4g}".format(i, mae, rmse))

    for time_range in [(1,7),(7,13)]:
        y = lightning_present[time_range][dataset]
        y_pred = models_lightning[time_range].predict(past_features[dataset])
        xent = cross_entropy(y, y_pred)
        binary = np.nanmean(y_pred.round() != y)
        frac_true = np.count_nonzero(y)/len(y)
        print("Lightning steps={}-{} cross-entropy={:.4g} error_rate={:.4g} frac_true={:.4g}".format(
            time_range[0], time_range[1]-1, xent, binary, frac_true))
        
    for time_range in [(1,7),(7,13)]:
        y = echo45_exists[time_range][dataset]
        y_pred = models_echo45_exists[time_range].predict(past_features[dataset])
        xent = cross_entropy(y, y_pred)
        binary = np.nanmean(y_pred.round() != y)        
        frac_true = np.count_nonzero(y)/len(y)
        print("Echo 45 dBZ presence steps={}-{} cross-entropy={:.4g} error_rate={:.4g} frac_true={:.4g}".format(
            time_range[0], time_range[1]-1, xent, binary, frac_true))

    for time_range in [(1,7),(7,13)]:
        y = echo45_height[time_range][dataset]
        y_pred = models_echo45_height[time_range].predict(past_feat_echo45[time_range][dataset])
        diff = y_pred-y
        mae = np.nanmean(abs(diff))
        rmse = np.sqrt(np.nanmean(diff**2))
        std_true = y.std()
        print("Echo 45 dBZ height steps={}-{} MAE={:.4g} RMSE={:.4g} std_true={:.4g}".format(
            time_range[0], time_range[1]-1, mae, rmse, std_true))


def output_importances(
        out_dir,
        models_maxz, models_lightning,
        models_echo45_exists, models_echo45_height, feature_names
    ):

    models_maxz_keys = sorted(
        models_maxz.keys(),
        key=lambda k: int(k.split("::")[-1])
    )
    print(models_maxz_keys)
    importance_maxz = {
        int(k.split("::")[-1]):
        gradboost.total_feat_importance(models_maxz[k].gb_model, feature_names)
        for k in models_maxz_keys
    } 
    importance_lightning = {
        k:
        gradboost.total_feat_importance(model, feature_names)
        for (k, model) in models_lightning.items()
    }
    importance_echo45_exists = {
        k:
        gradboost.total_feat_importance(model, feature_names)
        for (k, model) in models_echo45_exists.items()
    }
    importance_echo45_height = {
        k: 
        gradboost.total_feat_importance(model, feature_names)
        for (k, model) in models_echo45_height.items()
    }    

    def output_importance(importance, out_fn):
        with open(out_fn, 'w') as f:
            for (gain, feature) in importance:
                (source, feature_name) = feature.split("::")
                source = plots.get_source_name(source)
                feature_name = plots.get_feature_name(feature_name)

                f.write("{:.3f},{},{}\n".format(gain, source, feature_name))

    for k in importance_maxz.keys():
        out_fn = os.path.join(
            out_dir,
            "importance_maxz_{:02d}.csv".format(k)
        )
        output_importance(importance_maxz[k], out_fn)

    for k in importance_lightning.keys():
        out_fn = os.path.join(
            out_dir,
            "importance_lightning_{:02d}-{:02d}.csv".format(k[0],k[1]-1)
        )
        output_importance(importance_lightning[k], out_fn)

    for k in importance_echo45_exists.keys():
        out_fn = os.path.join(
            out_dir,
            "importance_echo45exists_{:02d}-{:02d}.csv".format(k[0],k[1]-1)
        )
        output_importance(importance_echo45_exists[k], out_fn)

    for k in importance_echo45_height.keys():
        out_fn = os.path.join(
            out_dir,
            "importance_echo45height_{:02d}-{:02d}.csv".format(k[0],k[1]-1)
        )
        output_importance(importance_echo45_height[k], out_fn)


def all_experiments(nc_file=None, pickle_file=None):
    (past_features, future_features) = load_data(
        pickle_file=pickle_file, nc_file=nc_file)

    (maxz, lightning_present, 
        echo45_exists, echo45_height) = target_variables(future_features)
    (maxz_past, lightning_present_past, echo45_exists_past, 
        echo45_height_past) = target_variables(past_features, maxz_start_index=1)

    past_feat_echo45 = {
        time_range:
        {
            dataset:
            past_features[dataset][echo45_exists[time_range][dataset]]
            for dataset in ["train", "valid", "test"]
        }
        for time_range in [(1,7),(7,13)]
    }

    (
        models_maxz, models_lightning, 
        models_echo45_exists, models_echo45_height
    ) = train_models(
        past_features, past_feat_echo45, future_features,
        lightning_present, echo45_exists, echo45_height
    )

    importances = feature_importance(
        models_maxz, models_lightning,
        models_echo45_exists, models_echo45_height,
        list(past_features["train"].keys())
    )

    combination_metrics = combination_errors(
        past_features, past_feat_echo45, maxz,
        lightning_present, echo45_exists, echo45_height
    )

    all_plots(
        maxz, lightning_present, echo45_exists,
        models_maxz, models_lightning, models_echo45_exists,
        past_features,
        importances,
        combination_metrics
    )

    print_errors(
        maxz, lightning_present, echo45_exists, echo45_height,
        models_maxz, models_lightning, models_echo45_exists, models_echo45_height,
        past_features, past_feat_echo45
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nc_file', type=str,
        help="NetCDF4 file with dataset", default="")
    parser.add_argument('--pickle_file', type=str,
        help="Pickle file with dataset", default="")

    args = parser.parse_args()
    nc_file = args.nc_file
    if not nc_file:
        nc_file = None
    pickle_file = args.pickle_file
    if not pickle_file:
        pickle_file = None

    all_experiments(nc_file=nc_file, pickle_file=pickle_file)
