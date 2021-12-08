import warnings
import pickle
import argparse

from config import load_config
from load_data import LoadData

import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run Training Pipeline")
parser.add_argument(
    "-l",
    "--local",
    help="Run local or in colab",
    action='store_true',
    default=False,
)
args = parser.parse_args()

LOCAL = args.local

if LOCAL:
    model_key = "model_local"
    spec = load_config("config.yaml")
    DATA_PATH = spec["general"]["data_path"]
    HYPERPARMETERS_PATH = spec[model_key]["hyperparameters_path"]
else:
    model_key = "model"
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    spec = load_config("/content/time-series-forecasting/config.yaml")
    DATA_PATH = "/content/time-series-forecasting/" + spec["general"]["data_path"]
    HYPERPARMETERS_PATH = spec[model_key]["hyperparameters_path"]

FOLDER_LIST = spec["general"]["folder_list"]

BATCH_SIZE = spec[model_key]["batch_size"]
MAX_EPOCHS = spec[model_key]["max_epochs"]
GPUS = spec[model_key]["gpus"]
LEARNING_RATE = spec[model_key]["learning_rate"]
HIDDEN_SIZE = spec[model_key]["hidden_size"]
DROPOUT = spec[model_key]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec[model_key]["hidden_continuous_size"]
GRADIENT_CLIP_VAL = spec[model_key]["gradient_clip_val"]

lags = spec[model_key]["lags"]
sma = spec[model_key]["sma"]

time_varying_known_reals = spec[model_key]["time_varying_known_reals"]

if lags:
    lags_columns = [f"lag_{lag}" for lag in range(lags, 0, -1)]
    time_varying_known_reals = time_varying_known_reals + lags_columns

if sma:
    sma_columns = [f"sma_{sma}" for sma in sma]
    time_varying_known_reals = time_varying_known_reals + sma_columns


time_varying_known_categoricals = spec[model_key]["time_varying_known_categoricals"]
max_prediction_length = spec[model_key]["max_prediction_length"]
max_encoder_length = spec[model_key]["max_encoder_length"]
sample = spec[model_key]["sample"]
cutoff = spec[model_key]["cutoff"]

if __name__ == "__main__":

    train_data, _ = LoadData(
        data_path=DATA_PATH,
        folder_list=FOLDER_LIST,
        cutoff=cutoff,
        sample=sample,
        sma=sma,
        lags=lags
    ).load_data()

    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="value",
        group_ids=["id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["id"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["value"],
        target_normalizer=GroupNormalizer(
            groups=["id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

    )

    pl.seed_everything(42)

    # create validation set (predict=True) which means to predict the
    # last max_prediction_length points in time for each series
    validation = TimeSeriesDataSet.from_dataset(
        training,
        train_data,
        predict=True,
        stop_randomization=True
    )

    # create dataloaders for model
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0)
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 10,
        num_workers=0
    )

    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=150,
        max_epochs=35,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.0003, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )

    # save study results - also we can resume tuning at a later point in time
    with open(HYPERPARMETERS_PATH, "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)
