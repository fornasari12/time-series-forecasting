import warnings
import argparse
import pandas as pd

from config import load_config
from load_data import LoadData

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

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
else:
    model_key = "model"
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    spec = load_config("/content/temporal-fusion-transformer/config.yaml")
    DATA_PATH = "/content/temporal-fusion-transformer/" + spec["general"]["data_path"]

FOLDER_LIST = spec["general"]["folder_list"]
MODEL_PATH = spec[model_key]["model_path"]
BATCH_SIZE = spec[model_key]["batch_size"]
MAX_EPOCHS = spec[model_key]["max_epochs"]
GPUS = spec[model_key]["gpus"]
LEARNING_RATE = spec[model_key]["learning_rate"]
HIDDEN_SIZE = spec[model_key]["hidden_size"]
DROPOUT = spec[model_key]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec[model_key]["hidden_continuous_size"]
GRADIENT_CLIP_VAL = spec[model_key]["gradient_clip_val"]
ATTENTION_HEAD_SIZE = spec[model_key]["attention_head_size"]
LSTM_LAYERS = spec[model_key]["lstm_layers"]

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

    for column in train_data.columns:
        if train_data[column].dtype == object:
            train_data[column] = train_data[column].astype(str).astype("category")
        # if train_data[column].dtype == "float64":
        #     train_data[column] = pd.to_numeric(train_data[column], downcast="float")
        # if train_data[column].dtype == "int64":
        #     train_data[column] = pd.to_numeric(train_data[column], downcast="integer")

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

    # configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )

    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gpus=GPUS,
        weights_summary="top",
        gradient_clip_val=GRADIENT_CLIP_VAL,
        # limit_train_batches=30,
        # fast_dev_run=True,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=20,
    )

    # fit network
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    torch.save(tft.state_dict(), MODEL_PATH)
