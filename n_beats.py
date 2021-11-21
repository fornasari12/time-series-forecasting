import os
import warnings

from config import load_config
from load_data import LoadData

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE

warnings.filterwarnings("ignore")

spec = load_config("config.yaml")
DATA_PATH = spec["general"]["data_path"]
FOLDER_LIST = spec["general"]["folder_list"]
MODEL_PATH = spec["model_local"]["model_path"]
BATCH_SIZE = spec["model_local"]["batch_size"]
MAX_EPOCHS = spec["model_local"]["max_epochs"]
GPUS = spec["model_local"]["gpus"]
LEARNING_RATE = spec["model_local"]["learning_rate"]
HIDDEN_SIZE = spec["model_local"]["hidden_size"]
DROPOUT = spec["model_local"]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec["model_local"]["hidden_continuous_size"]
GRADIENT_CLIP_VAL = spec["model_local"]["gradient_clip_val"]

max_prediction_length = spec["model_local"]["max_prediction_length"]
max_encoder_length = spec["model_local"]["max_encoder_length"]
sample = spec["model_local"]["sample"]
cutoff = spec["model_local"]["cutoff"]

if __name__ == "__main__":

    train_data, test_data = LoadData(
        data_path=DATA_PATH,
        folder_list=FOLDER_LIST,
        cutoff=cutoff,
        sample=sample,
        date_features=False,
        sma=None,
        lags=None
    ).load_data()

    train_data["value"] = train_data["value"].astype(float)

    print("A")

    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="value",
        categorical_encoders={"id": NaNLabelEncoder().fit(train_data.id)},
        group_ids=["id"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

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
    pl.seed_everything(42)

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
        max_epochs=100,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.01,
        callbacks=[early_stop_callback],
        limit_train_batches=30,
    )

    net = NBeats.from_dataset(
        training,
        learning_rate=4e-3,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=1.0,
    )

    trainer.fit(
        net,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    net.load_state_dict(torch.load("model/n_beats/n_beats.pt"))




