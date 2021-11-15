import warnings

from config import load_config
from load_data import LoadData

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

warnings.filterwarnings("ignore")

spec = load_config("/content/temporal-fusion-transformer/config.yaml")
DATA_PATH = spec["general"]["data_path"]
FOLDER_LIST = spec["general"]["folder_list"]
BATCH_SIZE = spec["model"]["batch_size"]
MAX_EPOCHS = spec["model"]["max_epochs"]
GPUS = spec["model"]["gpus"]
LEARNING_RATE = spec["model"]["learning_rate"]
HIDDEN_SIZE = spec["model"]["hidden_size"]
DROPOUT = spec["model"]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec["model"]["hidden_continuous_size"]
GRADIENT_CLIP_VAL = spec["model"]["gradient_clip_val"]
max_prediction_length = spec["model"]["max_prediction_length"]
max_encoder_length = spec["model"]["max_encoder_length"]

if __name__ == "__main__":

    train_data, _ = LoadData(
        data_path="/content/temporal-fusion-transformer/" + DATA_PATH,
        folder_list=FOLDER_LIST,
        cutoff=0.70
    ).load_data()

    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="value",
        group_ids=["dataset", "id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["id"],
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["hour", "month", "day_of_week"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["value"],
        target_normalizer=GroupNormalizer(
            groups=["dataset", "id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

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

    # calculate baseline mean absolute error, i.e. predict next value as the
    # last available value from the history
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    (actuals - baseline_predictions).abs().mean().item()

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
        attention_head_size=1,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # fit network
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    torch.save(tft.state_dict(), "/content/model/tft_regressor_test.pt")
