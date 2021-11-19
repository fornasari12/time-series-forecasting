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

    train_data, _ = LoadData(
        data_path=DATA_PATH,
        folder_list=FOLDER_LIST,
        cutoff=cutoff,
        sample=sample,
        sma=[12, 6],
        lags=2
    ).load_data()

    time_varying_known_categoricals = [
        "hour",
        "month",
        "day_of_week",
        "day",
        "weekofyear"
    ]

    time_varying_known_reals = [
        # "time_idx",
        "sma_12",
        "sma_6",
        "(t-2)",
        "(t-1)"
    ]

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
        time_varying_known_reals=time_varying_known_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
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

    torch.save(tft.state_dict(), MODEL_PATH)

    print("a")
