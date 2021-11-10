import warnings

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss

from config import load_config

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

spec = load_config("/content/temporal-fusion-transformer/config.yaml")
BATCH_SIZE = spec["model"]["batch_size"]
MAX_EPOCHS = spec["model"]["max_epochs"]
GPUS = spec["model"]["gpus"]
LEARNING_RATE = spec["model"]["learning_rate"]
HIDDEN_SIZE = spec["model"]["hidden_size"]
DROPOUT = spec["model"]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec["model"]["hidden_continuous_size"]
GRADIENT_CLIP_VAL = spec["model"]["gradient_clip_val"]

data = pd.read_csv("/content/temporal-fusion-transformer/data/MERCHANT_NUMBER_OF_TRX.csv")
data = data[[
    "MERCHANT_1_NUMBER_OF_TRX",
    "MERCHANT_2_NUMBER_OF_TRX",
    "date"
]]
data = data.set_index("date").stack().reset_index()
data = data.rename(
    columns={
        'level_1': 'id',
        0: 'trx'
    }
)

# add time index
data["time_idx"] = pd.to_datetime(data.date).astype(int)
data["time_idx"] -= data["time_idx"].min()
data["time_idx"] = (data.time_idx / 3600000000000) + 1
data["time_idx"] = data["time_idx"].astype(int)

# add datetime variables
data["month"] = pd.to_datetime(data.date).dt.month\
    .astype(str)\
    .astype("category")
data["day_of_week"] = pd.to_datetime(data.date).dt.dayofweek\
    .astype(str)\
    .astype("category")
data["hour"] = pd.to_datetime(data.date).dt.hour\
    .astype(str)\
    .astype("category")

# cut atypical values at the end of the sample
train_data = data[:3200*2]
max_prediction_length = 24
max_encoder_length = 72
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    train_data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="trx",
    group_ids=["id"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["id"],
    time_varying_known_reals=["time_idx"],
    time_varying_known_categoricals=["hour", "month", "day_of_week"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["trx"],
    target_normalizer=GroupNormalizer(
        groups=["id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,

)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, train_data, predict=True, stop_randomization=True)

# create dataloaders for model
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 10, num_workers=0)

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()

# configure network and trainer
pl.seed_everything(42)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    gpus=GPUS,
    weights_summary="top",
    gradient_clip_val=GRADIENT_CLIP_VAL,
    # limit_train_batches=30,  # comment in for training, running validation every 30 batches
    # fast_dev_run=True,  # comment in to check that network or dataset has no serious bugs
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
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

torch.save(tft.state_dict(), "/content/model/tft_regressor.pt")
