import warnings
import pickle

import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

data = pd.read_csv("data/MERCHANT_NUMBER_OF_TRX.csv")
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

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)
