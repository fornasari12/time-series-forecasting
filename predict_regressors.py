import torch
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import load_config

spec = load_config("config.yaml")
BATCH_SIZE = spec["model"]["batch_size"]
MAX_EPOCHS = spec["model"]["max_epochs"]
GPUS = spec["model"]["gpus"]
LEARNING_RATE = spec["model"]["learning_rate"]
HIDDEN_SIZE = spec["model"]["hidden_size"]
DROPOUT = spec["model"]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec["model"]["hidden_continuous_size"]

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

model = TemporalFusionTransformer.from_dataset(
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

model.load_state_dict(torch.load("model/tft_regressor.pt"))

for target in ["MERCHANT_1_NUMBER_OF_TRX", "MERCHANT_2_NUMBER_OF_TRX"]:

    df = data[data["id"] == target].reset_index(drop=True)

    for start in range(3212, 3600, 24):
        test_data = df[start:(start + max_encoder_length)]
        y_obs = df[(start + max_encoder_length): (start + max_encoder_length + max_prediction_length)]

        y_hat = model.predict(
            test_data,
            mode="prediction",
            return_x=True
        )[0][0].tolist()

        fig, ax = plt.subplots()

        ax.plot(
            pd.Series(data=y_hat,
                      index=pd.to_datetime(y_obs.date)),
            label='forecast'
        )

        ax.plot(
            y_obs.set_index(pd.to_datetime(y_obs.date))["trx"],
            color='orange',
            alpha=0.8,
            label='observed',
            linestyle='--',
            linewidth=1.2
        )

        plt.title(f'{target}')
        plt.legend()
        plt.show()
