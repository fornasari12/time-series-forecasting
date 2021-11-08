import torch
import pandas as pd

from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

import matplotlib.pyplot as plt

data = pd.read_csv("data/MERCHANT_NUMBER_OF_TRX.csv")
data = data[["MERCHANT_1_NUMBER_OF_TRX", "date"]]
data["id"] = "M1"

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
train_data = data[:3200]
max_prediction_length = 24
max_encoder_length = 72
training_cutoff = train_data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    train_data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="MERCHANT_1_NUMBER_OF_TRX",
    group_ids=["id"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["id"],
    time_varying_known_reals=["time_idx"],
    time_varying_known_categoricals=["hour", "month", "day_of_week"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["MERCHANT_1_NUMBER_OF_TRX"],
    target_normalizer=GroupNormalizer(
        groups=["id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,

)

model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)

model.load_state_dict(torch.load("model/tft_regressor.pt"))

for start in range(3200, 3700, 72):
    test_data = data[start:(start + max_encoder_length)]
    y_obs = data[(start + max_encoder_length): (start + max_encoder_length + max_prediction_length)]

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
        y_obs.set_index(pd.to_datetime(y_obs.date))["MERCHANT_1_NUMBER_OF_TRX"],
        color='orange',
        alpha=0.8,
        label='observed',
        linestyle='--',
        linewidth=1.2
    )

    plt.show()
