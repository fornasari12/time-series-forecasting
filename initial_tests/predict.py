import torch
import pandas as pd

from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

data = pd.read_csv("data/MERCHANT_NUMBER_OF_TRX.csv")
data = data[["MERCHANT_1_NUMBER_OF_TRX", "date"]]
data["id"] = "M1"

# add time index
data["time_idx"] = pd.to_datetime(data.date).astype(int)
data["time_idx"] -= data["time_idx"].min()
data["time_idx"] = (data.time_idx / 3600000000000) + 1
data["time_idx"] = data["time_idx"].astype(int)
data = data[:3840]

max_prediction_length = 24
max_encoder_length = 72
training_cutoff = data["time_idx"].max() - max_prediction_length

test_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="MERCHANT_1_NUMBER_OF_TRX",
    group_ids=["id"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["id"],
    time_varying_known_reals=["time_idx"],
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
model.load_state_dict(torch.load("model/tft.pt"))
# model.eval()

new_raw_predictions, new_x = model.predict(
    test_data,
    mode="raw",
    return_x=True
)

model.plot_prediction(
    new_x,
    new_raw_predictions,
    idx=0,
    show_future_observed=False
)