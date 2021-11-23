import pickle
import pandas as pd
import matplotlib.pyplot as plt

import torch
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, NBeats
from pytorch_forecasting.data import GroupNormalizer

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from config import load_config
from load_data import LoadData

spec = load_config("config.yaml")
DATA_PATH = spec["general"]["data_path"]
FOLDER_LIST = spec["general"]["folder_list"]
MODEL_PATH = spec["model"]["model_path"]
BATCH_SIZE = spec["model"]["batch_size"]
MAX_EPOCHS = spec["model"]["max_epochs"]
GPUS = spec["model"]["gpus"]
LEARNING_RATE = spec["model"]["learning_rate"]
HIDDEN_SIZE = spec["model"]["hidden_size"]
DROPOUT = spec["model"]["dropout"]
HIDDEN_CONTINUOUS_SIZE = spec["model"]["hidden_continuous_size"]
GRADIENT_CLIP_VAL = spec["model"]["gradient_clip_val"]
lags = spec["model"]["lags"]
sma = spec["model"]["sma"]
sma_columns = [f"sma_{sma}" for sma in sma]

if lags != "None":
    lags_columns = [f"(t-{lag})" for lag in range(lags, 0, -1)]

    time_varying_known_reals = (
            spec["model"]["time_varying_known_reals"] +
            lags_columns +
            sma_columns
    )
if lags == "None":
    lags = None
    time_varying_known_reals = (
            spec["model"]["time_varying_known_reals"] +
            sma_columns
    )
else:
    time_varying_known_reals = (spec["model"]["time_varying_known_reals"])

time_varying_known_categoricals = spec["model"]["time_varying_known_categoricals"]
max_prediction_length = spec["model"]["max_prediction_length"]
max_encoder_length = spec["model"]["max_encoder_length"]
sample = spec["model"]["sample"]
cutoff = spec["model"]["cutoff"]

# _________________________________________________________________________________________________________________
# Load Data for PyTorch Models:
train_data, test_data = LoadData(
    data_path=DATA_PATH,
    folder_list=FOLDER_LIST,
    cutoff=cutoff,
    sample="60min",
    date_features=True,
    sma=sma,
    lags=lags,
    time_idx=True,
).load_data()

# _________________________________________________________________________________________________________________
# Load N-BEATS Model & Scaler_dict:
with open("model/n_beats/n_beats.pickle", "rb") as f:
    model_n_beats = pickle.load(f)

with open("model/n_beats/scaler.pickle", "rb") as f:
    scaler_dict = pickle.load(f)

# _________________________________________________________________________________________________________________
# Load Temporal Fusion Transformer Model:
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

model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=1,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
)

model.load_state_dict(torch.load("/Volumes/GoogleDrive/My Drive/Colab_Notebooks/model/tft.pt"))

for data_name in train_data.id.unique().tolist():

    # _________________________________________________________________________________________________________________
    # Load Exponential Smoothing Model:
    with open(f"model/exponential_smoothing/{data_name}.pickle", "rb") as f:
        model_es = pickle.load(f)

    # _________________________________________________________________________________________________________________
    # Exponential Smoothing Data:
    test_data_es = test_data[
        test_data["id"] == data_name
    ][["date", "value"]].set_index("date", drop=True)["value"]

    test_data_es.index.freq = pd.infer_freq(test_data_es.index)

    # _________________________________________________________________________________________________________________
    # Temporal Fusion Transformer Data:
    test_data_tft = test_data[
        (test_data["id"] == data_name)
        ].reset_index(drop=True)

    # _________________________________________________________________________________________________________________
    # N-BEATS Data:
    test_data_nbeats = test_data[
        (test_data["id"] == data_name)
        ][["date", "value"]].reset_index(drop=True)

    scaler = scaler_dict[data_name]

    test_data_nbeats = scaler.fit_transform(
        TimeSeries.from_dataframe(
            df=test_data_nbeats,
            time_col="date"
        )
    )
    # _________________________________________________________________________________________________________________
    # Forecast:
    for start in range(0, 80, 1):

        # Update ES with new Data.
        model_es.update(
            test_data_es[start:(start + max_encoder_length)],
            update_params=False,
        )

        # Make forecast for es & tft:
        y_hat_es = model_es.predict(list(range(1, max_prediction_length + 1)))
        y_hat_tft = pd.Series(
            index=y_hat_es.index,
            data=model.predict(
                test_data_tft[start:(start + max_encoder_length)],
                mode="prediction",
                return_x=True)[0][0].tolist()
        )
        y_hat_nbeats = model_n_beats.predict(
            n=max_prediction_length,
            series=test_data_nbeats.slice_n_points_after(
                start_ts=start,
                n=max_encoder_length
            )
        )

        y_hat_nbeats = scaler.inverse_transform(y_hat_nbeats)
        y_hat_nbeats = y_hat_nbeats.pd_dataframe()

        # Plot forecasts and observed values
        ax = test_data_es[start: start + max_encoder_length + max_prediction_length].plot(
            figsize=(10, 6),
            marker="o",
            color="black",
            label="observed"
        )
        y_hat_es.plot(ax=ax, style="--", marker="o", color="red",
                      label="exponential_smoothing")
        y_hat_tft.plot(ax=ax, style="--", marker="o", color="blue",
                       label="temporal_fusion_transformer")
        y_hat_nbeats.plot(ax=ax, style="--", marker="o", color="green",
                       label="N=BEATS")

        plt.title(f"Forecasts for {data_name}")
        plt.pause(0.05)

        plt.show()
