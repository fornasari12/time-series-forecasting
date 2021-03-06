from os import listdir
from os.path import isfile, join
import warnings

import pandas as pd
import matplotlib.pyplot as plt

import torch
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import load_config
from load_data import LoadData

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

lags = spec["model_local"]["lags"]
sma = spec["model_local"]["sma"]
sma_columns = [f"sma_{sma}" for sma in sma]

if lags != "None":
    lags_columns = [f"(t-{lag})" for lag in range(lags, 0, -1)]

    time_varying_known_reals = (
            spec["model_local"]["time_varying_known_reals"] +
            lags_columns +
            sma_columns
    )
if lags == "None":
    lags = None
    time_varying_known_reals = (
            spec["model_local"]["time_varying_known_reals"] +
            sma_columns
    )

time_varying_known_categoricals = spec["model_local"]["time_varying_known_categoricals"]

max_prediction_length = spec["model_local"]["max_prediction_length"]
max_encoder_length = spec["model_local"]["max_encoder_length"]
sample = spec["model_local"]["sample"]
cutoff = spec["model_local"]["cutoff"]

train_data, test_data = LoadData(
    data_path=DATA_PATH,
    folder_list=FOLDER_LIST,
    cutoff=cutoff,
    sample=sample,
    sma=sma,
    lags=lags
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

model.load_state_dict(torch.load("model/tft_regressor_test.pt"))

for folder in FOLDER_LIST:

    print(folder)
    folder_path = f"{DATA_PATH}/{folder}"
    file_list = [
        f for f in listdir(folder_path) if isfile(join(folder_path, f))
    ]

    for file in file_list:
        file = file.replace(".csv", "")

        df = test_data[
            (test_data["dataset"] == folder) &
            (test_data["id"] == file)
        ].reset_index(drop=True)

        start = 0
        test_data_df = df[start:(start + max_encoder_length)]
        y_obs = df[(start + max_encoder_length): (start + max_encoder_length + max_prediction_length)]

        y_hat = model.predict(
            test_data_df,
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
            y_obs.set_index(pd.to_datetime(y_obs.date))["value"],
            color='orange',
            alpha=0.8,
            label='observed',
            linestyle='--',
            linewidth=1.2
        )

        plt.title(f'{file}')
        plt.legend()
        plt.show()
