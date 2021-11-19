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
sample = spec["model"]["sample"]
cutoff = spec["model"]["cutoff"]

train_data, test_data = LoadData(
    data_path=DATA_PATH,
    folder_list=FOLDER_LIST,
    cutoff=cutoff,
    sample=sample
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
