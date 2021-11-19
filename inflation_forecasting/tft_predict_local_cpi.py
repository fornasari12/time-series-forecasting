import warnings

import pandas as pd
import matplotlib.pyplot as plt

import torch
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import load_config
from load_inflation_data import load_inflation_data

warnings.filterwarnings("ignore")

spec = load_config("config.yaml")
DATA_PATH = spec["general"]["data_path"]
CSV_NAME = spec["general"]["csv_name"]
CSV_LIST = spec["general"]["csv_list"]
weights = spec["general"]["weights"]

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
lags_columns = [f"(t-{lag})" for lag in range(lags, 0, -1)]
sma_columns = [f"sma_{sma}" for sma in sma]
time_varying_known_reals = (
        spec["model_local"]["time_varying_known_reals"]
)

time_varying_known_categoricals = spec["model_local"]["time_varying_known_categoricals"]

max_prediction_length = spec["model_local"]["max_prediction_length"]
max_encoder_length = spec["model_local"]["max_encoder_length"]
cutoff = spec["model_local"]["cutoff"]

train_data, test_data = load_inflation_data(
    data_path=DATA_PATH,
    csv_name=CSV_NAME,
    csv_list=CSV_LIST,
    cutoff=cutoff
)

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

model.load_state_dict(torch.load(MODEL_PATH))

recontstructed_cpi = pd.DataFrame()
for group in test_data["id"].unique():

    df = test_data[
        (test_data["id"] == group)
    ].reset_index(drop=True)

    start = 0
    test_data_df = df[start:(start + max_encoder_length)]
    y_obs = df[(start + max_encoder_length): (start + max_encoder_length + max_prediction_length)]

    y_hat = model.predict(
        test_data_df,
        mode="prediction",
        return_x=True
    )[0][0]

    if group != "cpi":
        group_weight = weights[group] if group != "cpi" else None
        # y_hat_cpi = y_hat * group_weight / 100

        y_hat_cpi = pd.Series(
            index=y_obs["date"],
            data=y_hat * group_weight / 100
        )

        recontstructed_cpi[group] = y_hat_cpi

    if group == "cpi":

        y_hat_cpi = pd.Series(
            index=y_obs["date"],
            data=y_hat
        )

        recontstructed_cpi[group] = y_hat_cpi
        recontstructed_cpi["cpi_observed"] = pd.Series(
            data=y_obs["value"].to_numpy(),
            index=y_obs["date"]
        )

    fig, ax = plt.subplots()

    ax.plot(
        pd.Series(data=y_hat.tolist(),
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

    plt.title(f'{group}')
    plt.legend()

plt.show()

columns_for_construct = [
    'alim_beb_no_alc',
    'beb_alc_tab_estup',
    'bienes_servicios_diversos',
    'comunicaciones',
    'educacion',
    'muebles_art_hogar',
    'recreacion_cultura',
    'restaurantes_hoteles',
    'ropa_calzado',
    'salud',
    'transporte',
    'vivienda'
]

recontstructed_cpi["new_cpi"] = recontstructed_cpi[columns_for_construct].sum(axis=1)

print("a")

