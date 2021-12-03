import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

from config import load_config
from load_data import LoadData

# https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.forecasting.exp_smoothing.ExponentialSmoothing.html#sktime.forecasting.exp_smoothing.ExponentialSmoothing.update


PLOT = True
spec = load_config("config.yaml")
DATA_PATH = spec["general"]["data_path"]
FOLDER_LIST = spec["general"]["folder_list"]
sample = spec["model_local"]["sample"]
cutoff = spec["model_local"]["cutoff"]
max_prediction_length = spec["model_local"]["max_prediction_length"]

train_data, test_data = LoadData(
    data_path=DATA_PATH,
    folder_list=FOLDER_LIST,
    cutoff=cutoff,
    sample="60min",
    date_features=False,
    sma=None,
    lags=None,
    time_idx=False,
).load_data(
    min_obs=700,
    reduce_memory=["cat", "float", "int"]
)

for data_name in train_data.id.unique().tolist():

    train_data_id = train_data[
        train_data["id"] == data_name
    ][["date", "value"]].set_index("date", drop=True)["value"]

    test_data_id = test_data[
        test_data["id"] == data_name
    ][["date", "value"]].set_index("date", drop=True)["value"]

    train_data_id[train_data_id == 0] = 1e-8
    train_data_id.index.freq = pd.infer_freq(train_data_id.index)
    test_data_id.index.freq = pd.infer_freq(test_data_id.index)

    fit = ExponentialSmoothing(
        sp=96,
        trend="add",
        seasonal="add",
        use_boxcox=True,
        initialization_method="estimated",
    ).fit(train_data_id)

    with open(f"model/exponential_smoothing/{data_name}.pickle", "wb") as f:
        pickle.dump(fit, f)

    if PLOT:
        ax = train_data_id[-48:].plot(
            figsize=(10, 6),
            marker="o",
            color="black",
            title=f"Forecasts for {data_name}",
        )

        fit.predict(list(range(max_prediction_length))).rename("Holt-Winters (add-add-seasonal)").plot(
            ax=ax,
            style="--",
            marker="o",
            color="red",
            legend=True
        )

        test_data_id[:max_prediction_length].plot(
            ax=ax,
            figsize=(10, 6),
            marker="o",
            color="blue",
            title=f"Forecasts for {data_name}")

        plt.show(block=False)
        plt.pause(0.000005)
        plt.close()




