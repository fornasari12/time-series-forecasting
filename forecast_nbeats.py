import pickle
import argparse
import pandas as pd

from darts import TimeSeries

from config import load_config
from load_data import LoadData

parser = argparse.ArgumentParser(description="Run Training Pipeline")
parser.add_argument(
    "-l",
    "--local",
    help="Run local or in colab",
    action='store_true',
    default=False,
)
args = parser.parse_args()

LOCAL = args.local

if LOCAL:
    model_key = "model_local"
else:
    model_key = "model"

spec = load_config("/content/temporal-fusion-transformer/config.yaml")

MODEL_PATH_NBEATS = "/content/drive/MyDrive/Colab_Notebooks/model/n_beats/n_beats.pickle"
SCALER_PATH_NBEATS = "/content/drive/MyDrive/Colab_Notebooks/model/n_beats/scaler.pickle"
DATA_PATH = "/content/temporal-fusion-transformer/" + spec["general"]["data_path"]
FOLDER_LIST = spec["general"]["folder_list"]

lags = spec[model_key]["lags"]
sma = spec[model_key]["sma"]
max_prediction_length = spec[model_key]["max_prediction_length"]
max_encoder_length = spec[model_key]["max_encoder_length"]
sample = spec[model_key]["sample"]
cutoff = spec[model_key]["cutoff"]

# _________________________________________________________________________________________________________________
# Load Data for PyTorch Models:
train_data, test_data = LoadData(
    data_path=DATA_PATH,
    folder_list=FOLDER_LIST,
    cutoff=cutoff,
    sample="60min",
    date_features=True,
    # sma=sma,
    # lags=None,
    # time_idx=True,
).load_data()

# _________________________________________________________________________________________________________________
# Load N-BEATS Model & Scaler_dict:

with open(MODEL_PATH_NBEATS, "rb") as f:
    model_n_beats = pickle.load(f)

with open(SCALER_PATH_NBEATS, "rb") as f:
    scaler_dict = pickle.load(f)


for data_name in train_data.id.unique().tolist():

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

        y_hat_nbeats = model_n_beats.predict(
            n=max_prediction_length,
            series=test_data_nbeats.slice_n_points_after(
                start_ts=start,
                n=max_encoder_length
            )
        )

        y_hat_nbeats = scaler.inverse_transform(y_hat_nbeats)
        y_hat_nbeats = y_hat_nbeats.pd_dataframe()

        y_hat_nbeats.to_csv(f"/content/drive/MyDrive/Colab_Notebooks/data_nbeats/{data_name}_{start}.csv")

