import argparse
import pickle

from darts.models import NBEATSModel

from load_data import LoadData
from config import load_config

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
    model_key = "nbeats_local"
    spec = load_config("config.yaml")
    DATA_PATH = spec["general"]["data_path"]
else:
    model_key = "nbeats"
    spec = load_config("/content/temporal-fusion-transformer/config.yaml")
    DATA_PATH = "/content/temporal-fusion-transformer/" + spec["general"]["data_path"]

FOLDER_LIST = spec["general"]["folder_list"]
MODEL_PATH = spec[model_key]["model_path"]
SCALER_PATH = spec[model_key]["scaler_path"]
INPUT_CHUNK_LENGTH = spec[model_key]["input_chunk_length"]
OUTPUT_CHUNK_LENGTH = spec[model_key]["output_chunk_length"]
NUM_STACKS = spec[model_key]["num_stacks"]
NUM_BLOCKS = spec[model_key]["num_blocks"]
NUM_LAYERS = spec[model_key]["num_layers"]
LAYER_WIDTHS = spec[model_key]["layer_widths"]
N_EPOCHS = spec[model_key]["n_epochs"]
NR_EPOCHS_VAL_PERIOD = spec[model_key]["nr_epochs_val_period"]
BATCH_SIZE = spec[model_key]["batch_size"]
MODEL_NAME = spec[model_key]["model_name"]
sample = spec[model_key]["sample"]
cutoff = spec[model_key]["cutoff"]

series_dict = LoadData(
    data_path=DATA_PATH,
    folder_list=FOLDER_LIST,
    cutoff=cutoff,
    sample=sample,
).load_data_darts(
    scaler_path=SCALER_PATH,
)

model_nbeats = NBEATSModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    generic_architecture=True,
    num_stacks=NUM_STACKS,
    num_blocks=NUM_BLOCKS,
    num_layers=NUM_LAYERS,
    layer_widths=LAYER_WIDTHS,
    n_epochs=N_EPOCHS,
    nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
    batch_size=BATCH_SIZE,
    model_name=MODEL_NAME,
    force_reset=True
)

series_train_list = [series_dict[serie][0] for serie in series_dict.keys()]
series_test_list = [series_dict[serie][1] for serie in series_dict.keys()]

model_nbeats.fit(series_train_list, verbose=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_nbeats, f)
