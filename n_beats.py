import os
import warnings

from config import load_config
from load_data import LoadData

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE

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

max_prediction_length = spec["model_local"]["max_prediction_length"]
max_encoder_length = spec["model_local"]["max_encoder_length"]
sample = spec["model_local"]["sample"]
cutoff = spec["model_local"]["cutoff"]

if __name__ == "__main__":

    train_data, test_data = LoadData(
        data_path=DATA_PATH,
        folder_list=FOLDER_LIST,
        cutoff=cutoff,
        sample=sample,
        date_features=False,
        sma=None,
        lags=None,
        time_idx=True
    ).load_data()

    train_data["value"] = train_data["value"].astype(float)
    test_data["value"] = test_data["value"].astype(float)

    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="value",
        # categorical_encoders={"id": NaNLabelEncoder().fit(train_data.id)},
        group_ids=["id"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        # allow_missing_timesteps=True,
    )
    training_cutoff = train_data["time_idx"].max() - max_prediction_length
    # create validation set (predict=True) which means to predict the
    # last max_prediction_length points in time for each series
    validation = TimeSeriesDataSet.from_dataset(training, train_data, min_prediction_idx=training_cutoff + 1)

    # create dataloaders for model
    batch_size = 512
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # configure network and trainer
    pl.seed_everything(42)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.01,
        callbacks=[early_stop_callback],
        limit_train_batches=30,
    )

    net = NBeats.from_dataset(
        training,
        learning_rate=0.01,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=1.0,
    )

    trainer.fit(
        net,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    import pickle
    with open(f"model/n_beats/training.pickle", "wb") as f:
        pickle.dump(training, f)

    torch.save(net.state_dict(), "model/n_beats/n_beats.pt")

    # best_model_path = trainer.checkpoint_callback.best_model_path
    # best_model = NBeats.load_from_checkpoint(best_model_path)
    #
    # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    # predictions = best_model.predict(val_dataloader)
    # (actuals - predictions).abs().mean()
    #
    # raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)
    #
    # best_model.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)
    df = test_data[:max_encoder_length]
    y_hat_tft = net.predict(
            df,
            mode="prediction",
            return_x=True)[0][0].tolist()

    print("a")


