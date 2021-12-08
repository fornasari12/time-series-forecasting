import os
import warnings

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt

import torch
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
import flash
from flash.core.utilities.imports import example_requires
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData
from flash.core.integrations.pytorch_forecasting import convert_predictions

from config import load_config
from load_data import LoadData

example_requires("tabular")
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

    # train_data, test_data = LoadData(
    #     data_path=DATA_PATH,
    #     folder_list=FOLDER_LIST,
    #     cutoff=cutoff,
    #     sample=sample,
    #     date_features=False,
    #     sma=None,
    #     lags=None,
    #     time_idx=True
    # ).load_data()
    #
    # train_data["value"] = train_data["value"].astype(float)
    # test_data["value"] = test_data["value"].astype(float)

    timesteps = 600

    data = generate_ar_data(seasonality=10.0, timesteps=timesteps, n_series=1, seed=42)
    # data["static"] = 2
    data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
    data.series = data.series.astype(str).astype("category")

    max_encoder_length = 12
    max_prediction_length = 24

    cutoff = timesteps * 0.70
    train_data = data[data["time_idx"] <= cutoff]
    test_data = data[data["time_idx"] > cutoff]

    training_cutoff = data["time_idx"].max() - max_prediction_length

    datamodule = TabularForecastingData.from_data_frame(
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        # only unknown variable is "value" - and N-Beats can also not take any additional variables
        time_varying_unknown_reals=["value"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        train_data_frame=train_data[lambda x: x.time_idx <= training_cutoff],
        val_data_frame=train_data,
        batch_size=1024,
    )

    # 2. Build the task
    model = TabularForecaster(
        datamodule.parameters,
        backbone="n_beats",
        backbone_kwargs={"widths": [32, 512], "backcast_loss_ratio": 0.1},
    )

    # 3. Create the trainer and train the model
    trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count(), gradient_clip_val=0.01)
    trainer.fit(model, datamodule=datamodule)

    test_df = test_data[:12]
    # start_time_idx = test_df.time_idx.max() + 1
    #
    # start_date = test_df.date.max() + pd.Timedelta(days=1)
    # date_range = pd.date_range(start_date, periods=max_prediction_length, freq="D")
    # import numpy as np
    # test = pd.DataFrame(
    #     {
    #         "series": [0 for i in range(max_prediction_length)],
    #         "time_idx": [i for i in range(start_time_idx, start_time_idx + max_prediction_length)],
    #         "value": [1.52345 for i in range(max_prediction_length)],
    #         "date": date_range
    #     })

    # test.series = test.series.astype("category")
    # test.value = test.value.astype(float)

    # test = pd.concat([test_df, test]).reset_index(drop=True)
    #
    # test.series = test.series.astype("category")
    # test.value = test.value.astype(float)

    # 4. Generate predictions
    # predictions = model.predict(test_df)
    predictions = model.forward(test_df)
    predictions, inputs = convert_predictions(predictions)
    model.pytorch_forecasting_model.plot_interpretation(inputs, predictions, idx=0)
    plt.show()
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("model/tabular.pt")




    # import pickle
    # with open(f"model/n_beats/training.pickle", "wb") as f:
    #     pickle.dump(training, f)
    #
    # torch.save(net.state_dict(), "model/n_beats/n_beats.pt")
    #
    # # best_model_path = trainer.checkpoint_callback.best_model_path
    # # best_model = NBeats.load_from_checkpoint(best_model_path)
    # #
    # # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    # # predictions = best_model.predict(val_dataloader)
    # # (actuals - predictions).abs().mean()
    # #
    # # raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)
    # #
    # # best_model.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)
    # df = test_data[:max_encoder_length]

    # test_data.series = test_data.series.astype(str).astype("category")

    # test_slice = test_data[test_data["series"] == '0'][:max_encoder_length]
    #
    # testing = TimeSeriesDataSet(
    #     test_slice,
    #     time_idx="time_idx",
    #     target="value",
    #     # categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    #     group_ids=["series"],
    #     # only unknown variable is "value" - and N-Beats can also not take any additional variables
    #     time_varying_unknown_reals=["value"],
    #     max_encoder_length=max_encoder_length,
    #     max_prediction_length=max_prediction_length,
    # )
    # test = TimeSeriesDataSet.from_dataset(testing, test_slice)
    # test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    #
    # y_hat_tft = net.predict(
    #         test_data[test_data["series"] == '0'][:max_encoder_length],
    #         mode="prediction",
    #         return_x=True)
    #
    # print("a")
    #

