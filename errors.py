import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

errors = pd.read_csv("model/n_beats/errors.csv", index_col=0)

for data_name in [
    # "Twitter_volume_AAPL",
    "Twitter_volume_UPS",
    "Twitter_volume_KO",
    "Twitter_volume_GOOG",
    "Twitter_volume_CVS",
    "Twitter_volume_FB",
    "Twitter_volume_IBM",
    "Twitter_volume_CRM",
    "Twitter_volume_PFE",
    "Twitter_volume_AMZN"
]:
    errors = pd.read_csv(f"model/n_beats/errors_{data_name}.csv", index_col=0)
    error_metrics = pd.DataFrame()
    for step, df_step in errors.groupby(by="step"):

        print("------------------------------------------\n",
              f"Step {step}")

        df_step = df_step.dropna()

        for model in ["tft", "ets", "nbeats"]:

            y_obs = df_step["observed"].values
            y_hat = df_step[model].values

            mse = mean_squared_error(y_obs, y_hat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_obs, y_hat)

            df = pd.DataFrame(
                {
                    "model": model,
                     "mse": mse,
                     "rmse": rmse,
                     "mae": mae,
                },
                index=[step]
            )

            error_metrics = pd.concat([error_metrics, df], axis=0)

            print(
                f"model: {model}\n",
                f"mean_squared_error: {mse}\n",
                f"root_mean_squared_error: {rmse}\n",
                f"mean_absolute_error: {mae}\n",
            )

    error_metrics = error_metrics.reset_index().rename(columns={"index": "step"})

    ax = pd.Series(
        index=range(1, 25, 1),
        data=error_metrics[error_metrics["model"] == "tft"]["rmse"].values
    ).plot(
        figsize=(10, 6),
        style="--",
        marker="o",
        color="green",
        label="Temporal Fusion Transformer",
        legend=True,
    )

    pd.Series(
        index=range(1, 25, 1),
        data=error_metrics[error_metrics["model"] == "ets"]["rmse"].values
    ).plot(
        ax=ax,
        style="--",
        marker="o",
        color="red",
        legend=True,
        label="Triple Exponential Smoothing"
    )

    pd.Series(
        index=range(1, 25, 1),
        data=error_metrics[error_metrics["model"] == "nbeats"]["rmse"].values
    ).plot(
        ax=ax,
        style="--",
        marker="o",
        color="blue",
        legend=True,
        label="N-BEATS"
    )

    plt.title('RMSE for 1-24 forecasting horizon', fontsize=14)
    plt.xlabel('Horizon (steps ahead)', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.grid(True)
    plt.show()

    print(data_name)

