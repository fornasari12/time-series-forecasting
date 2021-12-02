from load_data import convert_tsf_to_dataframe
import pandas as pd

loaded_data,\
frequency,\
forecast_horizon,\
contain_missing_values,\
contain_equal_length = convert_tsf_to_dataframe(
    "/Users/nicolasfornasari/Downloads/m4_hourly_dataset.tsf"
)

for serie_name, df_serie in loaded_data.groupby(by="series_name"):
    start_timestamp = df_serie.start_timestamp.values[0]
    serie = list(df_serie.series_value)

    if frequency == "hourly":
        freq = "H"

    df = pd.DataFrame(
        data=serie).T

    index = pd.date_range(
        start=start_timestamp,
        freq=freq,
        periods=len(df)
    )

    df = df.set_index(index)
    df = df.reset_index()
    df.columns = ["date", "value"]

    df.to_csv(
        f"data/M4Hourly/{serie_name}.csv",
        index=False
    )

    print(serie_name)
