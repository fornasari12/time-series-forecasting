import pandas as pd

DATA_PATH = "../data/inflation/"
CSV_NAME = "commodities.csv"

df_commodities = pd.read_csv(
    DATA_PATH + CSV_NAME,
    infer_datetime_format=True
).iloc[8:, :].rename(columns={"Indicador": "date"})

df_commodities = df_commodities.iloc[:, 1:].astype(float)

CSV_NAME = "cpi_br_arg.csv"

df_cpi_br_arg = pd.read_csv(
    DATA_PATH + CSV_NAME,
    infer_datetime_format=True
).iloc[8:, :].rename(columns={"Indicador": "date"})

df_cpi_br_arg = df_cpi_br_arg.iloc[:, 1:].astype(float)


print("A")