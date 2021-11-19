import pandas as pd


def load_econuy_data(
        data_path: str,
        csv_name: str,
):
    df = pd.read_csv(
        data_path + csv_name,
        infer_datetime_format=True
    ).iloc[8:, :].rename(columns={"Indicador": "date"})

    df = df.set_index(
        pd.to_datetime(df["date"])
    )
    df = df.drop(columns="date")
    df = df.astype(float)

    if csv_name == "commodities.csv":
        df.columns = [
            "meat", "celulose", "soy", "milk", "rice", "wood",
            "wool", "barey", "gold", "wheat"
        ]

    if csv_name == "unemployment_rate_uy.csv":
        df = df.rename(
            columns={"Tasa de desempleo: total": "unemployment_rate"}
        )

    if csv_name == "nominal_wages_uy.csv":
        df = df.rename(
            columns={"�ndice medio de salarios privados": "ims_pr"}
        )
    if csv_name == "exchange_rate_uy.csv":
        df = df.rename(
            columns={"Tipo de cambio venta, fin de per�odo": "er_eop",
                     "Tipo de cambio venta, promedio": "er_aop"
                     }
        )

    if csv_name == "exchange_rate_br_arg.csv":
        df = df.rename(
            columns={"Argentina - oficial": "er_ar_official",
                     "Argentina - informal": "er_ar_blue",
                     "Brasil": "er_br"
                     }
        )

        df = df.resample("M").sum()

    return df