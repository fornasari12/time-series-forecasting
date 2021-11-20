import pandas as pd
from utils import load_econuy_data


def load_inflation_data(
        data_path: str,
        csv_name: str,
        csv_list: [list, str],
        cutoff: float
):

    df = pd.read_csv(data_path + csv_name)
    df = df.set_index(pd.to_datetime(df["date"]))
    df = df.drop(columns="date")
    df = df.astype(float)
    df = df.stack().reset_index()
    df = df.rename(
        columns={
            'level_1': 'id',
            0: 'value'
        }
    )
    df = df.set_index("date")

    df_exog = pd.DataFrame()
    for csv_name in csv_list:
        print(csv_name)

        df_econuy = load_econuy_data(
            data_path=data_path,
            csv_name=csv_name
        )

        df_exog = pd.concat([df_exog, df_econuy], axis=1)

    df_exog = df_exog[
        (df_exog.index >= df.index.min()) &
        (df_exog.index <= df.index.max())
    ]

    df_exog = df_exog.reset_index()
    df = df.reset_index()

    result = df.merge(df_exog, on="date", how='outer').sort_values("date")

    relevant_vars = [
        'date', 'id', 'value', 'ims_pr',
        # 'meat', 'soy', 'milk',
        # 'rice', 'wool', 'wheat',
        'Argentina', 'Brasil',
        'er_eop', 'er_aop',
        # 'er_ar_official', 'er_ar_blue', 'er_br',
        # 'unemployment_rate'
    ]

    result = result[relevant_vars]
    result = result[
        (result["date"] <= pd.to_datetime("2021-06-30 00:00:00"))
        # & (result["date"] >= pd.to_datetime("2002-01-31 00:00:00"))
    ].fillna(method="bfill")

    result["month"] = result["date"].dt.month.astype(str).astype("category")

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for group, df_group in result.groupby(by="id"):

        train_split = int(len(df_group) * cutoff)

        df_group["time_idx"] = list(range(len(df_group)))

        for column in [
            'value', 'ims_pr',
            # 'meat', 'soy', 'milk',
            # 'rice', 'wool', 'wheat',
            'Argentina', 'Brasil',
            'er_eop', 'er_aop',
            # 'er_ar_official', 'er_ar_blue', 'er_br',
            # 'unemployment_rate'
        ]:

            df_group[column] = (
                    df_group[column].pct_change()*100
            ).fillna(method="bfill")

            for lag in [1, 6, 12]:
                df_group[f"{column}_lag{lag}"] = (
                    df_group[column].shift(lag).fillna(method="bfill")
                )

        df_train = df_group.iloc[:train_split, :]
        df_test = df_group.iloc[train_split:, :]

        train_data = pd.concat(
            objs=[train_data, df_train],
            axis=0
        )
        test_data = pd.concat(
            objs=[test_data, df_test],
            axis=0
        )

    return train_data, test_data
