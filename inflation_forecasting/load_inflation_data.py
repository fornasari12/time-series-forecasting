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

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for group, df_group in result.groupby(by="id"):

        train_split = int(len(df_group) * cutoff)

        df_group["time_idx"] = list(range(len(df_group)))

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