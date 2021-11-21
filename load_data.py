from os import listdir
from os.path import isfile, join
import pandas as pd


class LoadData:
    """
    This Class will handle and load CSV files into the
    PyTorch TimeSeriesDataSet

    """
    def __init__(
            self,
            data_path: str,
            folder_list: str,
            cutoff: float,
            sample: str,
            date_features: bool = True,
            sma: [list, int] = None,
            lags: int = None,
            time_idx: bool = True,
    ):
        self.data_path = data_path
        self.folder_list = folder_list
        self.cutoff = cutoff
        self.sample = sample
        self.date_features = date_features
        self.sma = sma
        self.lags = lags
        self.time_idx = time_idx
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

    def get_file_list(self, folder: str):

        folder_path = f"{self.data_path}/{folder}"
        file_list = [
            f for f in listdir(folder_path) if isfile(join(folder_path, f))
        ]

        return file_list

    def _load_dataframe(
            self,
            folder: str,
            file: str
    ):

        file_path = f"{self.data_path}/{folder}/{file}"
        df = pd.read_csv(file_path)

        # FIXME: Auto detect date column
        df = df.rename(columns={"timestamp": "date"})

        return df

    @staticmethod
    def _resample_df(
            df: pd.DataFrame,
            sample: str
    ):

        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.resample(sample).sum()
        df = df.reset_index(drop=False)

        return df

    @staticmethod
    def _create_id_columns(
            df: pd.DataFrame,
            folder: str,
            file: str):

        df["dataset"] = folder
        df["id"] = file.replace(".csv", "")

    @staticmethod
    def _create_date_features(df: pd.DataFrame):

        df["date"] = pd.to_datetime(df["date"])

        df["month"] = pd.to_datetime(df.date).dt.month \
            .astype(str) \
            .astype("category")
        df["day_of_week"] = pd.to_datetime(df.date).dt.dayofweek \
            .astype(str) \
            .astype("category")
        df["hour"] = pd.to_datetime(df.date).dt.hour \
            .astype(str) \
            .astype("category")
        df["day"] = pd.to_datetime(df.date).dt.day \
            .astype(str) \
            .astype("category")
        df["weekofyear"] = pd.to_datetime(df.date).dt.isocalendar().week \
            .astype(str) \
            .astype("category")

    @staticmethod
    def _create_moving_average(
            serie: pd.Series,
            window: int
    ):
        serie_sma = serie.rolling(window=window).mean()

        return serie_sma

    @staticmethod
    def _create_lagged_variables(
            df: pd.DataFrame,
            column_name: str,
            lags: int
    ):

        df_lags = pd.DataFrame()
        for lag in range(lags, 0, -1):
            df_lags[f'(t-{lag})'] = \
                df[column_name].shift(lag).fillna(method="bfill")

        return pd.concat([df_lags, df], axis=1)

    @staticmethod
    def _create_time_idx(df: pd.DataFrame):
        df["time_idx"] = list(range(len(df)))

    @staticmethod
    def _train_test_split(
            df,
            train_split
    ):
        df_train = df.iloc[:train_split, :]
        df_test = df.iloc[train_split:, :]

        print(len(df_train), len(df_test))

        return df_train, df_test

    def _concatenate_df(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame
    ):
        self.train_data = pd.concat(
            objs=[self.train_data, df_train],
            axis=0
        )
        self.test_data = pd.concat(
            objs=[self.test_data, df_test],
            axis=0
        )

        self.train_data = self.train_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)

    def load_data(self):

        for folder in self.folder_list:

            file_list = self.get_file_list(
                folder=folder
            )

            for file in file_list:
                df = self._load_dataframe(folder=folder, file=file)
                df = self._resample_df(df=df, sample=self.sample)

                train_split = int(len(df) * self.cutoff)

                self._create_id_columns(
                    df=df,
                    folder=folder,
                    file=file
                )

                if self.date_features:
                    self._create_date_features(df=df)

                if self.sma:

                    for window in self.sma:
                        df[f"sma_{window}"] = self._create_moving_average(
                            serie=df["value"],
                            window=window
                        ).fillna(method="bfill")

                if self.lags:
                    df = self._create_lagged_variables(
                        df=df,
                        column_name="value",
                        lags=self.lags
                    )

                if self.time_idx:
                    self._create_time_idx(df=df)

                df_train, df_test = self._train_test_split(
                    df=df,
                    train_split=train_split
                )

                self._concatenate_df(
                    df_train=df_train,
                    df_test=df_test
                )

        return self.train_data, self.test_data
