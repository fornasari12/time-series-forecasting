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
            sample: str
    ):
        self.data_path = data_path
        self.folder_list = folder_list
        self.cutoff = cutoff
        self.sample = sample
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
                self._create_date_features(df=df)
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
