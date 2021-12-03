from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
from datetime import datetime
from numpy import distutils
import distutils.util
import numpy as np


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

        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    df[column] = pd.to_datetime(df[column])
                except ValueError:
                    continue

        # Get name of all datetime columns:
        datetime_columns = list(
            df.select_dtypes(include=['datetime64']).columns
        )

        assert len(datetime_columns) == 1, \
            "There could only be one datetime column, please check!"

        datetime_column = datetime_columns[0]
        df = df.rename(columns={datetime_column: "date"})

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
    def _create_date_features(
            df: pd.DataFrame,
    ):

        df["date"] = pd.to_datetime(df["date"])

        df["month"] = df["date"].dt.month.astype(str)
        df["day_of_week"] = df["date"].dt.dayofweek.astype(str)
        df["hour"] = df["date"].dt.hour.astype(str)
        df["day"] = df["date"].dt.day.astype(str)
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(str)

        return df

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
            df_lags[f'lag_{lag}'] = \
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

    @staticmethod
    def _reduce_memory_usage(
            df: pd.DataFrame,
            reduce_memory: [str, list]
    ):

        category = True if "cat" in reduce_memory else False
        floater = True if "float" in reduce_memory else False
        integer = True if "int" in reduce_memory else False

        for column in df.columns:

            if df[column].dtype == object and category:
                df[column] = df[column].astype(str).astype("category")

            if df[column].dtype == "float64" and floater:
                df[column] = pd.to_numeric(df[column], downcast="float")

            if df[column].dtype == "int64" and integer:
                df[column] = pd.to_numeric(df[column], downcast="integer")

        return df

    def load_data(
            self,
            min_obs: int,
            reduce_memory: [str, list]
    ):

        for folder in self.folder_list:

            file_list = self.get_file_list(
                folder=folder
            )

            for file in file_list:
                df = self._load_dataframe(folder=folder, file=file)
                df = self._resample_df(df=df, sample=self.sample)

                if len(df) < min_obs:
                    print(
                        f"{folder} - {file}: Not enough obs. ({len(df)})"
                    )
                    continue

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

        self.train_data = self._reduce_memory_usage(
            self.train_data,
            reduce_memory=reduce_memory
        )

        self.test_data = self._reduce_memory_usage(
            self.test_data,
            reduce_memory=reduce_memory
        )

        return self.train_data, self.test_data

    def load_data_darts(
            self,
            scaler_path,
    ):

        from darts import TimeSeries
        from darts.dataprocessing.transformers import Scaler

        scaler_dict = {}
        series_dict = {}

        for folder in self.folder_list:

            file_list = self.get_file_list(
                folder=folder
            )

            for file in file_list:
                df = self._load_dataframe(folder=folder, file=file)
                df = self._resample_df(df=df, sample=self.sample)

                df.value = df.value.astype(np.float32)

                scaler = Scaler()
                serie = scaler.fit_transform(
                    TimeSeries.from_dataframe(df=df, time_col="date")
                )
                train_split = int(len(serie.pd_dataframe()) * self.cutoff)

                train, test = serie.split_after(train_split)

                scaler_dict[file.replace(".csv", "")] = scaler
                series_dict[file.replace(".csv", "")] = [train, test]

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_dict, f)

        return series_dict


def convert_tsf_to_dataframe(
        full_file_path_and_name,
        replace_missing_vals_with='NaN',
        value_column_name="series_value"
):

    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:
                                raise Exception(
                                    "Invalid meta-data specification."
                                )

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:
                                raise Exception(
                                    "Invalid meta-data specification."
                                )

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    distutils.util.strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(
                                    distutils.util.strtobool(line_content[1])
                                )

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section "
                                "must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. "
                            "Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception(
                                "Missing attributes/values in series."
                            )

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma "
                                "separated numeric values. At least one numeric"
                                " value should be there in a series. Missing "
                                "values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(
                                replace_missing_vals_with
                        ) == len(numeric_series):

                            raise Exception(
                                "All series values are missing. A given series"
                                " should contains a set of comma separated "
                                "numeric values. At least one numeric value "
                                "should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i],
                                    '%Y-%m-%d %H-%M-%S'
                                )
                            else:
                                raise Exception("Invalid attribute type.")

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return loaded_data, frequency, forecast_horizon,  \
            contain_missing_values, contain_equal_length
