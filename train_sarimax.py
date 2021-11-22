import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
# from statsmodels.tsa.arima_model import ARIMA

warnings.filterwarnings("ignore")

data = pd.read_csv("data/poc.csv")

target = "MERCHANT_2_NUMBER_OF_TRX"
data = data[[
    target,
    "TIME"
]]
data = data.rename(columns={'TIME': 'date'})
# data["MERCHANT_1_NUMBER_OF_TRX"] = np.log(data["MERCHANT_1_NUMBER_OF_TRX"] + 1e-8)

# add datetime variables
data["month"] = pd.to_datetime(data.date).dt.month
data["day_of_week"] = pd.to_datetime(data.date).dt.dayofweek
data["hour"] = pd.to_datetime(data.date).dt.hour
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")

# cut atypical values at the end of the sample
start = 3000 - 6*24
train_data = data[:start]
test_data = data[start + 1:3600]
max_prediction_length = 24*15

AR, I, MA = 5, 1, 5
order = (AR, I, MA)

SAR, SI, SMA, seas = 1, 0, 1, 24
Sorder = (SAR, SI, SMA, seas)

endog = train_data[target]

exog = train_data.iloc[:, 1:]

model = sm.tsa.statespace.SARIMAX(endog=endog,
                                  exog=exog,
                                  order=order,
                                  seasonal_order=Sorder,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False,
                                  trend='c')

res = model.fit(disp=True)

exog_graph = None if exog is None else test_data[:max_prediction_length]

fig, ax = plt.subplots(figsize=(7, 3))

train_data[target][-200:].plot(ax=ax)
test_data[target][:100].plot(ax=ax)

# Construct the forecasts
fcast = res.get_forecast(
    steps=max_prediction_length,
    exog=exog_graph.iloc[:, 1:]
).summary_frame()
fcast['mean'].plot(ax=ax, style='k--')
# ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)
plt.show()
