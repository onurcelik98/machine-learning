import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pmdarima as pm
from sklearn.metrics import mean_squared_error

def get_stationarity(timeseries):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(timeseries['Passengers'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


df = pd.read_csv("sample_time_series.csv", parse_dates = ['Month'], index_col = ['Month'])
print(df.head())
plt.xlabel('Date')
plt.ylabel('Value')
plt.plot(df)
plt.show()


# model = pm.auto_arima(df.Passengers, start_p=1, start_q=1,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=10, max_q=10, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=False,   # No Seasonality
#                       start_P=0,
#                       D=0,
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=True)
#
# print(model.summary())
#
get_stationarity(df)



#%%
df_log = np.log(df)
plt.plot(df_log)
plt.show()
rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)
get_stationarity(df_log_minus_mean)

plt.plot(df + 10)
plt.plot(np.exp(df_log_minus_mean + rolling_mean))
plt.show()

#%%
train = df_log_minus_mean[:120]
test = df_log_minus_mean[120:]

decomposition = seasonal_decompose(df_log)
model = ARIMA(train, order=(2, 0, 1))
results = model.fit()
plt.plot(train)
plt.plot(results.fittedvalues, color='red')
plt.show()


#%%
fc = results.forecast(15, alpha=0.05)
fc_series = pd.Series(fc, index=test.index)
plt.plot(test)
plt.plot(fc_series)
plt.show()
rolling_mean_s = rolling_mean["Passengers"]

plt.plot(np.exp(train + rolling_mean))
plt.plot(np.exp(rolling_mean))
plt.plot(np.exp(test + rolling_mean))
plt.plot(np.exp(fc_series + rolling_mean_s))
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend(["Train data", "Rolling mean", "Test data", "Forecast"])
plt.show()


rmse = np.sqrt(mean_squared_error(test, fc_series))
print(rmse)

# #%%
# predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
# print(predictions_ARIMA_diff)
#
# #%%
# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# print(predictions_ARIMA_diff_cumsum)
#
# #%%
# predictions_ARIMA_log = pd.Series(df_log['Passengers'].iloc[0], index=df_log.index)
# print(predictions_ARIMA_log)
#
# #%%
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
# print(predictions_ARIMA_log)
#
# #%%
# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# print(predictions_ARIMA)
#
# #%%
# plt.plot(df)
# plt.plot(predictions_ARIMA)
# plt.show()
