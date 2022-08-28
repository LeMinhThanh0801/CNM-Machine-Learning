import os
import websocket, json
from re import T
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import rcParams
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime

import requests
from binance.client import Client
import pandas as pd

import matplotlib.pyplot as plt
root_url = 'https://api.binance.com/api/v1/klines'
def get_bars(symbol, interval = '1m'):
   url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&limit=1000'
   data = json.loads(requests.get(url).text)
   df = pd.DataFrame(data)
   df.columns = ['open_time',
                 'o', 'h', 'l', 'c', 'v',
                 'close_time', 'qav', 'num_trades',
                 'taker_base_vol', 'taker_quote_vol', 'ignore']
   df.index = [dt.datetime.fromtimestamp(x//1000.0) for x in df.close_time]
   return df
btcusdt = get_bars('BTCUSDT')

df0=pd.DataFrame(btcusdt)
df0.to_csv('_btcusdt.csv')


df = pd.read_csv("_btcusdt.csv")
df.head()
df["close_time"] = pd.to_datetime(df.close_time, format="%Y-%m-%d")
df.index = df['close_time']



data = df.sort_index(ascending=True, axis=0)
# new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=[
                           'close_time', 'c'])

for i in range(0, len(data)):
    new_dataset["close_time"][i] = data['c'][i]
    # new_dataset["Close"][i] = data["Close"][i]
    new_dataset["c"][i] = data["c"][i]


new_dataset.index = new_dataset.close_time
new_dataset.drop("close_time", axis=1, inplace=True)

final_dataset = new_dataset.values

train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]

# print('train_data: ', train_data)
# print('len train: ', len(train_data))
# print("-----------------------------------------")
# print('len valid: ', len(valid_data))
# print('valid_data: ', valid_data)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

print( scaled_data)
print(len(train_data))
x_train_data, y_train_data = [], []

for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

x_train_data = np.reshape(
    x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True,
               input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))


lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)
print("input data shape[0]: ", inputs_data.shape[0])

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

lstm_model.save("saved_model_rate_of_change.h6")


train_data = new_dataset[:987]
valid_data = new_dataset[987:]
valid_data['Predictions'] = predicted_closing_price
# print('valid_data: ', valid_data)
# print("valid_data: ", valid_data.index[0])
# print("valid_data: ", (valid_data.index[0] + 1))

# plt.plot(train_data["Close"])
# plt.plot(valid_data[['Close', "Predictions"]])
plt.plot(valid_data[['c', "Predictions"]])
plt.legend()
plt.show()
