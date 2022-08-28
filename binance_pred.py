import os
import websocket, json

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