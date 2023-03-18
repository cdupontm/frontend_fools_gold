import streamlit as st
import joblib
import os


import pandas as pd
import math
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


from PIL import Image

# Confit
st.set_page_config(page_title='Gold Price Prediction Tool', page_icon=':bar_chart:', layout='wide')


#opening the image
app_path=os.path.dirname(__file__)
root_path=os.path.dirname(app_path)
#image_path=os.path.join(root_path,'raw_data','shutterstock_1797061261-1.jpg')
image_path=os.path.join(root_path,'raw_data','gold_banner.jpg')
image = Image.open(image_path)
gold_banner = image.resize((1200, 200))




st.image(gold_banner, caption='')
st.info('**Gold Price Predictions**', icon="🧠")

model_name= st.sidebar.selectbox('Select a model', ( 'Decision Tree Regressor', 'Lasso','Sarima'))

# To make a dataframe from today's price
# Gold price (target)
df_gold = pd.DataFrame(yf.download('GC=F', period = '1'))
df_gold.columns= ['Open gold', 'High gold', 'Low gold', 'Close gold', 'Adj Close gold', 'Volume gold']
#Gold producer ETFs
df_ring = pd.DataFrame(yf.download('RING', period = '1'))
df_ring.columns= ['Open RING', 'High RING', 'Low RING', 'Close RING', 'Adj Close RING', 'Volume RING']
#Oil Price
df_brent = pd.DataFrame(yf.download('CL=F', period = '1'))
df_brent.columns= ['Open Brent', 'High Brent', 'Low Brent', 'Close Brent', 'Adj Close Brent', 'Volume Brent']
df_silver = pd.DataFrame(yf.download('SI=F', period = '1'))
df_silver.columns= ['Open Silver', 'High Silver', 'Low Silver', 'Close Silver', 'Adj Close Silver', 'Volume Silver']
df_palladium = pd.DataFrame(yf.download('PA=F', period = '1'))
df_palladium.columns= ['Open Palladium', 'High Palladium', 'Low Palladium', 'Close Palladium', 'Adj Close Palladium', 'Volume Palladium']
# Golden ocean share price as a proxy for dry bulk freight costs
df_freight = pd.DataFrame(yf.download('GOGL', period = '1'))
df_freight.columns= ['Open Freight', 'High Freight', 'Low Freight', 'Close Freight', 'Adj Close Freight', 'Volume Freight']
# Canadian stock market index
df_ca_stocks = pd.DataFrame(yf.download('^GSPTSE', period = '1'))
df_ca_stocks.columns= ['Open CA', 'High CA', 'Low CA', 'Close CA', 'Adj Close CA', 'Volume CA']
# 10y treasury yield
df_tnx = pd.DataFrame(yf.download('^TNX', period = '1'))
df_tnx.columns= ['Open TNX', 'High TNX', 'Low TNX', 'Close TNX', 'Adj Close TNX', 'Volume TNX']
plat_data = yf.download('PL=F',period = '1')
plat_data.columns= ["Open PLAT", "High PLAT", "Low PLAT", "Close PLAT", "Adj Close PLAT", "Volume PLAT"]
copper_data = yf.download('HG=F', period = '1')
copper_data.columns= ["Open Copper", "High Copper", "Low Copper", "Close Copper", "Adj Close Copper", "Volume Copper"]
aluminum_data = yf.download('ALI=F', period = '1')
aluminum_data.columns= ["Open ALI", "High ALI", "Low ALI", "Close ALI", "Adj Close ALI", "Volume ALI"]
spx_data = yf.download('^GSPC', period = '1')
spx_data.columns= ["Open SPX", "High SPX", "Low SPX", "Close SPX", "Adj Close SPX", "Volume SPX"]
nasdaq_data = yf.download('^IXIC', period = '1')
nasdaq_data.columns= ["Open NDQ", "High NDQ", "Low NDQ", "Close NDQ", "Adj Close NDQ", "Volume NDQ"]
btc_data = yf.download('BTC-USD', period = '1')
btc_data.columns= ["Open BTC", "High BTC", "Low BTC", "Close BTC", "Adj Close BTC", "Volume BTC"]
df_vix = yf.download('^VIX', period = '1')
df_vix.columns= ["Open VIX", "High VIX", "Low VIX", "Close VIX", "Adj Close VIX", "Volume VIX"]
df_eurusd = yf.download('EURUSD=X', period = '1')
df_eurusd.columns= ["Open EUR/USD", "High EUR/USD", "Low EUR/USD", "Close EUR/USD", "Adj Close EUR/USD", "Volume EUR/USD"]
df_usdjpy = yf.download('JPY=X', period = '1')
df_usdjpy.columns= ["Open USD/JPY", "High USD/JPY", "Low USD/JPY", "Close USD/JPY", "Adj Close USD/JPY", "Volume USD/JPY"]
df_usdchf = yf.download('CHF=X', period = '1')
df_usdchf.columns= ["Open USD/CHF", "High USD/CHF", "Low USD/CHF", "Close USD/CHF", "Adj Close USD/CHF", "Volume USD/CHF"]


# convert date column to datetime
df_gold.index = pd.to_datetime(df_gold.index, format = '%Y-%m-%d')
df_ring.index = pd.to_datetime(df_ring.index, format = '%Y-%m-%d')
df_silver.index = pd.to_datetime(df_silver.index, format = '%Y-%m-%d')
df_palladium.index = pd.to_datetime(df_palladium.index, format = '%Y-%m-%d')
df_freight.index = pd.to_datetime(df_freight.index, format = '%Y-%m-%d')
df_ca_stocks.index = pd.to_datetime(df_ca_stocks.index, format = '%Y-%m-%d')
df_tnx.index = pd.to_datetime(df_tnx.index, format = '%Y-%m-%d')
df_vix.index = pd.to_datetime(df_vix.index, format = '%Y-%m-%d')
df_eurusd.index = pd.to_datetime(df_eurusd.index, format = '%Y-%m-%d')
df_usdjpy.index = pd.to_datetime(df_usdjpy.index, format = '%Y-%m-%d')
df_usdchf .index = pd.to_datetime(df_usdchf .index, format = '%Y-%m-%d')
plat_data.index = pd.to_datetime(plat_data.index, format = '%Y-%m-%d')
copper_data.index = pd.to_datetime(copper_data.index, format = '%Y-%m-%d')
aluminum_data.index = pd.to_datetime(aluminum_data.index, format = '%Y-%m-%d')
spx_data.index = pd.to_datetime(spx_data.index, format = '%Y-%m-%d')
nasdaq_data.index = pd.to_datetime(nasdaq_data.index, format = '%Y-%m-%d')
btc_data.index = pd.to_datetime(btc_data.index, format = '%Y-%m-%d')

# List of dataframes
dataframes = [df_gold,
df_ring,
df_brent,
df_silver,
df_palladium,
df_freight,
df_ca_stocks,
df_tnx,
df_vix,
plat_data,
copper_data,
aluminum_data,
spx_data,
nasdaq_data,
btc_data,
df_eurusd,
df_usdjpy,
df_usdchf]

# Concatenate dfs using a for loop
for i in range(len(dataframes)):
    if i == 0:
        merged_df = dataframes[i]
    else:
        merged_df = pd.merge(merged_df, dataframes[i], left_index=True, right_index=True, how= 'outer')

merged_df = pd.DataFrame(merged_df)

# drop any rows where all values are null
df = merged_df.dropna(how='all')

# create new column with day of week
df['day_of_week'] = df.index.dayofweek

# filter out rows corresponding to Saturday and Sunday
df = df[(df['day_of_week'] != 5) & (df['day_of_week'] != 6)]

# drop day_of_week column
df = df.drop(columns=['day_of_week'])

df = df.drop(['Adj Close BTC', 'Adj Close ALI'], axis=1)
df = df.drop(['Adj Close RING'], axis=1)


df = df.filter(like='Adj Close')

# joblib load the scaler
scaler = joblib.load("model/scaler.pkl")

# scaler transform the raw data -> your X_preproc
inputs=scaler.transform(df)

# joblib load the model(whatever you like)
model_path=os.path.join(root_path,'model')
if model_name=='Lasso':
    model = joblib.load(os.path.join(model_path,'Lasso.pkl'))
    gold_pred=model.predict(inputs)[0]
    gold_pred=round(gold_pred,1)
elif model_name=='Sarima':
    model = joblib.load(os.path.join(model_path,'sarima.pkl'))
    results = model.get_forecast(1, alpha=0.05)
    gold_pred = (round(np.exp(results.predicted_mean),1)).iloc[0]
    confidence_int = results.conf_int()
else:
    model = joblib.load(os.path.join(model_path,'DecisionTreeRegressor.pkl'))
    gold_pred=model.predict(inputs)[0]
    gold_pred=round(gold_pred,1)

# model.predict(X_preproc) -> output
#gold_pred=model.predict(inputs)[0]
#gold_pred=round(gold_pred,1)

st.write("**Features**")
#st.write(df.round(2))

test=df.round(1)
st.write(test)

gold_price=round(df['Adj Close gold'][0],2)

pct_change=round((gold_pred/gold_price-1)*100,2)


col1, col2 = st.columns(2)
col1.metric("Gold Price", gold_price)
col2.metric("Gold Prediction", gold_pred, f'{pct_change}%')

st.text(model)

st.info('Disclaimer: **All investments carry significant risk.**')
#st.text("Past performance is not necessarily indicative of future results.")
#st.text("All investment decisions of an individual remain the specific responsibility of that individual")

df_gold = pd.DataFrame(yf.download('GC=F', period = 10))
