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

from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator


from PIL import Image

# Confit
st.set_page_config(page_title='Gold Price Prediction Tool', page_icon=':bar_chart:', layout='wide')


#opening the image
app_path=os.path.dirname(__file__)
root_path=os.path.dirname(app_path)
#image_path=os.path.join(root_path,'raw_data','shutterstock_1797061261-1.jpg')
image_path=os.path.join(root_path,'raw_data','gold_banner.jpg')
image = Image.open(image_path)
gold_banner = image.resize((2000, 200))


st.image(gold_banner, caption='')
st.info('**Gold Price Predictions**', icon="🧠")


######## NEW CODE ##########

period='120d'

# To make a dataframe from today's price
# Gold price (target)
df_gold = pd.DataFrame(yf.download('GC=F', period = period))
df_gold.columns= ['Open gold', 'High gold', 'Low gold', 'Close gold', 'Adj Close gold', 'Volume gold']
#Gold producer ETFs
df_ring = pd.DataFrame(yf.download('RING', period = period))
df_ring.columns= ['Open RING', 'High RING', 'Low RING', 'Close RING', 'Adj Close RING', 'Volume RING']
#Oil Price
df_brent = pd.DataFrame(yf.download('CL=F', period = period))
df_brent.columns= ['Open Brent', 'High Brent', 'Low Brent', 'Close Brent', 'Adj Close Brent', 'Volume Brent']
df_silver = pd.DataFrame(yf.download('SI=F', period = period))
df_silver.columns= ['Open Silver', 'High Silver', 'Low Silver', 'Close Silver', 'Adj Close Silver', 'Volume Silver']
df_palladium = pd.DataFrame(yf.download('PA=F', period = period))
df_palladium.columns= ['Open Palladium', 'High Palladium', 'Low Palladium', 'Close Palladium', 'Adj Close Palladium', 'Volume Palladium']
# Golden ocean share price as a proxy for dry bulk freight costs
df_freight = pd.DataFrame(yf.download('GOGL', period = period))
df_freight.columns= ['Open Freight', 'High Freight', 'Low Freight', 'Close Freight', 'Adj Close Freight', 'Volume Freight']
# Canadian stock market index
df_ca_stocks = pd.DataFrame(yf.download('^GSPTSE', period = period))
df_ca_stocks.columns= ['Open CA', 'High CA', 'Low CA', 'Close CA', 'Adj Close CA', 'Volume CA']
# 10y treasury yield
df_tnx = pd.DataFrame(yf.download('^TNX', period = period))
df_tnx.columns= ['Open TNX', 'High TNX', 'Low TNX', 'Close TNX', 'Adj Close TNX', 'Volume TNX']
plat_data = yf.download('PL=F',period = period)
plat_data.columns= ["Open PLAT", "High PLAT", "Low PLAT", "Close PLAT", "Adj Close PLAT", "Volume PLAT"]
copper_data = yf.download('HG=F', period = period)
copper_data.columns= ["Open Copper", "High Copper", "Low Copper", "Close Copper", "Adj Close Copper", "Volume Copper"]
#aluminum_data = yf.download('ALI=F', period = period)
#aluminum_data.columns= ["Open ALI", "High ALI", "Low ALI", "Close ALI", "Adj Close ALI", "Volume ALI"]
spx_data = yf.download('^GSPC', period = period)
spx_data.columns= ["Open SPX", "High SPX", "Low SPX", "Close SPX", "Adj Close SPX", "Volume SPX"]
nasdaq_data = yf.download('^IXIC', period = period)
nasdaq_data.columns= ["Open NDQ", "High NDQ", "Low NDQ", "Close NDQ", "Adj Close NDQ", "Volume NDQ"]
#btc_data = yf.download('BTC-USD', period = period)
#btc_data.columns= ["Open BTC", "High BTC", "Low BTC", "Close BTC", "Adj Close BTC", "Volume BTC"]
df_vix = yf.download('^VIX', period = period)
df_vix.columns= ["Open VIX", "High VIX", "Low VIX", "Close VIX", "Adj Close VIX", "Volume VIX"]
df_eurusd = yf.download('EURUSD=X', period = period)
df_eurusd.columns= ["Open EUR/USD", "High EUR/USD", "Low EUR/USD", "Close EUR/USD", "Adj Close EUR/USD", "Volume EUR/USD"]
df_usdjpy = yf.download('JPY=X', period = period)
df_usdjpy.columns= ["Open USD/JPY", "High USD/JPY", "Low USD/JPY", "Close USD/JPY", "Adj Close USD/JPY", "Volume USD/JPY"]
df_usdchf = yf.download('CHF=X', period = period)
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
#aluminum_data.index = pd.to_datetime(aluminum_data.index, format = '%Y-%m-%d')
spx_data.index = pd.to_datetime(spx_data.index, format = '%Y-%m-%d')
nasdaq_data.index = pd.to_datetime(nasdaq_data.index, format = '%Y-%m-%d')
#btc_data.index = pd.to_datetime(btc_data.index, format = '%Y-%m-%d')

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
spx_data,
nasdaq_data,
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
df = pd.DataFrame(merged_df)

# drop any rows where all values are null
#df = merged_df.dropna(how='any')

# create new column with day of week
df['day_of_week'] = df.index.dayofweek

# filter out rows corresponding to Saturday and Sunday
df = df[(df['day_of_week'] != 5) & (df['day_of_week'] != 6)]

# drop day_of_week column
df = df.drop(columns=['day_of_week'])

#df = df.drop(['Adj Close BTC', 'Adj Close ALI'], axis=1)
#df = df.drop(['Adj Close RING'], axis=1)


df = df.filter(like='Adj Close')

#st.write(df)

#sam notebook
data_gold = yf.download("GC=F", period = period)
data_gold_extra = yf.download("GC=F", period = period)

#pull in DXY data
data_dxy = yf.download("DX-Y.NYB", period = period)
#pull in Junk bond data
data_jnk = yf.download("JNK", period = period)
#pull in 5 year tips bond data ETF
data_TDTF = yf.download("TDTF", period = period)
#pull in Hang Seng Index
data_HSI = yf.download("^HSI", period = period)
#pull in 13 weeks t-bill yield
data_IRX = yf.download("^IRX", period = period)
#pull in 10 year yield
data_TNX = yf.download("^TNX", period = period)
#take just the close price of DXY
data_dxy_close = data_dxy['Adj Close']
df_new_feat = pd.merge(df, data_dxy_close, left_index=True, right_index=True, how= 'outer')
df_new_feat = df_new_feat.rename(columns={'Adj Close': 'Adj Close DXY'})
#take just the close price of JNK
data_jnk_close = data_jnk['Adj Close']
df_new_feat = pd.merge(df_new_feat, data_jnk_close, left_index=True, right_index=True, how= 'outer')
df_new_feat = df_new_feat.rename(columns={'Adj Close': 'Adj Close JNK'})
#take just the close price of TDTF
data_TDTF = data_TDTF['Adj Close']
df_new_feat = pd.merge(df_new_feat, data_TDTF, left_index=True, right_index=True, how= 'outer')
df_new_feat = df_new_feat.rename(columns={'Adj Close': 'Adj Close TDTF'})
#take just the close price of HSI
data_HSI = data_HSI['Adj Close']
df_new_feat = pd.merge(df_new_feat, data_HSI, left_index=True, right_index=True, how= 'outer')
df_new_feat = df_new_feat.rename(columns={'Adj Close': 'Adj Close HSI'})
#take just the close price of IRX
data_IRX = data_IRX['Adj Close']
df_new_feat = pd.merge(df_new_feat, data_IRX, left_index=True, right_index=True, how= 'outer')
df_new_feat = df_new_feat.rename(columns={'Adj Close': 'Adj Close IRX'})
# create a yield curve inversion
yield_inversion = data_TNX['Adj Close'] - data_IRX
df_new_feat = pd.merge(df_new_feat, yield_inversion, left_index=True, right_index=True, how= 'outer')
df_new_feat = df_new_feat.rename(columns={'Adj Close': 'yield_inversion'})

#adding other gold features - volume, low etc
data_gold_low = data_gold['Low']
data_gold_high = data_gold['High']
data_gold_volume = data_gold['Volume']

df_new_feat = pd.merge(df_new_feat, data_gold_low, left_index=True, right_index=True, how= 'outer')
df_new_feat = pd.merge(df_new_feat, data_gold_high, left_index=True, right_index=True, how= 'outer')
df_new_feat = pd.merge(df_new_feat, data_gold_volume, left_index=True, right_index=True, how= 'outer')

# Bollinger Bands
indicator_bb = BollingerBands(data_gold_extra['Adj Close'])
bb = data_gold_extra
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Close','bb_h','bb_l']]
bb_h = bb['bb_h']
bb_l = bb['bb_l']
df_new_feat = pd.merge(df_new_feat, bb_h, left_index=True, right_index=True, how= 'outer')
df_new_feat = pd.merge(df_new_feat, bb_l, left_index=True, right_index=True, how= 'outer')

# Moving Average Convergence Divergence
macd = MACD(data_gold_extra['Adj Close']).macd()
df_new_feat = pd.merge(df_new_feat, macd, left_index=True, right_index=True, how= 'outer')

# Resistence Strength Indicator
rsi = RSIIndicator(data_gold_extra['Adj Close']).rsi()
df_new_feat = pd.merge(df_new_feat, rsi, left_index=True, right_index=True, how= 'outer')

df = df_new_feat.dropna(how='any')

df = df.drop(['Adj Close IRX', 'Adj Close EUR/USD','Adj Close USD/JPY','Adj Close USD/CHF'], axis=1)

df_today=df.tail(1)

# joblib load the scaler
scaler = joblib.load("model/scaler_sam.pkl")

# scaler transform the raw data -> your X_preproc
inputs_new=scaler.transform(df_today)

model_name= st.sidebar.selectbox('Select a model', ('XGBoost','RandomForest', 'Lasso','Sarima'))

app_path=os.path.dirname(__file__)
root_path=os.path.dirname(app_path)
#root_path=os.path.dirname(root_path)


# joblib load the model(whatever you like)
model_path=os.path.join(root_path,'model')
if model_name=='Lasso':
    model = joblib.load(os.path.join(model_path,'lasso_clf_feat.pkl'))
    g=model.predict(inputs_new)
    pred=np.column_stack((g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g))
    gold_pred = round(scaler.inverse_transform(pred)[0][0],1)
elif model_name=='Sarima':
    model = joblib.load(os.path.join(model_path,'sarima.pkl'))
    results = model.get_forecast(1, alpha=0.05)
    gold_pred = (round(np.exp(results.predicted_mean),1)).iloc[0]
    confidence_int = results.conf_int()
elif model_name=='XGBoost':
    #model = joblib.load(os.path.join(model_path,'best_xgb_fit.pkl'))
    model = joblib.load(os.path.join(model_path,'xgb.pkl'))
    g=model.predict(inputs_new)
    pred=np.column_stack((g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g))
    gold_pred = round(scaler.inverse_transform(pred)[0][0],1)
else:
    model = joblib.load(os.path.join(model_path,'random_forest_clf_feat.pkl'))
    g=model.predict(inputs_new)
    pred=np.column_stack((g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g))
    gold_pred = round(scaler.inverse_transform(pred)[0][0],1)


# joblib load the scaler
#scaler = joblib.load("model/scaler.pkl")

# scaler transform the raw data -> your X_preproc
#inputs=scaler.transform(df)

# joblib load the model(whatever you like)
#model_path=os.path.join(root_path,'model')
#if model_name=='Lasso':
#    model = joblib.load(os.path.join(model_path,'Lasso.pkl'))
#    gold_pred=model.predict(inputs)[0]
#    gold_pred=round(gold_pred,1)
#elif model_name=='Sarima':
#    model = joblib.load(os.path.join(model_path,'sarima.pkl'))
#    results = model.get_forecast(1, alpha=0.05)
#    gold_pred = (round(np.exp(results.predicted_mean),1)).iloc[0]
#    confidence_int = results.conf_int()
#else:
#    model = joblib.load(os.path.join(model_path,'DecisionTreeRegressor.pkl'))
#    gold_pred=model.predict(inputs)[0]
#    gold_pred=round(gold_pred,1)

# model.predict(X_preproc) -> output
#gold_pred=model.predict(inputs)[0]
#gold_pred=round(gold_pred,1)

st.write("**Features**")
#st.write(df.round(2))

test=df_today.round(2)
test.columns = test.columns.str.replace('Adj Close','')
test1=test.iloc[:, : 15]
test2=test.iloc[:,13 :]
st.write(test1)
#st.write(test2)

gold_price=round(df_today['Adj Close gold'][0],2)

pct_change=round((gold_pred/gold_price-1)*100,2)

gold_pred2=(int(10*gold_pred-0.5)+1) / 10.0
#st.write(gold_pred2)

col1, col2, col3 = st.columns(3)
col1.metric("Gold Price", gold_price)
col2.metric("Gold Prediction", gold_pred2, f'{pct_change}%')
with col3:
        st.image("https://media.giphy.com/media/3ohzdIuqJoo8QdKlnW/giphy.gif",width=300)


st.write(model_name)

st.info('Disclaimer: **All investments carry significant risk.**')
#st.text("Past performance is not necessarily indicative of future results.")
#st.text("All investment decisions of an individual remain the specific responsibility of that individual")
