
import streamlit as st
import joblib
import os
import pandas as pd
import math
import yfinance as yf
import numpy as np

from PIL import Image



# Confit
st.set_page_config(page_title='Model Features', page_icon=':bar_chart:', layout='wide')

#opening the image
app_path=os.path.dirname(__file__)
root_path=os.path.dirname(app_path)
root_path=os.path.dirname(root_path)

banner_path=os.path.join(root_path,'raw_data','banner.jpg')
banner = Image.open(banner_path)
new_banner = banner.resize((1800, 150))

st.image(new_banner, caption='')
st.info('**Model Statistics**', icon="🧠")

model_name= st.sidebar.selectbox('Select a model', ('XGBoost','Decision Tree Regressor', 'Lasso','Sarima'))

if model_name=='XGBoost':
    features_path=os.path.join(root_path,'raw_data','XGBoostFeatures.png')
    pnl_path=os.path.join(root_path,'raw_data','XGBoostPnL.png')
    test_path=os.path.join(root_path,'raw_data','XGBoostTest.png')

    pnl= Image.open(pnl_path)
    pnl = pnl.resize((600, 400))
    test= Image.open(test_path)
    test = test.resize((600, 400))
    features= Image.open(features_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(test, caption='')
    with col2:
        st.image(pnl, caption='')


elif model_name=='Sarima':
    model = joblib.load(os.path.join(model_path,'sarima.pkl'))
    results = model.get_forecast(1, alpha=0.05)
    gold_pred = (round(np.exp(results.predicted_mean),1)).iloc[0]
    confidence_int = results.conf_int()
elif model_name=='Decision Tree Regressor':
    model = joblib.load(os.path.join(model_path,'DecisionTreeRegressor.pkl'))
    gold_pred=model.predict(inputs)[0]
    gold_pred=round(gold_pred,1)

st.info("Past performance is not necessarily indicative of future results.")
