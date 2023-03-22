
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
new_banner = banner.resize((2000, 150))

st.image(new_banner, caption='')
st.info('**Model Statistics**', icon="🧠")

model_name= st.sidebar.selectbox('Select a model', ('XGBoost','RandomForest', 'Lasso','Sarima'))

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
    st.write("XGBoost works by training a number of decision trees (flowcharts of series of related events) that explain the test target values. The predictions of each tree are combined to form the final prediction.")

    #st.image(features, caption='')

elif model_name=='Lasso':
    pnl_path=os.path.join(root_path,'raw_data','LassoPnL.png')
    test_path=os.path.join(root_path,'raw_data','LassoTest.png')
    pnl= Image.open(pnl_path)
    pnl = pnl.resize((600, 400))
    test= Image.open(test_path)
    test = test.resize((600, 400))

    col1, col2 = st.columns(2)
    with col1:
        st.image(test, caption='')
    with col2:
        st.image(pnl, caption='')
    st.write("Lasso is a regularization technique which shrinks data around a central mean. Lasso can also automate certain parts of model selection, like variable selection/parameter elimination.")


elif model_name=='Sarima':
    pnl_path=os.path.join(root_path,'raw_data','SarimaPnL.png')
    test_path=os.path.join(root_path,'raw_data','SarimaTest.png')
    pnl= Image.open(pnl_path)
    pnl = pnl.resize((600, 400))
    test= Image.open(test_path)
    test = test.resize((600, 400))

    col1, col2 = st.columns(2)
    with col1:
        st.image(test, caption='')
    with col2:
        st.image(pnl, caption='')
    st.write("Sarima only looks at the historical price of gold to predict the future price.")

elif model_name=='RandomForest':
    pnl_path=os.path.join(root_path,'raw_data','RandomForestPnL.png')
    test_path=os.path.join(root_path,'raw_data','RandomForestTest.png')
    pnl= Image.open(pnl_path)
    pnl = pnl.resize((600, 400))
    test= Image.open(test_path)
    test = test.resize((600, 400))

    col1, col2 = st.columns(2)
    with col1:
        st.image(test, caption='')
    with col2:
        st.image(pnl, caption='')
    st.write("RandomForest takes a random subset of features when forming a decision tree (flowcharts of series of related events) and only had access to a random set of the training data points.")


st.info("Past performance is not necessarily indicative of future results.")
