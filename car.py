import pickle
import streamlit as st
import requests
import pandas
import numpy  as np
import sklearn
# import mymodel as m

st.title('CAR-PRICE-PREDICTOR')
pipe= pickle.load(open('D:\\PROJECTS\\pp\\car_price_Model.pkl','rb'))
df=pickle.load(open('D:\\PROJECTS\\pp\\df.pkl','rb'))
# st.button('Select company')
# st.button('Select name')
# cars_list = cars['company'].values
company = st.selectbox(
    "Type or select a company",
    df['company'].unique()
)
year = st.selectbox(
    "Type or select a year",
    df['year'].unique()
)
kms_driven = st.selectbox(
    "Type or select a ",
    df['kms_driven'].unique()
)
fuel_type = st.selectbox(
    "Type or select a fuel-type",
    df['fuel_type'].unique()
)
model = st.selectbox(
    "Type or select a model-name",
    df['name'].unique()
)
if st.button('PREDICT-PRICE'):
    query=np.array([company,model,fuel_type,kms_driven,year])
    # query=query.reshape(1,5)
    st.title(pipe.predict(query))
# window=st.slider("hello")
# st.write(m.run(window=window))
if(st.button("About")):
    st.text("by pritam soni")