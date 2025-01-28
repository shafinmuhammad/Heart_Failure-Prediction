#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

filename = 'logistic_regression_heart.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('heart failure Prediction App')
st.subheader('Please enter your data:')

df = pd.read_csv('features.csv')
columns_list = df.columns.to_list()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    object_columns = df.select_dtypes(include=['object']).columns

    for col in object_columns: 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    transformer = MinMaxScaler()
    df[numerical_columns] = transformer.fit_transform(df[numerical_columns])


    df_preprocessed = df[columns_list].fillna(0)

    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.write(prediction_text)




#%%
