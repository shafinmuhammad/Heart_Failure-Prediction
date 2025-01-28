#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,
from sklearn.linear_model import LogisticRegression
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
    
    object_encoder= LabelEncoder()
    df[object_columns] = object_encoder.fit_transform(df[object_columns])
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_n=numerical_columns
    
    transformer= MinMaxScaler()

    df_n = pd.DataFrame(transformer.fit_transform(df[columns]axis=1))
    
    df_preprocessed = pd.concat([df[df_n],df['object_columns']], axis=1)
    
    df_preprocessed = df_preprocessed.reindex(columns=columns_list, fill_value=0)

    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.write(prediction_text)

