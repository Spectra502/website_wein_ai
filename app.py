import streamlit as st
import os
import pandas as pd
import ydata_profiling as ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, load_model
import pandas_profiling


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', sep=";", index_col=None) #, , index_col=None

with st.sidebar:
    st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiKsjCzOdbjOdEwRoolS0WIBBkE1qOzSJhz85suDtxoh9NkOqlEfbmfS-omjLzF6WQGOkXSCxDteKHeOi9TePoLrNDKTR6YXyoz0uilfuHYDlP8j08WnOqs8NRgWDMt_QFfkdmt27zpY896QkpDM5tXZ3P8d6BB8eWYY-IxWleplFDuRDKdRmR_0Ptpm7uk/w480-h300/logo.png")
    st.title("Wine Quality Prediction")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Machine Learning"])


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, sep=";", index_col=None) #index_col=None
        df.to_csv('dataset.csv', sep=";",index=None) #,, index=None 
        st.dataframe(df)
    st.write("Loaded dataset:")
    st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Machine Learning": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        #save_model(best_model, 'best_model')