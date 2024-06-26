import streamlit as st
import os
import pandas as pd
import ydata_profiling as ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, load_model
import pandas_profiling
from pycaret.classification import load_model, predict_model

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', sep=";", index_col=None) #, , index_col=None

with st.sidebar:
    st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiKsjCzOdbjOdEwRoolS0WIBBkE1qOzSJhz85suDtxoh9NkOqlEfbmfS-omjLzF6WQGOkXSCxDteKHeOi9TePoLrNDKTR6YXyoz0uilfuHYDlP8j08WnOqs8NRgWDMt_QFfkdmt27zpY896QkpDM5tXZ3P8d6BB8eWYY-IxWleplFDuRDKdRmR_0Ptpm7uk/w480-h300/logo.png")
    st.title("Vorhersage der Weinqualität")
    choice = st.radio("Navigation", ["Hochladen", "Analyze", "Machine Learning", "Vorhersage"])


if choice == "Hochladen":
    st.title("Laden Sie Ihren Dataset hoch")
    file = st.file_uploader("Laden Sie Ihren Dataset hoch")
    if file: 
        df = pd.read_csv(file, sep=";", index_col=None) #index_col=None
        df.to_csv('dataset.csv', sep=";",index=None) #,, index=None 
        st.dataframe(df)
    st.write("Geladener Dataset:")
    st.dataframe(df)

if choice == "Analyze": 
    st.title("Explorative Datenanalyse")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Machine Learning": 
    #chosen_target = st.selectbox('Choose the Target Column', df.columns)
    chosen_target = "quality"
    st.title('Zielspalte: Qualität')
    if st.button('Modellierung ausführen'): 
        categories = df.columns.to_list()
        categories.remove(chosen_target)
        setup(df, target=chosen_target, categorical_features=categories)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models(budget_time=0.9)
        compare_df = pull()
        st.write("Bestes Modell ganz oben:")
        st.dataframe(compare_df)

if choice == "Vorhersage": 
    st.title("Qualität Ihres Weins")
    loaded_model = load_model('best_model')
    fixed_acidity = st.slider('Fixed Acidity (Fester Säuregehalt)', float(df['fixed acidity'].min()), float(df['fixed acidity'].max()), float(df['fixed acidity'].mean()))
    volatile_acidity = st.slider('Volatile Acidity (Flüchtige Säure)', float(df['volatile acidity'].min()), float(df['volatile acidity'].max()), float(df['volatile acidity'].mean()))
    citric_acid = st.slider('Citric Acid (Zitronensäure)', float(df['citric acid'].min()), float(df['citric acid'].max()), float(df['citric acid'].mean()))
    residual_sugar = st.slider('Residual Sugar (Restzucker)', float(df['residual sugar'].min()), float(df['residual sugar'].max()), float(df['residual sugar'].mean()))
    chlorides = st.slider('Chlorides (Chloride)', float(df['chlorides'].min()), float(df['chlorides'].max()), float(df['chlorides'].mean()))
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide (Freies Schwefeldioxid)', float(df['free sulfur dioxide'].min()), float(df['free sulfur dioxide'].max()), float(df['free sulfur dioxide'].mean()))
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide (Gesamtschwefeldioxid)', float(df['total sulfur dioxide'].min()), float(df['total sulfur dioxide'].max()), float(df['total sulfur dioxide'].mean()))
    density = st.slider('Density (Dichte)', float(df['density'].min()), float(df['density'].max()), float(df['density'].mean()))
    pH = st.slider('pH', float(df['pH'].min()), float(df['pH'].max()), float(df['pH'].mean()))
    sulphates = st.slider('Sulphates (Sulfate)', float(df['sulphates'].min()), float(df['sulphates'].max()), float(df['sulphates'].mean()))
    alcohol = st.slider('Alcohol (Alkohol)', float(df['alcohol'].min()), float(df['alcohol'].max()), float(df['alcohol'].mean()))

    # Button to make prediction
    if st.button('Vorhersage der Qualität'):
        # Create an array of the input data
        input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
        input_df = pd.DataFrame(input_data, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

        # Get the prediction
        prediction = predict_model(loaded_model ,input_df)
        st.dataframe(prediction)
        if alcohol <= 8:
            st.title("Danke für eure Aufmerksamkeit")