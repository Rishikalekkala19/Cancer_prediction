import streamlit as st
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle

model = pickle.load(open("estimator_pl.pkl","rb"))

#st.image(r"innomatics_logo.png")

st.title("Cancer Prediction Project")

Age = st.number_input("Enter the age:")
Gender = st.text_input("Enter the gender:")
Tumor_Size= st.number_input("Enter the tumor size:")
Tumor_Grade= st.text_input("Enter the tumor grade:")
Symptoms_Severity=st.text_input("Enter the symptoms severity:")
Family_History = st.text_input("Enter if there is any family history:")
Smoking_History= st.text_input("Enter the smoking status:")
Alcohol_Consumption=st.text_input("Enter the alcohol consumption rate:")
Exercise_Frequency = st.text_input("Exercise routine:")



if st.button("Submit"):
    prediction = model.predict([[Age,Gender,Tumor_Size,Tumor_Grade,Symptoms_Severity,Family_History,Smoking_History,Alcohol_Consumption,Exercise_Frequency]])[0]
    st.write("The predicted value is:", prediction)
