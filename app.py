import streamlit as st
import joblib

# Load the model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Class Prediction")
st.write("Enter flower measurements to predict species.")

# Inputs
sl = st.number_input("Sepal length (cm)", min_value=0.0, format="%.2f")
sw = st.number_input("Sepal width (cm)", min_value=0.0, format="%.2f")
pl = st.number_input("Petal length (cm)", min_value=0.0, format="%.2f")
pw = st.number_input("Petal width (cm)", min_value=0.0, format="%.2f")

if st.button("Predict"):
    prediction = model.predict([[sl, sw, pl, pw]])[0]
    species = ["Setosa", "Versicolor", "Virginica"][prediction]
    st.success(f"Prediction: **{species}**")
