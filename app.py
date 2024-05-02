#maria
import streamlit as st
import numpy as np
import pickle

# Cargar el modelo
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

# Función para realizar la predicción
def predict_price(vars):
    prediction = model.predict(vars)
    return prediction

# Título de la aplicación
st.title("Predicción de precios de inmuebles")

# Widgets para ingresar la información requerida
var1 = st.number_input("Variable 1", value=0)
var2 = st.number_input("Variable 2", value=0)
var3 = st.number_input("Variable 3", value=0)

# Realizar la predicción
if st.button("Predecir"):
    vars = np.array([[var1, var2, var3]])
    prediction = predict_price(vars)
    st.write(f"El precio estimado del inmueble es: {prediction}")
