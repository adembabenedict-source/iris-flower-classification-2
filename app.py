import streamlit as st
import pickle
import numpy as np

st.title("Iris Flower Classifier - 100% Accuracy")
st.write("Enter flower measurements to predict the species")

# Load the model you just trained
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# Predict button
if st.button('Predict Species'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f'Predicted Species: **{species[prediction]}**')
    st.write("This model scored 100% on test data")