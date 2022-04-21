from regex import D
import streamlit as st
import altair as altc
import pandas as pd
import numpy as np
import os
import urllib
import cv2
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from skimage.transform import resize
import tensorflow as tf
import plotly.express as px
import pickle


with open('standardscaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


with open('knnclassifier.pkl', 'rb') as file:
    knn = pickle.load(file)

page = st.sidebar.selectbox(


    "Select Activity", ["ML Prediction"])

img = Image.open("Heart-Attack-blog.jpg")
st.sidebar.image(img)
d = {
    "Male": 1,
    "Female": 0,
    "normal": 1,
    "fixed defect": 2,
    "reversable defect": 3
}

if page == "ML Prediction":

    form = st.form(key='my_form1')

    x1 = form.text_input(label="Age")
    form.text(" \n")

    x2 = form.selectbox("Sex", ["Male", "Female"])
    form.text(" \n")

    x3 = form.text_input(label="Chest pain Level")
    form.text(" \n")

    x4 = form.text_input(label="resting blood pressure")
    form.text(" \n")

    x5 = form.text_input(label="serum cholesterol in mg/dl")
    form.text(" \n")

    x6 = form.text_input(label="fasting blood sugar > 120 mg/dl")
    form.text(" \n")

    x7 = form.text_input(
        label="resting electrocardiographic results (values 0,1,2)")
    form.text(" \n")

    x8 = form.text_input(label="thalach (maximum heart rate achieved)")
    form.text(" \n")

    x9 = form.text_input(label="exercise induced angina")
    form.text(" \n")

    x10 = form.text_input(
        label="ST depression induced by exercise relative to rest")
    form.text(" \n")

    x11 = form.text_input(
        label="the slope of the peak exercise ST segment")
    form.text(" \n")

    x12 = form.text_input(
        label="number of major vessels (0-3)")
    form.text(" \n")

    x13 = form.selectbox(
        "thal", ["normal", "fixed defect", "reversable defect"])
    form.text(" \n")

    submit_button = form.form_submit_button(
        label='Prediction')

    if submit_button:
        x2 = d[x2]

        x13 = d[x13]

        l = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]
        l1 = []
        for x in l:
            l1.append(float(x))

        X = scaler.transform([l1])

        out = knn.predict(X)[0]

        if out == 0:
            st.header("No Heart Attack")
        else:
            st.header("Heart Attack")
