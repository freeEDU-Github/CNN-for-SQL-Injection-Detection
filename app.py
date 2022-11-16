import sys
import pandas as pd
import numpy as np
import glob
import time
import nltk
from PIL import Image

nltk.download('stopwords')
import tensorflow as tf
import pickle
import streamlit as st

mymodel = tf.keras.models.load_model('my_model_cnn.h5')
myvectorizer = pickle.load(open("vect_cnn_2", 'rb'))

MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100

def clean_data(input_val):
    input_val = input_val.replace('\n', '')
    input_val = input_val.replace('%20', ' ')
    input_val = input_val.replace('=', ' = ')
    input_val = input_val.replace('((', ' (( ')
    input_val = input_val.replace('))', ' )) ')
    input_val = input_val.replace('(', ' ( ')
    input_val = input_val.replace(')', ' ) ')
    input_val = input_val.replace('1 ', 'numeric')
    input_val = input_val.replace(' 1', 'numeric')
    input_val = input_val.replace("'1 ", "'numeric ")
    input_val = input_val.replace(" 1'", " numeric'")
    input_val = input_val.replace('1,', 'numeric,')
    input_val = input_val.replace(" 2 ", " numeric ")
    input_val = input_val.replace(' 3 ', ' numeric ')
    input_val = input_val.replace(' 3--', ' numeric--')
    input_val = input_val.replace(" 4 ", ' numeric ')
    input_val = input_val.replace(" 5 ", ' numeric ')
    input_val = input_val.replace(' 6 ', ' numeric ')
    input_val = input_val.replace(" 7 ", ' numeric ')
    input_val = input_val.replace(" 8 ", ' numeric ')
    input_val = input_val.replace('1234', ' numeric ')
    input_val = input_val.replace("22", ' numeric ')
    input_val = input_val.replace(" 8 ", ' numeric ')
    input_val = input_val.replace(" 200 ", ' numeric ')
    input_val = input_val.replace("23 ", ' numeric ')
    input_val = input_val.replace('"1', '"numeric')
    input_val = input_val.replace('1"', '"numeric')
    input_val = input_val.replace("7659", 'numeric')
    input_val = input_val.replace(" 37 ", ' numeric ')
    input_val = input_val.replace(" 45 ", ' numeric ')

    return input_val


def main():
    st.title("SQL Injection Detection with a Machine Learning Approach")

    st.subheader("SQL Injection")
    st.markdown("SQL injection (SQLi) is a web security vulnerability that allows an attacker to interfere with database queries made by an application. It generally allows an attacker to view data that they would not normally be able to retrieve. This may include data belonging to other users or any other data that the application has access to. In many cases, an attacker can modify or delete this data, causing persistent changes to the application's content or behavior.")

    image = Image.open('sql.jpg')
    st.image(image, caption='SQL injection (SQLi)')

    st.subheader(
        "The primary goal of this project is to identify whether the inputted data by users contains SQLi vulnerabilities.")

    sample_dataset = pd.read_csv("sqli.csv", encoding='utf-16')
    st.dataframe(sample_dataset)

    input_val = st.text_input("Give me some sentences to work on : ")


    if st.button("Predict"):
        input_val = clean_data(input_val)
        input_val = [input_val]
        input_val = myvectorizer.transform(input_val).toarray()
        input_val.shape = (1, 64, 64, 1)
        result = mymodel.predict(input_val)

        if result < 0.5:
            st.success("This is safe")

        elif result > 0.5:
            st.error("ALERT - This can be SQL injection")


if __name__ == '__main__':
    main()