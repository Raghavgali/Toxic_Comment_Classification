import streamlit as st
import pickle
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


def load_tfidf():
    tfidf = pickle.load(open('tf_idf.pkt', 'rb'))
    return tfidf

def load_model():
    lgb_model = pickle.load(open('toxicity_model.pkt', 'rb'))
    return lgb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    lgb_model = load_model()
    prediction = lgb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 0 else "Non-Toxic"
    return class_name

st.header("Toxic Comment Classifier App")

st.subheader('Enter a comment to check for any toxicity')

user_input = st.text_input('Enter your comment')

if user_input is not None:
    if st.button('Analyse'):
        result = toxicity_prediction(user_input)
        st.subheader('Result:')
        st.info("The comment is " + result)