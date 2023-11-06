import numpy as np
import pandas as pd
import re
import nltk
import streamlit as st
import pickle
import joblib
import base64

# Download stopwords and initialize PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# Set the page title and icon
st.set_page_config(page_title="Restaurant Review Using Sentiment Analysis", page_icon=":spoon:", layout="centered")

#Add background image
@st.cache(allow_output_mutation=True)
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return data

img = get_img_as_base64("background-image.png")
img_base64 = base64.b64encode(img).decode()

# Define the background style
background_style = f"""
<style>
[data-testid="stAppViewContainer"]
{{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: Arial, sans-serif;
    background-color: rgba(0, 0, 0, 0);
}}

[data-testid="stHeader"]
{{
    background-color:rgba(0,0,0,0);
}}
</style>
"""
# Include custom CSS for styling
st.markdown(background_style, unsafe_allow_html=True)

# Greeting

# Wrap all content in a <div> with a background color
st.markdown("""
    <div style='background-color: rgba(251, 228, 216, 0.7);padding: 20px;border-radius: 5px;'>
        <h1 style='text-align: center; font-weight: bold; color: #190019;'>Welcome to Restaurant Review System using Sentiment Analysis</h1>
        <h2 style='text-align: center;color: #2B124C'>Enter your review:</h2>
    """, unsafe_allow_html=True)

# Add custom CSS for the "Predict" button and the "Type your review here" text
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: rgba(251, 228, 216, 0.7);
        color: purpel;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input for the review
review = st.text_area("","Type your review here:", height=150)


# Css styling part ended

# Define the perform_sentiment_analysis function here
def perform_sentiment_analysis(str1):
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', str1)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

    # Loading BoW dictionary
    from sklearn.feature_extraction.text import CountVectorizer
    cvFile = 'analysis.pkl'
    cv = pickle.load(open(cvFile, "rb"))
    X_fresh = cv.transform(corpus).toarray()
    return X_fresh

# Predictions (via sentiment classifier)
def load_model():
    classifier = joblib.load('classifier model')
    return classifier

 
y_pred = []
# Add a "Predict" button
if st.button("Predict"):
    if review != 'Type your review here':
        with st.spinner('Predicting...'):
            classifier = load_model()
            y_pred = classifier.predict(perform_sentiment_analysis(review))
                
if len(y_pred) != 0 and y_pred[0] == '1':
    st.markdown("<div style='font-weight: bold; color: green; background-color: rgba(69, 25, 82, 0.7); padding: 10px; border-radius: 5px;'>The review is likely positive</div>", unsafe_allow_html=True)
elif len(y_pred) != 0 and y_pred[0] == '0':
    st.markdown("<div style='font-weight: bold; color: red; background-color:rgba(69, 25, 82, 0.7); padding: 10px; border-radius: 5px;'>The review is likely negative</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='font-weight: bold; color: orange; background-color:rgba(69, 25, 82, 0.7); padding: 10px; border-radius: 5px;'>Please enter a review to make predictions</div>", unsafe_allow_html=True)

# Wrap the "Thank You" section in a <div> with its own background color
st.markdown("""
    <div style='background-color:rgba(251, 228, 216, 0.7);border-radius: 5px; margin-top:1rem;'>
        <h3 style='text-align: center; font-weight: bold; color: #522B5B;'>Thank you for using our sentiment analysis tool!</h3>
    </div>
    """, unsafe_allow_html=True)


# Close the wrapping <div>
st.markdown("</div>", unsafe_allow_html=True)