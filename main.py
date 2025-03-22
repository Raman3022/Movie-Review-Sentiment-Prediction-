import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load the IMBD dataset
model = load_model("./simple_enn_imbd.h5")
")

# step2 : helper function


def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# fuction to preprocess user input
def preprocess_review(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## step 3 : Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_review(
        review
    )  # Ensure this function is working correctly
    prediction = model.predict(preprocessed_input)

    print(f"Raw Prediction Output: {prediction}")  # Debugging

    sentiment = "Positive" if prediction[0][0] > 0.55 else "Negative"
    return sentiment, prediction[0][0]


## stream;lit app
import streamlit as st

st.title("IMBD Movie Review Sentiment Analysis")
st.write("Enter a movie review and we'll predict the sentiment!")


# User input

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocess_input = preprocess_review(user_input)

    ##make prediction
    prediction = model.predict(preprocess_input)
    sentiment = "Postive" if prediction[0][0] > 0.55 else "Negative"

    # display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a movie review to classify.")
