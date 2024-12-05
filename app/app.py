import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join back to form the preprocessed text
    return ' '.join(lemmatized_tokens)

# Load model and TF-IDF vectorizer
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Streamlit app UI
st.title('Personality Prediction from Text')

user_input = st.text_area("Enter text")

if st.button('Predict Personality Traits'):
    if user_input.strip():
        # Preprocess the input text
        preprocessed_input = preprocess_text(user_input)

        # Vectorize the input text
        input_tfidf = tfidf_vectorizer.transform([preprocessed_input])

        # Debugging: Print the input vector
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = input_tfidf.toarray()[0]

        # Create a list of tuples (feature, score) and sort them by score
        feature_importance = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

        # Make predictions
        prediction = model.predict(input_tfidf)

        # Output the predictions
        st.write(f"Openness: {prediction[0][0]:.2f}")
        st.write(f"Conscientiousness: {prediction[0][1]:.2f}")
        st.write(f"Extraversion: {prediction[0][2]:.2f}")
        st.write(f"Agreeableness: {prediction[0][3]:.2f}")
        st.write(f"Neuroticism: {prediction[0][4]:.2f}")
    else:
        st.write("Please enter some text.")
