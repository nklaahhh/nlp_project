import streamlit as st
import spacy
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import en_core_web_sm

# Load the transformer model pipeline for text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit app title
st.title("DISASTER TWEET CLASSIFICATION")

# Load SpaCy model
# nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_sm.load()
# Text areas for input
url_input = st.text_area("Enter URL")
paragraph_input = st.text_area("Enter a paragraph")

# Button for analysis
if st.button("Analyze"):
    if url_input:
        try:
            response = requests.get(url_input)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            doc = nlp(text)
            entities = [(X.text, X.label_) for X in doc.ents]
            # Perform classification
            result = classifier(text)
            label = result[0]['label']
            score = result[0]['score']
            classification = "Disaster" if label == "LABEL_1" else "Non-disaster"
            st.success(f"Entities: {entities}")
            st.success(f"Classification: {classification} with score {score:.2f}")
        except:
            st.error("Error opening URL. Please make sure it's valid.")
    elif paragraph_input:
        doc = nlp(paragraph_input)
        entities = [(X.text, X.label_) for X in doc.ents]
        # Perform classification
        result = classifier(paragraph_input)
        label = result[0]['label']
        score = result[0]['score']
        classification = "Disaster" if label == "LABEL_1" else "Non-disaster"
        st.success(f"Entities: {entities}")
        st.success(f"Classification: {classification} with score {score:.2f}")
    else:
        st.warning("Please enter a URL or a paragraph.")
