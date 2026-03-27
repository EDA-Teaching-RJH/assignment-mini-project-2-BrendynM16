import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class NLPEngine:
    def __init__(self, stop_words='english'):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix = None

    def clean_text(self, text):
        """
        this uses Regex we learned to strip out punctuation, numbers, and special characters,
        leaving only lowercase letters and spaces for better NLP matching.
        """
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
        return cleaned.lower()

    def train(self, corpus):
        """Trains the TF-IDF model on the csv file(dating_data.csv)."""
        cleaned_corpus = [self.clean_text(doc) for doc in corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_corpus)

    def get_similarity(self, query):
        """Returns an array of similarity scores for the query."""
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been trained yet.")
            
        cleaned_query = self.clean_text(query)
        query_vec = self.vectorizer.transform([cleaned_query])
        return cosine_similarity(query_vec, self.tfidf_matrix)[0]
