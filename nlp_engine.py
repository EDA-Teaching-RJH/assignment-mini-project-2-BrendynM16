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
