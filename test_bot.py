import unittest
from nlp_engine import NLPEngine

class TestNLPEngine(unittest.TestCase):
    def setUp(self):
        """will run before every test to set up the correct environment."""
        self.nlp = NLPEngine()
        self.corpus = [
            "They take hours to text back.",
            "We fight constantly over small things."
        ]
        self.nlp.train(self.corpus)
