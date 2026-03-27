import unittest
from nlp_engine import NLPEngine

class TestNLPEngine(unittest.TestCase):
    def setUp(self):
        """Runs before every test to set up the environment."""
        self.nlp = NLPEngine()
        self.corpus = [
            "They take hours to text back.",
            "We fight constantly over small things."
        ]
        self.nlp.train(self.corpus)

    def test_clean_text_regex(self):
        """Tests our regex implementation for stripping punctuation."""
        raw_text = "He text$s me at 3:00 AM!!!"
        cleaned = self.nlp.clean_text(raw_text)
        self.assertEqual(cleaned, "he texts me at  am")

    def test_similarity_match(self):
        """Tests that the vectorizer correctly scores a matching sentence higher."""
        scores = self.nlp.get_similarity("Why do they take so long to text back?")
        #should match the first sentence in the corpus better than the second for it to be valid
        self.assertTrue(scores[0] > scores[1]) 
