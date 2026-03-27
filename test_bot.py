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
