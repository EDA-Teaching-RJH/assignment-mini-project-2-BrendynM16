import unittest
from nlp_engine import NLPEngine

class TestNLPEngine(unittest.TestCase):
    def setUp(self):
        #Runs before every test to set up the environment
        self.nlp = NLPEngine()
        self.corpus = [
            "They take hours to text back.",
            "We fight constantly over small things."
        ]
        self.nlp.train(self.corpus)

    def test_clean_text_regex(self):
        #Tests our regex for punctuation
        raw_text = "He onl$y text*s me at 3:00 AM!!!"
        cleaned = self.nlp.clean_text(raw_text)
        self.assertEqual(cleaned, "he only texts me at  am")

    def test_similarity_match(self):
        #Tests the vectorizer, scores a matching sentence.
        scores = self.nlp.get_similarity("Why do they take so long to text back?")
        #this should match the first sentence in the corpus better than the second for it to be valid
        self.assertTrue(scores[0] > scores[1]) 

    def test_edge_case_gibberish(self):
        #if the user inputs total gibberish with no actual wors 
        scores = self.nlp.get_similarity("xyz abc fgh")
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)

if __name__ == '__main__':
    unittest.main()
