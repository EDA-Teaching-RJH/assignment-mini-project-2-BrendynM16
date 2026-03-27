import pandas as pd
import sys
from datetime import datetime

#Importing my newly created custom library
from nlp_engine import NLPEngine

# 1. OOP: The Superclass base which will e the basebot for the algorithm
class BaseBot:
    def __init__(self, bot_name):
        self.bot_name = bot_name

    def greet(self):
        print(f"🤖 Welcome to the {self.bot_name}! 🤖")
        print("Type 'quit' at any time to exit.\n")

