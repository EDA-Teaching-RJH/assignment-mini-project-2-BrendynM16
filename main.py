import pandas as pd
import sys
from datetime import datetime

#Importing my custom library
from nlp_engine import NLPEngine

# 1. OOP: The Superclass base, which will be the basebot for the algorithm
class BaseBot:
    def __init__(self, bot_name):
        self.bot_name = bot_name

    def greet(self):
        print(f"🤖 Welcome to the {self.bot_name}! 🤖")
        print("Type 'quit' at any time to exit.\n")

    def log_interaction(self, user_input):
        #Writes the user's input to a log file
        with open("chat_history.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"[{timestamp}] User Asked: {user_input}\n")
