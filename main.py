import pandas as pd
import sys
from datetime import datetime

#Importing my custom library
from nlp_engine import NLPEngine

# 1.The Superclass base, which will be the basebot for the algorithm
class BaseBot:
    def __init__(self, bot_name):
        self.bot_name = bot_name

    def greet(self):
        print(f"🤖 Welcome to the {self.bot_name}! 🤖")
        print("Type 'quit' at any time to exit.\n")

    def log_interaction(self, user_input):
        #Writes the user's input to a log file to document so i can further train the bot
        with open("chat_history.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"[{timestamp}] User Asked: {user_input}\n")
class DatingAdviceBot(BaseBot):
    def __init__(self, csv_filename='dating_data.csv'):
        # Call the superclass constructor
        super().__init__("Data-Driven Dating Advice Bot")
        
        
        try:
            self.df = pd.read_csv(csv_filename)
        except FileNotFoundError:
            print(f"❌ Error: Could not find '{csv_filename}'.")
            sys.exit()
        
        #training the Custom NLP Library
        self.nlp = NLPEngine()
        self.nlp.train(self.df['Situation'])

    def get_advice(self, user_input, threshold=0.15):
            # Use custom library to get scores
            similarity_scores = self.nlp.get_similarity(user_input)
        
            best_match_index = similarity_scores.argmax()
            highest_score = similarity_scores[best_match_index]
        
            if highest_score > threshold:
                row = self.df.iloc[best_match_index]
                return row['Situation'], row['Good Advice'], row['Bad Advice'], highest_score
            else:
                return None, None, None, highest_score

    def run(self):
        self.greet() #from my basebot
        
        while True:
            user_input = input("Tell me about your dating situation or problem: ")
            
            if user_input.lower().strip() == 'quit':
                print("Good luck out there! Goodbye.")
                break
                
            self.log_interaction(user_input)
            
            matched_sit, good_adv, bad_adv, score = self.get_advice(user_input, threshold=0.15)
            
            print("-" * 50)
            if matched_sit:
                match_percent = round(score * 100, 1)
                print(f"🔍 Match Found ({match_percent}% confidence).")
                print(f"📝 Situation: '{matched_sit}'\n")
                print(f"✅ GOOD ADVICE:\n   {good_adv}\n")
                print(f"❌ BAD ADVICE:\n   {bad_adv}")
            else:
                print(f"Hmm, it seems like closest match was {round(score * 100, 1)}%. Too low for me to be confident.")
            print("-" * 50 + "\n")
