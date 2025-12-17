import pandas as pd
import os
from transformers import pipeline
import numpy as np

DATA_DIR = "data"
#load in books data with categories
#see text_classification.py for how this file was created
books = pd.read_csv(os.path.join(DATA_DIR, "books_with_categories.csv"))

#model for emotion classification/sentiment analysis
classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      top_k = None)

#decided to split description into individual sentences for better emotion detection
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

#function to calculate max emotion scores across sentences in description
def calculate_max_emotion_scores(predictions):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

#look through all books and get max emotion scores for each book description
for i in range(len(books)):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

#create dataframe and merge with books data
emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn
books = pd.merge(books, emotions_df, on = "isbn13")
books.to_csv("books_with_emotions.csv", index = False)

