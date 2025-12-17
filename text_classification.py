import pandas as pd
import os
import torch
from transformers import pipeline
import numpy as np

DATA_DIR = "data"

#load in cleaned books data
books = pd.read_csv(os.path.join(DATA_DIR, "books_cleaned.csv"))

#create category mapping for top 10 categories inot Fiction and Nonfiction
#see data_exploration.ipynb
category_mapping = {'Fiction' : "Fiction",
 'Juvenile Fiction': "Children's Fiction",
 'Biography & Autobiography': "Nonfiction",
 'History': "Nonfiction",
 'Literary Criticism': "Nonfiction",
 'Philosophy': "Nonfiction",
 'Religion': "Nonfiction",
 'Comics & Graphic Novels': "Fiction",
 'Drama': "Fiction",
 'Juvenile Nonfiction': "Children's Nonfiction",
 'Science': "Nonfiction",
 'Poetry': "Fiction"}

#map categories to new simplified categories
books["simple_categories"] = books["categories"].map(category_mapping)

#create pipeline for zero-shot classification
fiction_categories = ["Fiction", "Nonfiction"]

pipe = pipeline("zero-shot-classification",
                model="facebook/bart-large-mnli")

#function to generate predictions
def generate_predictions(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    max_label = predictions["labels"][max_index]
    return max_label

#Evaulation: How good is this model at doing this task?
actual_cats = []
predicted_cats = []

#test on 300 fiction and 300 nonfiction books
for i in range(0, 300):
    sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Fiction"]

for i in range(0, 300):
    sequence = books.loc[books["simple_categories"] == "Nonfiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Nonfiction"]

#create dataframe to calculate accuracy
predictions_df = pd.DataFrame({"actual_categories": actual_cats, "predicted_categories": predicted_cats})

predictions_df["correct_prediction"] = (
    np.where(predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0)
)
accuracy = predictions_df["correct_prediction"].sum() / len(predictions_df) 
#result - 77.83% accuracy

#generate predictions for missing simple categories
isbns = []
predicted_cats = []

missing_cats = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)

for i in range(0, len(missing_cats)): 
    sequence = missing_cats["description"][i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    isbns += [missing_cats["isbn13"][i]]

missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})

#merge back to original books dataframe
books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"]) #when the original simple categories column is missing, use the predicted categories
#only a subset of the books have predicted categories - accuracy increases
books = books.drop(columns=["predicted_categories"])

books.to_csv("books_with_categories.csv", index=False)