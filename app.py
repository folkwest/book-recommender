import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings

#Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()

#data directory variable, not used in space
DATA_DIR = "data"

#load in books data with emotions
#see sentiment_analysis.py for how this file was created
def load_books(): #speed up build time
    #return pd.read_csv(os.path.join(DATA_DIR, "books_with_emotions.csv"))
    return pd.read_csv("books_with_emotions.csv")

books = load_books()

#handle missing thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
     books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

#prepare tagged descriptions for vector DB
#raw_documents = TextLoader(os.path.join(DATA_DIR, "tagged_description.txt"), encoding="utf-8").load()
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

#create or load vector DB
persist_dir = "./chroma"

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    # Load existing DB
    db_books = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
else:
    # Build DB from documents and persist
    db_books = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=persist_dir)

#get recommendations based on semantic search
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    #dropdown, sort by tone (emotion)
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

#recommend books function to be called on button click
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split() #truncate to 30 words (limited space)
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2: #two authors -> use "and"
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2: #more than two authors -> use commas and "and"
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else: #only one author
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}" #captions under book thumbnail image
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["None", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"] + ["All"]

#gradio interface
with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book recommender") #title of the dashboard

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness") #textbox for user to enter book description
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All") #category dropdown
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All") #tones dropdown
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860, theme = gr.themes.Soft())

'''
if __name__ == "__main__":
    dashboard.run(host="0.0.0.0", port=7860)
    '''

