from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DATA_DIR = "data"
load_dotenv(override=True)

#load in cleaned books data
books = pd.read_csv(os.path.join(DATA_DIR, "books_cleaned.csv"))

#embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#prepare tagged descriptions for vector DB
np.savetxt(
    "tagged_description.txt",
    books["tagged_description"].dropna().astype(str).values,
    fmt="%s",
    encoding="utf-8"
)

#create vector DB
raw_documents = TextLoader(os.path.join(DATA_DIR, "tagged_description.txt")).load()
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

db_books = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

#get recommendations based on semantic search
def retrieve_semantic_recommendations(
        query: str,
        top_k: int = 10,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k = 50)
    books_list = []
    for i in range(0, len(recs)):
        books_list += [int(recs[i].page_content.strip('"').split()[0])]
    return books[books["isbn13"].isin(books_list)].head(top_k)

#test function
retrieve_semantic_recommendations("A book to teach children about nature")

