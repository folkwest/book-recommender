---
title: BookWyrm
sdk: gradio
app_file: app.py
startup_duration_timeout: 1hr
---

## BookWyrm

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/HF-BookWyrm-orange?logo=huggingface)](https://huggingface.co/spaces/cternero/book-wyrm)

**Semantic Book Recommender** that suggests books based on your description, with optional filtering by category and emotional tone.

![Demo Screenshot](cover-not-found.jpg)

---

## Features

- **Semantic Search**: Find books similar to your description using embeddings.  
- **Category Filter**: Narrow results by book category.  
- **Emotional Tone Filter**: Sort results by tone (Happy, Sad, Suspenseful, Angry, Surprising).  
- **Visual Recommendations**: Displays book covers with information and short descriptions.

---

## Live Demo

Try it now on Hugging Face Spaces: [BookWyrm Demo](https://huggingface.co/spaces/cternero/book-wyrm)

---

## Tech Stack

- **Python 3.10+**  
- **Gradio** – Web interface  
- **LangChain** – Semantic embeddings  
- **Chroma** – Vector database for fast similarity search  
- **pandas / numpy** – Data processing  

---

## How it works
- Load books metadata and descriptions.
- Split text into chunks (CharacterTextSplitter).
- Generate embeddings using HuggingFaceEmbeddings.
- Store embeddings in Chroma for similarity search.
- Zero-shot text classification with facebook/bart-large-mnli
- Sentiment analysis with j-hartmann/emotion-english-distilroberta-base
- User inputs a query → retrieve similar books → filter by category/tone → display results.

---

## Future Improvements
- Update database (current data only has 7k books)
- Add ability to filter by page numbers

---

## Quick Start

```bash
# Clone repo
git clone https://github.com/yourusername/book-wyrm.git
cd book-wyrm

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## License
MIT License


