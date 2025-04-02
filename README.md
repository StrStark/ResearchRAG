# **Enhancing Small LLM Responses with RAG over Scientific Documents on Computational Complexity**

This project presents a Retrieval-Augmented Generation (RAG) pipeline tailored for scientific PDFs. It transforms PDFs into structured JSON, extracts key information, embeds text using Sentence Transformers, stores embeddings in ChromaDB, and leverages a generative model (e.g., T5) to answer questions based on retrieved content.

---

## Overview

This is a complete RAG system designed for scientific research workflows. It operates through four main stages:

1. **Ingestion:** Parses scientific PDFs into JSON format by extracting, chunking text, and computing relevant keywords.
2. **Vector Storage:** Creates embeddings for document chunks using Sentence Transformers and saves them in ChromaDB.
3. **Retrieval:** Fetches the most semantically relevant chunks from the vector database based on a given query.
4. **Generation:** Uses a generative model to produce informative, context-aware answers using the retrieved document context.

---

## Repository Structure

```
ResearchRAG/
├── data/
│   ├── pdfFiles/         # Input PDF documents
│   ├── JSONFiles/        # JSON output from PDF ingestion
│   └── chroma_db/        # Persistent vector database (ChromaDB)
├── src/
│   ├── ingestion.py      # Script for PDF to JSON conversion
│   ├── vector_store.py   # Embedding and storage logic
│   ├── retrieval.py      # Semantic search and document retrieval
│   └── generation.py     # Answer generation using retrieved content
├── notebooks/            # (Optional) Jupyter notebooks for experimentation
├── requirements.txt      # Dependency list
└── README.md             # Project overview and usage guide
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/StrStark/ResearchRAG.git
   cd ResearchRAG
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv env
   # Activate it with:
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

The project requires Python 3.7+ and the following libraries (see `requirements.txt` for exact versions):

- PyMuPDF
- yake
- chromadb
- sentence-transformers
- transformers
- torch
- tqdm

---

## Usage

### 1. Ingest PDFs

- **Add your PDFs:**  
  Place your scientific PDF files in the `data/pdfFiles/` folder.

- **Run the ingestion process:**  
  Execute the script to extract and chunk text and output JSON:

  ```bash
  python src/ingestion.py
  ```

  The JSON files will appear in `data/JSONFiles/`.

### 2. Store Embeddings

- **Generate and save embeddings:**  
  Run the following script to compute embeddings and store them in ChromaDB:

  ```bash
  python src/vector_store.py
  ```

### 3. Retrieve and Generate

- **Get answers using RAG:**  
  Use this script to retrieve relevant chunks and generate responses:

  ```bash
  python src/generation.py
  ```

  It will retrieve the most relevant text chunks from ChromaDB, combine them with your query, and pass them to a generative model for response generation.

---

## Acknowledgements

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)