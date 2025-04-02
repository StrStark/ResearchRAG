# ResearchRAG

ResearchRAG is a Retrieval-Augmented Generation (RAG) pipeline designed for scientific documents. It ingests PDF files, converts them into structured JSON with text chunking and keyword extraction, embeds the text using Sentence Transformers, stores the embeddings in a ChromaDB vector database, and finally uses a generative model (e.g., T5) to answer questions based on retrieved document context.

---

## Overview

This project demonstrates an end-to-end RAG system tailored for scientific research. The pipeline consists of four primary steps:

1. **Ingestion:** Convert scientific PDFs into JSON files by extracting and chunking text and computing keywords.
2. **Vector Storage:** Generate embeddings for each document chunk using Sentence Transformers and store them in a persistent ChromaDB.
3. **Retrieval:** Query the vector database to retrieve the most relevant document chunks based on semantic similarity.
4. **Generation:** Use a generative model with the retrieval context to produce detailed, context-aware answers.

---

## Repository Structure

```
ResearchRAG/
├── data/
│   ├── pdfFiles/         # Raw PDF documents
│   ├── JSONFiles/        # Output JSON files from PDF ingestion
│   └── chroma_db/        # Persistent ChromaDB database
├── src/
│   ├── ingestion.py      # Code to convert PDFs to JSON
│   ├── vector_store.py   # Embedding generation & storage in ChromaDB
│   ├── retrieval.py      # Retrieval of relevant document chunks
│   └── generation.py     # Generation (RAG) with and without retrieval context
├── notebooks/            # Optional Jupyter notebooks for prototyping
├── requirements.txt      # Project dependencies
└── README.md             # Project overview and instructions
```

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/StrStark/ResearchRAG.git
   cd ResearchRAG
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv env
   # Activate the environment:
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Ingest PDFs

- **Place your PDFs:**  
  Put your scientific PDF files in the `data/pdfFiles/` directory.

- **Convert PDFs to JSON:**  
  Run the ingestion script to extract text, chunk it, and generate JSON files:

  ```bash
  python src/ingestion.py
  ```

  JSON files will be created in the `data/JSONFiles/` directory.

### 2. Store Data in ChromaDB

- **Embed and Store:**  
  Run the vector store script to generate embeddings from JSON and store them in ChromaDB:

  ```bash
  python src/vector_store.py
  ```

### 3. Retrieve and Generate Answers

- **Test Retrieval & Generation:**  
  The generation script demonstrates both a direct model output and a retrieval-augmented answer:

  ```bash
  python src/generation.py
  ```

  The script retrieves the top relevant document chunks from ChromaDB and feeds their context along with your query into a generative model to produce an answer.

---

## Customization

- **Ingestion:**  
  Modify `src/ingestion.py` to tweak text cleaning, chunking size, or keyword extraction settings.

- **Embedding & Retrieval:**  
  Adjust settings in `src/vector_store.py` and `src/retrieval.py` if you want to experiment with different embedding models or retrieval configurations.

- **Generation:**  
  Enhance `src/generation.py` by refining prompt templates, adjusting generation parameters (like `max_new_tokens`), or swapping in a model fine-tuned on QA tasks.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## Contact

For any questions or suggestions, please feel free to open an issue or contact the repository maintainer.
