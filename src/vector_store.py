# src/vector_store.py

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_model():
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model loaded successfully.")
    return model

def connect_chromadb(db_path="/home/moahamdreza/Desktop/Ai-Rag/data/chroma_db", collection_name="research_chunks"):
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    print("Connected to ChromaDB.")
    return collection

def embed_and_store_json(data_path, model, collection):
    json_files = [f for f in os.listdir(data_path) if f.endswith(".json")]
    print(f"{len(json_files)} JSON files found in {data_path}.")
    
    for file_name in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(data_path, file_name)
        print(f"\nProcessing file: {file_name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        if not isinstance(documents, list):
            print(f"Skipped {file_name}: JSON root is not a list.")
            continue

        for idx, doc in enumerate(documents):
            # Check for required keys
            if not all(key in doc for key in ["chunk_text", "id", "title", "metadata"]):
                print(f"Skipped chunk {idx} in {file_name}: missing required keys.")
                continue

            try:
                text = doc["chunk_text"]
                chunk_id = doc["id"]
                title = doc["title"]
                metadata = doc.get("metadata", {})

                author = metadata.get("author", "Unknown")
                keywords = metadata.get("keywords", [])
                keywords_str = ", ".join(keywords)

                print(f"Embedding chunk {idx + 1}/{len(documents)} in {file_name}...")
                vector = model.encode(text).tolist()

                collection.add(
                    ids=[chunk_id],
                    embeddings=[vector],
                    metadatas=[{
                        "article_id": chunk_id,
                        "title": title,
                        "author": author,
                        "keywords": keywords_str
                    }]
                )
                print(f"Chunk {chunk_id} inserted successfully.")
            except Exception as e:
                print(f"Error inserting chunk {chunk_id}: {e}")

if __name__ == "__main__":
    # Example usage:
    model = load_model()
    collection = connect_chromadb()
    DATA_PATH = "/home/moahamdreza/Desktop/Ai-Rag/data/JSONFiles"  # Adjust this path as needed
    embed_and_store_json(DATA_PATH, model, collection)
