# src/ingestion.py

import fitz
import re
import uuid
import json
import yake
import os

def extract_keywords(text, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(n=1, top=max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def clean(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunkDeviding(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def give_data_structure(chunks, metadata):
    structured_chunks = []
    for chunk in chunks:
        structured_chunks.append({
            "id": str(uuid.uuid4()),
            "title": metadata.get("title", ""),
            "chunk_text": chunk,
            "metadata": {
                "author": metadata.get("author", ""),
                "keywords": extract_keywords(chunk, 10)
            }
        })
    return structured_chunks

def store_data_from_pdf_to_json(pdf_path, destination):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + " "
    clean_text = clean(full_text)
    chunks = chunkDeviding(clean_text)
    structured_data = give_data_structure(chunks, doc.metadata)
    output_path = os.path.join(destination, f"{uuid.uuid4()}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)

def pdf_to_json(pdf_folder, json_destination):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    for pdf in pdf_files:
        store_data_from_pdf_to_json(os.path.join(pdf_folder, pdf), json_destination)

if __name__ == "__main__":
    pdf_folder = "/home/moahamdreza/Desktop/Ai-Rag/data/pdfFiles"  # adjust path if necessary
    json_destination = "/home/moahamdreza/Desktop/Ai-Rag/data/JSONFiles"
    pdf_to_json(pdf_folder, json_destination)
