# src/retrieval.py

from sentence_transformers import SentenceTransformer

# Assuming you use the same model as in vector_store
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_query(query, model):
    return model.encode(query).tolist()

def retrieve_chunks(query, model, collection, top_k=5):
    query_embedding = encode_query(query, model)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

# Example usage:
if __name__ == "__main__":
    from vector_store import connect_chromadb  # Import the connection function from vector_store
    model = load_model()
    collection = connect_chromadb()
    query = "How do temporal reachability graphs help in network analysis?"
    results = retrieve_chunks(query, model, collection)
    print("Retrieved results:", results)
