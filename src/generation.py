# src/generation.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_generator(model_name="t5-base"):
    """
    Load the tokenizer and model for generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_answer_direct(query, tokenizer, model):
    """
    Generate a detailed answer directly from the model without retrieval context.
    """
    prompt = f"Please provide a detailed explanation for the following question:\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,  # Allow generation of 150 new tokens beyond the prompt
        num_beams=10,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def build_context(retrieved_results):
    """
    Build a context string from the metadata in the retrieved results.
    """
    context_parts = []
    # Assuming retrieved_results['metadatas'] is a nested list; iterate over the first list element.
    for meta in retrieved_results.get("metadatas", [])[0]:
        title = meta.get("title", "No Title")
        keywords = meta.get("keywords", "")
        context_parts.append(f"Title: {title}\nKeywords: {keywords}\n")
    return "\n".join(context_parts)

def generate_answer_with_context(query, context, tokenizer, model):
    """
    Generate a detailed answer using the retrieval context along with the query.
    """
    prompt = (
        "You are an expert in network analysis. "
        "Please provide a detailed explanation based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,  # Allow generation of 150 new tokens beyond the prompt
        num_beams=10,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # Example usage:
    # Import retrieval functions and ChromaDB connection from your other modules.
    from retrieval import load_model, retrieve_chunks
    from vector_store import connect_chromadb

    # Load the generative model
    tokenizer, gen_model = load_generator()

    # Define the query
    query = "How do temporal reachability graphs help in network analysis?"

    # 1. Get the direct answer (without retrieval context)
    direct_answer = generate_answer_direct(query, tokenizer, gen_model)
    
    # 2. Get the RAG answer (with retrieval context)
    retrieval_model = load_model()
    collection = connect_chromadb(db_path="/home/moahamdreza/Desktop/Ai-Rag/data/chroma_db")
    
    # Retrieve the top 5 relevant results
    retrieved_results = retrieve_chunks(query, retrieval_model, collection, top_k=5)
    context = build_context(retrieved_results)
    rag_answer = generate_answer_with_context(query, context, tokenizer, gen_model)
    
    # Print both outputs for comparison
    print("Direct Generation Answer:", direct_answer)
    print("RAG Answer:", rag_answer)
