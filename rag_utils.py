import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_rag_documents(path="rag_data.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_relevant_docs(query, documents, top_n=3):
    if not documents:
        return []
    texts = [doc["text"] if isinstance(doc, dict) else doc for doc in documents]
    vectorizer = TfidfVectorizer().fit(texts + [query])
    doc_vectors = vectorizer.transform(texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return [texts[i] for i in top_indices]