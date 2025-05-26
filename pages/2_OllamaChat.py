import streamlit as st
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from explanation_modes import get_prompt_template, list_prompt_modes

st.set_page_config(page_title="Chat with Ollama", layout="centered")
st.title("🤖 Chat with Local LLM (Ollama)")

@st.cache_resource
def load_rag_documents():
    try:
        with open("rag_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("rag_data.json not found.")
        return []

@st.cache_resource
def get_local_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return ["llama3", "mistral"]
    except:
        return ["llama3", "mistral"]

def retrieve_relevant_docs(query, documents, top_k=3, min_score=0.1):
    texts = [doc["text"] if isinstance(doc, dict) else doc for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [query])
    similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    top_indices = similarity_scores.argsort()[::-1]
    filtered_indices = [i for i in top_indices if similarity_scores[i] > min_score]
    return [texts[i] for i in filtered_indices[:top_k]]

documents = load_rag_documents()
available_models = get_local_models()

st.markdown("### Prompt Configuration")

selected_model = st.selectbox("Select an Ollama model:", options=available_models)

use_rag = st.checkbox("📚 Use Retrieval-Augmented Generation (RAG)", value=True)

prompt_modes = list_prompt_modes()
selected_mode = st.selectbox("🧠 Select a Prompt Template:", options=prompt_modes)
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""

if "last_mode" not in st.session_state:
    st.session_state.last_mode = ""


if selected_mode != st.session_state.last_mode:
    st.session_state.last_mode = selected_mode
    context = ""
    if use_rag and selected_mode in ["Market-Based (RAG)", "Investor Focus"]:
        top_docs = retrieve_relevant_docs("market trends", documents)
        context = "\n\n".join(top_docs)
    st.session_state.prompt_text = get_prompt_template(selected_mode, rag_context=context)

custom_prompt = st.text_area("✍️ Edit Prompt (optional):", key="prompt_text", height=200)

if st.button("Ask Model"):
    if not custom_prompt.strip():
        st.warning("Please enter or select a prompt.")
    else:
        try:
            with st.spinner("Generating response..."):
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": selected_model,
                        "prompt": custom_prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    st.success("Response:")
                    st.markdown(f"\n{answer}\n")
                else:
                    st.error(f"Error from Ollama ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Could not connect to Ollama: {e}")
