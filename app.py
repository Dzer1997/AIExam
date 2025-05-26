import streamlit as st


st.set_page_config(page_title="Local AI App", layout="centered")


st.title("Welcome to the Local AI App")

st.write("Use the **sidebar** to navigate between the tools.")


st.markdown("### Available Tools:")
st.markdown("This app helps users explore and understand the housing market in Denmark by using local AI models to answer questions and predict prices.")
st.markdown("**Example questions you can ask:**")
st.markdown("- What are the housing trends in Denmark since 2000?")
st.markdown("- Why is housing more expensive in Copenhagen?")
st.markdown("- How do mortgage rates affect house prices?")
st.markdown("- What are the typical house sizes in different regions?")
st.markdown("- What makes Cluster 2 homes more valuable?")

st.markdown(" **House Price Predictor**")
st.markdown("**Chat with Local LLM (Ollama + RAG)**")
