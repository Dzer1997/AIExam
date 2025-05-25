import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_utils import load_rag_documents, retrieve_relevant_docs

MODEL_PATH = "src/house_price_model.pkl"
CATEGORIES_PATH = "scr/categories_seen.pkl"

st.title("House Price Predictor (DKK)")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Please train it first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    categories = joblib.load(CATEGORIES_PATH) if os.path.exists(CATEGORIES_PATH) else {}
    return model, categories

model, categories_seen = load_model()

def explain_prediction_with_rag(prompt, documents, model_name="llama3", top_n=3):
    texts = [doc["text"] if isinstance(doc, dict) else doc for doc in documents]
    vectorizer = TfidfVectorizer().fit(texts + [prompt])
    doc_vectors = vectorizer.transform(texts)
    query_vector = vectorizer.transform([prompt])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    context = "\n\n".join([texts[i] for i in top_indices])

    full_prompt = (
        f"Based on the following information:\n{context}\n\n"
        f"Explain the reasoning behind this house price prediction:\n{prompt}"
    ) if context else prompt

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": full_prompt,
            "stream": False
        }
    )
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No explanation returned.")
    else:
        return f"Error from Ollama API: {response.status_code}"

with st.form("prediction_form"):
    st.header("Enter House Details")

    quarter = st.selectbox("Quarter", options=categories_seen.get("quarter", ["2024Q4", "2024Q3"]))
    house_type = st.selectbox("House Type", options=categories_seen.get("house_type", ["Villa", "Apartment"]))
    sales_type = st.selectbox("Sales Type", options=categories_seen.get("sales_type", ["regular_sale"]))
    area = st.selectbox("Area", options=categories_seen.get("area", ["Capital, Copenhagen"]))
    region = st.selectbox("Region", options=categories_seen.get("region", ["Zealand"]))

    year_build = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
    house_age = 2024 - year_build

    prediction_year = st.number_input("Prediction Year", min_value=1990, max_value=2030, value=2024)

    col1, col2, col3 = st.columns(3)
    with col1:
        no_rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=3)
        sqm = st.number_input("Square Meters", min_value=10.0, max_value=500.0, value=100.0)
    with col2:
        zip_code = st.number_input("Zip Code", min_value=1000, max_value=9990, value=2100)
        price_per_m2 = st.number_input("Economic Price per m²", value=30000)
    with col3:
        interest_rate = st.number_input("Nominal Interest Rate (%)", value=3.1)
        inflation = st.number_input("Annual Inflation Rate (%)", value=2.0)
        yield_bonds = st.number_input("Mortgage Credit Bond Yield (%)", value=4.2)

    q_change = st.number_input("Quarterly Price Change (%)", value=0.02)
    y_change = st.number_input("Yearly Price Change (%)", value=0.05)
    since_1992_change = st.number_input("Change since 1992 (%)", value=4.0)

    sold_homes = st.number_input("Homes Sold", value=9000)
    sold_q = st.number_input("Sold Quarter Change", value=-0.02)
    sold_y = st.number_input("Sold YoY Change", value=0.03)
    sold_since_1992 = st.number_input("Sold Change since 1992", value=0.3)

    offer_diff = st.number_input("Offer vs Purchase (%)", value=-1.5)

    month = st.number_input("Month", value=10, min_value=1, max_value=12)
    quarter_from_date = st.number_input("Quarter From Date", value=4)

    submitted = st.form_submit_button("Predict Price")

if submitted and model:
    house_age = prediction_year - year_build

    new_house = pd.DataFrame([{
        'quarter': quarter,
        'house_type': house_type,
        'sales_type': sales_type,
        'year_build': year_build,
        '%_change_between_offer_and_purchase': offer_diff,
        'no_rooms': no_rooms,
        'sqm': sqm,
        'zip_code': zip_code,
        'area': area,
        'region': region,
        'nom_interest_rate%': interest_rate,
        'dk_ann_infl_rate%': inflation,
        'yield_on_mortgage_credit_bonds%': yield_bonds,
        'price_per_m2_econ': price_per_m2,
        'quarterly_change%': q_change,
        'yearly_change%': y_change,
        'change_since_1992%': since_1992_change,
        'homes_sold': sold_homes,
        'sold_q_change': sold_q,
        'sold_yoy_change': sold_y,
        'sold_change_since_1992': sold_since_1992,
        'year': prediction_year,
        'month': month,
        'quarter_from_date': quarter_from_date,
        'house_age': house_age
    }])

    for col in ['quarter', 'house_type', 'sales_type', 'area', 'region']:
        val = new_house.at[0, col]
        known = categories_seen.get(col, [])
        if known and val not in known:
            st.error(f"Unknown '{col}' value: '{val}'. Must be one of: {known}")
            st.stop()

    log_price = model.predict(new_house)
    dkk_price = np.power(10, log_price)

    st.success(f"Predicted DKK price: **{dkk_price[0]:,.0f} kr.**")
    st.write(f"Log10 price: {log_price[0]:.6f}")

    st.subheader("Entered House Information:")
    st.dataframe(new_house)

    if st.button("Explain Prediction"):
        explanation_prompt = (
            f"House details: {new_house.to_dict(orient='records')[0]}\n"
            f"Predicted price (log10): {log_price[0]:.6f}"
        )
        rag_docs = load_rag_documents()
        explanation = explain_prediction_with_rag(explanation_prompt, rag_docs, "llama3")
        st.subheader("Prediction Explanation:")
        st.write(explanation)

if st.button("Retrain Model"):
    st.info("To retrain, run: `python src/train_model.py` from your terminal.")
