import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
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
documents = load_rag_documents()

def house_summary(row):
    summary = (
        f"Quarter: {row['quarter']}, House Type: {row['house_type']}, Sales Type: {row['sales_type']}, "
        f"Year Built: {row['year_build']}, Rooms: {row['no_rooms']}, Area: {row['area']}, Region: {row['region']}, "
        f"Square Meters: {row['sqm']}, Zip Code: {row['zip_code']}, Interest Rate: {row['nom_interest_rate%']}%, "
        f"Inflation: {row['dk_ann_infl_rate%']}%, Mortgage Bond Yield: {row['yield_on_mortgage_credit_bonds%']}%, "
        f"Price per m2 (econ): {row['price_per_m2_econ']}, Quarterly Change: {row['quarterly_change%']*100:.2f}%, "
        f"Yearly Change: {row['yearly_change%']*100:.2f}%"
    )
    return summary

with st.form("prediction_form"):
    st.header("Enter House Details")

    quarter = st.selectbox("Quarter", options=categories_seen.get("quarter", ["2024Q4", "2024Q3"]))
    house_type = st.selectbox("House Type", options=categories_seen.get("house_type", ["Villa", "Apartment"]))
    sales_type = st.selectbox("Sales Type", options=categories_seen.get("sales_type", ["regular_sale"]))
    area = st.selectbox("Area", options=categories_seen.get("area", ["Capital, Copenhagen"]))
    region = st.selectbox("Region", options=categories_seen.get("region", ["Zealand"]))

    year_build = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
    prediction_year = st.number_input("Prediction Year", min_value=1990, max_value=2030, value=2024)

    col1, col2, col3 = st.columns(3)
    with col1:
        no_rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=3)
        sqm = st.number_input("Square Meters", min_value=10.0, max_value=500.0, value=100.0)
    with col2:
        zip_code = st.number_input("Zip Code", min_value=1000, max_value=9990, value=2100)
        price_per_m2 = st.number_input("Economic Price per mÂ²", value=30000)
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

    use_rag = st.checkbox("Use RAG to explain prediction", value=True)

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

    if use_rag:
        summary_text = house_summary(new_house.iloc[0])
        top_docs = retrieve_relevant_docs(summary_text, documents)
        rag_context = "\n\n".join([doc["text"] if isinstance(doc, dict) else doc for doc in top_docs])
    else:
        rag_context = ""

    full_prompt = (
        f"Explain the predicted house price based on the following house details:\n{summary_text}\n\n"
        f"Use this market context:\n{rag_context}\n"
        "Provide a clear and concise explanation suitable for a home buyer."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",  # or use a selected model variable if you want
                "prompt": full_prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            result = response.json()
            explanation = result.get("response", "").strip()
            st.markdown("### AI Explanation:")
            st.write(explanation)
        else:
            st.error(f"Error from Ollama: {response.status_code}")
    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")

if st.button("Retrain Model"):
    st.info("To retrain, run: `python src/train_model.py` from your terminal.")
