import pandas as pd
import numpy as nps
import joblib
import os
import sys

MODEL_PATH = r"C:\Users\Rabee Abla\Documents\GitHub\AIExam\src\house_price_model.pkl"
CATEGORIES_PATH = r"C:\Users\Rabee Abla\Documents\GitHub\AIExam\src\categories_seen.pkl"

if not os.path.exists(MODEL_PATH):
    print(f" Model file not found at: {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(CATEGORIES_PATH):
    print(f"Category mapping file not found at: {CATEGORIES_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
categories_seen = joblib.load(CATEGORIES_PATH)

new_house = pd.DataFrame([{
    'quarter': '2024Q4',
    'house_type': 'Apartment',
    'sales_type': 'regular_sale',
    'year_build': 2008,
    '%_change_between_offer_and_purchase': -1.5,
    'no_rooms': 3,
    'sqm': 95.0,
    'zip_code': 2100,
    'area': 'Capital, Copenhagen',
    'region': 'Zealand',
    'nom_interest_rate%': 3.2,
    'dk_ann_infl_rate%': 2.1,
    'yield_on_mortgage_credit_bonds%': 4.1,
    'price_per_m2_econ': 41500,
    'quarterly_change%': 0.045,
    'yearly_change%': 0.072,
    'change_since_1992%': 4.2,
    'homes_sold': 9800,
    'sold_q_change': -0.03,
    'sold_yoy_change': 0.065,
    'sold_change_since_1992': 0.34,
    'year': 2024,
    'month': 11,
    'quarter_from_date': 5,
    'house_age': 16
}])


categorical_columns = ['quarter', 'house_type', 'sales_type', 'area', 'region']

for col in categorical_columns:
    new_val = new_house.at[0, col]
    known_vals = categories_seen.get(col, [])

    if new_val not in known_vals:
        print(f"Unknown value in '{col}': '{new_val}'")
        print(f"Allowed values for '{col}': {known_vals}")
        sys.exit(1)

log_price_pred = model.predict(new_house)
price_pred = np.power(10, log_price_pred)

print("\n Prediction Complete:")
print(f"Predicted log10 price: {log_price_pred[0]:.6f}")
print(f"Predicted DKK price: {price_pred[0]:,.0f} DKK")
