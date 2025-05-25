# AIExam
python -m venv LocalAI
LocalAI\Scripts\activate eller source ./LocalAI/Scripts/activate -> which python -> streamlit run app.py --server.address 127.0.0.1
deactivate
pip install -r requirements.txt
# Run Streamlit
streamlit run app.py
python src\train_model.py
This script will:

Load your cleaned dataframe (homes_cleaned_log)

Drop leaky columns like log_price, purchase_price, etc.

Split the dataset into training/testing

Train the stacked model

Save the model to house_price_model.pkl

Save all seen categories to categories_seen.pkl
Predict a New House Price
Once trained, you can make predictions on new house data using:
python src\predict_house.py
Example new_house input (defined inside predict_house.py):
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
The script will:

Load the trained model and categories

Validate new inputs for unseen categorical values

Predict the log10 price and convert to DKK

Display clean output
Category Validation
Before prediction, the script checks:

If all categorical values in new_house are among those seen during training.

If any unknown category is found, it will raise an error to avoid incorrect predictions.

gitignore

# Virtual environments
LocalAI/

# Model files
*.pkl

# Python cache
__pycache__/
*.pyc


