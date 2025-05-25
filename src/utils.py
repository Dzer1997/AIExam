import os
import joblib

def validate_new_house(new_house, model_input_columns, category_path=None):
    # Check all required columns exist
    if not all(col in new_house.columns for col in model_input_columns):
        missing = set(model_input_columns) - set(new_house.columns)
        raise ValueError(f" Missing columns: {missing}")
    if new_house.isnull().any().any():
        raise ValueError(" New house data has missing values.")

    if category_path and os.path.exists(category_path):
        categories_seen = joblib.load(category_path)
        for col in categories_seen:
            val = new_house[col].iloc[0]
            if val not in categories_seen[col]:
                raise ValueError(f"Unknown value '{val}' in column '{col}'")

    print("Validation passed. New house data looks good.")
