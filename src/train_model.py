import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

basePath = r"C:\Users\ermin\Documents\GitHub\AIExam\Data"
filePath = os.path.join(basePath, "DKHousingPricesSample100k.csv")

homes = pd.read_csv(filePath)
homes = homes.drop(columns=['city'])
homes['dk_ann_infl_rate%'] = homes['dk_ann_infl_rate%'].fillna(homes['dk_ann_infl_rate%'].median())
homes['yield_on_mortgage_credit_bonds%'] = homes['yield_on_mortgage_credit_bonds%'].fillna(homes['yield_on_mortgage_credit_bonds%'].median())
filePath = os.path.join(basePath, "economic_data.xlsx")
economic_data = pd.read_excel(filePath)
economic_data.rename(columns={
    'Price per m²': 'price_per_m2_econ',
    'Change (from previous quarter)': 'quarterly_change%',
    'Change from 1 year ago': 'yearly_change%',
    'Change since 1992': 'change_since_1992%',
    'house_type': 'house_type'
}, inplace=True)
merged_data = pd.merge(homes, economic_data, on=['quarter', 'house_type'], how='left')
filePath = os.path.join(basePath, "number_Home_sold.xlsx")
number_sold = pd.read_excel(filePath)

number_sold = number_sold.rename(columns={
    "Number of home sold": "homes_sold",
    "Change (from previous quarter)": "sold_q_change",
    "Change from 1 year ago": "sold_yoy_change",
    "Change since 1992": "sold_change_since_1992"
})

final_df = merged_data.merge(number_sold, on=["quarter", "house_type"], how="left")
final_df['date'] = pd.to_datetime(final_df['date'], errors='coerce')
final_df['log_price'] = np.log10(final_df['purchase_price'])


Q1_log = final_df['log_price'].quantile(0.25)
Q3_log = final_df['log_price'].quantile(0.75)
IQR_log = Q3_log - Q1_log

lower_log = Q1_log - 1.5 * IQR_log
upper_log = Q3_log + 1.5 * IQR_log

print(f"Lower whisker on log scale: {lower_log:.2f}")
print(f"Upper whisker on log scale: {upper_log:.2f}")

upper_original = 10 ** upper_log
print(f"Upper whisker in original scale: {upper_original:,.0f}")

outliers_log = final_df[(final_df['log_price'] < lower_log) | (final_df['log_price'] > upper_log)]
low_outliers_log = final_df[final_df['log_price'] < lower_log]
high_outliers_log = final_df[final_df['log_price'] > upper_log]

homes_cleaned_log = final_df[
    (final_df['log_price'] >= lower_log) &
    (final_df['log_price'] <= upper_log)
].copy()
import numpy as np

homes_cleaned_log['log10_price'] = np.log10(homes_cleaned_log['purchase_price'])

upper_whisker_log10 = 7.11

outliers_log = homes_cleaned_log[homes_cleaned_log['log10_price'] > upper_whisker_log10]
homes_cleaned_log = homes_cleaned_log.drop(
    columns=['log_purchase_price', 'log_price', 'log10_price'], 
    errors='ignore'
)
homes_cleaned_log['log_price'] = np.log10(homes_cleaned_log['purchase_price'])
homes_cleaned_log = homes_cleaned_log.copy()

homes_cleaned_log['date'] = pd.to_datetime(homes_cleaned_log['date'])

homes_cleaned_log['year'] = homes_cleaned_log['date'].dt.year
homes_cleaned_log['month'] = homes_cleaned_log['date'].dt.month
homes_cleaned_log['quarter_from_date'] = homes_cleaned_log['date'].dt.quarter  

homes_cleaned_log['house_age'] = homes_cleaned_log['year'] - homes_cleaned_log['year_build']

homes_cleaned_log['price_per_room'] = homes_cleaned_log['purchase_price'] / homes_cleaned_log['no_rooms']

homes_cleaned_log['room_density'] = homes_cleaned_log['sqm'] / homes_cleaned_log['no_rooms']

homes_cleaned_log = homes_cleaned_log.drop(columns=[
    col for col in ['log_purchase_price', 'log_price', 'log10_price'] if col in homes_cleaned_log.columns
])

homes_cleaned_log['log_price'] = np.log10(homes_cleaned_log['purchase_price'])
homes_cleaned_log = homes_cleaned_log.drop(columns=['address', 'house_id'])
y = homes_cleaned_log['log_price']


leaky_columns = [
    'purchase_price', 'log_price', 'log_purchase_price', 'log10_price',
    'sqm_price', 'price_per_room', 'room_density', 'date'
]

X = homes_cleaned_log.drop(columns=[col for col in leaky_columns if col in homes_cleaned_log.columns], errors='ignore')

categorical_features = ['quarter', 'house_type', 'sales_type', 'area', 'region']
numeric_features = [col for col in X.columns if col not in categorical_features]

CATEGORIES_PATH = 'src/categories_seen.pkl'
categories_seen = {col: X[col].unique().tolist() for col in categorical_features}
joblib.dump(categories_seen, CATEGORIES_PATH)

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
])

best_rf_params = {
    'n_estimators': 200,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': None,
    'max_depth': 30,
    'bootstrap': True,
    'random_state': 42
}

best_xgb_params = {
    'n_estimators': 400,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'reg_alpha': 1,
    'reg_lambda': 1.5,
    'random_state': 42
}

base_estimators = [
    ('rf', RandomForestRegressor(**best_rf_params)),
    ('xgb', XGBRegressor(**best_xgb_params))
]

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stack', StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        passthrough=True,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

log_preds = model_pipeline.predict(X_test)
price_preds = np.power(10, log_preds)
actual_prices = np.power(10, y_test)

rmse = np.sqrt(mean_squared_error(actual_prices, price_preds))
r2 = r2_score(actual_prices, price_preds)

print(f" RMSE: {rmse:,.2f}")
print(f" R² Score: {r2:.4f}")

MODEL_PATH = 'src/house_price_model.pkl'
joblib.dump(model_pipeline, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
print(f"Categories saved to: {CATEGORIES_PATH}")


