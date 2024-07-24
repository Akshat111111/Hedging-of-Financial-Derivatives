import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load the dataset
data_path = '/kaggle/input/sd254_users.csv'  # Update this path if necessary
data = pd.read_csv(data_path)

# Inspect the data
print(data.head())
print(data.info())
print(data.describe())

# Assuming the dataset has the following columns: 'Date', 'StockPrice', 'GDP', 'UnemploymentRate', 'InterestRate', 'InflationRate'
data['Date'] = pd.to_datetime(data['Date'])

# Create additional time-based features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Drop the original Date column
data.drop(columns=['Date'], inplace=True)

# Extract features and target
X = data[['GDP', 'UnemploymentRate', 'InterestRate', 'InflationRate', 'Year', 'Month', 'Day']]
y = data['StockPrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical data
num_cols = ['GDP', 'UnemploymentRate', 'InterestRate', 'InflationRate', 'Year', 'Month', 'Day']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols)
    ])

# Define the full pipeline including the regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'regressor': [LinearRegression(), RandomForestRegressor()],
    'regressor__n_estimators': [100, 200],  # Only for RandomForest
    'regressor__max_depth': [None, 10, 20]  # Only for RandomForest
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best model: {grid_search.best_params_}")

# Evaluate the best model on test data
y_pred = best_model.predict(X_test)

# Regression metrics
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Plot true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Stock Prices')
plt.show()

# Feature importance analysis (only for RandomForest)
if isinstance(best_model.named_steps['regressor'], RandomForestRegressor):
    feature_importance = best_model.named_steps['regressor'].feature_importances_
    feature_names = num_cols
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.show()
