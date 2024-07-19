import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load the dataset
data_path = '/kaggle/input/sd254_users.csv'  # Update this path if necessary
data = pd.read_csv(data_path)

# Inspect the data
print(data.head())
print(data.info())
print(data.describe())

# Assuming the dataset has the following columns: 'TransactionDate', 'StockSymbol', 'TransactionType', 'SharesTraded', 'SharePrice', 'InsiderRole'
# Create a target column based on insider trading activity (e.g., 1 for insider trading, 0 for normal trading)
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['Month'] = data['TransactionDate'].dt.month
data['Year'] = data['TransactionDate'].dt.year

# Create a target variable for insider trading
# Assuming 'InsiderRole' indicates whether the transaction is by an insider
data['InsiderTrading'] = np.where(data['InsiderRole'].notnull(), 1, 0)

# Extract features and target
X = data[['StockSymbol', 'TransactionType', 'SharesTraded', 'SharePrice', 'Month', 'Year']]
y = data['InsiderTrading']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define preprocessing steps for numerical and categorical data
num_cols = ['SharesTraded', 'SharePrice', 'Month', 'Year']
cat_cols = ['StockSymbol', 'TransactionType']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Define the full pipeline including the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier': [LogisticRegression(), RandomForestClassifier()],
    'classifier__C': [0.1, 1.0, 10.0],  # Only for LogisticRegression
    'classifier__n_estimators': [100, 200]  # Only for RandomForest
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# Best model and its parameters
best_model = grid_search.best_estimator_
print(f"Best model: {grid_search.best_params_}")

# Evaluate the best model on test data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Feature importance analysis (only for RandomForest)
if isinstance(best_model.named_steps['classifier'], RandomForestClassifier):
    feature_importance = best_model.named_steps['classifier'].feature_importances_
    feature_names = num_cols + list(best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.show()
