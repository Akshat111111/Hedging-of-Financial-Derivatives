import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Title: Analyzing the Impact of User Demographics on Financial Metrics

# Step 1: Load the dataset
file_path = '/mnt/data/sd254_users.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 2: Understand the structure and content of the data
# Display the summary of the dataset
print("\nSummary of the dataset:")
data.info()

# Step 3: Perform Exploratory Data Analysis (EDA)
# 3.1 Summary Statistics
print("\nSummary statistics for numerical columns:")
print(data.describe())

# 3.2 Visualizations
# Plot the distribution of age groups
plt.figure(figsize=(12, 6))
sns.histplot(data['age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the distribution of income levels
plt.figure(figsize=(12, 6))
sns.histplot(data['income'], kde=True, bins=30)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 4: Analyze Relationships
# 4.1 Correlation Matrix
# Compute the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 4.2 Pair Plot
# Plot pairwise relationships
plt.figure(figsize=(14, 10))
sns.pairplot(data[['age', 'income', 'spending_score']])
plt.title('Pairwise Relationships')
plt.show()

# Step 5: Data Preprocessing and Feature Engineering
# Handle missing values and scaling
# 5.1 Handle Missing Values
data.fillna(data.median(), inplace=True)  # Impute missing values with median

# 5.2 Feature Engineering
# Create new features if applicable
# For demonstration, we'll create an interaction term
data['age_income_interaction'] = data['age'] * data['income']

# Feature selection using SelectKBest
X = data[['age', 'income', 'age_income_interaction']]
y = data['spending_score']

# Apply feature selection
feature_selector = SelectKBest(score_func=f_regression, k='all')
X_selected = feature_selector.fit_transform(X, y)
selected_features = X.columns[feature_selector.get_support()]

print("\nSelected Features:")
print(selected_features)

# 5.3 Prepare Data for Modeling
X = data[selected_features]
y = data['spending_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build a Predictive Model
# 6.1 Create a Pipeline
pipeline = make_pipeline(
    StandardScaler(),  # Scale features
    LinearRegression() # Linear Regression model
)

# Train the model
pipeline.fit(X_train, y_train)

# 6.2 Evaluate the Model
# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

# 6.3 Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f'\nCross-validated R-squared scores: {cv_scores}')
print(f'Average Cross-validated R-squared score: {cv_scores.mean():.2f}')

# Step 7: Visualize Predictions vs Actual
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Actual vs Predicted Spending Score')
plt.xlabel('Actual Spending Score')
plt.ylabel('Predicted Spending Score')
plt.grid(True)
plt.show()
