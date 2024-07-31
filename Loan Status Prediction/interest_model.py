import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """
    Load dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    """
    Preprocess the dataset for modeling.
    
    Parameters:
    df (DataFrame): The input dataset.
    target_column (str): The name of the target column.
    
    Returns:
    DataFrame, Series: Features and target variable.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_model(X, y):
    """
    Train the interest rate prediction model.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target variable.
    
    Returns:
    model: Trained model.
    """
    model = MLPRegressor(hidden_layer_sizes=(32, 16, 8), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=500)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Parameters:
    model: Trained model.
    X_test (array-like): Test feature matrix.
    y_test (array-like): Test target variable.
    
    Returns:
    dict: Dictionary containing mean squared error and R^2 score.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mean_squared_error": mse, "r2_score": r2}

if __name__ == "__main__":
    file_path = "interest_rate_dataset.csv"
    target_column = "interest_rate"
    
    # Load and preprocess data
    df = load_data(file_path)
    X, y = preprocess_data(df, target_column)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluation_results = evaluate_model(model, X_test, y_test)
    print("Evaluation Results:")
    print(evaluation_results)
