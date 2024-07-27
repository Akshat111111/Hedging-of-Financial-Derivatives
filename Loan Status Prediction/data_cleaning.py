import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """
    Load dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    df (DataFrame): The input dataset.
    strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
    
    Returns:
    DataFrame: Dataset with missing values handled.
    """
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def encode_categorical_data(df):
    """
    Encode categorical features in the dataset.
    
    Parameters:
    df (DataFrame): The input dataset.
    
    Returns:
    DataFrame: Dataset with categorical features encoded.
    """
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders

def normalize_data(df):
    """
    Normalize the dataset.
    
    Parameters:
    df (DataFrame): The input dataset.
    
    Returns:
    DataFrame: Normalized dataset.
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def preprocess_data(file_path, impute_strategy='mean'):
    """
    Complete preprocessing pipeline: load data, handle missing values, encode categorical data, and normalize data.
    
    Parameters:
    file_path (str): Path to the CSV file.
    impute_strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
    
    Returns:
    DataFrame: Preprocessed dataset.
    """
    df = load_data(file_path)
    df = handle_missing_values(df, strategy=impute_strategy)
    df, label_encoders = encode_categorical_data(df)
    df = normalize_data(df)
    return df, label_encoders

if __name__ == "__main__":
    file_path = "kidney_disease_dataset.csv"
    preprocessed_df, encoders = preprocess_data(file_path)
    print("Preprocessed DataFrame:")
    print(preprocessed_df.head())
