import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values, encode categorical features,
    and scale numerical features.
    """
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply the preprocessing steps
    df_preprocessed = preprocessor.fit_transform(df)

    # Convert the result to a DataFrame
    df_preprocessed = pd.DataFrame(df_preprocessed)

    return df_preprocessed

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Filepath to the dataset
    filepath = 'path/to/your/dataset.csv'
    
    # Load the data
    df = load_data(filepath)
    
    # Preprocess the data
    df_preprocessed = preprocess_data(df)
    
    # Split the data
    target_column = 'your_target_column'
    X_train, X_test, y_train, y_test = split_data(df_preprocessed, target_column)
    
    # Save the preprocessed and split data to CSV files
    X_train.to_csv('path/to/X_train.csv', index=False)
    X_test.to_csv('path/to/X_test.csv', index=False)
    y_train.to_csv('path/to/y_train.csv', index=False)
    y_test.to_csv('path/to/y_test.csv', index=False)
