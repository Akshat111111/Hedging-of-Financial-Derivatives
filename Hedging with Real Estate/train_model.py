import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'path/to/model.pkl')
    return model

if __name__ == "__main__":
    df = pd.read_csv('path/to/features_data.csv')
    model = train_model(df)
