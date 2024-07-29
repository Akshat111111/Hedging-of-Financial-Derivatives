import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
def evaluate_model(df, model):
    X = df.drop('target', axis=1)
    y = df['target']
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')
if __name__ == "__main__":
    df = pd.read_csv('path/to/features_data.csv')
    model = joblib.load('path/to/model.pkl')
    evaluate_model(df, model)
