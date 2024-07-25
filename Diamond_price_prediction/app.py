import streamlit as st
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
import joblib


# Function to predict diamond price
def predict_diamond_price(model, input_data):
    # Example of categorical columns and their encoding mapping
    # Replace these with your actual column names and mappings
    categorical_cols = ['cut', 'color', 'clarity']
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        input_data[col] = label_encoders[col].fit_transform(input_data[col])

    # Predict the price
    prediction = model.predict(input_data)

    return prediction[0] / 100  # Assuming the model predicts in cents, converting to lakhs




# Function to display scraped information
def display_diamond_information():
    info = fetch_diamond_information()
    st.subheader('Diamond Information from the Web')
    st.write(info)


def main():
    # Load the trained model
    model = joblib.load('model_rf.pkl')

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Diamond Price Prediction', 'Dataset View'])

    if page == 'Diamond Price Prediction':
        # Title and description
        st.title('Diamond Price Prediction')
        st.write("""
           This app predicts the price of a diamond based on its characteristics.
           Please fill in the following details to get the prediction.
           """)
        # User input fields
        carat = st.number_input('Enter the weight of the Diamond in carats (ex : 0.24) : ', min_value=0.0, step=0.01)
        cut = st.selectbox('Enter the Cutting Type of Diamond : ', ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'])
        color = st.selectbox('Enter the Color Scheme of the Diamond: ', ['E', 'I', 'J', 'H', 'F', 'G', 'D'])
        clarity = st.selectbox('Enter the Clarity type of the Diamond: ',
                               ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'])
        depth = st.number_input('Enter the Depth of the Diamond structure (ex: 61.5) : ', min_value=0.0, step=0.1)
        table = st.number_input('Enter the table(top view length) of the Diamond Structure (ex : 55.0) : ',
                                min_value=0.0, step=0.1)
        x = st.number_input('Enter the X dimension of the diamond (ex : 1.00) : ', min_value=0.0, step=0.1)
        y = st.number_input('Enter the Y dimension of the diamond (ex : 1.00) :', min_value=0.0, step=0.1)
        z = st.number_input('Enter the Z dimension of the diamond (ex : 1.00) :', min_value=0.0, step=0.1)

        # Predict button with styled button class
        if st.button('Predict Price'):
            # Create a DataFrame with the user input
            input_data = pd.DataFrame({
                'carat': [carat],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z]
            })

            # Predict diamond price
            predicted_price = predict_diamond_price(model, input_data)
            st.success(f"Predicted Price: {predicted_price:,.2f} lakhs")

    elif page == 'Dataset View':
        # Title and description
        st.title('Dataset view')
        # Load and display dataset (example using a placeholder CSV)
        diamond_df = pd.read_csv('diamonds.csv')
        st.dataframe(diamond_df)


if __name__ == "__main__":
    main()
