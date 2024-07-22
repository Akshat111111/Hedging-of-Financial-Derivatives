import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the Random Forest model
model = joblib.load('rf_model.pkl')

# Define preprocessing function
def preprocess_input(location, cuisine, rating, seating_capacity, followers, chef_experience, reviews,
                     avg_review_length, ambience_score, service_quality_score, parking, weekend_reservations,
                     weekday_reservations, Marketing_Budget_in_Rupees, Average_Meal_Price_in_Rupees):
    data = {
        'Location': [location],
        'Cuisine': [cuisine],
        'Rating': [rating],
        'Seating Capacity': [seating_capacity],
        'Social Media Followers': [followers],
        'Chef Experience Years': [chef_experience],
        'Number of Reviews': [reviews],
        'Avg Review Length': [avg_review_length],
        'Ambience Score': [ambience_score],
        'Service Quality Score': [service_quality_score],
        'Parking Availability': [parking],
        'Weekend Reservations': [weekend_reservations],
        'Weekday Reservations': [weekday_reservations],
        'Marketing Budget in Rupees': [Marketing_Budget_in_Rupees],
        'Average Meal Price in Rupees': [Average_Meal_Price_in_Rupees]
    }

    df = pd.DataFrame(data)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Location', 'Cuisine', 'Parking Availability'])

    # Add log-transformed columns
    df['Marketing Budget (Log)'] = np.log1p(df['Marketing Budget in Rupees'])
    df['Social Media Followers (Log)'] = np.log1p(df['Social Media Followers'])

    # Define expected columns
    expected_cols = ['Rating', 'Seating Capacity', 'Social Media Followers', 'Chef Experience Years',
                     'Number of Reviews', 'Avg Review Length', 'Ambience Score', 'Service Quality Score',
                     'Weekend Reservations', 'Weekday Reservations', 'Marketing Budget in Rupees',
                     'Average Meal Price in Rupees', 'Location_Downtown', 'Location_Rural', 'Location_Suburban',
                     'Cuisine_American', 'Cuisine_French', 'Cuisine_Indian', 'Cuisine_Italian', 'Cuisine_Japanese',
                     'Cuisine_Mexican', 'Parking Availability_No', 'Parking Availability_Yes', 'Marketing Budget (Log)',
                     'Social Media Followers (Log)']

    # Add missing columns with default values
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[expected_cols]

    return df

# Define prediction function
def predict_revenue(location, cuisine, rating, seating_capacity, followers, chef_experience, reviews, avg_review_length,
                    ambience_score, service_quality_score, parking, weekend_reservations, weekday_reservations,
                    Marketing_Budget_in_Rupees, Average_Meal_Price_in_Rupees):
    preprocessed_df = preprocess_input(location, cuisine, rating, seating_capacity, followers, chef_experience, reviews,
                                       avg_review_length, ambience_score, service_quality_score, parking,
                                       weekend_reservations, weekday_reservations, Marketing_Budget_in_Rupees,
                                       Average_Meal_Price_in_Rupees)
    prediction = model.predict(preprocessed_df)
    return prediction[0]

# Streamlit app
def main():
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            color: yellow;
            font-size: 40px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: green;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: darkgreen;
            color: white;
        }
        .stNumberInput, .stSelectbox, .stSlider {
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">Restaurant Revenue Prediction</div>', unsafe_allow_html=True)

    # Input fields
    with st.form(key='input_form'):
        location = st.selectbox("Enter Location:", ["Rural", "Downtown", "Suburban"])
        cuisine = st.selectbox("Enter Cuisine:", ["Japanese", "Mexican", "Italian", "Indian", "French", "American"])
        rating = st.slider("Enter Rating:", 0.0, 5.0, 4.0, step=0.1)
        seating_capacity = st.slider("Enter Seating Capacity:", 0, 1000, 60, step=1)
        followers = st.number_input("Enter Social Media Followers:", min_value=0)
        chef_experience = st.number_input("Enter Chef Experience in Years:", min_value=0)
        reviews = st.number_input("Enter Number of Reviews:", min_value=0)
        avg_review_length = st.number_input("Enter average review length of all Reviews:", min_value=0)
        ambience_score = st.slider("Enter Ambience Score:", 0.0, 10.0, 5.0, step=0.1)
        service_quality_score = st.slider("Enter Service Quality Score:", 0.0, 10.0, 5.0, step=0.1)
        parking = st.selectbox("Enter Parking Availability:", ["Yes", "No"])
        weekend_reservations = st.number_input("Enter Weekend Reservations:", min_value=0)
        weekday_reservations = st.number_input("Enter Weekday Reservations:", min_value=0)
        Marketing_Budget_in_Rupees = st.number_input("Enter Marketing Budget in Rupees:", min_value=0)
        Average_Meal_Price_in_Rupees = st.number_input("Enter Average Meal Price in Rupees:", min_value=0)

        submit_button = st.form_submit_button(label='Predict Revenue')

    if submit_button:
        revenue = predict_revenue(location, cuisine, rating, seating_capacity, followers, chef_experience, reviews,
                                  avg_review_length, ambience_score, service_quality_score, parking,
                                  weekend_reservations, weekday_reservations, Marketing_Budget_in_Rupees,
                                  Average_Meal_Price_in_Rupees)
        st.success(f"Predicted Restaurant Revenue: Rs {revenue:.2f}")

if __name__ == '__main__':
    main()
