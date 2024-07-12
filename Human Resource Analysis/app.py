import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load the trained XGBoost model
model = joblib.load('salary_prediction_model.pkl')

# Define unique values for categorical columns
unique_values = {
    'Gender': ['Male', 'Female'],
    'Education Level': ["Bachelor's", "Master's", 'PhD'],
    'Job Title': [
        'Others', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Marketing Analyst',
        'Product Manager', 'Sales Manager', 'Marketing Coordinator', 'Senior Scientist',
        'HR Manager', 'Project Manager', 'Operations Manager', 'Senior Engineer',
        'Business Analyst', 'Recruiter', 'HR Generalist', 'Administrative Assistant',
        'Director of Marketing', 'Customer Service Manager', 'Event Coordinator',
        'Director of Operations', 'Senior Data Scientist', 'Junior Accountant',
        'Senior Financial Analyst', 'Senior Software Engineer', 'Junior Account Manager',
        'Senior Project Manager', 'Senior Marketing Manager', 'Junior Software Developer',
        'Senior HR Manager', 'Senior Project Coordinator', 'Senior Marketing Analyst',
        'Senior Business Analyst', 'Junior Marketing Analyst', 'Junior HR Generalist',
        'Senior Product Manager', 'Junior Operations Analyst', 'Senior Software Developer',
        'Junior Sales Representative', 'Junior Marketing Manager', 'Junior Business Analyst',
        'Senior Sales Manager', 'Junior Marketing Specialist', 'Junior Project Manager',
        'Senior Accountant', 'Senior Business Development Manager', 'Senior Product Designer',
        'Junior Financial Analyst', 'Senior Operations Manager', 'Director of Human Resources',
        'Senior Sales Representative', 'Senior Marketing Coordinator',
        'Senior Human Resources Manager', 'Junior Business Development Associate',
        'Junior HR Coordinator', 'Director of Finance', 'Junior Marketing Coordinator',
        'Senior Operations Analyst', 'Senior UX Designer', 'Junior Product Manager',
        'Senior Marketing Specialist', 'Senior Data Analyst', 'Senior IT Consultant',
        'Senior Financial Advisor', 'Junior Business Operations Analyst',
        'Junior Operations Manager', 'Senior Financial Manager', 'Senior Data Engineer',
        'Senior Operations Coordinator', 'Director of Engineering'
    ]
}

# Define X.columns from your trained model
X_columns = [
    'Age', 'Years of Experience', 'Gender_Male', "Education Level_Master's",
    'Education Level_PhD', 'Job Title_Business Analyst',
    'Job Title_Customer Service Manager', 'Job Title_Data Analyst',
    'Job Title_Director of Engineering', 'Job Title_Director of Finance',
    'Job Title_Director of Human Resources',
    'Job Title_Director of Marketing', 'Job Title_Director of Operations',
    'Job Title_Event Coordinator', 'Job Title_HR Generalist',
    'Job Title_HR Manager', 'Job Title_Junior Account Manager',
    'Job Title_Junior Accountant', 'Job Title_Junior Business Analyst',
    'Job Title_Junior Business Development Associate',
    'Job Title_Junior Business Operations Analyst',
    'Job Title_Junior Financial Analyst', 'Job Title_Junior HR Coordinator',
    'Job Title_Junior HR Generalist', 'Job Title_Junior Marketing Analyst',
    'Job Title_Junior Marketing Coordinator',
    'Job Title_Junior Marketing Manager',
    'Job Title_Junior Marketing Specialist',
    'Job Title_Junior Operations Analyst',
    'Job Title_Junior Operations Manager',
    'Job Title_Junior Product Manager', 'Job Title_Junior Project Manager',
    'Job Title_Junior Sales Representative',
    'Job Title_Junior Software Developer', 'Job Title_Marketing Analyst',
    'Job Title_Marketing Coordinator', 'Job Title_Operations Manager',
    'Job Title_Others', 'Job Title_Product Manager',
    'Job Title_Project Manager', 'Job Title_Recruiter',
    'Job Title_Sales Associate', 'Job Title_Sales Manager',
    'Job Title_Senior Accountant', 'Job Title_Senior Business Analyst',
    'Job Title_Senior Business Development Manager',
    'Job Title_Senior Data Analyst', 'Job Title_Senior Data Engineer',
    'Job Title_Senior Data Scientist', 'Job Title_Senior Engineer',
    'Job Title_Senior Financial Advisor',
    'Job Title_Senior Financial Analyst',
    'Job Title_Senior Financial Manager', 'Job Title_Senior HR Manager',
    'Job Title_Senior Human Resources Manager',
    'Job Title_Senior IT Consultant', 'Job Title_Senior Manager',
    'Job Title_Senior Marketing Analyst',
    'Job Title_Senior Marketing Coordinator',
    'Job Title_Senior Marketing Manager',
    'Job Title_Senior Marketing Specialist',
    'Job Title_Senior Operations Analyst',
    'Job Title_Senior Operations Coordinator',
    'Job Title_Senior Operations Manager',
    'Job Title_Senior Product Designer', 'Job Title_Senior Product Manager',
    'Job Title_Senior Project Coordinator',
    'Job Title_Senior Project Manager', 'Job Title_Senior Sales Manager',
    'Job Title_Senior Sales Representative', 'Job Title_Senior Scientist',
    'Job Title_Senior Software Developer',
    'Job Title_Senior Software Engineer', 'Job Title_Senior UX Designer'
]

# Streamlit App
st.title('Salary Prediction App')

# Custom CSS for improved styling
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        color: #ff5722; /* Deep orange */
        text-align: center;
        border-bottom: 2px solid #ff5722; /* Deep orange */
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .prediction-container {
        background-color: #013220; /* Grey */
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
    }
    .prediction-description {
        font-size: 16px;
        color: white;
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields for user data
age = st.slider('Age', min_value=0, max_value=100, value=30, step=1)
gender = st.selectbox('Gender', unique_values['Gender'])
education_level = st.selectbox('Education Level', unique_values['Education Level'])
job_title = st.selectbox('Job Title', unique_values['Job Title'])
years_experience = st.slider('Years of Experience', min_value=0, max_value=50, value=5, step=1)

# Function to predict salary
def predict_salary(model, input_data):
    # Convert input data to DataFrame
    user_df = pd.DataFrame([input_data])

    # One-hot encode categorical variables
    categorical_features = ['Gender', 'Education Level', 'Job Title']
    user_df_encoded = pd.get_dummies(user_df, columns=categorical_features, drop_first=True)

    # Ensure columns match the trained model
    missing_cols = set(X_columns) - set(user_df_encoded.columns)
    for col in missing_cols:
        user_df_encoded[col] = 0  # Add missing columns with default value

    user_df_encoded = user_df_encoded[X_columns]  # Ensure column order is the same as X_columns

    # Predict using the model
    salary_prediction = model.predict(user_df_encoded)

    return salary_prediction[0]

# Predict button
if st.button('Predict Salary'):
    input_data = {
        'Age': age,
        'Gender': gender,
        "Education Level": education_level,
        'Job Title': job_title,
        'Years of Experience': years_experience
    }

    predicted_salary = predict_salary(model, input_data)

    # Display prediction result in a green container with bold text
    st.markdown(
        f'<div style="background-color:#00FF00; padding:10px; border-radius:10px;"><h2 style="color:black; text-align:center;">Monthly Salary :<b>{predicted_salary :.2f} Rs</b></h2></div>',
        unsafe_allow_html=True)

