import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('xgboost_model.pkl')


# Function to get user input
def get_user_input():
    Loan_Amount_Requested = st.number_input("Enter Loan Amount in Rupees", min_value=0)
    Length_Employed = st.number_input("Enter Work Experience in Years", min_value=0)
    Home_Owner = st.selectbox("Select Home Owner status", ['Rent', 'Mortgage', 'Other', 'Own'])
    Income_Verified = st.selectbox("Select Income Verification status",
                                   ['NOT VERIFIED', 'VERIFIED - income', 'VERIFIED - income source'])
    Purpose_Of_Loan = st.selectbox("Select Purpose of Loan",
                                   ['car', 'debt_consolidation', 'credit_card', 'home_improvement',
                                    'major_purchase', 'other', 'medical', 'small_business', 'moving',
                                    'wedding', 'vacation', 'house', 'educational', 'renewable_energy'])
    Gender = st.selectbox("Select Gender", ['Male', 'Female'])
    Annual_Income = st.number_input("Enter Annual Income", min_value=0.0)
    Debt_To_Income = st.number_input("Enter Debt to Income ratio", min_value=0.0)
    Inquiries_Last_6Mo = st.number_input("Enter Inquiries in Last 6 Months", min_value=0)
    Months_Since_Deliquency = st.number_input("Enter Months Since Delinquency", min_value=0)
    Number_Open_Accounts = st.number_input("Enter Number of Open Accounts", min_value=0)
    Total_Accounts = st.number_input("Enter Total Accounts", min_value=0)

    return {
        'Loan_Amount_Requested': Loan_Amount_Requested,
        'Length_Employed': Length_Employed,
        'Home_Owner': Home_Owner,
        'Annual_Income': Annual_Income,
        'Income_Verified': Income_Verified,
        'Purpose_Of_Loan': Purpose_Of_Loan,
        'Debt_To_Income': Debt_To_Income,
        'Inquiries_Last_6Mo': Inquiries_Last_6Mo,
        'Months_Since_Deliquency': Months_Since_Deliquency,
        'Number_Open_Accounts': Number_Open_Accounts,
        'Total_Accounts': Total_Accounts,
        'Gender': Gender,
    }


# Function to preprocess user input
def preprocess_input(user_input):
    user_df = pd.DataFrame([user_input])

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to categorical columns
    cat_cols = ['Home_Owner', 'Income_Verified', 'Purpose_Of_Loan', 'Gender']
    for col in cat_cols:
        user_df[col] = label_encoder.fit_transform(user_df[col])

    return user_df


def main():
    st.title('Loan Interest Rate Prediction')

    # Get user input
    user_input = get_user_input()

    # Preprocess the input
    processed_input = preprocess_input(user_input)

    # Predict the interest rate
    if st.button('Predict Interest Rate'):
        prediction = model.predict(processed_input)
        st.markdown('<div style="padding: 10px; border: 2px solid #1EBEA5; border-radius: 5px; background-color: #1EBEA5; color: white; text-align: center; font-weight: bold;">'
                    f'Predicted Loan Interest Rate: {prediction[0]:.2f}'
                    '</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
