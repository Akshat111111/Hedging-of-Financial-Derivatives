import streamlit as st
from utils import model
import pandas as pd

heading_html = '''
    <style>
        .text-glow {
    text-align: center;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', 'source-code-pro', monospace;
}

.text-glow::before {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 1.5;
    width: 87%;
    height: 2px;
    background-color: #fff;
    filter: drop-shadow(0px 0px 5px #00ff00);
    transition: filter 0.3s ease-in-out;
}

.text-glow:hover::before {
    filter: drop-shadow(0px 0px 10px #00ff00);
}       
    </style>
'''

with st.form(key='Form', clear_on_submit=True):
    st.markdown("<h1 class='text-glow'>Loan Status Prediction</h1>", unsafe_allow_html=True)
    st.html(heading_html) 
    # Gender
    st.markdown('<h4>Select Gender</h4>', unsafe_allow_html=True)
    gender = st.selectbox(label='None', options=['Male', 'Female'], label_visibility='collapsed', key='gender')

    # Married
    st.markdown('<h4>Are you Married?</h4>', unsafe_allow_html=True)
    married = st.selectbox(label='None', options=['Yes', 'No'], label_visibility='collapsed', key='married')

    # Dependents
    st.markdown('<h4>Select no.of.dependents</h4>', unsafe_allow_html=True)
    dependents = st.selectbox(label='None', options=['1', '0', '2', '3+'], label_visibility='collapsed', key='dependents')

    # Education
    st.markdown('<h4>Select your education</h4>', unsafe_allow_html=True)
    education = st.selectbox(label='None', options=['Graduate', 'Not Graduate'], label_visibility='collapsed', key='education')

    # Self Employed
    st.markdown('<h4>Are you Self Employed?</h4>', unsafe_allow_html=True)
    self_employed = st.selectbox(label='None', options=['Yes', 'No'], label_visibility='collapsed', key='self_employed')

    # Applicant Income
    st.markdown('<h4>Enter your Income</h4>', unsafe_allow_html=True)
    applicant_income = st.number_input(label='None', label_visibility='collapsed', key='applicant_income')

    # Coapplicant Income
    st.markdown('<h4>Enter Co-applicant Income</h4>', unsafe_allow_html=True)
    coapplicant_income = st.number_input(label='None', label_visibility='collapsed', key='coapplicant_income')

    # Loan Amount
    st.markdown('<h4>Enter Loan amount</h4>', unsafe_allow_html=True)
    loan_amount = st.number_input(label='None', label_visibility='collapsed', key='loan_amount')

    # Loan Amount Term
    st.markdown('<h4>Enter Loan Amount Term</h4>', unsafe_allow_html=True)
    loan_amount_term = st.number_input(label='None', label_visibility='collapsed', key='loan_amount_term')

    # Credit History
    st.markdown('<h4>Select your Credit History</h4>', unsafe_allow_html=True)
    credit_history = int(st.selectbox(label='None', options=['Good', 'Bad'], label_visibility='collapsed', key='credit_history') == "Good")

    # Property Area
    st.markdown('<h4>Area of Living</h4>', unsafe_allow_html=True)
    property_area = st.selectbox(label='None', options=['Rural', 'Urban', 'Semiurban'], label_visibility='collapsed', key='property_area')
    input_values = [[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]]
    column_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    
    input = pd.DataFrame(input_values , columns = column_names)
    if st.form_submit_button(label='Prediction', use_container_width=True):
        if model.predict(input)[0]:
            st.success('Loan Status Approved')
        else:
            st.error('Loan Status not approved')

