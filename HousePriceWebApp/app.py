import streamlit as st
from utils import DATA, PIPELINE
import pandas as pd

AREA_TYPE = DATA.get_area_type()
LOCATION = DATA.get_location()

st.header("House Price Prediction :house:", divider="blue")

st.markdown("<h4>Select the Area Type :</h4>", unsafe_allow_html=True)
area_type = st.selectbox("Select the Area Type", AREA_TYPE, placeholder="None", label_visibility="collapsed", help="Select the type of area")
st.write("")

st.markdown("<h4>Are you ready to Move currently ?</h4>", unsafe_allow_html=True)
availability = int(st.selectbox("Are you ready to Move currently ?", ("Yes", "No"), placeholder="None", label_visibility="collapsed", help="Select whether you are ready to move currently") == "Yes")
st.write("")

st.markdown("<h4>Select your Location 📍:</h4>", unsafe_allow_html=True)
location = st.selectbox("Select your Location", LOCATION, placeholder="None", label_visibility="collapsed", help="Select the location of the property")
st.write("")

st.markdown("<h4>Select total area in sqft.</h4>", unsafe_allow_html=True)
total_sqft = st.number_input("Select total area in sqft.", key="total_sqft", min_value=300.0 , max_value=30000.0 , label_visibility="collapsed", help="Select the total area in square feet" , step = 100.0)
st.write("")

st.markdown("<h4>Select Number of bathrooms 🚿</h4>", unsafe_allow_html=True)
bath = st.slider("Select Number of bathrooms", key="bath", min_value=1, max_value=16, label_visibility="collapsed", help="Select the number of bathrooms")
st.write("")

st.markdown("<h4>Select BHK </h4>", unsafe_allow_html=True)
bhk = st.slider("Select BHK", key="bhk", min_value=1, max_value=16, label_visibility="collapsed", help="Select the number of bedrooms")
st.write("")

if st.button("Submit"):
    column_names = DATA.get_column_names()
    input = pd.DataFrame([[area_type, availability, location, total_sqft, bath, bhk]], columns=column_names)
    st.dataframe(input)

    prediction = PIPELINE.predict(input)[0]
    c = st.container()
    c.write(f"## **House price is:**  `Rs.{prediction * 1000 :.1f}`")
