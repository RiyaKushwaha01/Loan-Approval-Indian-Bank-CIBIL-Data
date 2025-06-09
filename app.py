import streamlit as st
import numpy as np
import pandas as pd
import pickle
import zipfile
from io import BytesIO

# --- Login Section ---
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "User" and password == "User@123":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid username or password")

# Session authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    st.set_page_config(page_title="Loan Approval Prediction App", layout="wide")

    @st.cache_data
    def load_data(file):
        with zipfile.ZipFile(file) as z:
            file_names = z.namelist()
            df = pd.read_excel(z.open([f for f in file_names if "Unseen_Dataset" in f][0]))
        return df

    # Upload file
    uploaded_file = st.file_uploader("Upload ZIP file containing Unseen_Dataset", type=["zip"])
    if uploaded_file:
        df1 = load_data(uploaded_file)
        st.write("Sample of Uploaded Data")
        st.dataframe(df1.head())

    # --- Loan Approval Prediction App ---
    st.title("Loan Approval Prediction App")
    st.header("Enter Feature Values")

    # Input fields for each feature
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    Total_TL = st.number_input("Total TL", step=1)
    Age_Oldest_TL = st.number_input("Age of Oldest TL", step=1)
    num_std_6mts = st.number_input("Number of Standard Accounts in 6M", step=1)
    num_std_12mts = st.number_input("Number of Standard Accounts in 12M", step=1)
    max_recent_level_of_deliq = st.number_input("Max Level of Delinquency Recently", step=1)
    Tot_Closed_TL = st.number_input("Total Closed TL", step=1)
    num_std = st.number_input("Number of Standard Accounts", step=1)
    enq_L12m = st.number_input("Total Enquiries in Last 12M", step=1)
    tot_enq = st.number_input("Total Enquiry", step=1)
    PL_enq = st.number_input("Personal Loan Enquiry", step=1)
    PROSPECTID = st.number_input("Applicant ID", step=1)
    PL_enq_L12m = st.number_input("PL Enquiries in Last 12M", step=1)
    enq_L6m = st.number_input("Enquiry in Last 6M", step=1)
    Secured_TL = st.number_input("Secured TL", step=1)
    time_since_recent_enq = st.number_input("Days Since Recent Enquiry", step=1)

    # Collect input in a DataFrame
    input_data = pd.DataFrame([{
        "Credit Score": Credit_Score,
        "Total TL": Total_TL,
        "Age of Oldest TL": Age_Oldest_TL,
        "Number of Standard Accounts in 6M": num_std_6mts,
        "Number of Standard Accounts in 12M": num_std_12mts,
        "Maximum level of delinquency in recent times": max_recent_level_of_deliq,
        "Total Closed TL": Tot_Closed_TL,
        "Number of standard accounts": num_std,
        "Total enquires in Last 12 months": enq_L12m,
        "Total Enquiry": tot_enq,
        "Personal Loan Enquiry": PL_enq,
        "Applicant ID": PROSPECTID,
        "Personal Loan enquiry in Last 12 months": PL_enq_L12m,
        "Enquiries in last 6 months": enq_L6m,
        "Secured TL": Secured_TL,
        "Days since the most recent enquiry": time_since_recent_enq
    }])

    # Load the model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict and display result
    if st.button("Predict Loan Approval"):
        try:
            prediction = model.predict(input_data)
            st.success(f"Estimated Approval Class: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("""
        <hr>
        <small>Developed with ❤️ using Streamlit</small>
    """, unsafe_allow_html=True)
