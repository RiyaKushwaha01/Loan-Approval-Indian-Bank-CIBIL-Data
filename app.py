import streamlit as st
import pandas as pd
import pickle

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
        return pd.read_excel(file)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("Sample of Uploaded Data")
        st.dataframe(df.head())

        # --- Load Model ---
        try:
            with open("Model.pkl", "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            st.error("Model file not found. Please make sure 'Model.pkl' is available in the app directory.")
            st.stop()

        # --- Input Section ---
        st.title("Loan Approval Prediction")
        st.header("Enter Feature Values")

        # Manual input for features (based on your model)
        input_fields = {
            "recent_level_of_deliq": st.number_input("Most recent delinquency level", step=1),
            "Credit_Score": st.number_input("Credit Score", min_value=300, max_value=900, step=1),
            "tot_enq": st.number_input("Total Enquiry", step=1),
            "Secured_TL": st.number_input("Secured TL", step=1),
            "PL_enq_L12m": st.number_input("PL Enquiries in Last 12M", step=1),
            "enq_L12m": st.number_input("Enquiry in Last 12M", step=1),
            "Total_TL": st.number_input("Total TL", step=1),
            "time_since_recent_enq": st.number_input("Days since most recent enquiry", step=1),
            "enq_L6m": st.number_input("Enquiry in Last 6M", step=1),
            "num_std_6mts": st.number_input("Number of Standard Accounts in 6M", step=1),
            "num_std_12mts": st.number_input("Number of Standard Accounts in 12M", step=1),
            "enq_L3m": st.number_input("Enquiry in Last 3M", step=1),
            "Age_Oldest_TL": st.number_input("Age of Oldest TL", step=1),
            "PL_enq": st.number_input("Personal Loan Enquiry", step=1),
            "Tot_Closed_TL": st.number_input("Total Closed TL", step=1),
            "num_std": st.number_input("Number of Standard Accounts", step=1)
        }

        input_df = pd.DataFrame([input_fields])

        # --- Prediction ---
        if st.button("Predict Loan Approval"):
            try:
                # Ensure columns are in the order model expects
                input_df = input_df[model.feature_names_in_]
                prediction = model.predict(input_df)[0]

                st.success(f"Predicted Approval Class: {prediction}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("""<hr><small>Developed with ❤️ using Streamlit</small>""", unsafe_allow_html=True)

    else:
        st.warning("📄 Please upload an Excel file to begin.")
