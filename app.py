import streamlit as st
import pandas as pd
import zipfile
import joblib

# --- Login Section ---
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "Riya" and password == "Riya@123":
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

    # Sidebar with help description
    st.sidebar.title("Help Box")
    st.sidebar.info(
        """
        **📄 Total TL**: Total number of trade lines (credit accounts) held.
        
        **📝 PL Enquiries in Last 12M** : Personal loan enquiries in last 12 months(1 year).
        
        **📆 Enquiry in Last 3M** : Total enquiries in the last 3 months.
        
        **🎂 Age** : Age in years.
        
        **✅ Number of Standard Accounts** : Number of "standard" accounts (good standing).
        
        **🔍 Total Enquiry** : Total credit enquiries.
        
        **⏱️ Time since recent enquiry** : Days since the most recent enquiry.
        
        **🔐 Secured TL** : Secured Term Loan 
        
        (- A secured TL is a loan taken by giving something valuable as a guarantee — like your house papers, car, gold, or fixed deposit.
        
         - If you don’t repay the loan, the bank can take that item to get back its money.)
         
        **📊 Percentage of Current Balance** : How much loan amount is still unpaid, shown in percentage.
        
        **📊 Age of Oldest TL** : The number of months or years since the borrower opened their very first loan account(in months).
        
        **💳 Credit Score** : A number that shows how good applicant are at paying back loans and credit card bills.
        
        **📆 Enquiry in Last 12M** : Total enquiries in the last 12 months or 1 year.
        
        **🟢 P1**: Best customers with high credit scores and clean repayment history. 
        
        (Very low risk – highly trustworthy)
        
        **🟡 P2**: Good customers with minor and no risk.  
        
        (Generally reliable with small issues, if any)
        
        **🟠 P3**: Mid-risk customers, may have had delinquencies.  
        
        (Moderate credit risk – needs review)
        
        **🔴 P4**: High-risk approvals – most prone to credit issues. 
        
        (Very risky – careful evaluation needed)
        """
    )

    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("Sample of Uploaded Data")
        st.dataframe(df.head())

    # --- Load model and preprocessing objects (outside of file check) ---
    zip_path = "Model.zip"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("extracted_model")

    with open("extracted_model/Model.pkl", "rb") as f:
        bundle = joblib.load(f)
        model = bundle["model"]
        label_encoder = bundle["encoder_y"]
        Selected_features = bundle["encoder_x"]
        selector_dict = bundle["selector"]

    # --- Input Section ---
    st.title("Loan Approval Prediction")
    st.header("Enter Feature Values")

    input_fields = {
        "Total_TL": st.number_input("Total TL", step=1),
        "PL_enq_L12m": st.number_input("PL Enquiries in Last 12M", step=1),
        "enq_L3m": st.number_input("Enquiry in Last 3M", step=1),
        "AGE" : st.number_input("Age", step=1),
        "num_std": st.number_input("Number of Standard Accounts", step=1),
        "tot_enq": st.number_input("Total Enquiry", step=1),
        "time_since_recent_enq": st.number_input("Time since recent enquiry", step=1),
        "Secured_TL": st.number_input("Secured TL", step=1),
        "pct_currentBal_all_TL" : st.number_input("Percentage of Current Balance",step=0.1),
        "Age_Oldest_TL": st.number_input("Age of Oldest TL", step=1),
        "Credit_Score": st.number_input("Credit Score", min_value=300, max_value=900, step=1),
        "enq_L12m": st.number_input("Enquiry in Last 12M", step=1)
    }

    input_df = pd.DataFrame([input_fields])

    # --- Prediction ---
    if st.button("Predict Loan Approval"):
        try:
            input_df = input_df[model.feature_names_in_]
            prediction_numeric = model.predict(input_df)[0]
            prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]
            st.success(f"Predicted Approval Class: {prediction_label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("""<hr><small>Developed with ❤️ using Streamlit</small>""", unsafe_allow_html=True)
