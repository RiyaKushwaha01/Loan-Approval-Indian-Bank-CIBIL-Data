
import streamlit as st
import pickle
import pandas as pd
import streamlit_authenticator as stauth

# ----- USER AUTHENTICATION SETUP -----
names = ['John Doe']
usernames = ['johndoe']
passwords = ['12345']  # plain text, for demo only

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    {"usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
        }
    }},
    "myapp", "abcdef", cookie_expiry_days=1
)

name, auth_status, username = authenticator.login('Login', 'main')

if auth_status:
    st.sidebar.success(f"Welcome, {name} 👋")
    authenticator.logout('Logout', 'sidebar')

    # Load the model and encoder
    with open("linear_regression_model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        encoder = bundle["encoder"]
        selected_features = bundle["selected_features"]

    # Streamlit UI
    st.title("🛒 Retail Demand Prediction App")

    st.markdown("### Enter Input Features:")

    category = st.selectbox("Category", ["Food", "Furniture", "Clothing", "Toy", "Groceries"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Snowy"])
    promotion = st.selectbox("Promotion", ["Yes", "No"])
    seasonality = st.selectbox("Seasonality", ["Spring", "Summer", "Autumn", "Winter"])

    inventory = st.number_input("Inventory", step=0.1)
    sales = st.number_input("Sales", step=0.1)
    price = st.number_input("Price", step=0.1)
    discount = st.number_input("Discount", step=0.1)
    orders = st.number_input("Orders", step=0.1)
    competitor_pricing = st.number_input("Competitor Pricing", step=0.1)

    if st.button("Predict Demand"):
        try:
            # Prepare input data
            categorical_data = {
                'Category': category,
                'Region': region,
                'Weather': weather,
                'Promotion': promotion,
                'Seasonality': seasonality
            }

            numeric_data = {
                'Inventory': inventory,
                'Sales': sales,
                'Price': price,
                'Discount': discount,
                'Orders': orders,
                'Competitor_Pricing': competitor_pricing
            }

            cat_df = pd.DataFrame([categorical_data])
            num_df = pd.DataFrame([numeric_data])

            # Encode categorical columns only
            encoded_cat = encoder.transform(cat_df)
            encoded_cat_df = pd.DataFrame(
                encoded_cat.toarray(),
                columns=encoder.get_feature_names_out()
            )

            # Concatenate and predict
            final_input = pd.concat([encoded_cat_df, num_df], axis=1)

            # Selecting features
            final_input = final_input[selected_features]
            prediction = model.predict(final_input)

            st.success(f"📦 Predicted Demand: {prediction[0]:.2f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

elif auth_status is False:
    st.error("Username or password is incorrect")

elif auth_status is None:
    st.warning("Please enter your username and password")
