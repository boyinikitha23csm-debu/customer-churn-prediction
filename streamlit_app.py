import streamlit as st
import os
import json

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# Snowflake connection
conn = st.connection(
    "snowflake",
    ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL")
)

session = conn.session()

# Customer inputs
st.subheader("Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Income",
    min_value=0.0,
    value=40000.0
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=50,
    value=2
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=85.0
)

# Prediction
if st.button("🔮 Predict Churn"):

    try:

        input_df = session.create_dataframe(
            [[age, income, tenure, monthly_charges]],
            schema=[
                "AGE",
                "INCOME",
                "TENURE",
                "MONTHLY_CHARGES"
            ]
        )

        input_df.create_or_replace_temp_view("CHURN_INPUT")

        query = """
        SELECT MODEL(
            ML_PROJECT.ML_SCHEMA.CUSTOMER_CHURN_MODEL,
            V2
        )!predict(
            AGE,
            INCOME,
            TENURE,
            MONTHLY_CHARGES
        ) AS PREDICTION
        FROM CHURN_INPUT
        """

        result = session.sql(query).to_pandas()

        # Snowflake returns:
        # {"output_feature_0": 1}

        prediction_json = result.iloc[0, 0]

        if isinstance(prediction_json, str):
            prediction_data = json.loads(prediction_json)
            prediction = int(prediction_data["output_feature_0"])
        else:
            prediction = int(prediction_json)

        # Display result
        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Customer may churn")

            st.write(
                "The model predicts that this customer "
                "is likely to churn."
            )

        else:

            st.success("✅ Customer may stay")

            st.write(
                "The model predicts that this customer "
                "is likely to stay."
            )

        # Customer information
        st.subheader("Customer Information")

        st.write(f"**Age:** {age}")
        st.write(f"**Income:** {income}")
        st.write(f"**Tenure:** {tenure}")
        st.write(f"**Monthly Charges:** {monthly_charges}")

    except Exception as e:

        st.error("Prediction failed.")
        st.code(str(e))


st.divider()

st.subheader("About this Project")

st.write(
    """
    This application uses a Random Forest machine learning model
    trained on customer data stored in Snowflake.

    The trained model is registered in the Snowflake Model Registry
    and is used to predict customer churn.
    """
)

st.write("**Model:** Random Forest Classifier")
st.write("**Target:** CHURN")
st.write("**0:** Customer may stay")
st.write("**1:** Customer may churn")
