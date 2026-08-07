import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Page Configuration
st.set_page_config(page_title="Kenyan Land Sector Survey & ML Tool", layout="wide")

# Load Data and Model (Cached for performance)
@st.cache_data
def load_data():
    # Replace with your cleaned dataset path
    return pd.read_csv("cleaned_survey_data.csv")

@st.cache_resource
def load_model():
    # Replace with your trained model path
    return joblib.load("capacity_predictor_model.pkl")

# Navigation Menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:", ["Interactive Survey Explorer", "Model Predictor"])

# ==========================================
# MODULE 1: INTERACTIVE SURVEY EXPLORER
# ==========================================
if page == "Interactive Survey Explorer":
    st.title("📊 Kenyan Land Sector Survey Explorer")
    st.write("Explore responses across sectors, specializations, and capacity indicators.")

    df3 = load_data()

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    sector_filter = st.sidebar.multiselect(
        "Select Sector:", 
        options=df3["Location"].unique(), 
        default=df3["Location"].unique()
    )
    
    # Filter Data
    filtered_df = df3[df3["Location"].isin(sector_filter)]

    # Metrics Overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Respondents", len(filtered_df))
    col2.metric("Sectors Represented", filtered_df["Location"].nunique())
    avg_exp = pd.to_numeric(filtered_df["Years_Experience"], errors="coerce").mean()

    col3.metric("Avg Experience (Years)", round(avg_exp, 1) if pd.notna(avg_exp) else "N/A")

    st.markdown("---")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("KGRS Familiarity by Location")
        fig1 = px.histogram(filtered_df, x="Location", color="KGRS_Familiarity", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.subheader("AFREF Understanding Breakdown")
        fig2 = px.bar(
            filtered_df["AFREF_Understanding"].value_counts().reset_index(),
             x="AFREF_Understanding",
             y="count",
            labels={"count": "Respondents"}
        )
        st.plotly_chart(fig2, use_container_width=True)
# ==========================================
# MODULE 2: MODEL PREDICTOR
# ==========================================
elif page == "Model Predictor":
    st.title("🤖 Capacity & Adoption Predictor")

    # Load Model and Encoders
    try:
        model = joblib.load("capacity_predictor_model.pkl")
        encoders = joblib.load("encoders.pkl")
    except Exception as e:
        st.error("Model files not found. Run the training script in work.ipynb first!")
        st.stop()

    df = load_data()

    # User Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender:", df["Gender"].unique())
            location = st.selectbox("Location:", df["Location"].unique())
            specialization = st.selectbox("Specialization:", df["Specialization"].unique())

        with col2:
            level = st.selectbox("Level of Practice:", df["Level_of_Practice"].unique())
            experience = st.number_input("Years of Experience:", min_value=0, max_value=50, value=5)
            policy = st.selectbox("Policy Familiarity:", df["Policy_Familiarity"].unique())

        submit = st.form_submit_button("Predict KGRS Usage Level")

    if submit:
        # Construct input DataFrame matching original features
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Location': location,
            'Specialization': specialization,
            'Level_of_Practice': level,
            'Years_Experience': experience,
            'Policy_Familiarity': policy
        }])

       # Encode only categorical columns using loaded encoders
        for col, le in encoders.items():
            if col in input_data.columns and col != "Years_Experience":
                input_data[col] = le.transform(input_data[col].astype(str))
                
        # Make Prediction
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.success(f"Predicted KGRS Usage / Adoption Class: **{prediction}**")