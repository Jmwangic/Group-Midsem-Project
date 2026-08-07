import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Page Configuration
st.set_page_config(
    page_title="Kenyan Land Sector Survey & ML Tool", 
    page_icon="🛰️",
    layout="wide"
)

# Load Data and Model (Cached for performance)
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_survey_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("capacity_predictor_model.pkl")

# Navigation Menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:", ["Interactive Survey Explorer", "Model Predictor"])

# Defined scale order for ordinal survey fields
ORDINAL_ORDER = ["Very Low", "Low", "Neutral", "High", "Very High"]

# ==========================================
# MODULE 1: INTERACTIVE SURVEY EXPLORER
# ==========================================
if page == "Interactive Survey Explorer":
    st.title("📊 Kenyan Land Sector Survey Explorer")
    st.write("Explore responses across sectors, specializations, and capacity indicators.")

    df3 = load_data()

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    
    # Filter 1: Location/Sector
    sector_filter = st.sidebar.multiselect(
        "Select Location / Sector:", 
        options=df3["Location"].dropna().unique(), 
        default=df3["Location"].dropna().unique()
    )
    
    # Filter 2: Specialization
    specialization_filter = st.sidebar.multiselect(
        "Select Specialization:",
        options=df3["Specialization"].dropna().unique(),
        default=df3["Specialization"].dropna().unique()
    )

    # Filter 3: Level of Practice
    practice_filter = st.sidebar.multiselect(
        "Select Level of Practice:",
        options=df3["Level_of_Practice"].dropna().unique(),
        default=df3["Level_of_Practice"].dropna().unique()
    )

    # Apply Filters Dynamically
    filtered_df = df3[
        (df3["Location"].isin(sector_filter)) &
        (df3["Specialization"].isin(specialization_filter)) &
        (df3["Level_of_Practice"].isin(practice_filter))
    ]

    # Metrics Overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Respondents", len(filtered_df))
    col2.metric("Sectors Represented", filtered_df["Location"].nunique())
    
    # Coerce numeric type safely
    avg_exp = pd.to_numeric(filtered_df["Years_Experience"], errors="coerce").mean()
    col3.metric("Avg Experience (Years)", round(avg_exp, 1) if pd.notna(avg_exp) else "N/A")

    st.markdown("---")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("KGRS Familiarity by Location")
        fig1 = px.histogram(
            filtered_df, 
            x="Location", 
            color="KGRS_Familiarity", 
            barmode="group",
            category_orders={"KGRS_Familiarity": ORDINAL_ORDER},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(xaxis_title="Location", yaxis_title="Respondents", legend_title="Familiarity")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.subheader("AFREF Understanding Breakdown")
        afref_counts = filtered_df["AFREF_Understanding"].value_counts().reset_index()
        afref_counts.columns = ["AFREF_Understanding", "count"]
        
        fig2 = px.bar(
            afref_counts,
            x="AFREF_Understanding",
            y="count",
            labels={"count": "Respondents", "AFREF_Understanding": "Understanding Level"},
            category_orders={"AFREF_Understanding": ORDINAL_ORDER},
            color_discrete_sequence=["#2b5c8f"]
        )
        fig2.update_layout(xaxis_title="Understanding Level", yaxis_title="Respondents")
        st.plotly_chart(fig2, use_container_width=True)

    # Data Inspector & Export Container
    st.markdown("---")
    with st.expander("🔍 View & Export Filtered Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_geodetic_survey_data.csv",
            mime="text/csv"
        )

# ==========================================
# MODULE 2: MODEL PREDICTOR
# ==========================================
elif page == "Model Predictor":
    st.title("🤖 Capacity & Adoption Predictor")
    st.caption("Machine Learning inference engine for evaluating KGRS adoption based on practitioner demographic profiles.")

    # Load Model and Encoders
    try:
        model = joblib.load("capacity_predictor_model.pkl")
        encoders = joblib.load("encoders.pkl")
    except Exception as e:
        st.error("Model files not found. Ensure capacity_predictor_model.pkl and encoders.pkl exist in the working directory!")
        st.stop()

    df = load_data()

    # Overview Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Model Architecture", value="Decision Tree", help="Classifier evaluated on KGRS survey profiles.")
    with m2:
        st.metric(label="Target Label", value="KGRS Usage", help="Predicts high/low adoption rating.")
    with m3:
        st.metric(label="Available Features", value="6 Predictors", help="Demographics & Practice parameters.")

    st.markdown("---")

    # Split into Two Columns: Form on Left, Dynamic Output on Right
    col_input, col_info = st.columns([1.2, 0.8], gap="medium")

    with col_input:
        st.subheader("📋 Practitioner Parameters")
        st.write("Adjust the features below to run a new prediction.")
        
        with st.form("prediction_form", border=True):
            f1, f2 = st.columns(2)
            
            with f1:
                gender = st.selectbox("Gender:", df["Gender"].dropna().unique())
                location = st.selectbox("Sector / Location:", df["Location"].dropna().unique())
                specialization = st.selectbox("Specialization:", df["Specialization"].dropna().unique())

            with f2:
                level = st.selectbox("Level of Practice:", df["Level_of_Practice"].dropna().unique())
                experience = st.number_input("Years of Experience:", min_value=0, max_value=50, value=5)
                policy = st.selectbox("Policy Familiarity:", df["Policy_Familiarity"].dropna().unique())

            submit = st.form_submit_button("🚀 Predict KGRS Usage Level", use_container_width=True)

    with col_info:
        st.subheader("💡 Model Information")
        st.info(
            """
            **How this model works:**
            * Encodes categorical demographic data into numerical representations.
            * Passes experience and policy awareness through the trained decision boundary.
            * Predicts KGRS usage class to highlight capacity building needs.
            """
        )

    # Dynamic Prediction Results Display
    if submit:
        # Construct input DataFrame matching original feature order
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Location': location,
            'Specialization': specialization,
            'Level_of_Practice': level,
            'Years_Experience': experience,
            'Policy_Familiarity': policy
        }])

        # Encode categorical columns using loaded encoders
        for col, le in encoders.items():
            if col in input_data.columns and col != "Years_Experience":
                input_data[col] = le.transform(input_data[col].astype(str))
                
        # Make Prediction
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("🎯 Model Prediction Results")
        
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.success(f"Predicted KGRS Adoption Class:\n\n### **{prediction}**")

        with res_col2:
            # Display Probability Breakdown (if model supports predict_proba)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_
                
                prob_df = pd.DataFrame({"Adoption Class": classes, "Probability": probabilities})
                
                fig_prob = px.bar(
                    prob_df, 
                    x="Adoption Class", 
                    y="Probability", 
                    range_y=[0, 1],
                    color="Probability",
                    color_continuous_scale="Viridis",
                    title="Prediction Confidence Score"
                )
                fig_prob.update_layout(height=280)
                st.plotly_chart(fig_prob, use_container_width=True)