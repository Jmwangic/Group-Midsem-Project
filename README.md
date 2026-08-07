# Group-Midsem-Project
Group Midsem Project focusing on assessing the skills gap and utilization of the Kenya Geodetic Reference System (KGRS). This repository contains data cleaning, exploratory data analysis (EDA), preprocessing, machine learning models, and an interactive Streamlit web dashboard evaluating survey professionals' experiences and backgrounds in Kenya.

# 🛰️ Project: Analyzing Geodetic Infrastructure Utilization & Capacity in Kenya
## 🌧️ Overview
Kenya's Geodetic Reference System (KGRS) is critical infrastructure for spatial referencing. However, a lack of professional familiarity and inadequate access points limit the system's return on investment. This project implements data analytics, machine learning classification, and an interactive web dashboard to assess professionals' backgrounds, identify capacity building needs, and evaluate the geodetic reference system's usage across the government and private sectors.

## 🔍 Problem Statement
A skills gap and suboptimal utilization of the KGRS among land sector professionals hinder the efficiency, accuracy, and interoperability of land operations in Kenya. This can result in project delays, high rework costs, data inconsistencies, and land boundary disputes. This study models professional training profiles to identify factors predicting low system familiarity and high geodetic resource underutilization.

## 💾 Dataset Summary
Source: Industry-focused Geodetic Reference Frame survey data (GEODETIC_2.csv / cleaned_survey_data.csv)

Size: 333 rows × 130 columns (comprehensive survey responses processed down to key operational indicators)

Key Columns:

Sex / Gender, Highest Education Level, Area of specialization / Specialization

Where do you practice? / Location, Years of practice / Years_Experience

Capacity needs: Technical skills and knowledge gaps (e.g., GNSS, AI, Land policies)

System Familiarity & Adoption: Policy_Familiarity, AFREF_Understanding, AFREF_Reference_Level, KGRS_Familiarity, KGRS_Usage_Level

## 🔄 Step 1: Business Understanding
### 💡 Why This Matters:

Standardization: High-quality geodetic grids reduce land disputes.

Capacity Building: Highlighting explicit technical demands (like AI and GNSS) allows academic institutions and government bodies to tailor professional development.

Infrastructure Strategy: Assessing the physical network's adequacy highlights areas where the National Gravimetric Network falls short.

## 🧹 Step 2: Data Cleaning & Preprocessing
### 🧼 Actions Taken:
Capitalization & Column Normalization: Script systematically loops through columns, formats text metrics uniformly, and maps verbose survey questions into clean, standardized feature names (Location, Specialization, Years_Experience).

Handling Missingness: Handled optional string entries (such as Name) and conditional survey questions (such as Gravimetric applications).

Multilabel Splitting: Cleaned semi-colon separated string attributes indicating multiple knowledge gaps or technical skills.

Data Type Coercion & Encoding: Converted numerical attributes (such as Years_Experience) using pd.to_numeric(..., errors='coerce') for aggregation, and label-encoded categorical attributes for machine learning inference.

### Sample script action demonstrating data loading and capitalization
import pandas as pd
df = pd.read_csv("GEODETIC_2.csv")
df.columns = df.columns.str.title()  # Formatting column indexes

## 📊 Step 3: Exploratory Data Analysis (EDA) & Interactive Dashboard
### 📊 Key Discoveries:
Diverse Specializations: The professional landscape is heavily composed of Land Surveying, Civil Engineering, Planning, and Geospatial Systems.
Sector Representation: Responses span Private Sector, County Government, and National Agencies.
Emerging Demands: A significant portion of professionals indicated a strong desire to learn technical skills like AI for Geospatial Practice and GNSS Data Processing.

## 🖥️ Interactive Web Dashboard (Dashboard.py):
Interactive Survey Explorer: Dynamic filter sidebar allowing custom evaluation of metrics by practitioner location and specialization.
KPI Metrics Overview: Live summary showing total respondents (333), represented sectors, and average experience level.
Plotly Visualizations: Interactive histograms and bar charts breaking down KGRS usage levels and AFREF understanding.

## 🤖 Step 4: Machine Learning, Modeling & Deployment
To identify the core profiles of land sector professionals and predict which demographics suffer the most from resource limitations, the following classification models were initialized and deployed:
### 🔗 Algorithms Used:
Logistic Regression: To establish a baseline profile of factors contributing to low network familiarity.
Decision Trees: To capture non-linear interactions of a professional's years of practice, sector, and education level.

### 🛠️ Evaluated Metrics & Deployment Artifacts:
Confusion Matrix Evaluation
ROC-AUC Curves
Classification Reports (Precision, Recall, F1-Score)
Model Persistence & Live Inference: Saved model pipeline (capacity_predictor_model.pkl) and encoders integrated directly into an interactive prediction tab inside the Streamlit web application.

## 🚀 Running the Streamlit App Locally
Install Dependencies:

### Bash
pip install -r requirements.txt
Run the Dashboard:

### Bash
python -m streamlit run Dashboard.py

## 📦 Project Structure
Plaintext
├── Dashboard.py                  # Main Streamlit web application code
├── cleaned_survey_data.csv       # Preprocessed geodetic survey dataset
├── capacity_predictor_model.pkl  # Serialized machine learning model artifact
├── requirements.txt              # Project dependencies for cloud deployment
└── README.md                     # Project documentation

## 📚 Recommendations & Future Directions
Targeted Capacity Building: Institutions should introduce focused training courses on GNSS applications and AI integration in geodetic survey workflows.
Policy Harmonization: Standardizing regulatory operations at the national level to assist County-level practitioners.
Upgrade Gravimetric Grids: Expand the physical density of geodetic and gravimetric reference stations based on the high rate of "low/very low" adequacy ratings from active survey practitioners.

## 👨‍🔬 Authors
Mwangi John Eric Chege
Melchizedeke Mwangi

Geospatial Data Scientists & Geodetic System Analysts

📍 Kenya

✉️ https://github.com/Jmwangic/Group-Midsem-Project
