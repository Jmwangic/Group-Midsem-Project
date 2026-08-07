import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib


st.set_page_config(
    page_title="KGRS Capacity & Readiness Dashboard",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORDINAL_ORDER = ["Very Low", "Low", "Neutral", "High", "Very High"]
ORDINAL_SCORES = dict(zip(ORDINAL_ORDER, [0, 25, 50, 75, 100]))
READINESS_FIELDS = [
    "KGRS_Familiarity",
    "KGRS_Usage_Level",
    "AFREF_Understanding",
    "AFREF_Reference_Level",
]


@st.cache_data
def load_data():
    """Load the cleaned survey responses and derive dashboard-ready measures."""
    data = pd.read_csv("cleaned_survey_data.csv")

    for column in READINESS_FIELDS:
        data[f"{column}_Score"] = data[column].map(ORDINAL_SCORES)
    data["Policy_Familiarity_Score"] = data["Policy_Familiarity"].map({"Yes": 100, "No": 0})

    score_columns = [f"{column}_Score" for column in READINESS_FIELDS] + [
        "Policy_Familiarity_Score"
    ]
    data["Readiness_Score"] = data[score_columns].mean(axis=1)
    data["Readiness_Band"] = pd.cut(
        data["Readiness_Score"],
        bins=[-1, 40, 60, 75, 100],
        labels=["Critical", "High priority", "Moderate", "Strong"],
    )
    data["At_Risk"] = data["Readiness_Score"] < 50
    return data


@st.cache_resource
def load_model_assets():
    return joblib.load("capacity_predictor_model.pkl"), joblib.load("encoders.pkl")


def apply_filters(data):
    """Render common filters and return the selected subset."""
    st.sidebar.header("Survey filters")
    filters = {}
    for column, label in [
        ("Location", "Sector / employer"),
        ("Specialization", "Specialization"),
        ("Level_of_Practice", "Level of practice"),
        ("Years_Experience", "Experience band"),
    ]:
        options = sorted(data[column].dropna().unique())
        filters[column] = st.sidebar.multiselect(label, options, default=options)

    subset = data.copy()
    for column, selection in filters.items():
        subset = subset[subset[column].isin(selection)]
    return subset


def percentage(value):
    return f"{value:.0%}" if pd.notna(value) else "N/A"


def intervention_for(readiness, risk_rate):
    if pd.isna(readiness):
        return "Collect more responses"
    if readiness < 40 or risk_rate >= 0.60:
        return "Priority: foundational KGRS & AFREF training"
    if readiness < 60 or risk_rate >= 0.35:
        return "Targeted practice and policy workshop"
    if readiness < 75:
        return "Applied KGRS adoption support"
    return "Peer champions and advanced applications"


data = load_data()

st.sidebar.title("KGRS Dashboard")
page = st.sidebar.radio(
    "Module",
    ["Executive overview", "Survey explorer", "Model predictor"],
)
st.sidebar.caption("Capacity, readiness, and adoption insights for the Kenyan land sector.")

if page != "Model predictor":
    filtered_data = apply_filters(data)


if page == "Executive overview":
    st.title("KGRS Capacity & Readiness Overview")
    st.caption("Use the filters to focus the evidence on a sector, specialty, or experience group.")

    if filtered_data.empty:
        st.warning("No responses match the selected filters. Adjust a filter to continue.")
        st.stop()

    low_usage = filtered_data["KGRS_Usage_Level"].isin(["Very Low", "Low"]).mean()
    low_familiarity = filtered_data["KGRS_Familiarity"].isin(["Very Low", "Low"]).mean()
    policy_rate = (filtered_data["Policy_Familiarity"] == "Yes").mean()

    metrics = st.columns(5)
    metrics[0].metric("Responses", len(filtered_data))
    metrics[1].metric("Readiness index", f"{filtered_data['Readiness_Score'].mean():.0f}/100")
    metrics[2].metric("Low KGRS usage", percentage(low_usage))
    metrics[3].metric("Low familiarity", percentage(low_familiarity))
    metrics[4].metric("Policy familiarity", percentage(policy_rate))

    st.markdown("---")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Capacity-gap heatmap by sector")
        heatmap_columns = {
            "KGRS Familiarity": "KGRS_Familiarity_Score",
            "KGRS Usage": "KGRS_Usage_Level_Score",
            "AFREF Understanding": "AFREF_Understanding_Score",
            "AFREF Reference": "AFREF_Reference_Level_Score",
            "Policy Familiarity": "Policy_Familiarity_Score",
        }
        heatmap = filtered_data.groupby("Location")[list(heatmap_columns.values())].mean()
        heatmap.columns = list(heatmap_columns.keys())
        heatmap = heatmap.reindex(sorted(heatmap.index))
        figure = px.imshow(
            heatmap,
            text_auto=".0f",
            color_continuous_scale="RdYlGn",
            zmin=0,
            zmax=100,
            aspect="auto",
            labels={"x": "Indicator", "y": "Sector / employer", "color": "Score"},
        )
        figure.update_layout(height=360, coloraxis_colorbar_title="Score / 100")
        st.plotly_chart(figure, use_container_width=True)

    with right:
        st.subheader("Readiness distribution")
        readiness_counts = (
            filtered_data["Readiness_Band"]
            .value_counts()
            .reindex(["Critical", "High priority", "Moderate", "Strong"], fill_value=0)
            .rename_axis("Readiness band")
            .reset_index(name="Respondents")
        )
        figure = px.bar(
            readiness_counts,
            x="Readiness band",
            y="Respondents",
            color="Readiness band",
            color_discrete_map={
                "Critical": "#b91c1c",
                "High priority": "#f97316",
                "Moderate": "#eab308",
                "Strong": "#16a34a",
            },
        )
        figure.update_layout(height=360, showlegend=False, xaxis_title=None)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Priority intervention opportunities")
    priorities = (
        filtered_data.groupby(["Location", "Specialization"], dropna=False)
        .agg(
            Respondents=("Username", "size"),
            Readiness_Score=("Readiness_Score", "mean"),
            At_Risk_Rate=("At_Risk", "mean"),
            Low_Usage_Rate=("KGRS_Usage_Level", lambda x: x.isin(["Very Low", "Low"]).mean()),
        )
        .reset_index()
    )
    priorities = priorities[priorities["Respondents"] >= 3].copy()
    priorities["Recommended action"] = priorities.apply(
        lambda row: intervention_for(row["Readiness_Score"], row["At_Risk_Rate"]), axis=1
    )
    priorities["Readiness score"] = priorities.pop("Readiness_Score").round(1)
    priorities["At-risk rate"] = priorities.pop("At_Risk_Rate").map(percentage)
    priorities["Low usage rate"] = priorities.pop("Low_Usage_Rate").map(percentage)
    priorities = priorities.sort_values("Readiness score").head(12)
    st.dataframe(priorities, use_container_width=True, hide_index=True)

    with st.expander("Data quality and interpretation"):
        normalized_users = filtered_data["Username"].fillna("").str.strip().str.lower()
        duplicate_records = filtered_data.duplicated().sum()
        st.write(
            f"Filtered records: {len(filtered_data):,} | "
            f"Unique respondents: {normalized_users.nunique():,} | "
            f"Exact duplicate records: {duplicate_records:,}"
        )
        st.caption(
            "The readiness index is a descriptive score, not a validated competency test. "
            "Treat small groups and repeated submissions cautiously."
        )


elif page == "Survey explorer":
    st.title("Interactive Survey Explorer")
    st.caption("Explore adoption and understanding patterns across practitioner groups.")

    if filtered_data.empty:
        st.warning("No responses match the selected filters. Adjust a filter to continue.")
        st.stop()

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("KGRS familiarity by sector")
        figure = px.histogram(
            filtered_data,
            x="Location",
            color="KGRS_Familiarity",
            barmode="group",
            category_orders={"KGRS_Familiarity": ORDINAL_ORDER},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        figure.update_layout(xaxis_title=None, yaxis_title="Respondents", legend_title="Familiarity")
        st.plotly_chart(figure, use_container_width=True)

    with chart_right:
        st.subheader("Familiarity-to-usage pathway")
        flow = filtered_data.groupby(["KGRS_Familiarity", "KGRS_Usage_Level"]).size().reset_index(name="Respondents")
        flow["KGRS_Familiarity"] = pd.Categorical(
            flow["KGRS_Familiarity"], categories=ORDINAL_ORDER, ordered=True
        )
        flow["KGRS_Usage_Level"] = pd.Categorical(
            flow["KGRS_Usage_Level"], categories=ORDINAL_ORDER, ordered=True
        )
        figure = px.parallel_categories(
            flow,
            dimensions=["KGRS_Familiarity", "KGRS_Usage_Level"],
            color="Respondents",
            color_continuous_scale="Viridis",
        )
        figure.update_layout(height=430)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Readiness by experience band")
    experience_summary = (
        filtered_data.groupby("Years_Experience", as_index=False)
        .agg(**{"Average readiness score": ("Readiness_Score", "mean")})
    )
    figure = px.bar(
        experience_summary,
        x="Years_Experience",
        y="Average readiness score",
        color="Average readiness score",
        color_continuous_scale="RdYlGn",
        range_y=[0, 100],
    )
    figure.update_layout(xaxis_title="Experience band", yaxis_title="Average readiness / 100")
    st.plotly_chart(figure, use_container_width=True)

    with st.expander("View and export filtered data"):
        display_columns = [column for column in filtered_data.columns if column != "Username"]
        st.dataframe(filtered_data[display_columns], use_container_width=True, hide_index=True)
        anonymized_csv = filtered_data[display_columns].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download anonymised filtered data (CSV)",
            anonymized_csv,
            file_name="kgrs_filtered_anonymised_data.csv",
            mime="text/csv",
        )


elif page == "Model predictor":
    st.title("KGRS Usage Risk Predictor")
    st.caption("A profile-based Random Forest prediction to support capacity-building decisions.")

    try:
        model, encoders = load_model_assets()
    except Exception:
        st.error("The model or encoder files could not be loaded. Confirm both .pkl files are in this folder.")
        st.stop()

    overview = st.columns(3)
    overview[0].metric("Model", type(model).__name__.replace("Classifier", ""))
    overview[1].metric("Target", "KGRS usage level")
    overview[2].metric("Prediction classes", len(model.classes_))

    st.info(
        "Use this as a screening indicator, not an individual assessment. Probability expresses model confidence, not certainty or causation."
    )

    input_column, explanation_column = st.columns([1.15, 0.85], gap="large")
    with input_column:
        st.subheader("Practitioner profile")
        with st.form("prediction_form", border=True):
            first, second = st.columns(2)
            with first:
                gender = st.selectbox("Gender", encoders["Gender"].classes_)
                location = st.selectbox("Sector / employer", encoders["Location"].classes_)
                specialization = st.selectbox("Specialization", encoders["Specialization"].classes_)
            with second:
                practice_level = st.selectbox("Level of practice", encoders["Level_of_Practice"].classes_)
                experience = st.selectbox("Experience band", encoders["Years_Experience"].classes_)
                policy = st.selectbox("Policy familiarity", encoders["Policy_Familiarity"].classes_)
            submitted = st.form_submit_button("Predict KGRS usage level", use_container_width=True)

    with explanation_column:
        st.subheader("What drives the model?")
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame(
                {"Feature": model.feature_names_in_, "Importance": model.feature_importances_}
            ).sort_values("Importance")
            figure = px.bar(importance, x="Importance", y="Feature", orientation="h", range_x=[0, 1])
            figure.update_layout(height=290, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figure, use_container_width=True)
        st.caption("Importance is based on the trained model and does not prove a causal relationship.")

    if submitted:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Location": location,
            "Specialization": specialization,
            "Level_of_Practice": practice_level,
            "Years_Experience": experience,
            "Policy_Familiarity": policy,
        }])
        for column, encoder in encoders.items():
            input_data[column] = encoder.transform(input_data[column].astype(str))

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        probability_data = pd.DataFrame({"KGRS usage class": model.classes_, "Probability": probabilities})

        st.markdown("---")
        result, probability_chart = st.columns([0.8, 1.2])
        with result:
            st.subheader("Prediction")
            st.success(f"Predicted KGRS usage: **{prediction}**")
            st.metric("Model confidence", percentage(probabilities.max()))
        with probability_chart:
            figure = px.bar(
                probability_data,
                x="KGRS usage class",
                y="Probability",
                range_y=[0, 1],
                color="Probability",
                color_continuous_scale="Viridis",
            )
            figure.update_layout(height=310, yaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(figure, use_container_width=True)
