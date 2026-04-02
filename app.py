"""
Income Level Prediction — Streamlit Web Application
====================================================
Author  : Adewale Adeagbo
GitHub  : https://github.com/cssadewale
LinkedIn: https://linkedin.com/in/adewalesamsonadeagbo

Model files are stored on Google Drive (too large for GitHub).
They are downloaded automatically when the app starts.
"""

import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Level Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Google Drive File IDs ─────────────────────────────────────────────────
# Extracted from the shareable Google Drive links.
# Link format: https://drive.google.com/file/d/FILE_ID/view
MODEL_FILE_ID  = "1Gu0twEPwLE3qqYFnJ8JfU2YQFi4k1l9B"
SCALER_FILE_ID = "1__bNEQfDKE2gLbZxs6xfkWFF0gs-Ddhe"

MODEL_PATH  = "income_prediction_rf_model.joblib"
SCALER_PATH = "income_prediction_scaler.joblib"


# ── Download and Load Model Files ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Download model and scaler from Google Drive if not already present,
    then load and return them.

    @st.cache_resource means this function runs only ONCE per app session.
    After the first run, the loaded model and scaler are reused for every
    subsequent user interaction — no repeated downloads or reloads.
    """

    # Download model file if not already on disk
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model file... (first load only, ~15 seconds)"):
            gdown.download(
                f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
                MODEL_PATH,
                quiet=False
            )

    # Download scaler file if not already on disk
    if not os.path.exists(SCALER_PATH):
        with st.spinner("Downloading scaler file..."):
            gdown.download(
                f"https://drive.google.com/uc?id={SCALER_FILE_ID}",
                SCALER_PATH,
                quiet=False
            )

    # Confirm both files downloaded correctly
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error(
            "❌ Model files could not be downloaded from Google Drive. "
            "Please ensure both files are shared as 'Anyone with the link' "
            "in Google Drive, then reload this page."
        )
        st.stop()

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()


# ── Training Column Schema ─────────────────────────────────────────────────
# These are the EXACT column names produced by the notebook's one-hot encoding.
# The input data must be transformed to match this schema before prediction.
TRAINING_COLUMNS = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week',
    'has_capital_gain', 'has_capital_loss',
    'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
    'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
    'workclass_State-gov', 'workclass_Without-pay',
    'education_11th', 'education_12th', 'education_1st-4th',
    'education_5th-6th', 'education_7th-8th', 'education_9th',
    'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
    'education_Doctorate', 'education_HS-grad', 'education_Masters',
    'education_Preschool', 'education_Prof-school', 'education_Some-college',
    'marital_status_Married-AF-spouse', 'marital_status_Married-civ-spouse',
    'marital_status_Married-spouse-absent', 'marital_status_Never-married',
    'marital_status_Separated', 'marital_status_Widowed',
    'occupation_Armed-Forces', 'occupation_Craft-repair',
    'occupation_Exec-managerial', 'occupation_Farming-fishing',
    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Other-service', 'occupation_Priv-house-serv',
    'occupation_Prof-specialty', 'occupation_Protective-serv',
    'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
    'relationship_Not-in-family', 'relationship_Other-relative',
    'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
    'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
    'sex_Male',
    'native_country_Cambodia', 'native_country_Canada', 'native_country_China',
    'native_country_Columbia', 'native_country_Cuba',
    'native_country_Dominican-Republic', 'native_country_Ecuador',
    'native_country_El-Salvador', 'native_country_England',
    'native_country_France', 'native_country_Germany', 'native_country_Greece',
    'native_country_Guatemala', 'native_country_Haiti', 'native_country_Honduras',
    'native_country_Hong', 'native_country_Hungary', 'native_country_India',
    'native_country_Iran', 'native_country_Ireland', 'native_country_Italy',
    'native_country_Jamaica', 'native_country_Japan', 'native_country_Laos',
    'native_country_Mexico', 'native_country_Nicaragua',
    'native_country_Outlying-US(Guam-USVI-etc)', 'native_country_Peru',
    'native_country_Philippines', 'native_country_Poland',
    'native_country_Portugal', 'native_country_Puerto-Rico',
    'native_country_Scotland', 'native_country_South', 'native_country_Taiwan',
    'native_country_Thailand', 'native_country_Trinadad&Tobago',
    'native_country_United-States', 'native_country_Vietnam',
    'native_country_Yugoslavia',
]

NUMERICAL_COLS = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']


# ── Preprocessing Function ────────────────────────────────────────────────
def preprocess_input(raw_input: dict) -> pd.DataFrame:
    """
    Convert raw user input into a model-ready DataFrame.

    Step 1 — Build a single-row DataFrame from the input dictionary
    Step 2 — Engineer the two binary capital flag features
    Step 3 — One-hot encode all categorical columns (drop_first=True,
              matching the notebook's exact encoding)
    Step 4 — Add any missing dummy columns as 0 (handles categories
              the user did not select — they become all-zero columns)
    Step 5 — Reorder columns to match the training schema exactly
    Step 6 — Scale the four numerical columns using the saved scaler
    """
    df = pd.DataFrame([raw_input])

    # Step 2
    df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'] > 0).astype(int)

    # Step 3
    cat_cols = [
        'workclass', 'education', 'marital_status', 'occupation',
        'relationship', 'race', 'sex', 'native_country'
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Step 4
    for col in TRAINING_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Step 5
    df = df[TRAINING_COLUMNS]

    # Step 6
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    return df


# ─────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────
st.title("💰 Income Level Predictor")
st.markdown(
    "Predict whether an individual earns **above or below \\$50,000/year** "
    "based on U.S. Census data. Powered by a **Tuned Random Forest Classifier** "
    "trained on 48,842 census records."
)
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
**Model:** Tuned Random Forest Classifier  
**Dataset:** UCI Adult Census Income (1994)  
**Records trained on:** 48,842  
**Features used:** 14 (including 2 engineered)  
**Primary metric:** ROC-AUC  

---

**Top 5 Predictors:**
1. 🎂 Age
2. 💹 Capital Gain
3. ⏰ Hours per Week
4. 💍 Marital Status
5. 🎓 Education & Occupation

---

**Author:** Adewale Adeagbo  
[GitHub](https://github.com/cssadewale) ·
[LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo)

---
*YouThrive Data Science Capstone · 2025*
    """)


# ── Input Form ────────────────────────────────────────────────────────────
st.subheader("📋 Step 1 — Enter Individual Details")
st.caption("Fill in all the fields below, then click the Predict button at the bottom.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**👤 Demographics**")
    age = st.slider(
        "Age", min_value=17, max_value=90, value=35, step=1,
        help="Age of the individual (17–90)"
    )
    sex = st.radio("Sex", ["Male", "Female"], index=0, horizontal=True)
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ])
    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany", "Canada",
        "Puerto-Rico", "El-Salvador", "India", "Cuba", "England",
        "Jamaica", "South", "China", "Italy", "Dominican-Republic",
        "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
        "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua",
        "Peru", "France", "Greece", "Ecuador", "Ireland",
        "Hong", "Cambodia", "Thailand", "Trinadad&Tobago",
        "Yugoslavia", "Outlying-US(Guam-USVI-etc)",
        "Hungary", "Honduras", "Scotland", "Laos"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    relationship = st.selectbox("Relationship Role", [
        "Husband", "Not-in-family", "Wife",
        "Own-child", "Unmarried", "Other-relative"
    ])

with col2:
    st.markdown("**💼 Employment**")
    workclass = st.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked"
    ])
    occupation = st.selectbox("Occupation", [
        "Prof-specialty", "Exec-managerial", "Craft-repair",
        "Adm-clerical", "Sales", "Other-service",
        "Machine-op-inspct", "Transport-moving", "Handlers-cleaners",
        "Farming-fishing", "Tech-support", "Protective-serv",
        "Priv-house-serv", "Armed-Forces"
    ])
    education = st.selectbox("Education Level", [
        "Bachelors", "Some-college", "HS-grad", "Masters",
        "Assoc-voc", "Assoc-acdm", "Prof-school", "Doctorate",
        "11th", "10th", "9th", "12th",
        "7th-8th", "5th-6th", "1st-4th", "Preschool"
    ])
    hours_per_week = st.slider(
        "Hours Worked per Week", min_value=1, max_value=99, value=40, step=1,
        help="Typical number of hours worked per week"
    )

    st.markdown("**💹 Investment Income**")
    st.caption("Enter 0 if none.")
    capital_gain = st.number_input(
        "Capital Gain ($)",
        min_value=0, max_value=99999, value=0, step=100,
        help="Income from investment gains (0 for most people)"
    )
    capital_loss = st.number_input(
        "Capital Loss ($)",
        min_value=0, max_value=4356, value=0, step=100,
        help="Losses from investments (0 for most people)"
    )


# ── Predict Button ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Step 2 — Get Prediction")
predict_clicked = st.button(
    "Predict Income Level",
    use_container_width=True,
    type="primary"
)

if predict_clicked:

    # Assemble raw input dictionary
    raw_input = {
        "age":            age,
        "workclass":      workclass,
        "education":      education,
        "marital_status": marital_status,
        "occupation":     occupation,
        "relationship":   relationship,
        "race":           race,
        "sex":            sex,
        "capital_gain":   capital_gain,
        "capital_loss":   capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }

    # Preprocess and predict
    X_input    = preprocess_input(raw_input)
    prediction = model.predict(X_input)[0]
    prob_high  = model.predict_proba(X_input)[0][1]   # P(income > 50K)
    prob_low   = model.predict_proba(X_input)[0][0]   # P(income <= 50K)

    # ── Result Display ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success("### ✅  Predicted Income:  **> $50,000 / year**")
    else:
        st.warning("### 📉  Predicted Income:  **≤ $50,000 / year**")

    # Two probability metrics side by side
    m1, m2 = st.columns(2)
    m1.metric(label="Probability > $50K",  value=f"{prob_high:.1%}")
    m2.metric(label="Probability ≤ $50K", value=f"{prob_low:.1%}")

    # Visual confidence bar
    st.markdown("**Model confidence — probability of earning > $50K:**")
    st.progress(float(prob_high))

    # Plain-English interpretation
    if prediction == 1:
        st.markdown(
            f"The model is **{prob_high:.1%} confident** this individual "
            f"earns **above \\$50,000** per year."
        )
    else:
        st.markdown(
            f"The model is **{prob_low:.1%} confident** this individual "
            f"earns **\\$50,000 or below** per year."
        )

    # Collapsible input summary
    with st.expander("📄 View Full Input Summary"):
        summary = pd.DataFrame({
            "Feature": [
                "Age", "Sex", "Race", "Native Country",
                "Marital Status", "Relationship Role",
                "Work Class", "Occupation", "Education",
                "Hours / Week", "Capital Gain", "Capital Loss"
            ],
            "Value Entered": [
                age, sex, race, native_country,
                marital_status, relationship,
                workclass, occupation, education,
                f"{hours_per_week} hrs",
                f"${capital_gain:,}",
                f"${capital_loss:,}"
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This prediction is based on a model trained on 1994 "
        "U.S. Census data and is intended for **educational and portfolio "
        "demonstration** purposes only. It must not be used to make decisions "
        "affecting real individuals without a formal fairness audit."
    )


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by **Adewale Adeagbo** · "
    "[GitHub](https://github.com/cssadewale) · "
    "[LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo) · "
    "YouThrive Data Science Capstone · 2025"
)
