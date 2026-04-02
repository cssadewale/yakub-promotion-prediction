"""
╔══════════════════════════════════════════════════════════════╗
║   YAKUB TRADING GROUP — STAFF PROMOTION PREDICTION SYSTEM    ║
║   Built by: Adewale Adeagbo                                  ║
║   GitHub  : github.com/cssadewale                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YTG Promotion Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background-color: #F7F6F2;
    }

    /* ── Hide default Streamlit header/footer ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1C1C2E;
        border-right: 3px solid #C9A84C;
    }
    [data-testid="stSidebar"] * {
        color: #F0EDE4 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #C9A84C !important;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1C1C2E 0%, #2D2D44 60%, #1C1C2E 100%);
        border-bottom: 3px solid #C9A84C;
        padding: 2.2rem 2.5rem 1.8rem 2.5rem;
        border-radius: 0 0 12px 12px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        color: #F0EDE4;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #C9A84C;
        margin-top: 0.4rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 500;
    }
    .hero-tagline {
        font-size: 0.88rem;
        color: #9E9BB5;
        margin-top: 0.6rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E4E1D9;
        border-left: 4px solid #C9A84C;
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #1C1C2E;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #7B7A8A;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.3rem;
        font-weight: 600;
    }

    /* ── Result box: PROMOTED ── */
    .result-promoted {
        background: linear-gradient(135deg, #1A3C2E 0%, #1F4D38 100%);
        border: 2px solid #3DAB6B;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-promoted .result-icon { font-size: 3rem; }
    .result-promoted .result-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #6FD9A0;
        margin: 0.5rem 0 0.2rem 0;
    }
    .result-promoted .result-prob {
        font-size: 1.05rem;
        color: #A8E8C5;
    }

    /* ── Result box: NOT PROMOTED ── */
    .result-not-promoted {
        background: linear-gradient(135deg, #3C1A1A 0%, #4D1F1F 100%);
        border: 2px solid #C0392B;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-not-promoted .result-icon { font-size: 3rem; }
    .result-not-promoted .result-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #F08080;
        margin: 0.5rem 0 0.2rem 0;
    }
    .result-not-promoted .result-prob {
        font-size: 1.05rem;
        color: #F4BABA;
    }

    /* ── Section header ── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: #1C1C2E;
        border-bottom: 2px solid #C9A84C;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Info box ── */
    .info-box {
        background: #FFFDF4;
        border: 1px solid #E8D98A;
        border-left: 4px solid #C9A84C;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        font-size: 0.86rem;
        color: #5A5640;
        margin: 0.8rem 0;
    }

    /* ── Feature importance bar ── */
    .fi-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }
    .fi-label { width: 180px; color: #3A3A4A; font-weight: 500; }
    .fi-bar-bg {
        flex: 1;
        background: #ECEAE3;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .fi-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #C9A84C, #E8C96A);
        border-radius: 4px;
    }
    .fi-pct { width: 38px; text-align: right; color: #7B7A8A; }

    /* ── Footer ── */
    .app-footer {
        margin-top: 3rem;
        padding: 1.2rem 0;
        border-top: 1px solid #D8D5CC;
        text-align: center;
        font-size: 0.78rem;
        color: #9E9B90;
    }
    .app-footer a { color: #C9A84C; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "model/best_model_pipeline.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()


# ─── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏢 Yakub Trading Group</div>
    <div class="hero-subtitle">Staff Promotion Eligibility Prediction System</div>
    <div class="hero-tagline">
        An evidence-based, data-driven tool to support fair and transparent promotion decisions.
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Model Missing Warning ─────────────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ **Model file not found.**\n\n"
        "Expected location: `model/best_model_pipeline.pkl`\n\n"
        "Please run the training notebook end-to-end to generate the model file, "
        "then place it in the `model/` folder and restart the app."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Employee Input Form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Employee Profile")
    st.markdown("---")

    st.markdown("### 🏢 Structural Information")

    division = st.selectbox("Division", [
        "Commercial Sales and Marketing",
        "Customer Support and Field Operations",
        "Information and Strategy",
        "Business Finance Operations",
        "Research and Innovation",
        "People and HR Management",
        "Supply Chain and Procurement",
        "Regulatory and Legal Services",
        "IT and Solution Support",
    ])

    qualification = st.selectbox("Qualification", [
        "First Degree or HND",
        "MSc  MBA and PhD",
        "Non University Education",
    ])

    gender = st.selectbox("Gender", ["Male", "Female"])

    channel = st.selectbox("Channel of Recruitment", [
        "Agency and others",
        "Direct Internal process",
        "Referral and Special Candidates",
    ])

    state_of_origin = st.selectbox("State of Origin", [
        "ABIA", "ADAMAWA", "AKWA IBOM", "ANAMBRA", "BAUCHI", "BAYELSA",
        "BENUE", "BORNO", "CROSS RIVER", "DELTA", "EBONYI", "EDO",
        "EKITI", "ENUGU", "FCT", "GOMBE", "IMO", "JIGAWA", "KADUNA",
        "KANO", "KATSINA", "KEBBI", "KOGI", "KWARA", "LAGOS", "NASARAWA",
        "NIGER", "OGUN", "ONDO", "OSUN", "OYO", "PLATEAU", "RIVERS",
        "SOKOTO", "TARABA", "YOBE", "ZAMFARA",
    ])

    foreign_schooled = st.selectbox("Foreign Schooled", ["No", "Yes"])
    marital_status   = st.selectbox("Marital Status", ["Married", "Single", "Not_Sure"])

    st.markdown("---")
    st.markdown("### 📈 Performance & Experience")

    targets_met      = st.selectbox("Targets Met This Year?", [0, 1],
                                    format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
    previous_award   = st.selectbox("Has a Previous Award?", [0, 1],
                                    format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
    last_perf_score  = st.slider("Last Performance Score", 0.0, 12.5, 7.5, step=0.5)
    training_score   = st.slider("Training Score Average", 31, 91, 55)
    trainings_att    = st.slider("Trainings Attended", 2, 11, 2)
    prev_employers   = st.slider("No. of Previous Employers", 0, 6, 1)

    st.markdown("---")
    st.markdown("### 📅 Career Timeline")

    year_of_recruitment = st.slider("Year of Recruitment", 1982, 2024, 2015)
    age                 = st.slider("Age (years)", 22, 75, 33)

    st.markdown("---")
    st.markdown("### 📋 Background Flags")

    disciplinary = st.selectbox("Past Disciplinary Action?",
                                ["No", "Yes"])
    intra_move   = st.selectbox("Previous Intra-Departmental Movement?",
                                ["No", "Yes"])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Promotion Eligibility", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Prediction Result", "📖 How It Works", "ℹ️ About"])


# ─── TAB 1: Prediction ────────────────────────────────────────────────────────
with tab1:

    if not predict_btn:
        st.markdown("""
        <div class="info-box">
            👈 &nbsp; Fill in the employee's details in the sidebar on the left,
            then tap <strong>"Predict Promotion Eligibility"</strong> to generate a prediction.
        </div>
        """, unsafe_allow_html=True)

        # Show model summary metrics
        st.markdown('<div class="section-header">Model Performance Summary</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            ("0.411", "F1-Score"),
            ("0.891", "ROC-AUC"),
            ("0.950", "Precision"),
            ("0.936", "Accuracy"),
            ("38,312", "Training Records"),
        ]
        for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1.5rem;">
            <strong>Selected Model:</strong> Random Forest Classifier
            &nbsp;|&nbsp;
            <strong>Class Weighting:</strong> Balanced (handles 10.8:1 imbalance)
            &nbsp;|&nbsp;
            <strong>Primary Metric:</strong> F1-Score
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Build input DataFrame ──────────────────────────────────────────
        input_data = pd.DataFrame([{
            "division":                           division,
            "qualification":                      qualification,
            "gender":                             gender,
            "channel_of_recruitment":             channel,
            "trainings_attended":                 trainings_att,
            "last_performance_score":             last_perf_score,
            "year_of_recruitment":                year_of_recruitment,
            "targets_met":                        targets_met,
            "previous_award":                     previous_award,
            "training_score_average":             training_score,
            "state_of_origin":                    state_of_origin,
            "foreign_schooled":                   foreign_schooled,
            "marital_status":                     marital_status,
            "past_disciplinary_action":           disciplinary,
            "previous_intradepartmental_movement": intra_move,
            "no_of_previous_employers":           prev_employers,
            "Age":                                age,
        }])

        # ── Apply log1p to the three skewed features ───────────────────────
        for col in ["trainings_attended", "training_score_average", "last_performance_score"]:
            input_data[col] = np.log1p(input_data[col])

        # ── Generate prediction ────────────────────────────────────────────
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        pct         = probability * 100

        # ── Result display ─────────────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="result-promoted">
                <div class="result-icon">🏆</div>
                <div class="result-title">Promotion Recommended</div>
                <div class="result-prob">
                    Promotion probability: <strong>{pct:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-not-promoted">
                <div class="result-icon">📋</div>
                <div class="result-title">Not Yet Eligible</div>
                <div class="result-prob">
                    Promotion probability: <strong>{pct:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability gauge ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Promotion Probability Breakdown</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label="Probability of Promotion",
                      value=f"{pct:.1f}%")
            st.progress(float(probability))
        with col_b:
            st.metric(label="Probability of Not Being Promoted",
                      value=f"{(1 - probability)*100:.1f}%")
            st.progress(float(1 - probability))

        # ── Input Summary ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Employee Profile Summary</div>',
                    unsafe_allow_html=True)

        summary_cols = st.columns(3)
        summary_items = [
            ("Division", division),
            ("Qualification", qualification),
            ("Gender", gender),
            ("Recruitment Channel", channel),
            ("Targets Met", "Yes ✅" if targets_met == 1 else "No ❌"),
            ("Previous Award", "Yes ✅" if previous_award == 1 else "No ❌"),
            ("Last Performance Score", f"{last_perf_score} / 12.5"),
            ("Training Score Avg", f"{training_score} / 91"),
            ("Age", f"{age} years"),
            ("Year of Recruitment", str(year_of_recruitment)),
            ("Trainings Attended", str(trainings_att)),
            ("No. of Prev. Employers", str(prev_employers)),
        ]
        for i, (label, value) in enumerate(summary_items):
            with summary_cols[i % 3]:
                st.markdown(f"**{label}**")
                st.markdown(f"<span style='color:#1C1C2E;font-size:0.93rem;'>{value}</span>",
                            unsafe_allow_html=True)
                st.markdown("---")

        # ── HR Guidance ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">HR Interpretation Guide</div>',
                    unsafe_allow_html=True)

        if pct >= 50:
            st.success(
                f"✅ **This employee is flagged as promotion-eligible** "
                f"with a {pct:.1f}% probability score.\n\n"
                "**Recommended action:** Escalate to the divisional HR review panel "
                "for final assessment and sign-off."
            )
        elif pct >= 25:
            st.warning(
                f"⚠️ **Borderline case** — promotion probability is {pct:.1f}%.\n\n"
                "**Recommended action:** Review performance trajectory over the next cycle. "
                "Focus development on target achievement and training engagement."
            )
        else:
            st.info(
                f"📋 **Not yet eligible** — promotion probability is {pct:.1f}%.\n\n"
                "**Recommended action:** The employee would benefit from a structured "
                "development plan focused on meeting annual targets and improving "
                "training scores before the next review cycle."
            )


# ─── TAB 2: How It Works ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">How the Prediction Model Works</div>',
                unsafe_allow_html=True)

    st.markdown("""
    This application uses a **Random Forest Classifier** trained on **38,312 employee records**
    from Yakub Trading Group. The model was built to replace a subjective, manager-discretion-heavy
    promotion process with a transparent, evidence-based scoring system.
    """)

    st.markdown('<div class="section-header">The 5 Most Important Features</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        These are the features the model relies on most heavily, in rank order.
        Features are ranked by their <strong>Mean Decrease in Gini Impurity</strong>
        across all 200 decision trees in the Random Forest.
    </div>
    """, unsafe_allow_html=True)

    feature_importances = [
        ("Targets Met",             0.192),
        ("Training Score Average",  0.163),
        ("Last Performance Score",  0.152),
        ("Previous Award",          0.138),
        ("Year of Recruitment",     0.110),
        ("Age",                     0.092),
        ("No. of Prev. Employers",  0.058),
        ("Division",                0.041),
        ("Trainings Attended",      0.034),
        ("Other Features",          0.020),
    ]

    max_val = max(v for _, v in feature_importances)
    for label, val in feature_importances:
        pct_width = int((val / max_val) * 100)
        st.markdown(f"""
        <div class="fi-row">
            <div class="fi-label">{label}</div>
            <div class="fi-bar-bg">
                <div class="fi-bar-fill" style="width:{pct_width}%;"></div>
            </div>
            <div class="fi-pct">{val:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Findings from the Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎯 Targets Met**
        Promoted employees met their annual targets at **70.6%** vs **32.0%** for those not promoted —
        a 2.2× difference and the single clearest separator in the dataset.

        **🏅 Previous Award**
        Award holders were promoted at **44.9%** vs **7.6%** without an award — a 5.9× difference.
        Only 2.3% of employees have an award, but it is the most decisive individual signal.
        """)
    with col2:
        st.markdown("""
        **📚 Training Score Average**
        Promoted employees score **8.6 points higher** on average across all training sessions.

        **📊 Last Performance Score**
        Promoted employees score **1.61 points higher** on average, with a median of 10.0
        vs 7.5 for the non-promoted group.
        """)

    st.markdown('<div class="section-header">Model Performance</div>',
                unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        "Model":      ["Logistic Regression", "Random Forest ✅"],
        "Accuracy":   ["76.5%", "93.6%"],
        "Precision":  ["24.3%", "95.0%"],
        "Recall":     ["84.3%", "26.2%"],
        "F1-Score":   ["0.378", "0.411"],
        "ROC-AUC":    ["0.874", "0.891"],
    })
    st.table(perf_df.set_index("Model"))

    st.markdown("""
    <div class="info-box">
        <strong>Why F1-Score is the primary metric:</strong> The dataset has a severe
        10.8:1 class imbalance — a model predicting "Not Promoted" for everyone achieves
        91.5% accuracy without learning anything useful. F1-Score balances Precision and
        Recall and is the appropriate measure for imbalanced classification.
    </div>
    """, unsafe_allow_html=True)


# ─── TAB 3: About ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">About This Project</div>',
                unsafe_allow_html=True)

    st.markdown("""
    This application is part of a portfolio project built to demonstrate an end-to-end
    machine learning pipeline — from raw data through cleaning, exploratory analysis,
    feature engineering, modelling, evaluation, and deployment.

    **The business problem:** Yakub Trading Group, a multi-sector Nigerian conglomerate,
    needed to replace an informal, subjective staff promotion process with a transparent,
    data-driven system. This tool is the deployed result of that project.
    """)

    st.markdown('<div class="section-header">Project Workflow</div>',
                unsafe_allow_html=True)

    steps = [
        ("Step 1", "Data Loading & Inspection",
         "Loaded 38,312 records × 18 columns. Identified 1,679 missing values in Qualification only."),
        ("Step 2", "Data Cleaning & Feature Engineering",
         "Mode imputation, column standardisation, duplicate check, and Age feature engineering."),
        ("Step 3", "Exploratory Data Analysis",
         "Univariate, bivariate, and temporal analysis. Confirmed 10.8:1 class imbalance."),
        ("Step 4", "Preprocessing Pipeline",
         "log1p skewness treatment, stratified 80/20 split, scikit-learn ColumnTransformer pipeline."),
        ("Step 5", "Model Building & Evaluation",
         "Logistic Regression vs Random Forest with class_weight='balanced'. Random Forest selected."),
        ("Step 6", "Business Insights",
         "Translated all findings into 4 actionable recommendations for the leadership team."),
    ]
    for step, title, desc in steps:
        with st.expander(f"**{step}: {title}**"):
            st.markdown(desc)

    st.markdown('<div class="section-header">Built By</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **Adewale Adeagbo** — Data Scientist | Machine Learning Engineer | STEM Educator

    Lagos, Nigeria · B.Sc/Ed Computer Science, Lagos State University

    With over 10 years of experience teaching Mathematics, Further Mathematics, Physics,
    Chemistry, and Computer Science, I bring a precision-first approach to data science —
    breaking down complex problems step by step, communicating findings clearly, and
    building solutions that are both rigorous and understandable.

    📫 buildingmyictcareer@gmail.com
    🔗 [LinkedIn](https://www.linkedin.com/in/adewalesamsonadeagbo)
    🐙 [GitHub — github.com/cssadewale](https://github.com/cssadewale)
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built by <strong>Adewale Adeagbo</strong> &nbsp;|&nbsp;
    <a href="https://github.com/cssadewale" target="_blank">github.com/cssadewale</a>
    &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/adewalesamsonadeagbo" target="_blank">LinkedIn</a>
    &nbsp;|&nbsp; Yakub Trading Group · Staff Promotion Prediction System
</div>
""", unsafe_allow_html=True)
