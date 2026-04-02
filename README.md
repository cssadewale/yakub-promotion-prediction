# 🏢 Algorithmic Staff Promotion Prediction — Yakub Trading Group

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

**Yakub Trading Group** is a multi-sector Nigerian conglomerate facing a persistent internal challenge: its staff promotion process relies on informal, manager-discretion-heavy decisions that employees perceive as subjective and biased.

As the **Lead Data Scientist**, I was engaged to replace this system with a **transparent, evidence-based promotion eligibility model** — one grounded entirely in measurable performance, training, and structural employee data.

This project delivers:
- A fully documented **binary classification pipeline** predicting whether an employee will be promoted
- A **deployment-ready model** saved as a serialised scikit-learn Pipeline
- **Actionable business recommendations** for the leadership team

---

## 🎯 Business Problem

> *"Which employee characteristics genuinely drive promotion eligibility — and can we build a system that is fairer and more defensible than subjective manager decisions?"*

**Target variable:** `Promoted_or_Not` → `1` = Promoted, `0` = Not Promoted

---

## 📂 Repository Structure

```
yakub-promotion-prediction/
│
├── notebook/
│   └── Promotion_Prediction_Portfolio.ipynb   ← Full analysis notebook
│
├── model/
│   └── best_model_pipeline.pkl                ← Saved Random Forest Pipeline
│
├── data/
│   └── README.md                              ← Data source instructions
│
├── requirements.txt                           ← Python dependencies
├── .gitignore                                 ← Files excluded from Git
└── README.md                                  ← This file
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Records | 38,312 employee records |
| Features | 17 predictive features + 1 target |
| Missing Values | `qualification` only — 1,679 rows (4.38%) |
| Class Balance | **10.8 : 1** (Not Promoted : Promoted) — severely imbalanced |
| Source | Provided by Yakub Trading Group HR Department |

**Features include:** Division, Qualification, Gender, Channel of Recruitment, Trainings Attended, Year of Birth, Last Performance Score, Year of Recruitment, Targets Met, Previous Award, Training Score Average, State of Origin, Foreign Schooled, Marital Status, Past Disciplinary Action, Previous Intra-Departmental Movement, No. of Previous Employers.

---

## 🔁 Project Workflow

```
Step 1: Data Loading & Inspection
        ↓
Step 2: Data Cleaning & Feature Engineering
        ↓
Step 3: Exploratory Data Analysis (EDA)
        ↓
Step 4: Feature Engineering & Preprocessing Pipeline
        ↓
Step 5: Model Building & Evaluation
        ↓
Step 6: Business Insights & Recommendations
```

---

## 🧪 Models Trained

Both models used `class_weight='balanced'` to handle the 10.8:1 class imbalance.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.765 | 0.243 | 0.843 | 0.378 | 0.874 |
| **Random Forest** ✅ | **0.936** | **0.950** | **0.262** | **0.411** | **0.891** |

**Selected model:** Random Forest — highest F1-Score (0.411) and ROC-AUC (0.891).

> **Why not use accuracy?** A naïve model predicting "Not Promoted" for every employee achieves 91.54% accuracy without learning anything. F1-Score and ROC-AUC are the appropriate metrics for this imbalanced problem.

---

## 🔑 Key Findings

### Top Predictors of Promotion (from Random Forest feature importances)

| Rank | Feature | Insight |
|------|---------|---------|
| 1 | `targets_met` | Promoted employees met targets at 70.6% vs 32.0% — a **2.2× difference** |
| 2 | `previous_award` | Award holders promoted at 44.9% vs 7.6% — a **5.9× difference** |
| 3 | `training_score_average` | Promoted employees scored **8.6 points higher** on average |
| 4 | `last_performance_score` | Promoted employees scored **1.61 points higher** on average |
| 5 | `year_of_recruitment` | Earlier cohorts (longer tenure) have significantly higher promotion rates |

### Business Highlights

- **No strong demographic bias** found in the raw promotion rates: gender gap is only 0.63 percentage points (8.90% female vs 8.27% male)
- **Referral channel employees** have a 12.1% promotion rate — **44% above the company average** — warranting leadership scrutiny
- **Divisional disparity** exists: IT & Solution Support (10.7%) vs Regulatory & Legal (5.6%) — nearly a 2× gap
- **Demographic features** (gender, marital status, foreign schooled) rank at the bottom of feature importances — the model is primarily performance-driven

---

## ⚙️ How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/cssadewale/yakub-promotion-prediction.git
cd yakub-promotion-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook notebook/Promotion_Prediction_Portfolio.ipynb
```

### 4. Load the saved model (for inference)
```python
import joblib
import pandas as pd

# Load the full Pipeline (preprocessor + classifier)
model = joblib.load('model/best_model_pipeline.pkl')

# Predict on new employee data
# Input must match the original feature columns
prediction = model.predict(new_employee_df)
probability = model.predict_proba(new_employee_df)[:, 1]
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Data visualisation |
| scikit-learn | Preprocessing, modelling, evaluation |
| Joblib | Model serialisation |
| Jupyter Notebook | Interactive development environment |

---

## 📈 Evaluation Approach

Given the **10.8:1 class imbalance**, raw accuracy was deliberately excluded as a primary metric. Evaluation focused on:

- **F1-Score** — harmonic mean of Precision and Recall; primary selection criterion
- **ROC-AUC** — overall discriminative ability across all thresholds
- **Precision-Recall Curve** — most honest metric under severe imbalance
- **Confusion Matrix** — business-interpretable breakdown of error types

---

## 💼 Business Recommendations (Summary)

1. **Formalise performance-based criteria** around the four validated predictors: targets met, training score, performance score, and previous award
2. **Audit the referral recruitment channel** — its 44% above-average promotion rate may reflect informal network advantages
3. **Conduct divisional calibration** — the near-2× promotion gap across divisions suggests inconsistent standards
4. **Deploy as a scoring tool** — use promotion probability scores to flag candidates for HR review each cycle

Full recommendations with exact figures are in the notebook's **Step 6: Business Insights**.

---

## ⚠️ Limitations

- Historical data may encode past biases — a fairness audit is recommended before production deployment
- `state_of_origin` is included as a feature but is a legally protected characteristic in some jurisdictions — consider removing before deployment
- The model predicts *whether* an employee is promoted, not *at what grade* — a companion ordinal model is recommended
- Annual retraining is recommended to prevent temporal drift as HR patterns evolve

---

## 👤 Author

**Adewale Adeagbo** — Data Scientist | Machine Learning Engineer | STEM Educator  
Lagos, Nigeria

I bring a unique perspective to data science — over 10 years of teaching Mathematics, Further Mathematics, Physics, Chemistry, and Computer Science has trained me to break down complex problems with precision, communicate findings clearly, and build solutions that are both rigorous and understandable. I hold a B.Sc/Ed in Computer Science from Lagos State University and am now building an evidence-based portfolio of end-to-end machine learning projects.

📫 **Email:** buildingmyictcareer@gmail.com  
🔗 **LinkedIn:** [linkedin.com/in/adewalesamsonadeagbo](https://www.linkedin.com/in/adewalesamsonadeagbo)  
🐙 **GitHub:** [github.com/cssadewale](https://github.com/cssadewale)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
