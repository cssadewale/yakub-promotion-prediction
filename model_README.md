# 📁 Model Directory

## About This Folder

This folder stores the **serialised trained model** — the final Random Forest Pipeline saved as a `.pkl` file using `joblib`.

---

## File: `best_model_pipeline.pkl`

This file is generated at the end of **Step 5 (Model Building & Evaluation)** in the main notebook. Run the notebook in full to generate it.

The saved file contains the **complete scikit-learn Pipeline**, which includes:
1. The `ColumnTransformer` preprocessor (imputation + scaling for numerical features; imputation + one-hot encoding for categorical features)
2. The fitted `RandomForestClassifier` with `class_weight='balanced'` and `n_estimators=200`

---

## Loading and Using the Model

```python
import joblib
import pandas as pd

# ── Step 1: Load the pipeline ──────────────────────────────
model = joblib.load('model/best_model_pipeline.pkl')

# ── Step 2: Prepare new employee data ─────────────────────
# The input must be a DataFrame with the same columns as the
# training data (excluding 'employeeno' and 'year_of_birth',
# which were dropped before modelling)

new_employee = pd.DataFrame([{
    'division':                          'Commercial Sales and Marketing',
    'qualification':                     'First Degree or HND',
    'gender':                            'Male',
    'channel_of_recruitment':            'Direct Internal process',
    'trainings_attended':                2,
    'last_performance_score':            10.0,
    'year_of_recruitment':               2015,
    'targets_met':                       1,
    'previous_award':                    0,
    'training_score_average':            72,
    'state_of_origin':                   'LAGOS',
    'foreign_schooled':                  'No',
    'marital_status':                    'Married',
    'past_disciplinary_action':          'No',
    'previous_intradepartmental_movement': 'No',
    'no_of_previous_employers':          1,
    'Age':                               33
}])

# ── Step 3: Generate prediction and probability ────────────
prediction  = model.predict(new_employee)         # 0 or 1
probability = model.predict_proba(new_employee)[:, 1]  # 0.0 to 1.0

print(f"Promotion Prediction : {'Promoted' if prediction[0] == 1 else 'Not Promoted'}")
print(f"Promotion Probability: {probability[0]:.2%}")
```

---

## Model Performance (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | 0.9364 |
| Precision | 0.9497 |
| Recall | 0.2623 |
| **F1-Score** | **0.4111** |
| **ROC-AUC** | **0.8914** |

> The model is deployment-ready. For Streamlit deployment, load this `.pkl` file at application startup and pass raw user input directly into `model.predict()` and `model.predict_proba()`.

---

## 👤 Author

**Adewale Adeagbo** — [github.com/cssadewale](https://github.com/cssadewale) | [linkedin.com/in/adewalesamsonadeagbo](https://www.linkedin.com/in/adewalesamsonadeagbo)
