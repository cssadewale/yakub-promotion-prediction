# 📁 Data Directory

## About This Folder

The raw dataset (`data.csv`) is **not stored in this repository** for two reasons:
1. The file size is ~5.4 MB — better served via direct download
2. The `.gitignore` is configured to exclude `.csv` files from version control

---

## How to Get the Dataset

The dataset is hosted on Google Drive and can be downloaded in two ways:

### Option A — Automatic (inside the notebook)
The notebook's **Step 1 (Data Loading)** handles the download automatically using `gdown`:

```python
import gdown
url = "https://drive.google.com/file/d/1ZvMu-zl7FySIFy61rnPOa9EQfkkE8KfK/view?usp=drivesdk"
gdown.download(url, "data.csv", fuzzy=True)
```

Simply run the notebook from the beginning and the file will be downloaded to the working directory.

### Option B — Manual download
1. Open this link: [Google Drive Dataset](https://drive.google.com/file/d/1ZvMu-zl7FySIFy61rnPOa9EQfkkE8KfK/view?usp=drivesdk)
2. Click **Download**
3. Save the file as `data.csv` inside this `data/` folder

---

## Dataset Description

| Property | Detail |
|----------|--------|
| File name | `data.csv` |
| Records | 38,312 employee records |
| Columns | 19 (17 features + 1 target + 1 identifier) |
| File size | ~5.4 MB |
| Format | CSV (comma-separated) |

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `EmployeeNo` | String | Unique employee identifier — dropped before modelling |
| `Division` | Categorical | Business unit (9 categories) |
| `Qualification` | Categorical | Highest education level (3 categories) |
| `Gender` | Categorical | Male / Female |
| `Channel_of_Recruitment` | Categorical | How the employee was hired (3 categories) |
| `Trainings_Attended` | Integer | Number of trainings attended (2–11) |
| `Year_of_birth` | Integer | Employee birth year — used to engineer `Age` |
| `Last_performance_score` | Float | Most recent performance rating (0–12.5) |
| `Year_of_recruitment` | Integer | Year the employee joined the company |
| `Targets_met` | Binary | Whether the employee met annual targets (0/1) |
| `Previous_Award` | Binary | Whether the employee has received a prior award (0/1) |
| `Training_score_average` | Integer | Average score across all training sessions (31–91) |
| `State_Of_Origin` | Categorical | Employee's state of origin (37 Nigerian states) |
| `Foreign_schooled` | Categorical | Whether the employee was educated abroad (Yes/No) |
| `Marital_Status` | Categorical | Marital status (Married / Single / Not_Sure) |
| `Past_Disciplinary_Action` | Categorical | Whether employee has a disciplinary record (Yes/No) |
| `Previous_IntraDepartmental_Movement` | Categorical | Prior department transfer (Yes/No) |
| `No_of_previous_employers` | Integer | Number of employers before this company (0–6) |
| `Promoted_or_Not` | Binary | **Target variable** — promoted (1) or not (0) |

---

## Class Distribution

| Class | Label | Count | Proportion |
|-------|-------|-------|-----------|
| 0 | Not Promoted | 35,071 | 91.54% |
| 1 | Promoted | 3,241 | 8.46% |

**Imbalance ratio: 10.8 : 1** — both models use `class_weight='balanced'` to compensate.
