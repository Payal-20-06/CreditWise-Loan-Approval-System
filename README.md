# 🏦 CreditWise — AI-Powered Loan Approval System

> **Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval.  
> Implemented binary classification along with EDA, feature engineering & model evaluation (Precision, Recall, F1).**

---

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-creditwise--insights.streamlit.app-10B981?style=for-the-badge&logo=streamlit&logoColor=white)](https://creditwise-insights.streamlit.app/)

> 🌐 **Live App:** [https://creditwise-insights.streamlit.app/](https://creditwise-insights.streamlit.app/)

---
## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Project Solution](#project-solution)
- [Live App Features](#live-app-features)
- [Tech Stack](#tech-stack)
- [Machine Learning Models](#machine-learning-models)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Serialised Artifacts](#serialised-artifacts)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Developer](#developer)

---

## 🔴 Problem Statement

Banks and financial institutions receive thousands of loan applications every day. Manually evaluating each application is:

- **Time-consuming** — loan officers spend hours reviewing documents
- **Inconsistent** — human bias and fatigue lead to subjective decisions
- **Costly** — operational overhead from manual processing
- **Slow** — applicants wait days or weeks for decisions

Traditional rule-based systems (credit score thresholds, income cutoffs) fail to capture the **complex, non-linear relationships** between borrower characteristics and default risk.

> **The challenge:** Build an intelligent system that can accurately predict whether a loan application should be **Approved** or **Rejected** based on applicant financial and demographic data — instantly, consistently, and at scale.

---

## ✅ Project Solution

**CreditWise** is a complete, end-to-end AI-powered loan approval decision support system that:

1. **Trains** multiple supervised ML classifiers on historical banking data
2. **Evaluates** models using Accuracy, Precision, Recall, and F1-Score
3. **Deploys** the best model (Logistic Regression, 86% accuracy) in an interactive Streamlit dashboard
4. **Provides** instant, explainable loan decisions with confidence scores
5. **Visualises** applicant risk profiles, portfolio demographics, and model performance

The system reduces loan decision time from **days → seconds** while maintaining consistent, data-driven credit evaluation.

---

## 🚀 Live App Features

### 🔮 Page 1 — Loan Prediction Engine
- **Interactive input form** in the sidebar for all 18 applicant features
- **3 Risk Profile Presets** — Low Risk (Prime), Medium Risk (Standard), High Risk (Subprime)
- **Instant AI decision** — Approved ✅ or Rejected ❌ with model confidence %
- **Credit Score Gauge** — Visual CIBIL score indicator with rating (Poor → Excellent)
- **Income vs Loan Bar Chart** — Loan-to-Income ratio assessment
- **Financial Health Radar Chart** — 6-dimension applicant profile vs. bank benchmark
- **Feature Vector Inspector** — Expandable view of the processed model input

### 🏠 Page 2 — Dashboard
- **6 KPI Cards** — Monthly Income, Loan Amount, Credit Score, DTI Ratio, EMI Estimate, Collateral
- **Portfolio Demographics** — Approval Rate by Employment Status (bar chart)
- **Average Loan Amount by Purpose** — Business, Car, Education, Home, Personal
- **Credit Score Distribution by Approval Status** — Violin plot
- **Applicant vs Portfolio Benchmarks** — Side-by-side comparison table

### 📊 Page 3 — Data Insights
- **Credit Score Distribution** — Histogram split by approval/rejection
- **Income vs Loan Amount Scatter** — Color-coded by approval status
- **Loan Purpose Breakdown** — Pie chart of loan categories in the portfolio
- **Dynamic filters** — Select Employment Status, Loan Purpose for drill-down

### 📈 Page 4 — Model Analytics
- **Logistic Regression Coefficient Chart** — 28 feature weights visualised
- **Model Performance Comparison** — Accuracy, Precision, Recall, F1 for all 3 classifiers
- **Confusion Matrix Visualiser** — Heatmap for Logistic Regression, Naive Bayes, KNN
- **Approval Rate by Employment Status** — Grouped bar chart

### ℹ️ Page 5 — About Project
- Technical architecture overview
- Feature preprocessing pipeline description
- Academic context and model selection rationale

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| **Web Framework** | Streamlit | ≥ 1.35.0 |
| **Data Processing** | Pandas | ≥ 2.1.0 |
| **Numerical Computing** | NumPy | ≥ 1.26.0 |
| **Machine Learning** | Scikit-learn | ≥ 1.4.0 |
| **Interactive Charts** | Plotly | ≥ 5.20.0 |
| **Static Visualisation** | Matplotlib + Seaborn | ≥ 3.8.0 / ≥ 0.13.0 |
| **Model Serialisation** | Joblib | ≥ 1.3.2 |
| **PDF Export** | ReportLab | ≥ 4.1.0 |
| **Language** | Python | 3.10+ |
| **Notebook / EDA** | Jupyter Notebook | — |

### 🎨 UI Design
- **Dark FinTech Theme** — Deep navy (`#0F172A`) background with emerald green (`#10B981`) accents
- **3 switchable themes** — Default (Dark FinTech), Light, Dark Purple
- **Google Fonts** — Inter + Playfair Display
- **Glassmorphism cards**, animated hover effects, gradient buttons
- **Responsive layout** — Wide sidebar + multi-column main content

---

## 🤖 Machine Learning Models

### Primary Model — Logistic Regression ⭐
| Metric | Score |
|---|---|
| Accuracy | **86%** |
| Precision | 78% |
| Recall | 84% |
| F1-Score | 81% |

Chosen as the **production model** for its:
- Best balance of Accuracy and Recall
- Interpretable coefficients (explainability)
- Low inference latency (suitable for real-time predictions)
- Stable generalisation on the test split

### Benchmark Models Evaluated

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **86%** | **78%** | **84%** | **81%** |
| Naive Bayes | 83% | 71% | 80% | 75% |
| K-Nearest Neighbours | 79% | 68% | 57% | 62% |

### Top Predictive Features (by coefficient weight)
| Feature | Weight | Impact |
|---|---|---|
| `Applicant_Income_log` | +2.39 | ↑ Higher income → Approved |
| `Credit_Score_sq` | +1.95 | ↑ Higher credit score → Approved |
| `Employer_Category_MNC` | +0.19 | ↑ MNC employer → Approved |
| `DTI_Ratio_sq` | −2.37 | ↓ High debt ratio → Rejected |
| `Applicant_Income` (raw) | −1.72 | Captured non-linearly via log |
| `Gender_Male` | −0.33 | Slight negative coefficient |
| `Loan_Amount` | −0.27 | ↓ Larger loan → More scrutiny |

---

## 📂 Dataset

### File: `loan_approval_data.csv`
- **Size:** ~126 KB
- **Records:** Synthetic banking dataset representing real-world loan application patterns
- **Target Variable:** `Loan_Status` — Binary (1 = Approved, 0 = Rejected)

### Input Features (18 Raw Features)

| Feature | Type | Description |
|---|---|---|
| `Applicant_Income` | Numerical | Primary applicant monthly gross income (₹) |
| `Coapplicant_Income` | Numerical | Co-applicant monthly income (₹), 0 if none |
| `Age` | Numerical | Applicant's current age (18–75) |
| `Dependents` | Numerical | Number of financially dependent family members |
| `Existing_Loans` | Numerical | Count of currently active loan accounts |
| `Savings` | Numerical | Total liquid savings balance (₹) |
| `Collateral_Value` | Numerical | Current market value of pledged asset (₹) |
| `Loan_Amount` | Numerical | Total loan amount requested (₹) |
| `Loan_Term` | Categorical | Repayment duration in months (12–84) |
| `Credit_Score` | Numerical | CIBIL / bureau credit score (300–900) |
| `DTI_Ratio` | Numerical | Monthly debt obligations / gross monthly income |
| `Gender` | Categorical | Male / Female |
| `Marital_Status` | Categorical | Single / Married |
| `Education` | Categorical | Graduate / Not Graduate |
| `Employment_Status` | Categorical | Salaried / Self-employed / Contract / Unemployed |
| `Employer_Category` | Categorical | Government / MNC / Private / Business / Unemployed |
| `Loan_Purpose` | Categorical | Home / Car / Education / Personal / Business |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |

---

## ⚙️ Feature Engineering

The `preprocess_inputs()` pipeline transforms raw inputs into model-ready features:

```python
# Log transformation to reduce income skewness
Applicant_Income_log = np.log1p(Applicant_Income)

# Polynomial features for non-linear relationships
DTI_Ratio_sq    = DTI_Ratio ** 2
Credit_Score_sq = Credit_Score ** 2

# Ordinal encoding
Education_Level = 1 if Graduate else 0

# Manual One-Hot Encoding for all categorical variables
Employment_Status → [Salaried, Self-employed, Unemployed]
Marital_Status   → [Single]
Loan_Purpose     → [Car, Education, Home, Personal]
Property_Area    → [Semiurban, Urban]
Gender           → [Male]
Employer_Category→ [Government, MNC, Private, Unemployed]
```

**Total engineered features fed to the model: 28**

The feature vector is then **StandardScaler-normalised** (`scaler.pkl`) before prediction to match the training distribution.

---

## 🔑 Serialised Artifacts

| File | Size | Description |
|---|---|---|
| `model.pkl` | 1.1 KB | Trained Logistic Regression model (scikit-learn) |
| `scaler.pkl` | 2.1 KB | StandardScaler fitted on training data |
| `feature_names.pkl` | 0.6 KB | Ordered list of 28 feature column names |
| `loan_approval_data.csv` | 126 KB | Full historical loan application dataset |
| `Credit_wise.ipynb` | 781 KB | Full EDA, feature engineering & model training notebook |

---

## 📁 Project Structure

```
CreditWise_Loan_Approval/
│
├── app.py                    # Main Streamlit application (1760+ lines)
├── Credit_wise.ipynb         # Jupyter notebook — EDA, training, evaluation
│
├── model.pkl                 # Trained Logistic Regression model
├── scaler.pkl                # Fitted StandardScaler
├── feature_names.pkl         # Feature column name list (28 features)
│
├── loan_approval_data.csv    # Historical loan dataset
├── requirements.txt          # Python package dependencies
├── test_load.py              # Artifact verification script
└── README.md                 # This file
```

---

## ▶️ How to Run

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd CreditWise_Loan_Approval
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

> ⚠️ **Required files:** Ensure `model.pkl`, `scaler.pkl`, `feature_names.pkl`, and `loan_approval_data.csv` are in the **same folder** as `app.py` before running.

---

## 📊 Model Performance

### Confusion Matrix — Logistic Regression (Test Set)

```
                  Predicted Rejected   Predicted Approved
Actual Rejected        125                  14
Actual Approved         10                  51
```

- **True Positives (Approved correctly):** 51
- **True Negatives (Rejected correctly):** 125
- **False Positives (Wrongly approved):** 14
- **False Negatives (Wrongly rejected):** 10

### Why Logistic Regression?

Logistic Regression was selected over KNN and Naive Bayes because:
- ✅ **Highest overall accuracy** (86%)
- ✅ **Best Recall** — minimises false rejections of good applicants
- ✅ **Interpretable coefficients** — each feature's influence on approval is explainable
- ✅ **Fast inference** — suitable for real-time web deployment
- ✅ **Stable generalisation** — no overfitting on the test split

---

## 👩‍💻 Developer

**Payal Maina**  
B.Tech CSE (Data Science)

> *"Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval. Implemented binary classification along with EDA, feature engineering & model evaluation (Precision, Recall, F1)."*

---

## 📜 License

This project is developed for academic and portfolio purposes.

---

<div align="center">
  <b>🏦 CreditWise — Turning Data Into Decisions</b><br>
  <sub>AI-Powered Loan Approval System | Built with Streamlit + Scikit-learn</sub>
</div>
