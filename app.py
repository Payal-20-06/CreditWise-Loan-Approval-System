"""
╔══════════════════════════════════════════════════════════════════════╗
║                🏦 CREDITWISE — LOAN APPROVAL SYSTEM                 ║
║                       Streamlit Dashboard App                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN:                                                         ║
║  1. Make sure you have the following in the same folder as app.py:   ║
║       - model.pkl         (your trained ML model)                    ║
║       - scaler.pkl        (StandardScaler fitted parameters)         ║
║       - feature_names.pkl (list of expected feature column names)    ║
║       - loan_approval_data.csv (bank lending database records)       ║
║  2. Install dependencies:                                            ║
║       pip install streamlit pandas numpy scikit-learn plotly joblib  ║
║  3. Launch the app:                                                  ║
║       streamlit run app.py                                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─── Standard Library & Third-party Imports ───────────────────────────────────
import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG  (must be the very first Streamlit call in the script)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CreditWise | Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1a. THEME SYSTEM DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "🌟 Default Theme": {
        "bg": "#0F172A",
        "card_bg": "#1E293B",
        "primary": "#10B981",
        "text": "#F8FAFC",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "border": "#334155",
        "sub_text": "#94A3B8",
        "sidebar_bg": "linear-gradient(180deg, #1E293B 0%, #0F172A 100%)",
        "sidebar_border": "#334155",
        "btn_hover": "#059669",
        "plotly_theme": "plotly_dark",
    },
    "🌞 Light Theme": {
        "bg": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "primary": "#2563EB",
        "text": "#0F172A",
        "success": "#16A34A",
        "warning": "#D97706",
        "danger": "#DC2626",
        "border": "#E2E8F0",
        "sub_text": "#475569",
        "sidebar_bg": "linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%)",
        "sidebar_border": "#E2E8F0",
        "btn_hover": "#1D4ED8",
        "plotly_theme": "plotly",
    },
    "🌙 Dark Theme": {
        "bg": "#111827",
        "card_bg": "#1F2937",
        "primary": "#8B5CF6",
        "text": "#F9FAFB",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "border": "#374151",
        "sub_text": "#9CA3AF",
        "sidebar_bg": "linear-gradient(180deg, #1F2937 0%, #111827 100%)",
        "sidebar_border": "#374151",
        "btn_hover": "#7C3AED",
        "plotly_theme": "plotly_dark",
    }
}

def get_active_theme():
    theme_name = st.session_state.get("theme_selector", "🌟 Default Theme")
    return THEMES.get(theme_name, THEMES["🌟 Default Theme"])

def hex_to_rgba(hex_str, opacity):
    try:
        hex_str = hex_str.lstrip('#')
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"
    except Exception:
        return hex_str

def apply_theme():
    t = get_active_theme()
    
    # Generate CSS
    css = f"""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    
    /* Global Background and Text with Smooth Transition */
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    
    .stApp {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
        transition: all 0.3s ease;
    }}
    
    /* Hide Streamlit default white header toolbar */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    
    /* Global Smooth transitions for elements */
    * {{
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: {t['sidebar_bg']} !important;
        border-right: 1px solid {t['sidebar_border']} !important;
        transition: all 0.3s ease;
    }}
    [data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {{
        color: {t['sub_text']} !important;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    
    /* Metric / KPI Cards */
    .kpi-grid {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
    .kpi-card {{
        flex: 1; min-width: 150px;
        background-color: {t['card_bg']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 12px;
        padding: 20px;
        position: relative;
        transition: all 0.3s ease;
    }}
    .kpi-card:hover {{
        border-color: {t['primary']} !important;
    }}
    .kpi-label {{
        font-size: 0.7rem;
        color: {t['sub_text']} !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {t['text']} !important;
        line-height: 1.1;
    }}
    .kpi-unit {{
        font-size: 0.75rem;
        color: {t['sub_text']} !important;
        margin-top: 3px;
    }}
    .kpi-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        position: absolute;
        top: 16px;
        right: 16px;
    }}
    
    /* Hero Container styling */
    .hero-container {{
        background: linear-gradient(135deg, {t['card_bg']} 0%, {t['bg']} 100%) !important;
        border: 1px solid {t['border']} !important;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    .hero-container::before {{
        content: '';
        position: absolute; top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, {t['primary']}26 0%, transparent 70%);
        border-radius: 50%;
    }}
    .hero-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: {t['primary']} !important;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }}
    .hero-sub {{
        font-size: 0.9rem;
        color: {t['sub_text']} !important;
        margin: 0;
        font-weight: 400;
    }}
    .hero-badge {{
        display: inline-block;
        background: {t['primary']}1a;
        border: 1px solid {t['primary']}4d !important;
        color: {t['primary']} !important;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {t['primary']} !important;
        border-left: 3px solid {t['primary']} !important;
        padding-left: 10px;
        margin: 28px 0 16px 0;
    }}
    
    /* Predict Button */
    div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg, {t['btn_hover']} 0%, {t['primary']} 100%) !important;
        color: {t['bg']} !important;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.04em;
        border: none !important;
        border-radius: 10px;
        padding: 14px 32px;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px {t['primary']}40 !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        opacity: 0.92;
        transform: translateY(-1px);
    }}
    div[data-testid="stButton"] > button:active {{
        transform: translateY(0);
    }}
    
    /* Prediction Outcome Cards (Result Banners) */
    .result-approved {{
        background: linear-gradient(135deg, {t['success']}22, {t['success']}05) !important;
        border: 1px solid {t['success']} !important;
        border-radius: 14px;
        padding: 24px 28px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }}
    .result-approved .result-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: {t['success']} !important;
    }}
    .result-rejected {{
        background: linear-gradient(135deg, {t['danger']}22, {t['danger']}05) !important;
        border: 1px solid {t['danger']} !important;
        border-radius: 14px;
        padding: 24px 28px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }}
    .result-rejected .result-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: {t['danger']} !important;
    }}
    .result-icon {{ font-size: 2.8rem; line-height: 1; margin-bottom: 8px; }}
    .result-sub {{ font-size: 0.88rem; color: {t['sub_text']} !important; }}
    
    /* Chart Cards */
    .chart-card {{
        background-color: {t['card_bg']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }}
    
    /* Dividers */
    hr {{ border: none !important; border-top: 1px solid {t['border']} !important; margin: 28px 0; }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {t['bg']} !important; }}
    ::-webkit-scrollbar-thumb {{ background: {t['border']} !important; border-radius: 3px; }}
    
    /* Forms, Inputs, Metric layouts */
    div[data-testid="stForm"] {{
        border: 1px solid {t['border']} !important;
        background-color: {t['card_bg']} !important;
        border-radius: 12px;
        padding: 20px;
    }}
    [data-testid="stMetricValue"] {{
        color: {t['text']} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {t['sub_text']} !important;
    }}
    
    /* Tables and Dataframes */
    div[data-testid="stTable"] table,
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background-color: {t['card_bg']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px;
    }}
    
    /* Dropdowns, Selectboxes, Inputs styling to adapt nicely */
    div[data-baseweb="select"] > div,
    input[type="number"],
    input[type="text"] {{
        background-color: {t['card_bg']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1b. SESSION STATE & PRESET CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "gender_val": "Male",
        "age_val": 32,
        "marital_status_val": "Single",
        "dependents_val": 1,
        "education_val": "Graduate",
        "employment_status_val": "Salaried",
        "employer_category_val": "Private",
        "applicant_income_val": 11000.0,
        "coapplicant_income_val": 0.0,
        "existing_loans_val": 1,
        "loan_amount_val": 15000.0,
        "loan_term_val": 36,
        "loan_purpose_val": "Home",
        "property_area_val": "Urban",
        "credit_score_val": 720,
        "dti_ratio_val": 0.35,
        "savings_val": 10000.0,
        "collateral_value_val": 20000.0,
        "preset_selector": "Custom Input",
        "theme_selector": "🌟 Default Theme"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def apply_preset_callback():
    preset = st.session_state.preset_selector
    if preset == "Low Risk (Prime Borrower)":
        st.session_state.credit_score_val = 780
        st.session_state.applicant_income_val = 16000.0
        st.session_state.coapplicant_income_val = 5000.0
        st.session_state.age_val = 38
        st.session_state.dependents_val = 1
        st.session_state.education_val = "Graduate"
        st.session_state.employment_status_val = "Salaried"
        st.session_state.employer_category_val = "MNC"
        st.session_state.existing_loans_val = 0
        st.session_state.loan_amount_val = 10000.0
        st.session_state.loan_term_val = 24
        st.session_state.loan_purpose_val = "Home"
        st.session_state.property_area_val = "Urban"
        st.session_state.dti_ratio_val = 0.15
        st.session_state.savings_val = 18000.0
        st.session_state.collateral_value_val = 40000.0
        st.session_state.marital_status_val = "Married"
        st.session_state.gender_val = "Male"
    elif preset == "Medium Risk (Standard Borrower)":
        st.session_state.credit_score_val = 710
        st.session_state.applicant_income_val = 12000.0
        st.session_state.coapplicant_income_val = 3000.0
        st.session_state.age_val = 30
        st.session_state.dependents_val = 1
        st.session_state.education_val = "Graduate"
        st.session_state.employment_status_val = "Salaried"
        st.session_state.employer_category_val = "Private"
        st.session_state.existing_loans_val = 1
        st.session_state.loan_amount_val = 14000.0
        st.session_state.loan_term_val = 36
        st.session_state.loan_purpose_val = "Home"
        st.session_state.property_area_val = "Semiurban"
        st.session_state.dti_ratio_val = 0.28
        st.session_state.savings_val = 12000.0
        st.session_state.collateral_value_val = 25000.0
        st.session_state.marital_status_val = "Married"
        st.session_state.gender_val = "Male"
    elif preset == "High Risk (Subprime Borrower)":
        st.session_state.credit_score_val = 560
        st.session_state.applicant_income_val = 5000.0
        st.session_state.coapplicant_income_val = 0.0
        st.session_state.age_val = 22
        st.session_state.dependents_val = 3
        st.session_state.existing_loans_val = 4
        st.session_state.loan_amount_val = 35000.0
        st.session_state.loan_term_val = 72
        st.session_state.loan_purpose_val = "Personal"
        st.session_state.property_area_val = "Rural"
        st.session_state.dti_ratio_val = 0.55
        st.session_state.savings_val = 500.0
        st.session_state.collateral_value_val = 1000.0
        st.session_state.marital_status_val = "Single"
        st.session_state.gender_val = "Male"

def input_change_callback():
    st.session_state.preset_selector = "Custom Input"

init_session_state()
apply_theme()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL & ARTIFACTS LOADING
#     We use st.cache_resource so artifacts are loaded once per session.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    """Load the trained ML model, scaler, and expected feature name list from disk."""
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        with open("feature_names.pkl", "rb") as f:
            feature_names = joblib.load(f)
        return model, scaler, feature_names, None  # (model, scaler, features, error)
    except Exception as e:
        return None, None, None, str(e)

model, scaler, feature_names, load_error = load_artifacts()

# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET LOADING (Cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_raw_data():
    """Load the historical loan approval dataset for portfolio metrics and insights."""
    try:
        df = pd.read_csv("loan_approval_data.csv")
        return df
    except Exception:
        return None

raw_data = load_raw_data()

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PREPROCESSING FUNCTION
#     Takes raw sidebar inputs and outputs a model-ready, scaled DataFrame.
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_inputs(
    applicant_income, coapplicant_income, age, dependents, existing_loans,
    savings, collateral_value, loan_amount, loan_term, education,
    employment_status, marital_status, loan_purpose, property_area,
    gender, employer_category, credit_score, dti_ratio,
):
    """
    Build the model-ready feature vector from raw user inputs.
    Engineers log transforms, squared values, manually applies OHE, 
    and aligns features according to the expected feature list.
    """
    # ── Derived Features ──
    applicant_income_log = np.log1p(applicant_income)
    dti_ratio_sq = dti_ratio ** 2
    credit_score_sq = credit_score ** 2

    # ── Ordinal Encodings ──
    education_level = 1 if education == "Graduate" else 0

    # ── One-Hot Encodings (Manual) ──
    emp_salaried     = 1 if employment_status == "Salaried" else 0
    emp_self         = 1 if employment_status == "Self-employed" else 0
    emp_unemployed   = 1 if employment_status == "Unemployed" else 0

    marital_single   = 1 if marital_status == "Single" else 0

    purpose_car      = 1 if loan_purpose == "Car" else 0
    purpose_edu      = 1 if loan_purpose == "Education" else 0
    purpose_home     = 1 if loan_purpose == "Home" else 0
    purpose_personal = 1 if loan_purpose == "Personal" else 0

    area_semiurban   = 1 if property_area == "Semiurban" else 0
    area_urban       = 1 if property_area == "Urban" else 0

    gender_male      = 1 if gender == "Male" else 0

    emp_cat_govt     = 1 if employer_category == "Government" else 0
    emp_cat_mnc      = 1 if employer_category == "MNC" else 0
    emp_cat_private  = 1 if employer_category == "Private" else 0
    emp_cat_unemp    = 1 if employer_category == "Unemployed" else 0

    raw = {
        "Applicant_Income":               applicant_income,
        "Coapplicant_Income":             coapplicant_income,
        "Age":                            age,
        "Dependents":                     dependents,
        "Existing_Loans":                 existing_loans,
        "Savings":                        savings,
        "Collateral_Value":               collateral_value,
        "Loan_Amount":                    loan_amount,
        "Loan_Term":                      loan_term,
        "Education_Level":                education_level,
        "Employment_Status_Salaried":     emp_salaried,
        "Employment_Status_Self-employed": emp_self,
        "Employment_Status_Unemployed":   emp_unemployed,
        "Marital_Status_Single":          marital_single,
        "Loan_Purpose_Car":               purpose_car,
        "Loan_Purpose_Education":         purpose_edu,
        "Loan_Purpose_Home":              purpose_home,
        "Loan_Purpose_Personal":          purpose_personal,
        "Property_Area_Semiurban":        area_semiurban,
        "Property_Area_Urban":            area_urban,
        "Gender_Male":                    gender_male,
        "Employer_Category_Government":   emp_cat_govt,
        "Employer_Category_MNC":          emp_cat_mnc,
        "Employer_Category_Private":      emp_cat_private,
        "Employer_Category_Unemployed":   emp_cat_unemp,
        "DTI_Ratio_sq":                   dti_ratio_sq,
        "Credit_Score_sq":                credit_score_sq,
        "Applicant_Income_log":           applicant_income_log,
    }

    df = pd.DataFrame([raw])

    # Reindex to match exactly what the model was trained on
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    return df

# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTLY CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_plotly_layout(t):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t['sub_text'], family="Inter"),
        margin=dict(l=10, r=10, t=30, b=10),
    )

def gauge_credit_score(score: int) -> go.Figure:
    """Gauge chart showing the applicant's credit score with colored bands."""
    t = get_active_theme()
    if score < 580:
        bar_color = t['danger']
        rating    = "Poor"
    elif score < 670:
        bar_color = t['warning']
        rating    = "Fair"
    elif score < 740:
        bar_color = t['warning']
        rating    = "Good"
    elif score < 800:
        bar_color = t['primary']
        rating    = "Very Good"
    else:
        bar_color = t['success']
        rating    = "Excellent"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 750, "valueformat": ".0f",
               "increasing": {"color": t['success']},
               "decreasing": {"color": t['danger']}},
        title={"text": f"Credit Score — <b>{rating}</b>",
               "font": {"size": 14, "color": t['text']}},
        gauge={
            "axis": {"range": [300, 900], "tickwidth": 1,
                     "tickcolor": t['border'], "tickfont": {"size": 9, "color": t['sub_text']}},
            "bar":  {"color": bar_color, "thickness": 0.25},
            "bgcolor": t['bg'],
            "borderwidth": 0,
            "steps": [
                {"range": [300, 580], "color": hex_to_rgba(t['danger'], 0.1)},
                {"range": [580, 670], "color": hex_to_rgba(t['warning'], 0.1)},
                {"range": [670, 740], "color": hex_to_rgba(t['warning'], 0.1)},
                {"range": [740, 900], "color": hex_to_rgba(t['primary'], 0.1)},
            ],
            "threshold": {
                "line": {"color": t['primary'], "width": 3},
                "thickness": 0.75,
                "value": 750,
            },
        },
        number={"font": {"size": 36, "color": t['text']}},
     ))
    fig.update_layout(**get_plotly_layout(t), height=230)
    return fig


def bar_income_vs_loan(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
) -> go.Figure:
    """Horizontal bar comparing combined monthly income pool vs. requested loan amount."""
    t = get_active_theme()
    combined = applicant_income + coapplicant_income
    lti = loan_amount / max(combined, 1)

    bar_color_loan  = t['danger'] if lti > 5 else t['success']
    bar_color_combo = t['primary']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Combined Monthly Income",
        x=[combined],
        y=["Finances"],
        orientation="h",
        marker_color=bar_color_combo,
        text=[f"₹{combined:,.0f}"],
        textposition="inside",
        insidetextanchor="start",
        textfont={"color": t['bg'], "size": 11},
    ))
    fig.add_trace(go.Bar(
        name="Loan Amount Requested",
        x=[loan_amount],
        y=["Loan"],
        orientation="h",
        marker_color=bar_color_loan,
        text=[f"₹{loan_amount:,.0f}"],
        textposition="inside",
        insidetextanchor="start",
        textfont={"color": t['bg'], "size": 11},
    ))
    fig.add_annotation(
        x=0.01, y=1.15, xref="paper", yref="paper",
        text=f"Loan-to-Income Ratio: <b>{lti:.1f}×</b>  {'⚠️ High' if lti > 5 else '✅ Healthy'}",
        showarrow=False,
        font={"size": 11, "color": t['primary']},
        align="left",
    )
    fig.update_layout(
        **get_plotly_layout(t),
        barmode="overlay",
        height=180,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font={"size": 10, "color": t['text']}),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont={"color": t['text']}),
    )
    return fig


def radar_financial_health(
    credit_score: int,
    savings: float,
    collateral_value: float,
    applicant_income: float,
    loan_amount: float,
    existing_loans: int,
) -> go.Figure:
    """Radar chart comparing applicant profile indicators to banking benchmarks."""
    t = get_active_theme()
    cs_score   = min(100, (credit_score  - 300) / 6)
    sav_score  = min(100, savings / 5_000_000 * 100)
    col_score  = min(100, collateral_value / 10_000_000 * 100)
    inc_score  = min(100, applicant_income / 200_000 * 100)
    lti_score  = max(0, 100 - (loan_amount / max(applicant_income, 1) / 0.12))
    loan_score = max(0, 100 - existing_loans * 15)

    categories = [
        "Credit Score", "Savings",
        "Collateral", "Income",
        "LTI Ratio", "Loan Burden",
    ]
    values = [cs_score, sav_score, col_score, inc_score, lti_score, loan_score]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor=hex_to_rgba(t['primary'], 0.13),
        line=dict(color=t['primary'], width=2),
        name="Applicant",
    ))
    bench = [70] * (len(categories) + 1)
    fig.add_trace(go.Scatterpolar(
        r=bench,
        theta=cats_closed,
        mode="lines",
        line=dict(color=t['border'], width=1.5, dash="dot"),
        name="Bank Benchmark",
    ))
    fig.update_layout(
        **get_plotly_layout(t),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont={"size": 8, "color": t['sub_text']},
                gridcolor=t['border'], linecolor=t['border'],
            ),
            angularaxis=dict(
                tickfont={"size": 10, "color": t['text']},
                gridcolor=t['border'], linecolor=t['border'],
            ),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.08, font={"size": 10, "color": t['text']}),
        height=310,
    )
    return fig


def plot_model_coefficients() -> go.Figure:
    """Generate a horizontal bar chart of the model coefficients (explanatory weights)."""
    t = get_active_theme()
    coefficients = {
        "Applicant_Income_log":           2.3906,
        "Credit_Score_sq":                1.9549,
        "Employer_Category_MNC":          0.1883,
        "Property_Area_Urban":            0.2075,
        "Employer_Category_Unemployed":   0.1000,
        "Employer_Category_Private":      0.0952,
        "Marital_Status_Single":          0.0553,
        "Existing_Loans":                 0.0547,
        "Property_Area_Semiurban":        0.0531,
        "Savings":                        0.0389,
        "Coapplicant_Income":             0.0353,
        "Age":                            0.0004,
        "Employer_Category_Government":   -0.0228,
        "Collateral_Value":               -0.0341,
        "Loan_Purpose_Personal":          -0.0358,
        "Dependents":                     -0.1370,
        "Employment_Status_Self-employed": -0.1391,
        "Loan_Term":                      -0.1634,
        "Education_Level":                -0.1695,
        "Loan_Purpose_Home":              -0.1895,
        "Loan_Purpose_Car":               -0.2121,
        "Employment_Status_Unemployed":   -0.2348,
        "Employment_Status_Salaried":     -0.2396,
        "Loan_Purpose_Education":         -0.2518,
        "Loan_Amount":                    -0.2697,
        "Gender_Male":                    -0.3267,
        "Applicant_Income":               -1.7248,
        "DTI_Ratio_sq":                   -2.3684,
    }
    
    sorted_coefs = sorted(coefficients.items(), key=lambda x: x[1])
    features = [x[0] for x in sorted_coefs]
    weights = [x[1] for x in sorted_coefs]
    
    colors = [t['danger'] if w < 0 else t['success'] for w in weights]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weights,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{w:+.3f}" for w in weights],
        textposition="outside",
        textfont={"size": 10, "color": t['sub_text']}
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t['sub_text'], family="Inter"),
        margin=dict(l=150, r=40, t=10, b=10),
        xaxis=dict(gridcolor=t['border'], zerolinecolor=t['border'], title=dict(text="Coefficient Weight", font={"color": t['text']}), tickfont={"color": t['sub_text']}),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont={"size": 9, "color": t['text']}),
        height=620,
        showlegend=False
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SIDEBAR — Navigation & Input Form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    t = get_active_theme()

    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px 0 5px 0;">
          <h2 style="margin: 0; font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: bold; color: {t['primary']};">🏦 CreditWise</h2>
          <p style="margin: 0; font-size: 0.8rem; color: {t['sub_text']}; font-weight: 500;">AI-Powered Loan Approval System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("━━━━━━━━━━━━━━")
    st.markdown("##### 🎨 Theme")
    theme_choice = st.selectbox(
        "Theme Selector",
        ["🌟 Default Theme", "🌞 Light Theme", "🌙 Dark Theme"],
        key="theme_selector",
        label_visibility="collapsed"
    )
    st.markdown("━━━━━━━━━━━━━━")

    st.markdown("##### 📍 Navigation")
    page = st.radio(
        "Select Page",
        [
            "🔮 Loan Prediction",
            "🏠 Dashboard",
            "📊 Data Insights",
            "📈 Model Analytics",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )
    st.markdown("━━━━━━━━━━━━━━")

    st.markdown("##### ⚙️ Model Information")
    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: {t['text']}; line-height: 1.5;">
          <b>Algorithm:</b> Logistic Regression<br>
          <b>Accuracy:</b> 86%
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("━━━━━━━━━━━━━━")

    st.markdown("##### 👩💻 Developer")
    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: {t['text']}; line-height: 1.5;">
          <b>Payal Maina</b><br>
          B.Tech CSE (Data Science)
        </div>
        """,
        unsafe_allow_html=True
    )

    show_inputs = page in ["🔮 Loan Prediction", "🏠 Dashboard", "📊 Data Insights"]

    if show_inputs:
        st.markdown("━━━━━━━━━━━━━━")
        st.markdown("##### 🎯 Risk Profile Preset")
        preset = st.selectbox(
            "Preset Risk Profile",
            [
                "Custom Input",
                "Low Risk (Prime Borrower)",
                "Medium Risk (Standard Borrower)",
                "High Risk (Subprime Borrower)"
            ],
            key="preset_selector",
            on_change=apply_preset_callback
        )
        st.markdown("━━━━━━━━━━━━━━")

        st.markdown("##### 📋 Applicant Profile")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="gender_val",
            on_change=input_change_callback
        )

        age = st.slider(
            "Age (years)",
            min_value=18, max_value=75,
            key="age_val",
            step=1,
            on_change=input_change_callback,
            help="Applicant's current age."
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married"],
            key="marital_status_val",
            on_change=input_change_callback
        )

        dependents = st.slider(
            "Number of Dependents",
            0, 6,
            key="dependents_val",
            step=1,
            on_change=input_change_callback,
            help="Number of financially dependent family members."
        )

        education = st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"],
            key="education_val",
            on_change=input_change_callback,
            help="Graduate = 1, Not Graduate = 0 after encoding."
        )

        st.markdown("━━━━━━━━━━━━━━")
        st.markdown("##### 💼 Employment & Income")

        employment_status = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-employed", "Contract", "Unemployed"],
            key="employment_status_val",
            on_change=input_change_callback
        )

        employer_category = st.selectbox(
            "Employer Category",
            ["Government", "MNC", "Private", "Business", "Unemployed"],
            key="employer_category_val",
            on_change=input_change_callback
        )

        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)",
            min_value=1_000, max_value=25_000,
            key="applicant_income_val",
            step=500,
            on_change=input_change_callback,
            help="Gross monthly income of the primary applicant (scaled to match database range)."
        )

        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income (₹)",
            min_value=0, max_value=20_000,
            key="coapplicant_income_val",
            step=500,
            on_change=input_change_callback,
            help="Leave at 0 if there is no co-applicant (scaled to match database range)."
        )

        existing_loans = st.slider(
            "Existing Active Loans",
            0, 10,
            key="existing_loans_val",
            step=1,
            on_change=input_change_callback,
            help="Number of loans already running."
        )

        st.markdown("━━━━━━━━━━━━━━")
        st.markdown("##### 🏠 Loan & Financial Details")

        loan_amount = st.number_input(
            "Loan Amount Requested (₹)",
            min_value=1_000, max_value=50_000,
            key="loan_amount_val",
            step=500,
            on_change=input_change_callback,
            help="Total loan amount requested (scaled to match database range)."
        )

        loan_term = st.selectbox(
            "Loan Term (months)",
            [12, 24, 36, 48, 60, 72, 84],
            key="loan_term_val",
            on_change=input_change_callback
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Home", "Car", "Education", "Personal", "Business"],
            key="loan_purpose_val",
            on_change=input_change_callback
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"],
            key="property_area_val",
            on_change=input_change_callback
        )

        st.markdown("━━━━━━━━━━━━━━")
        st.markdown("##### 📊 Financial Health")

        credit_score = st.slider(
            "Credit Score",
            min_value=300, max_value=900,
            key="credit_score_val",
            step=1,
            on_change=input_change_callback,
            help="CIBIL / credit bureau score. 750+ is generally considered good."
        )

        dti_ratio = st.slider(
            "Debt-to-Income (DTI) Ratio",
            min_value=0.10, max_value=0.60,
            key="dti_ratio_val",
            step=0.01,
            on_change=input_change_callback,
            help="Ratio of applicant's monthly debt obligations to gross monthly income."
        )

        savings = st.number_input(
            "Total Savings (₹)",
            min_value=0, max_value=25_000,
            key="savings_val",
            step=500,
            on_change=input_change_callback,
            help="Total savings value (scaled to match database range)."
        )

        collateral_value = st.number_input(
            "Collateral Value (₹)",
            min_value=0, max_value=60_000,
            key="collateral_value_val",
            step=500,
            on_change=input_change_callback,
            help="Current market value of any asset offered as collateral (scaled to match database range)."
        )
    else:
        # Defaults to prevent NameErrors in calculations for non-input views
        gender = "Male"
        age = 32
        marital_status = "Single"
        dependents = 1
        education = "Graduate"
        employment_status = "Salaried"
        employer_category = "Private"
        applicant_income = 11000
        coapplicant_income = 0
        existing_loans = 1
        loan_amount = 15000
        loan_term = 36
        loan_purpose = "Home"
        property_area = "Urban"
        credit_score = 720
        dti_ratio = 0.35
        savings = 10000
        collateral_value = 20000

# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN PAGE HEADER & ROUTING
# ══════════════════════════════════════════════════════════════════════════════
badge_dict = {
    "🔮 Loan Prediction": "AI-Powered Loan Approval System",
    "🏠 Dashboard": "AI-Powered Loan Approval System",
    "📊 Data Insights": "AI-Powered Loan Approval System",
    "📈 Model Analytics": "AI-Powered Loan Approval System",
    "ℹ️ About Project": "AI-Powered Loan Approval System"
}

title_dict = {
    "🔮 Loan Prediction": "🏦 CreditWise Prediction Engine",
    "🏠 Dashboard": "🏦 CreditWise Dashboard",
    "📊 Data Insights": "🏦 CreditWise Portfolio Insights",
    "📈 Model Analytics": "🏦 CreditWise Model Analytics",
    "ℹ️ About Project": "🏦 About CreditWise System"
}

sub_dict = {
    "🔮 Loan Prediction": "Automated credit risk evaluation. Adjust inputs in the sidebar and run the prediction.",
    "🏠 Dashboard": "High-level overview of the loan applicant profile and overall bank lending trends.",
    "📊 Data Insights": "Visual analytics for the current applicant and global distribution analysis.",
    "📈 Model Analytics": "Deep dive into model coefficients, weights, and predictive metrics.",
    "ℹ️ About Project": "Technical design details, feature preprocessing pipeline, and architecture info."
}

st.markdown(
    f"""
    <div class="hero-container">
      <div class="hero-badge">{badge_dict[page]}</div>
      <h1 class="hero-title">{title_dict[page]}</h1>
      <p class="hero-sub">{sub_dict[page]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(
        f"⚠️ Could not load model artefacts: `{load_error}`\n\n"
        "Please ensure **model.pkl**, **scaler.pkl**, and **feature_names.pkl** are in the same "
        "directory as `app.py` and restart the app."
    )
    st.stop()

# ── 8a. Loan Prediction View ──
if page == "🔮 Loan Prediction":
    st.success("✅ Model and Scaler loaded successfully — ready to predict.", icon="🤖")
    st.markdown('<div class="section-header">LOAN DECISION ENGINE</div>', unsafe_allow_html=True)

    predict_col, result_col = st.columns([1, 2], gap="large")

    # Determine if we should auto-predict (active risk preset profile)
    auto_predict = st.session_state.get("preset_selector", "Custom Input") != "Custom Input"

    with predict_col:
        st.markdown(
            """
            <p style="color:#94A3B8; font-size:0.85rem; margin-bottom:16px;">
            Review the applicant details in the sidebar, then click the button to
            run the model and generate an instant decision.
            </p>
            """,
            unsafe_allow_html=True,
        )
        predict_clicked = st.button("⚡  Predict Loan Status", use_container_width=True)
        
        if auto_predict:
            st.markdown(
                f"""
                <div style="padding:12px; border:1px solid #334155; border-radius:8px; background:#1E293B; text-align:center; margin-top:10px;">
                  <span style="color:#10B981; font-size:0.85rem; font-weight:600;">Preset Active:</span><br>
                  <span style="color:#E2E8F0; font-size:0.8rem;">{st.session_state.preset_selector}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with result_col:
        if predict_clicked or auto_predict:
            input_df = preprocess_inputs(
                applicant_income=applicant_income,
                coapplicant_income=coapplicant_income,
                age=age,
                dependents=dependents,
                existing_loans=existing_loans,
                savings=savings,
                collateral_value=collateral_value,
                loan_amount=loan_amount,
                loan_term=loan_term,
                education=education,
                employment_status=employment_status,
                marital_status=marital_status,
                loan_purpose=loan_purpose,
                property_area=property_area,
                gender=gender,
                employer_category=employer_category,
                credit_score=credit_score,
                dti_ratio=dti_ratio,
            )

            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]

            proba_text = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(scaled_input)[0]
                confidence = proba[1] if len(proba) > 1 else proba[0]
                proba_text = f"<div class='result-sub'>Model confidence: <b>{confidence*100:.1f}%</b></div>"

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-approved">
                      <div class="result-icon">✅</div>
                      <div class="result-title">Loan Approved</div>
                      <div class="result-sub">
                        The applicant meets CreditWise's credit criteria.
                        Proceed to documentation and disbursement.
                      </div>
                      {proba_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if predict_clicked:
                    st.balloons()
            else:
                st.markdown(
                    f"""
                    <div class="result-rejected">
                      <div class="result-icon">❌</div>
                      <div class="result-title">Loan Rejected</div>
                      <div class="result-sub">
                        The application does not satisfy the minimum risk thresholds.
                        Advise the applicant to improve their credit profile and reapply.
                      </div>
                      {proba_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("🔍 View Processed Feature Vector"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
        else:
            st.markdown(
                """
                <div style="
                  border: 1px dashed #1E3A5F; border-radius: 14px;
                  padding: 28px; text-align: center; color: #475569;">
                  <div style="font-size: 2rem; margin-bottom: 8px;">💡</div>
                  <div style="font-size: 0.9rem;">
                    Adjust the inputs in the sidebar and click<br>
                    <b style="color:#F0C040">Predict Loan Status</b> or select a preset to see the result.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── 8b. Dashboard View ──
elif page == "🏠 Dashboard":
    t = get_active_theme()
    dti = dti_ratio
    emi_est = (loan_amount / max(loan_term, 1)) * 1.08

    st.markdown('<div class="section-header">APPLICANT SNAPSHOT</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-dot" style="background-color: {t['warning']}; box-shadow: 0 0 6px {t['warning']}80;"></div>
            <div class="kpi-label" style="color: {t['sub_text']};">Monthly Income</div>
            <div class="kpi-value" style="color: {t['text']};">₹{applicant_income:,.0f}</div>
            <div class="kpi-unit" style="color: {t['sub_text']};">+ ₹{coapplicant_income:,.0f} co-applicant</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-dot" style="background-color: {t['primary']}; box-shadow: 0 0 6px {t['primary']}80;"></div>
            <div class="kpi-label" style="color: {t['sub_text']};">Loan Requested</div>
            <div class="kpi-value" style="color: {t['text']};">₹{loan_amount/1_00_000:.1f}L</div>
            <div class="kpi-unit" style="color: {t['sub_text']};">{loan_term} months · {loan_purpose}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-dot" style="background-color: {t['success'] if credit_score >= 700 else t['danger']}; box-shadow: 0 0 6px {t['success'] if credit_score >= 700 else t['danger']}80;"></div>
            <div class="kpi-label" style="color: {t['sub_text']};">Credit Score</div>
            <div class="kpi-value" style="color: {t['text']};">{credit_score}</div>
            <div class="kpi-unit" style="color: {t['sub_text']};">{'Good standing' if credit_score >= 700 else 'Needs review'}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-dot" style="background-color: {t['success'] if dti < 0.4 else t['danger']}; box-shadow: 0 0 6px {t['success'] if dti < 0.4 else t['danger']}80;"></div>
            <div class="kpi-label" style="color: {t['sub_text']};">DTI Ratio</div>
            <div class="kpi-value" style="color: {t['text']};">{dti:.2f}</div>
            <div class="kpi-unit" style="color: {t['sub_text']};">{'Within limits' if dti < 0.4 else 'Elevated risk'}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-dot" style="background-color: {t['primary']}; box-shadow: 0 0 6px {t['primary']}80;"></div>
            <div class="kpi-label" style="color: {t['sub_text']};">Est. Monthly EMI</div>
            <div class="kpi-value" style="color: {t['text']};">₹{emi_est:,.0f}</div>
            <div class="kpi-unit" style="color: {t['sub_text']};">approx @ 8% p.a.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">PORTFOLIO STATISTICS</div>', unsafe_allow_html=True)
    if raw_data is not None:
        total_apps = len(raw_data)
        avg_cs = raw_data["Credit_Score"].mean()
        avg_inc = raw_data["Applicant_Income"].mean()
        avg_la = raw_data["Loan_Amount"].mean()
        approved_pct = (raw_data["Loan_Approved"].str.lower() == "yes").mean() * 100

        st.markdown(
            f"""
            <div class="kpi-grid">
              <div class="kpi-card">
                <div class="kpi-dot" style="background-color: {t['primary']}; box-shadow: 0 0 6px {t['primary']}80;"></div>
                <div class="kpi-label" style="color: {t['sub_text']};">Total Database Records</div>
                <div class="kpi-value" style="color: {t['text']};">{total_apps}</div>
                <div class="kpi-unit" style="color: {t['sub_text']};">Cumulative files checked</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-dot" style="background-color: {t['success']}; box-shadow: 0 0 6px {t['success']}80;"></div>
                <div class="kpi-label" style="color: {t['sub_text']};">Baseline Approval Rate</div>
                <div class="kpi-value" style="color: {t['text']};">{approved_pct:.1f}%</div>
                <div class="kpi-unit" style="color: {t['sub_text']};">Historical acceptance benchmark</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-dot" style="background-color: {t['warning']}; box-shadow: 0 0 6px {t['warning']}80;"></div>
                <div class="kpi-label" style="color: {t['sub_text']};">Avg Database Credit Score</div>
                <div class="kpi-value" style="color: {t['text']};">{avg_cs:.0f}</div>
                <div class="kpi-unit" style="color: {t['sub_text']};">Median credit history bureau rank</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-dot" style="background-color: {t['primary']}; box-shadow: 0 0 6px {t['primary']}80;"></div>
                <div class="kpi-label" style="color: {t['sub_text']};">Avg Loan Size Requested</div>
                <div class="kpi-value" style="color: {t['text']};">₹{avg_la/1_00_000:.1f}L</div>
                <div class="kpi-unit" style="color: {t['sub_text']};">₹{avg_la:,.0f} overall average</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-dot" style="background-color: {t['warning']}; box-shadow: 0 0 6px {t['warning']}80;"></div>
                <div class="kpi-label" style="color: {t['sub_text']};">Avg Base Applicant Income</div>
                <div class="kpi-value" style="color: {t['text']};">₹{avg_inc:,.0f}</div>
                <div class="kpi-unit" style="color: {t['sub_text']};">Excludes joint incomes</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Database `loan_approval_data.csv` could not be loaded for portfolio metrics.")

    # ── Portfolio Demographics Charts ──
    st.markdown('<div class="section-header">PORTFOLIO DEMOGRAPHICS</div>', unsafe_allow_html=True)
    if raw_data is not None:
        clean_raw_data = raw_data.dropna(subset=["Employment_Status", "Loan_Approved", "Loan_Purpose", "Loan_Amount"])
        
        dash_col1, dash_col2 = st.columns(2, gap="large")
        with dash_col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{t['sub_text']};font-size:0.78rem;text-transform:uppercase;"
                "letter-spacing:0.06em;font-weight:600;margin-bottom:8px;'>"
                "Approval Rate by Employment Status</p>",
                unsafe_allow_html=True,
            )
            emp_app = clean_raw_data.groupby("Employment_Status")["Loan_Approved"].apply(
                lambda x: (x.str.lower() == "yes").mean() * 100
            ).reset_index(name="Approval Rate (%)")
            
            fig_emp = px.bar(
                emp_app,
                x="Employment_Status",
                y="Approval Rate (%)",
                color="Employment_Status",
                color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B", "#EF4444"],
                text_auto=".1f"
            )
            fig_emp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t['sub_text'], family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None, tickfont={"color": t['text']}),
                yaxis=dict(gridcolor=t['border'], range=[0, 105], title=dict(text="Approval Rate (%)", font={"color": t['text']}), tickfont={"color": t['sub_text']}),
                showlegend=False,
                height=260
            )
            st.plotly_chart(fig_emp, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with dash_col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{t['sub_text']};font-size:0.78rem;text-transform:uppercase;"
                "letter-spacing:0.06em;font-weight:600;margin-bottom:8px;'>"
                "Average Loan Amount by Purpose</p>",
                unsafe_allow_html=True,
            )
            purp_loan = clean_raw_data.groupby("Loan_Purpose")["Loan_Amount"].mean().reset_index(name="Avg Loan Amount")
            purp_loan["Avg Loan Amount (Lakhs)"] = purp_loan["Avg Loan Amount"] / 100000
            
            fig_purp = px.bar(
                purp_loan,
                x="Loan_Purpose",
                y="Avg Loan Amount (Lakhs)",
                color="Loan_Purpose",
                color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B", "#8B5CF6", "#EF4444"],
                text_auto=".1f"
            )
            fig_purp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t['sub_text'], family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None, tickfont={"color": t['text']}),
                yaxis=dict(gridcolor=t['border'], title=dict(text="Avg Amount (₹ Lakhs)", font={"color": t['text']}), tickfont={"color": t['sub_text']}),
                showlegend=False,
                height=260
            )
            st.plotly_chart(fig_purp, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Portfolio demographic charts could not be loaded (missing raw CSV).")

# ── 8c. Data Insights View ──
elif page == "📊 Data Insights":
    t = get_active_theme()
    st.markdown('<div class="section-header">APPLICANT VISUAL ANALYTICS</div>', unsafe_allow_html=True)
    chart_col1, chart_col2, chart_col3 = st.columns([1.1, 1.2, 1.1], gap="large")

    with chart_col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(gauge_credit_score(credit_score), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:{t['sub_text']};font-size:0.78rem;text-transform:uppercase;"
            "letter-spacing:0.06em;font-weight:600;margin-bottom:8px;'>"
            "Income vs. Loan Comparison</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            bar_income_vs_loan(applicant_income, coapplicant_income, loan_amount),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(
            radar_financial_health(
                credit_score, savings, collateral_value,
                applicant_income, loan_amount, existing_loans,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">HISTORICAL PORTFOLIO DISTRIBUTIONS</div>', unsafe_allow_html=True)

    if raw_data is not None:
        dist_col1, dist_col2, dist_col3 = st.columns([1, 1, 1], gap="medium")

        with dist_col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{t['sub_text']};font-size:0.78rem;text-transform:uppercase;"
                "letter-spacing:0.06em;font-weight:600;margin-bottom:8px;'>"
                "Credit Score by Approval</p>",
                unsafe_allow_html=True,
            )
            fig_hist = px.histogram(
                raw_data,
                x="Credit_Score",
                color="Loan_Approved",
                nbins=20,
                color_discrete_map={"Yes": t['success'], "No": t['danger']},
                labels={"Credit_Score": "Credit Score", "count": "Count"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t['sub_text'], family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font={"color": t['text']}),
                height=260
            )
            fig_hist.update_xaxes(gridcolor=t['border'], title_font={"color": t['text']}, tickfont={"color": t['sub_text']})
            fig_hist.update_yaxes(gridcolor=t['border'], title_font={"color": t['text']}, tickfont={"color": t['sub_text']})
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with dist_col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{t['sub_text']};font-size:0.78rem;text-transform:uppercase;"
                "letter-spacing:0.06em;font-weight:600;margin-bottom:8px;'>"
                "Income vs. Loan Scatter (⭐=Current)</p>",
                unsafe_allow_html=True,
            )
            fig_scatter = px.scatter(
                raw_data,
                x="Applicant_Income",
                y="Loan_Amount",
                color="Loan_Approved",
                color_discrete_map={"Yes": t['success'], "No": t['danger']},
                labels={"Applicant_Income": "Income (₹)", "Loan_Amount": "Loan (₹)"},
            )
            fig_scatter.add_trace(go.Scatter(
                x=[applicant_income],
                y=[loan_amount],
                mode="markers",
                marker=dict(color=t['warning'], size=14, symbol="star", line=dict(color=t['bg'], width=1.5)),
                name="Current"
            ))
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t['sub_text'], family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font={"color": t['text']}),
                height=260
            )
            fig_scatter.update_xaxes(gridcolor=t['border'], title_font={"color": t['text']}, tickfont={"color": t['sub_text']})
            fig_scatter.update_yaxes(gridcolor=t['border'], title_font={"color": t['text']}, tickfont={"color": t['sub_text']})
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with dist_col3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:{t['sub_text']};font-size:0.78rem;text-transform:uppercase;"
                "letter-spacing:0.06em;font-weight:600;margin-bottom:8px;'>"
                "Historical Approvals Donut</p>",
                unsafe_allow_html=True,
            )
            fig_pie = px.pie(
                raw_data,
                names="Loan_Approved",
                color="Loan_Approved",
                color_discrete_map={"Yes": t['success'], "No": t['danger']},
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t['sub_text'], family="Inter"),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, title=None, font={"color": t['text']}),
                height=260
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Distribution charts could not be loaded (missing raw CSV).")

# ── 8d. Model Analytics View ──
elif page == "📈 Model Analytics":
    t = get_active_theme()
    st.markdown('<div class="section-header">MODEL DETAILS & HEALTH</div>', unsafe_allow_html=True)

    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-dot" style="background-color: {t['warning']}; box-shadow: 0 0 6px {t['warning']}80;"></div>
              <div class="kpi-label" style="color: {t['sub_text']};">Primary Algorithm</div>
              <div class="kpi-value" style="font-size:1.3rem; color: {t['text']};">Logistic Regression</div>
              <div class="kpi-unit" style="color: {t['sub_text']};">Production risk classifier</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_metrics2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-dot" style="background-color: {t['success']}; box-shadow: 0 0 6px {t['success']}80;"></div>
              <div class="kpi-label" style="color: {t['sub_text']};">Model Accuracy</div>
              <div class="kpi-value" style="color: {t['text']};">88.0%</div>
              <div class="kpi-unit" style="color: {t['sub_text']};">Actual test classification rate</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_metrics3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-dot" style="background-color: {t['primary']}; box-shadow: 0 0 6px {t['primary']}80;"></div>
              <div class="kpi-label" style="color: {t['sub_text']};">Feature Count</div>
              <div class="kpi-value" style="color: {t['text']};">28</div>
              <div class="kpi-unit" style="color: {t['sub_text']};">Post-one-hot engineered</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_metrics4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-dot" style="background-color: {t['warning']}; box-shadow: 0 0 6px {t['warning']}80;"></div>
              <div class="kpi-label" style="color: {t['sub_text']};">Data Preprocessing</div>
              <div class="kpi-value" style="font-size:1.3rem; color: {t['text']};">Standard Scaling</div>
              <div class="kpi-unit" style="color: {t['sub_text']};">Log & Square Transforms</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown('<hr>', unsafe_allow_html=True)
    
    comp_col, cm_col = st.columns([1.1, 0.9], gap="large")
    
    with comp_col:
        st.markdown('<div class="section-header">MODEL PERFORMANCE COMPARISON</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <p style="color:{t['sub_text']}; font-size:0.82rem; margin-bottom:12px;">
            Comparison of classifiers trained on CreditWise's engineered features.
            Logistic Regression provides the best balance of Accuracy and Recall.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        metrics_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Naive Bayes", "K-Nearest Neighbors"],
            "Accuracy": [0.880, 0.860, 0.785],
            "Precision": [0.785, 0.811, 0.673],
            "Recall": [0.836, 0.705, 0.574],
            "F1 Score": [0.810, 0.754, 0.619]
        })
        
        melted_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        fig_comp = px.bar(
            melted_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={
                "Logistic Regression": t['primary'],
                "Naive Bayes": hex_to_rgba(t['primary'], 0.8),
                "K-Nearest Neighbors": t['warning']
            },
            text_auto=".3f"
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t['sub_text'], family="Inter"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None, tickfont={"color": t['text']}),
            yaxis=dict(gridcolor=t['border'], range=[0, 1.05], title=dict(text="Score", font={"color": t['text']}), tickfont={"color": t['sub_text']}),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font={"color": t['text']})
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    with cm_col:
        st.markdown('<div class="section-header">CONFUSION MATRIX VISUALIZER</div>', unsafe_allow_html=True)
        selected_cm_model = st.selectbox(
            "Select Classifier for Heatmap",
            ["Logistic Regression", "Naive Bayes", "K-Nearest Neighbors"],
            label_visibility="collapsed"
        )
        
        if selected_cm_model == "Logistic Regression":
            z = [[125, 14], [10, 51]]
        elif selected_cm_model == "Naive Bayes":
            z = [[129, 10], [18, 43]]
        else:
            z = [[122, 17], [26, 35]]

        x_labels = ["Predicted Rejected", "Predicted Approved"]
        y_labels = ["Actual Rejected", "Actual Approved"]
        
        fig_cm = px.imshow(
            z,
            x=x_labels,
            y=y_labels,
            color_continuous_scale=[[0, t['card_bg']], [1, t['primary']]],
            aspect="auto"
        )
        
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(
                    x=j,
                    y=i,
                    text=str(z[i][j]),
                    showarrow=False,
                    font=dict(size=20, color=t['text'] if z[i][j] > 40 else t['sub_text'], weight="bold")
                )
                
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t['sub_text'], family="Inter"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=270,
            coloraxis_showscale=False,
            xaxis=dict(tickfont={"color": t['text']}),
            yaxis=dict(tickfont={"color": t['text']})
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">FEATURE WEIGHTS & COEFFICIENTS</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p style="color:{t['sub_text']}; font-size:0.85rem; margin-bottom:16px;">
        The chart below illustrates the coefficients of the Logistic Regression model. 
        <b style="color:{t['success']}">Green bars</b> represent features that increase the likelihood of loan approval 
        (e.g., higher credit score, log applicant income). <b style="color:{t['danger']}">Red bars</b> represent features 
        that increase the probability of loan rejection (e.g., higher DTI ratio, higher loan amount).
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_model_coefficients(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 8e. About Project View ──
elif page == "ℹ️ About Project":
    t = get_active_theme()
    st.markdown('<div class="section-header">PROJECT DETAILS & OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background:{t['card_bg']}; border:1px solid {t['border']}; border-radius:14px; padding:28px; color:{t['text']};">
          <h3 style="color:{t['primary']}; margin-top:0; font-family:'Playfair Display', serif;">🏦 CreditWise Risk Division</h3>
          <p>
            The <b>CreditWise Loan Approval System</b> is an advanced machine learning dashboard designed 
            to automate standard credit verification processes. By analyzing applicant credit bureau scores, 
            debt obligations, incomes, and collateral assets, the decision engine estimates credit risk 
            in real-time and determines loan qualification automatically.
          </p>
          <hr style="border-top:1px solid {t['border']};">
          <h4 style="color:{t['primary']};">🛠️ Preprocessing & Feature Engineering Pipeline</h4>
          <p>The model requires several non-linear transformations and encoding steps to make accurate classifications:</p>
          <ul>
            <li><b>Log Transformation</b>: Applied to <i>Applicant Income</i> to reduce right-skewness and normalize heavy tails.</li>
            <li><b>Squared Features</b>:
              <ul>
                <li><i>Credit Score (squared)</i>: Amplifies the difference between subprime credit and prime credit scores.</li>
                <li><i>DTI Ratio (squared)</i>: Exponentially penalizes applicants whose debt obligations consume a high portion of their income.</li>
              </ul>
            </li>
            <li><b>One-Hot Encoding (OHE)</b>: Categorical fields like <i>Employment Status, Marital Status, Loan Purpose, Property Area, Gender,</i> and <i>Employer Category</i> are mapped into individual dummy columns (excluding a reference group).</li>
            <li><b>Standard Scaling</b>: Coordinates all numerical vectors to have a mean of 0 and standard deviation of 1.</li>
          </ul>
          <hr style="border-top:1px solid {t['border']};">
          <h4 style="color:{t['primary']};">🧠 Model Architecture</h4>
          <p>
            The system employs a <b>Logistic Regression</b> classifier, chosen for its high transparency, 
            excellent explainability in lending audits, and robust probabilistic calibration. 
            All decision weights are audited and mapped directly to bank compliance standards.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# 9.  GLOBAL FOOTER
# ══════════════════════════════════════════════════════════════════════════════
t_footer = get_active_theme()
st.markdown(
    f"""
    <hr>
    <div style="text-align:center; color:{t_footer['sub_text']}; font-size:0.72rem; padding:12px 0;">
      🏦 <b style="color:{t_footer['text']}">CreditWise</b> &nbsp;·&nbsp;
      AI-Powered Loan Approval System &nbsp;·&nbsp;
      For authorised loan officers only &nbsp;·&nbsp;
      <i>Predictions are model-generated and subject to manual review.</i>
    </div>
    """,
    unsafe_allow_html=True,
)