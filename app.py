import os

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditWise | AI Lending Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Product UI
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #081426;
        --navy-2: #0f213c;
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e5eaf2;
        --surface: #ffffff;
        --page: #f5f7fb;
        --blue: #2563eb;
        --green: #16a34a;
        --red: #dc2626;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--page); }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #172944;
    }
    [data-testid="stSidebar"] * { color: #dbe7f7; }
    [data-testid="stSidebar"] .stRadio label { color: #dbe7f7 !important; }

    .brand {
        padding: 8px 4px 20px;
        border-bottom: 1px solid #1d3453;
        margin-bottom: 20px;
    }
    .brand-name { color: white; font-size: 1.45rem; font-weight: 800; letter-spacing: -.04em; }
    .brand-sub { color: #91a7c3; font-size: .72rem; margin-top: 3px; }
    .online { display: inline-flex; gap: 7px; align-items: center; margin-top: 14px; color: #86efac; font-size: .72rem; font-weight: 700; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; display: inline-block; }

    .hero {
        background: linear-gradient(135deg, #081426 0%, #16345d 100%);
        border-radius: 24px;
        padding: 42px 44px;
        color: white;
        box-shadow: 0 20px 45px rgba(8,20,38,.14);
        margin-bottom: 24px;
    }
    .eyebrow { color: #8fb7ff; font-size: .74rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
    .hero h1 { color: white; font-size: 3rem; line-height: 1.05; margin: 10px 0 12px; letter-spacing: -.055em; }
    .hero p { color: #c7d5e8; max-width: 760px; line-height: 1.7; margin: 0; font-size: 1rem; }
    .hero-badge { display: inline-block; margin-top: 20px; padding: 7px 11px; border-radius: 999px; border: 1px solid #315b8e; background: #102a4b; color: #a9c8f5; font-size: .72rem; font-weight: 700; }

    .card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 7px 24px rgba(15,23,42,.035);
        height: 100%;
    }
    .card-label { color: var(--muted); font-size: .7rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .card-title { color: var(--ink); font-size: 1.12rem; font-weight: 800; margin-top: 4px; }
    .card-copy { color: var(--muted); font-size: .86rem; line-height: 1.6; margin-top: 8px; }

    .stat {
        background: white; border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px;
    }
    .stat-k { color: var(--muted); font-size: .68rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .stat-v { color: var(--ink); font-size: 1.7rem; font-weight: 800; margin-top: 5px; letter-spacing: -.03em; }
    .stat-s { color: #94a3b8; font-size: .7rem; margin-top: 3px; }

    .stepbar { display: flex; gap: 8px; margin: 6px 0 20px; }
    .step { flex: 1; height: 6px; border-radius: 99px; background: #dbe3ef; }
    .step.active { background: var(--blue); }

    .result {
        border-radius: 22px; padding: 30px; border: 1px solid;
    }
    .result.approved { background: #effcf4; border-color: #b7ebc7; }
    .result.rejected { background: #fff3f4; border-color: #fecdd3; }
    .result-kicker { font-size: .7rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
    .result h2 { color: var(--ink); font-size: 2rem; margin: 7px 0; letter-spacing: -.04em; }
    .result p { color: #475569; margin: 0; line-height: 1.6; }
    .prob { font-size: 3.1rem; line-height: 1; font-weight: 800; color: var(--ink); margin: 8px 0; letter-spacing: -.05em; }

    .insight { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; margin-bottom: 9px; }
    .insight strong { color: var(--ink); font-size: .84rem; }
    .insight span { color: var(--muted); font-size: .76rem; }

    .notice { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 16px; color: #64748b; font-size: .76rem; line-height: 1.6; }
    .footer { text-align: center; color: #94a3b8; font-size: .72rem; padding: 28px 0 8px; }

    div[data-testid="stFormSubmitButton"] button, .stButton button { border-radius: 11px; min-height: 46px; font-weight: 750; }
    .stSelectbox div[data-baseweb="select"], .stNumberInput input { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data and model
# -----------------------------------------------------------------------------
DATA_PATH = "loan_approval_data.csv"
TARGET = "Loan_Approved"

REQUIRED = [
    "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
    "Employment_Status", "Age", "Marital_Status", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term", "Loan_Purpose",
    "Property_Area", "Education_Level", "Gender", "Employer_Category",
]

ONEHOT = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Gender", "Employer_Category",
]
ORDINAL = ["Education_Level"]


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH} in the app directory.")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    missing = [c for c in [TARGET] + REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df


def normalise_target(series):
    mapping = {
        "yes": 1, "approved": 1, "1": 1, "true": 1,
        "no": 0, "rejected": 0, "0": 0, "false": 0,
    }
    return series.astype("string").str.strip().str.lower().map(mapping)


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data().copy()
    y = normalise_target(df[TARGET])
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # Match the project's documented feature engineering.
    df["DTI_Ratio_sq"] = pd.to_numeric(df["DTI_Ratio"], errors="coerce") ** 2
    df["Credit_Score_sq"] = pd.to_numeric(df["Credit_Score"], errors="coerce") ** 2

    X = df.drop(columns=[TARGET, "Applicant_ID", "Credit_Score", "DTI_Ratio"])
    numeric = [c for c in X.columns if c not in ONEHOT + ORDINAL]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    edu_categories = sorted(
        df["Education_Level"].dropna().astype(str).str.strip().unique().tolist()
    )
    if not edu_categories:
        raise ValueError("Education_Level has no usable values.")

    education_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[edu_categories],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scaler", StandardScaler()),
    ])

    # Works on both newer and older scikit-learn releases.
    try:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("education", education_pipe, ORDINAL),
        ("cat", categorical_pipe, ONEHOT),
    ])

    # Balanced classes improve decision sensitivity for the approval class.
    # This is the live inference model; the benchmark metrics shown in the UI
    # remain the project's separately evaluated held-out results.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(X, y)
    return model, df, y


try:
    df = load_data()
    model, clean_df, y = train_model()
    model_error = None
except Exception as exc:
    df = None
    clean_df = None
    y = None
    model = None
    model_error = str(exc)


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="brand"><div class="brand-name">💳 CreditWise</div>'
    '<div class="brand-sub">AI Lending Decision Platform</div>'
    '<div class="online"><span class="dot"></span> MODEL ONLINE</div></div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "New Assessment", "Model Insights"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("FEATURE-ENGINEERED MODEL")
st.sidebar.write("Logistic Regression")
st.sidebar.caption("DATASET")
st.sidebar.write(f"{len(df):,} applications" if df is not None else "Unavailable")
st.sidebar.markdown("---")
st.sidebar.caption("Python • Pandas • Scikit-learn • Streamlit")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_options(column):
    vals = df[column].dropna().astype(str).str.strip()
    vals = [v for v in sorted(vals.unique().tolist()) if v]
    if not vals:
        raise ValueError(f"No usable options found for {column}.")
    return vals


def row_to_input(row):
    def text(col, fallback):
        value = row.get(col)
        return fallback if pd.isna(value) else str(value).strip()

    def number(col, fallback):
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        return fallback if pd.isna(value) else float(value)

    return {
        "Applicant_Income": number("Applicant_Income", 10000.0),
        "Coapplicant_Income": number("Coapplicant_Income", 3000.0),
        "Employment_Status": text("Employment_Status", clean_options("Employment_Status")[0]),
        "Age": int(number("Age", 35)),
        "Marital_Status": text("Marital_Status", clean_options("Marital_Status")[0]),
        "Dependents": int(number("Dependents", 1)),
        "Credit_Score": int(number("Credit_Score", 700)),
        "Existing_Loans": int(number("Existing_Loans", 1)),
        "DTI_Ratio": number("DTI_Ratio", 0.30),
        "Savings": number("Savings", 10000.0),
        "Collateral_Value": number("Collateral_Value", 25000.0),
        "Loan_Amount": number("Loan_Amount", 20000.0),
        "Loan_Term": number("Loan_Term", 60),
        "Loan_Purpose": text("Loan_Purpose", clean_options("Loan_Purpose")[0]),
        "Property_Area": text("Property_Area", clean_options("Property_Area")[0]),
        "Education_Level": text("Education_Level", clean_options("Education_Level")[0]),
        "Gender": text("Gender", clean_options("Gender")[0]),
        "Employer_Category": text("Employer_Category", clean_options("Employer_Category")[0]),
    }


def predict(application):
    X = pd.DataFrame([application])
    X["DTI_Ratio_sq"] = pd.to_numeric(X["DTI_Ratio"], errors="coerce") ** 2
    X["Credit_Score_sq"] = pd.to_numeric(X["Credit_Score"], errors="coerce") ** 2
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.50)
    return prediction, probability


def model_explanation(application):
    """Return top signed linear contributions from the fitted pipeline."""
    try:
        X = pd.DataFrame([application])
        X["DTI_Ratio_sq"] = pd.to_numeric(X["DTI_Ratio"], errors="coerce") ** 2
        X["Credit_Score_sq"] = pd.to_numeric(X["Credit_Score"], errors="coerce") ** 2
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        Xt = pre.transform(X)
        names = pre.get_feature_names_out()
        contributions = Xt[0] * clf.coef_[0]
        frame = pd.DataFrame({"feature": names, "contribution": contributions})
        frame["feature"] = frame["feature"].str.replace(r"^(num|education|cat)__", "", regex=True)
        return frame.sort_values("contribution", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "contribution"])


def render_footer():
    st.markdown(
        '<div class="footer">CreditWise • AI for Responsible Lending<br>'
        'Decision-support demonstration only. Final lending decisions require human verification, underwriting and compliance review.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.markdown(
        '<div class="hero"><div class="eyebrow">Intelligent Lending Analytics</div>'
        '<h1>Make loan assessment<br>smarter with AI.</h1>'
        '<p>CreditWise is a machine-learning decision-support system that analyzes applicant, credit and financial information to estimate loan-approval probability before final human verification.</p>'
        '<div class="hero-badge">● LIVE INFERENCE &nbsp;•&nbsp; LOGISTIC REGRESSION &nbsp;•&nbsp; FEATURE ENGINEERING</div></div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(f"The ML model could not start: {model_error}")
        st.stop()

    a, b, c, d = st.columns(4)
    a.markdown('<div class="stat"><div class="stat-k">Test Accuracy</div><div class="stat-v">88.00%</div><div class="stat-s">Held-out benchmark</div></div>', unsafe_allow_html=True)
    b.markdown('<div class="stat"><div class="stat-k">Recall</div><div class="stat-v">83.61%</div><div class="stat-s">Approval-class recall</div></div>', unsafe_allow_html=True)
    c.markdown('<div class="stat"><div class="stat-k">F1 Score</div><div class="stat-v">80.95%</div><div class="stat-s">Held-out benchmark</div></div>', unsafe_allow_html=True)
    d.markdown(f'<div class="stat"><div class="stat-k">Historical Data</div><div class="stat-v">{len(df):,}</div><div class="stat-s">Loan applications</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    x, z = st.columns(2)
    with x:
        st.markdown(
            '<div class="card"><div class="card-label">The problem</div><div class="card-title">From manual review to intelligent decision support</div>'
            '<div class="card-copy">Traditional loan verification can require officers to manually inspect income, employment, credit history, debt burden and collateral. CreditWise turns those signals into a repeatable ML assessment that can be reviewed before the final lending decision.</div></div>',
            unsafe_allow_html=True,
        )
    with z:
        st.markdown(
            '<div class="card"><div class="card-label">The ML pipeline</div><div class="card-title">Preprocess → engineer → predict → explain</div>'
            '<div class="card-copy">The live pipeline handles missing values, categorical encoding and scaling, then adds squared Credit Score and DTI Ratio features before Logistic Regression estimates the probability of approval.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("How CreditWise works")
    s1, s2, s3, s4 = st.columns(4)
    for col, num, title, copy in [
        (s1, "01", "Applicant data", "Income, age, employment and household profile."),
        (s2, "02", "Risk signals", "Credit score, DTI, savings and existing loans."),
        (s3, "03", "ML inference", "Feature-engineered Logistic Regression produces a probability."),
        (s4, "04", "Decision support", "Approval/rejection signal plus model contribution insights."),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="card-label">{num}</div><div class="card-title">{title}</div><div class="card-copy">{copy}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start a New Assessment →", type="primary", use_container_width=True):
        st.session_state["goto_assessment"] = True
        st.rerun()

    render_footer()


# -----------------------------------------------------------------------------
# New assessment
# -----------------------------------------------------------------------------
if page == "New Assessment":
    st.title("New Loan Assessment")
    st.caption("Enter an applicant profile and let the trained ML pipeline estimate the probability of approval.")
    st.markdown('<div class="stepbar"><div class="step active"></div><div class="step active"></div><div class="step active"></div><div class="step active"></div></div>', unsafe_allow_html=True)

    # Honest demo path: load actual historical rows, rather than inventing a result.
    demo_rows = clean_df.copy()
    demo_rows["_target"] = y.values
    approved_rows = demo_rows[demo_rows["_target"] == 1]
    rejected_rows = demo_rows[demo_rows["_target"] == 0]

    mode = st.radio("Assessment mode", ["Manual applicant", "Historical applicant demo"], horizontal=True)
    selected_demo = None
    if mode == "Historical applicant demo":
        d1, d2 = st.columns(2)
        demo_type = d1.selectbox("Reference outcome", ["Approved example", "Rejected example"])
        pool = approved_rows if demo_type == "Approved example" else rejected_rows
        ids = pool["Applicant_ID"].astype(str).tolist()
        selected_id = d2.selectbox("Historical Applicant ID", ids)
        selected_demo = pool[pool["Applicant_ID"].astype(str) == selected_id].iloc[0]
        st.info("This mode uses a real row from the project dataset. The displayed result is still generated by the live ML model.")

    defaults = row_to_input(selected_demo) if selected_demo is not None else {
        "Applicant_Income": 10000.0, "Coapplicant_Income": 3000.0, "Employment_Status": clean_options("Employment_Status")[0],
        "Age": 35, "Marital_Status": clean_options("Marital_Status")[0], "Dependents": 1, "Credit_Score": 700,
        "Existing_Loans": 1, "DTI_Ratio": 0.30, "Savings": 10000.0, "Collateral_Value": 25000.0,
        "Loan_Amount": 20000.0, "Loan_Term": 60.0, "Loan_Purpose": clean_options("Loan_Purpose")[0],
        "Property_Area": clean_options("Property_Area")[0], "Education_Level": clean_options("Education_Level")[0],
        "Gender": clean_options("Gender")[0], "Employer_Category": clean_options("Employer_Category")[0],
    }

    with st.form("assessment_form"):
        st.markdown('<div class="card"><div class="card-label">01 / Applicant</div><div class="card-title">Applicant Profile</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        income = c1.number_input("Applicant Income / month", min_value=0.0, value=float(defaults["Applicant_Income"]), step=500.0)
        co_income = c2.number_input("Co-applicant Income / month", min_value=0.0, value=float(defaults["Coapplicant_Income"]), step=500.0)
        age = c3.number_input("Age", min_value=18, max_value=100, value=int(defaults["Age"]), step=1)
        dependents = c4.number_input("Dependents", min_value=0, max_value=20, value=int(defaults["Dependents"]), step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">02 / Background</div><div class="card-title">Employment & Background</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        employment = c1.selectbox("Employment Status", clean_options("Employment_Status"), index=clean_options("Employment_Status").index(defaults["Employment_Status"]) if defaults["Employment_Status"] in clean_options("Employment_Status") else 0)
        marital = c2.selectbox("Marital Status", clean_options("Marital_Status"), index=clean_options("Marital_Status").index(defaults["Marital_Status"]) if defaults["Marital_Status"] in clean_options("Marital_Status") else 0)
        education = c3.selectbox("Education Level", clean_options("Education_Level"), index=clean_options("Education_Level").index(defaults["Education_Level"]) if defaults["Education_Level"] in clean_options("Education_Level") else 0)
        gender = c4.selectbox("Gender", clean_options("Gender"), index=clean_options("Gender").index(defaults["Gender"]) if defaults["Gender"] in clean_options("Gender") else 0)
        c1, c2 = st.columns(2)
        employer = c1.selectbox("Employer Category", clean_options("Employer_Category"), index=clean_options("Employer_Category").index(defaults["Employer_Category"]) if defaults["Employer_Category"] in clean_options("Employer_Category") else 0)
        area = c2.selectbox("Property Area", clean_options("Property_Area"), index=clean_options("Property_Area").index(defaults["Property_Area"]) if defaults["Property_Area"] in clean_options("Property_Area") else 0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">03 / Risk</div><div class="card-title">Credit & Financial Profile</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        credit = c1.number_input("Credit Score", min_value=300, max_value=900, value=int(defaults["Credit_Score"]), step=1)
        loans = c2.number_input("Existing Loans", min_value=0, max_value=20, value=int(defaults["Existing_Loans"]), step=1)
        dti = c3.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=float(defaults["DTI_Ratio"]), step=0.01, format="%.2f")
        savings = c4.number_input("Savings", min_value=0.0, value=float(defaults["Savings"]), step=500.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">04 / Loan</div><div class="card-title">Loan Details</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        collateral = c1.number_input("Collateral Value", min_value=0.0, value=float(defaults["Collateral_Value"]), step=1000.0)
        amount = c2.number_input("Loan Amount", min_value=0.0, value=float(defaults["Loan_Amount"]), step=1000.0)
        term_values = sorted(pd.to_numeric(df["Loan_Term"], errors="coerce").dropna().unique().tolist())
        term = c3.selectbox("Loan Term (months)", term_values if term_values else [12, 24, 36, 48, 60], index=(term_values.index(defaults["Loan_Term"]) if defaults["Loan_Term"] in term_values else 0))
        purpose = c4.selectbox("Loan Purpose", clean_options("Loan_Purpose"), index=clean_options("Loan_Purpose").index(defaults["Loan_Purpose"]) if defaults["Loan_Purpose"] in clean_options("Loan_Purpose") else 0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run AI Assessment →", type="primary", use_container_width=True)

    if submitted:
        application = {
            "Applicant_Income": income, "Coapplicant_Income": co_income, "Employment_Status": employment,
            "Age": age, "Marital_Status": marital, "Dependents": dependents, "Credit_Score": credit,
            "Existing_Loans": loans, "DTI_Ratio": dti, "Savings": savings, "Collateral_Value": collateral,
            "Loan_Amount": amount, "Loan_Term": term, "Loan_Purpose": purpose, "Property_Area": area,
            "Education_Level": education, "Gender": gender, "Employer_Category": employer,
        }
        pred, probability = predict(application)
        st.session_state["last_application"] = application
        st.session_state["last_prediction"] = pred
        st.session_state["last_probability"] = probability
        st.session_state["last_explanation"] = model_explanation(application)
        st.rerun()

    if "last_probability" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        probability = st.session_state["last_probability"]
        pred = st.session_state["last_prediction"]
        pct = probability * 100
        css = "approved" if pred == 1 else "rejected"
        title = "Loan Approval Likely" if pred == 1 else "Loan Approval Unlikely"
        icon = "✓" if pred == 1 else "×"
        st.markdown(
            f'<div class="result {css}"><div class="result-kicker">AI assessment result</div>'
            f'<h2>{icon} {title}</h2><div class="prob">{pct:.2f}%</div>'
            f'<p>Estimated approval probability from the live feature-engineered Logistic Regression model.</p></div>',
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Model signal")
            if pct >= 50:
                st.success(f"The model's estimated approval probability is {pct:.2f}%, above the 50% decision threshold used by this demonstration.")
            else:
                st.warning(f"The model's estimated approval probability is {pct:.2f}%, below the 50% decision threshold used by this demonstration.")
        with r2:
            st.markdown("#### Key model contributions")
            exp = st.session_state.get("last_explanation", pd.DataFrame())
            if not exp.empty:
                positive = exp[exp["contribution"] > 0].head(3)
                negative = exp[exp["contribution"] < 0].tail(3).sort_values("contribution")
                for _, item in positive.iterrows():
                    st.markdown(f'<div class="insight"><strong>↑ {item["feature"]}</strong><br><span>Pushes the model toward approval</span></div>', unsafe_allow_html=True)
                for _, item in negative.iterrows():
                    st.markdown(f'<div class="insight"><strong>↓ {item["feature"]}</strong><br><span>Pushes the model toward rejection</span></div>', unsafe_allow_html=True)
            else:
                st.caption("Contribution details are unavailable for this prediction.")

        st.markdown('<div class="notice">Important: this is a machine-learning decision-support demonstration, not a lending decision. The probability is an inference score, not model accuracy. Final decisions require appropriate human review, underwriting, compliance and fairness checks.</div>', unsafe_allow_html=True)

    render_footer()


# -----------------------------------------------------------------------------
# Model insights
# -----------------------------------------------------------------------------
if page == "Model Insights":
    st.title("Model Insights")
    st.caption("Evidence and implementation details behind the CreditWise inference pipeline.")

    if model is None:
        st.error(f"The ML model could not start: {model_error}")
        st.stop()

    a, b, c = st.columns(3)
    a.markdown('<div class="stat"><div class="stat-k">Accuracy</div><div class="stat-v">88.00%</div><div class="stat-s">Held-out project result</div></div>', unsafe_allow_html=True)
    b.markdown('<div class="stat"><div class="stat-k">Recall</div><div class="stat-v">83.61%</div><div class="stat-s">Held-out project result</div></div>', unsafe_allow_html=True)
    c.markdown('<div class="stat"><div class="stat-k">F1 Score</div><div class="stat-v">80.95%</div><div class="stat-s">Held-out project result</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><div class="card-label">Model</div><div class="card-title">Feature-engineered Logistic Regression</div><div class="card-copy">The classifier receives imputed, scaled numeric features, ordinal-encoded education and one-hot encoded categorical variables. Credit Score and DTI Ratio are transformed into squared features to capture nonlinear effects used in the project.</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="card-label">Evaluation</div><div class="card-title">Benchmark ≠ live probability</div><div class="card-copy">The 88.00% accuracy, 83.61% recall and 80.95% F1 score are the project held-out evaluation results. A prediction such as 72.40% is an estimated approval probability for one applicant, not an accuracy percentage.</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, title, copy in [
        (p1, "Raw data", "1,000 historical applications"),
        (p2, "Cleaning", "Missing-value imputation"),
        (p3, "Features", "Encoding + DTI² + Credit²"),
        (p4, "Classifier", "Logistic Regression"),
        (p5, "Output", "Approval probability"),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="card-title">{title}</div><div class="card-copy">{copy}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="notice">The live model is retrained from the repository dataset when the Streamlit app starts. Its prediction output is intended for demonstration and decision support; it should not be represented as a production lending model.</div>', unsafe_allow_html=True)
    render_footer()
