import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

st.set_page_config(
    page_title="CreditWise | AI Loan Risk",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Premium fintech-style UI
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #172033;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        background: linear-gradient(135deg, #0b1220 0%, #18243a 100%);
        border-radius: 22px;
        padding: 34px 38px;
        margin-bottom: 20px;
        box-shadow: 0 14px 35px rgba(15, 23, 42, .10);
    }

    .hero-kicker {
        color: #8fb7ff;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        color: white;
        font-size: 2.65rem;
        font-weight: 800;
        letter-spacing: -.04em;
        margin: 0;
    }

    .hero-subtitle {
        color: #c7d2e3;
        font-size: 1rem;
        margin-top: 8px;
        max-width: 720px;
        line-height: 1.6;
    }

    .live-pill {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(34, 197, 94, .12);
        color: #86efac;
        border: 1px solid rgba(134, 239, 172, .18);
        font-size: .78rem;
        font-weight: 700;
    }

    .section-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 18px;
        padding: 20px 22px 8px 22px;
        margin: 12px 0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .035);
    }

    .section-label {
        color: #64748b;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.18rem;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 92px;
    }

    .metric-name {
        color: #64748b;
        font-size: .75rem;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .result-approved {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 25px;
        margin-top: 12px;
    }

    .result-rejected {
        background: linear-gradient(135deg, #fff1f2, #fff7f7);
        border: 1px solid #fecdd3;
        border-radius: 20px;
        padding: 25px;
        margin-top: 12px;
    }

    .result-title {
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-text {
        color: #475569;
        line-height: 1.55;
        margin: 0;
    }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: .75rem;
        padding: 25px 0 10px 0;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 12px;
        min-height: 52px;
        font-weight: 750;
        font-size: 1rem;
    }

    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data + ML pipeline
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_data.csv")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def train_creditwise_model():
    df = load_data().copy()
    target = "Loan_Approved"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    target_text = df[target].astype("string").str.strip().str.lower()
    target_map = {
        "yes": 1,
        "approved": 1,
        "1": 1,
        "true": 1,
        "no": 0,
        "rejected": 0,
        "0": 0,
        "false": 0,
    }
    y = target_text.map(target_map)

    valid = y.notna()
    if valid.sum() == 0:
        raise ValueError("No valid Loan_Approved labels were found. Expected values such as Yes/No.")

    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    required = [
        "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
        "Employment_Status", "Age", "Marital_Status", "Dependents",
        "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender", "Employer_Category",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    # Project feature engineering: squared credit score and DTI ratio.
    df["DTI_Ratio_sq"] = pd.to_numeric(df["DTI_Ratio"], errors="coerce") ** 2
    df["Credit_Score_sq"] = pd.to_numeric(df["Credit_Score"], errors="coerce") ** 2
    X = df.drop(columns=[target, "Applicant_ID", "Credit_Score", "DTI_Ratio"])

    onehot_cols = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Gender", "Employer_Category",
    ]
    ordinal_cols = ["Education_Level"]
    numeric_cols = [c for c in X.columns if c not in onehot_cols + ordinal_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    education_categories = sorted(
        df["Education_Level"].dropna().astype(str).str.strip().unique().tolist()
    )
    if not education_categories:
        raise ValueError("Education_Level contains no usable categories.")

    education_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=[education_categories],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("education", education_pipe, ordinal_cols),
        ("cat", categorical_pipe, onehot_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    pipeline.fit(X, y)
    return pipeline


try:
    df = load_data()
    model = train_creditwise_model()
    model_error = None
except Exception as exc:
    df = load_data()
    model = None
    model_error = str(exc)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 💳 CreditWise")
st.sidebar.caption("AI Loan Risk Decision Support")
st.sidebar.markdown("---")
st.sidebar.markdown("**MODEL**")
st.sidebar.write("Feature-engineered Logistic Regression")
st.sidebar.markdown("**TASK**")
st.sidebar.write("Binary loan approval classification")
st.sidebar.markdown("**DATA**")
st.sidebar.write(f"{len(df):,} historical applications")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python • Pandas • Scikit-learn • Streamlit")


# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Intelligent Lending Analytics</div>
        <div class="hero-title">💳 CreditWise</div>
        <div class="hero-subtitle">
            AI-powered loan approval decision-support for faster, more consistent
            applicant assessment — designed to support loan officers before final
            human verification.
        </div>
        <div class="live-pill">● LIVE ML MODEL &nbsp;•&nbsp; FEATURE-ENGINEERED LOGISTIC REGRESSION</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(f"Model could not be loaded: {model_error}")
else:
    # High-level model evidence. These are the project's documented held-out test results.
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div class="metric-card"><div class="metric-name">TEST ACCURACY</div><div class="metric-value">88.00%</div></div>', unsafe_allow_html=True)
    m2.markdown('<div class="metric-card"><div class="metric-name">RECALL</div><div class="metric-value">83.61%</div></div>', unsafe_allow_html=True)
    m3.markdown('<div class="metric-card"><div class="metric-name">F1 SCORE</div><div class="metric-value">80.95%</div></div>', unsafe_allow_html=True)
    m4.markdown('<div class="metric-card"><div class="metric-name">MODEL</div><div class="metric-value">Logistic Reg.</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Applicant form
# -----------------------------------------------------------------------------
def options(column):
    values = df[column].dropna().astype(str).str.strip()
    values = sorted(v for v in values.unique().tolist() if v)
    if not values:
        raise ValueError(f"Column '{column}' contains no usable categories.")
    return values


with st.form("creditwise_form"):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 / Applicant</div><div class="section-title">Applicant Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    applicant_income = c1.number_input("Applicant Income (monthly)", min_value=0.0, value=10000.0, step=500.0)
    coapplicant_income = c2.number_input("Co-applicant Income (monthly)", min_value=0.0, value=5000.0, step=500.0)
    age = c3.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    dependents = c4.number_input("Dependents", min_value=0, max_value=20, value=1, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 / Background</div><div class="section-title">Employment & Background</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    employment_status = c1.selectbox("Employment Status", options("Employment_Status"))
    marital_status = c2.selectbox("Marital Status", options("Marital_Status"))
    education_level = c3.selectbox("Education Level", options("Education_Level"))
    gender = c4.selectbox("Gender", options("Gender"))
    c1, c2 = st.columns(2)
    employer_category = c1.selectbox("Employer Category", options("Employer_Category"))
    property_area = c2.selectbox("Property Area", options("Property_Area"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 / Risk</div><div class="section-title">Credit & Financial Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    credit_score = c1.number_input("Credit Score", min_value=300, max_value=900, value=700, step=1)
    existing_loans = c2.number_input("Existing Loans", min_value=0, max_value=20, value=1, step=1)
    dti_ratio = c3.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
    savings = c4.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 / Loan</div><div class="section-title">Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    collateral_value = c1.number_input("Collateral Value", min_value=0.0, value=25000.0, step=1000.0)
    loan_amount = c2.number_input("Loan Amount", min_value=0.0, value=20000.0, step=1000.0)
    loan_terms = sorted(pd.to_numeric(df["Loan_Term"], errors="coerce").dropna().unique().tolist())
    if not loan_terms:
        loan_terms = [12, 24, 36, 48, 60]
    loan_term = c3.selectbox("Loan Term (months)", loan_terms)
    loan_purpose = c4.selectbox("Loan Purpose", options("Loan_Purpose"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  ANALYZE APPLICATION", use_container_width=True)


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
if submitted and model is not None:
    applicant = pd.DataFrame([
        {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status,
            "Age": age,
            "Marital_Status": marital_status,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category,
        }
    ])

    applicant["DTI_Ratio_sq"] = applicant["DTI_Ratio"] ** 2
    applicant["Credit_Score_sq"] = applicant["Credit_Score"] ** 2
    applicant = applicant.drop(columns=["Credit_Score", "DTI_Ratio"])

    prediction = int(model.predict(applicant)[0])
    probability = float(model.predict_proba(applicant)[0, 1])

    st.markdown("---")
    st.markdown('<div class="section-label">05 / Decision Support</div><div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Credit Score", int(credit_score))
    r2.metric("DTI Ratio", f"{dti_ratio:.2f}")
    r3.metric("Estimated Approval Probability", f"{probability * 100:.1f}%")

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-approved">
                <div class="result-title">✅ Loan Approved</div>
                <p class="result-text">
                    CreditWise predicts a positive loan-approval outcome for this application.
                    Estimated approval probability: <strong>{probability * 100:.1f}%</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-rejected">
                <div class="result-title">❌ Loan Rejected</div>
                <p class="result-text">
                    CreditWise predicts a negative loan-approval outcome for this application.
                    Estimated approval probability: <strong>{probability * 100:.1f}%</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(probability, text=f"Estimated approval probability — {probability * 100:.1f}%")

    with st.expander("How CreditWise works"):
        st.markdown(
            """
            **ML pipeline**

            Applicant data → Missing-value handling → Categorical encoding → Feature scaling →
            Feature engineering → Logistic Regression → Approval probability

            The application is a **decision-support demonstration**. It does not replace
            final human verification or a bank's underwriting, compliance, or fairness review.
            """
        )

    st.caption(
        "The documented project evaluation reports 88.00% test accuracy, 83.61% recall and "
        "80.95% F1-score. The probability above is the model's estimate for this individual "
        "application and should not be interpreted as model accuracy. Final lending decisions "
        "require human verification."
    )

st.markdown(
    '<div class="footer-note">CreditWise • Machine Learning Decision Support • Python + Scikit-learn + Streamlit</div>',
    unsafe_allow_html=True,
)
