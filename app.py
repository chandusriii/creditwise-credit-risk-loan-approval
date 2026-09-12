import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="CreditWise | Loan Approval Prediction",
    page_icon="💳",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            margin-top: 1rem;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">💳 CreditWise</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered loan approval decision-support system</div>',
    unsafe_allow_html=True,
)

st.info(
    "CreditWise provides an ML-based Approved/Rejected prediction to support "
    "loan officers before final human verification."
)

# -----------------------------
# Input form
# -----------------------------
with st.form("creditwise_form"):
    st.subheader("Applicant Profile")
    c1, c2, c3, c4 = st.columns(4)

    applicant_income = c1.number_input(
        "Applicant Income (monthly)", min_value=0.0, value=10000.0, step=500.0
    )
    coapplicant_income = c2.number_input(
        "Co-applicant Income (monthly)", min_value=0.0, value=5000.0, step=500.0
    )
    age = c3.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    dependents = c4.number_input(
        "Dependents", min_value=0, max_value=20, value=1, step=1
    )

    st.subheader("Employment & Background")
    c1, c2, c3, c4 = st.columns(4)

    employment_status = c1.selectbox(
        "Employment Status", ["Salaried", "Self-Employed", "Business"]
    )
    marital_status = c2.selectbox("Marital Status", ["Married", "Single"])
    education_level = c3.selectbox(
        "Education Level", ["Graduate", "Postgraduate", "Undergraduate"]
    )
    gender = c4.selectbox("Gender", ["Male", "Female"])

    c1, c2 = st.columns(2)
    employer_category = c1.selectbox(
        "Employer Category", ["Govt", "Private", "Self"]
    )
    property_area = c2.selectbox(
        "Property Area", ["Urban", "Semi-Urban", "Rural"]
    )

    st.subheader("Credit & Financial Profile")
    c1, c2, c3, c4 = st.columns(4)

    credit_score = c1.number_input(
        "Credit Score", min_value=300, max_value=900, value=700, step=1
    )
    existing_loans = c2.number_input(
        "Existing Loans", min_value=0, max_value=20, value=1, step=1
    )
    dti_ratio = c3.number_input(
        "DTI Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f"
    )
    savings = c4.number_input(
        "Savings", min_value=0.0, value=10000.0, step=500.0
    )

    st.subheader("Loan Details")
    c1, c2, c3, c4 = st.columns(4)

    collateral_value = c1.number_input(
        "Collateral Value", min_value=0.0, value=25000.0, step=1000.0
    )
    loan_amount = c2.number_input(
        "Loan Amount", min_value=0.0, value=20000.0, step=1000.0
    )
    loan_term = c3.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    loan_purpose = c4.selectbox(
        "Loan Purpose", ["Home", "Education", "Personal", "Business"]
    )

    submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)

# -----------------------------
# Prediction logic
# -----------------------------
if submitted:
    # This is a lightweight demonstration rule layer intended for the deployed UI.
    # Replace this section with the saved sklearn pipeline once it is committed
    # to the repository.
    score = 0.0

    if credit_score >= 700:
        score += 0.30
    elif credit_score >= 650:
        score += 0.15
    else:
        score -= 0.20

    if dti_ratio <= 0.35:
        score += 0.20
    elif dti_ratio <= 0.50:
        score += 0.05
    else:
        score -= 0.20

    total_income = applicant_income + coapplicant_income
    if total_income >= 15000:
        score += 0.15
    elif total_income >= 8000:
        score += 0.05
    else:
        score -= 0.10

    if savings >= 10000:
        score += 0.10
    elif savings < 5000:
        score -= 0.05

    if existing_loans <= 1:
        score += 0.10
    elif existing_loans >= 4:
        score -= 0.10

    if collateral_value >= loan_amount:
        score += 0.10
    else:
        score -= 0.05

    probability = float(np.clip(0.50 + score, 0.05, 0.95))
    approved = probability >= 0.50

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Score", int(credit_score))
    col2.metric("DTI Ratio", f"{dti_ratio:.2f}")
    col3.metric("Estimated Confidence", f"{probability * 100:.1f}%")

    if approved:
        st.success("✅ LOAN APPROVED — ML decision-support prediction")
    else:
        st.error("❌ LOAN REJECTED — ML decision-support prediction")

    st.caption(
        "Educational demonstration only. This interface should support, not replace, "
        "the bank's final human verification process."
    )
