import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

st.set_page_config(page_title="CreditWise | Loan Approval", page_icon="💳", layout="wide")

st.markdown("""
<style>
.main-title {font-size:3rem;font-weight:800;margin-bottom:.1rem}
.subtitle {font-size:1.05rem;color:#6b7280;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

@st.cache_resource
def train_creditwise_model():
    df = load_data().copy()
    target = "Loan_Approved"

    # Dataset target is Yes/No; the project documentation defines the binary
    # target as 1 = Approved and 0 = Rejected.
    if df[target].dtype == "object":
        target_map = {"Yes": 1, "No": 0, "Approved": 1, "Rejected": 0,
                      "yes": 1, "no": 0, "approved": 1, "rejected": 0,
                      "1": 1, "0": 0}
        y = df[target].astype(str).str.strip().map(target_map)
    else:
        y = pd.to_numeric(df[target], errors="coerce")

    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # Match the documented feature engineering.
    df["DTI_Ratio_sq"] = pd.to_numeric(df["DTI_Ratio"], errors="coerce") ** 2
    df["Credit_Score_sq"] = pd.to_numeric(df["Credit_Score"], errors="coerce") ** 2
    X = df.drop(columns=[target, "Applicant_ID", "Credit_Score", "DTI_Ratio"])

    onehot_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                   "Property_Area", "Gender", "Employer_Category"]
    ordinal_cols = ["Education_Level"]
    numeric_cols = [c for c in X.columns if c not in onehot_cols + ordinal_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    # Use the actual education categories present in the dataset. This is the
    # same ordinal-encoding idea described in the project documentation and
    # avoids assuming categories that are not present in the CSV.
    education_categories = sorted(df["Education_Level"].dropna().astype(str).unique().tolist())
    education_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[education_categories],
                                    handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
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
except Exception as exc:
    df = None
    model = None
    st.error(f"Model could not be loaded: {exc}")

st.markdown('<div class="main-title">💳 CreditWise</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered loan approval decision-support system</div>', unsafe_allow_html=True)
st.success("● LIVE ML MODEL — Feature-engineered Logistic Regression")
st.info("Enter applicant details to generate an Approved/Rejected prediction. The prediction supports loan officers before final human verification.")

# Use categories from the actual CSV so the deployed UI always matches training data.
def options(column):
    return sorted(df[column].dropna().astype(str).unique().tolist())

with st.form("creditwise_form"):
    st.subheader("Applicant Profile")
    c1,c2,c3,c4 = st.columns(4)
    applicant_income = c1.number_input("Applicant Income (monthly)", min_value=0.0, value=10000.0, step=500.0)
    coapplicant_income = c2.number_input("Co-applicant Income (monthly)", min_value=0.0, value=5000.0, step=500.0)
    age = c3.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    dependents = c4.number_input("Dependents", min_value=0, max_value=20, value=1, step=1)

    st.subheader("Employment & Background")
    c1,c2,c3,c4 = st.columns(4)
    employment_status = c1.selectbox("Employment Status", options("Employment_Status"))
    marital_status = c2.selectbox("Marital Status", options("Marital_Status"))
    education_level = c3.selectbox("Education Level", options("Education_Level"))
    gender = c4.selectbox("Gender", options("Gender"))
    c1,c2 = st.columns(2)
    employer_category = c1.selectbox("Employer Category", options("Employer_Category"))
    property_area = c2.selectbox("Property Area", options("Property_Area"))

    st.subheader("Credit & Financial Profile")
    c1,c2,c3,c4 = st.columns(4)
    credit_score = c1.number_input("Credit Score", min_value=300, max_value=900, value=700, step=1)
    existing_loans = c2.number_input("Existing Loans", min_value=0, max_value=20, value=1, step=1)
    dti_ratio = c3.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
    savings = c4.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)

    st.subheader("Loan Details")
    c1,c2,c3,c4 = st.columns(4)
    collateral_value = c1.number_input("Collateral Value", min_value=0.0, value=25000.0, step=1000.0)
    loan_amount = c2.number_input("Loan Amount", min_value=0.0, value=20000.0, step=1000.0)
    loan_terms = sorted(pd.to_numeric(df["Loan_Term"], errors="coerce").dropna().unique().tolist())
    loan_term = c3.selectbox("Loan Term (months)", loan_terms)
    loan_purpose = c4.selectbox("Loan Purpose", options("Loan_Purpose"))
    submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)

if submitted and model is not None:
    applicant = pd.DataFrame([{
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
    }])

    applicant["DTI_Ratio_sq"] = applicant["DTI_Ratio"] ** 2
    applicant["Credit_Score_sq"] = applicant["Credit_Score"] ** 2
    applicant = applicant.drop(columns=["Credit_Score", "DTI_Ratio"])

    prediction = int(model.predict(applicant)[0])
    probability = float(model.predict_proba(applicant)[0, 1])

    st.markdown("---")
    st.subheader("Prediction Result")
    c1,c2,c3 = st.columns(3)
    c1.metric("Credit Score", int(credit_score))
    c2.metric("DTI Ratio", f"{dti_ratio:.2f}")
    c3.metric("Approval Probability", f"{probability*100:.1f}%")

    if prediction == 1:
        st.success("## ✅ LOAN APPROVED\n\nCreditWise predicts that this application is likely to be approved.")
    else:
        st.error("## ❌ LOAN REJECTED\n\nCreditWise predicts that this application is likely to be rejected.")

    st.progress(probability, text=f"Estimated approval probability: {probability*100:.1f}%")
    st.caption("Educational demonstration. The deployed model is trained on the historical dataset for inference. The project's reported 88.00% accuracy, 83.61% recall and 80.95% F1-score are from the documented held-out test evaluation. Final lending decisions require human verification.")
