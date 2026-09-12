import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

st.set_page_config(page_title="CreditWise | AI Lending Platform", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:#f6f9fd}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071426,#06111f);border-right:1px solid #12243d}
[data-testid="stSidebar"] *{color:#e6edf7}.block-container{max-width:1500px;padding-top:1.4rem}
.brand{padding:5px 2px 18px}.brand-row{display:flex;align-items:center;gap:11px}.brand-icon{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0ea5e9,#6366f1);font-size:21px}.brand-name{font-size:1.25rem;font-weight:800;color:#fff}.brand-tag{font-size:.65rem;color:#8da4c3;margin-top:3px}
.online{display:inline-flex;gap:7px;padding:8px 12px;border:1px solid #00d4a7;border-radius:999px;color:#42f0c4!important;font-size:.7rem;font-weight:700}.dot{width:8px;height:8px;border-radius:50%;background:#00e0a8;box-shadow:0 0 9px #00e0a8}
.nav{padding:10px;border-radius:10px;margin:5px 0;color:#a8bad2!important;font-size:.8rem}.active{background:linear-gradient(90deg,rgba(37,99,235,.32),rgba(37,99,235,.12));border:1px solid rgba(59,130,246,.42);color:#fff!important}
.side-card{margin-top:18px;border-radius:15px;padding:18px 16px;background:linear-gradient(145deg,#0b2344,#07162a);border:1px solid #14518b}.side-card h3{color:#fff;margin:0;font-size:1.1rem}.side-card span{color:#18c8ff}.side-card p{color:#91aac8;font-size:.72rem;line-height:1.5}.creator{border-top:1px solid #17304e;margin-top:20px;padding-top:15px;color:#fff;font-size:.76rem}.creator small{color:#7f97b5}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.title{font-size:1.45rem;font-weight:800;color:#0f1e35}.sub{font-size:.78rem;color:#708198}.pill{background:#06243b;color:#fff;border-radius:10px;padding:9px 13px;font-size:.7rem;font-weight:700}.pill b{color:#19d9a7}
.hero{border-radius:19px;padding:27px 30px;background:linear-gradient(120deg,#edf5ff,#f7fbff 60%,#eaf3ff);border:1px solid #dbe9fb;margin:12px 0 16px}.hero h1{font-size:2.7rem;letter-spacing:-.055em;color:#101f38;margin:0}.hero h1 span{color:#087cf5}.hero h3{font-size:1rem;color:#182b4a;margin:2px 0 5px}.hero p{color:#667a95;font-size:.8rem}.features{display:flex;gap:28px}.features b{color:#126de8;background:#dcecff;border-radius:50%;padding:6px;margin-right:6px;font-size:.7rem}
.progress{display:flex;align-items:center;margin:10px 2px 18px}.step{font-size:.68rem;color:#708198;display:flex;gap:6px;align-items:center;white-space:nowrap}.step b{width:28px;height:28px;border-radius:50%;background:#e4ebf4;display:flex;align-items:center;justify-content:center}.step.on{color:#123d78;font-weight:700}.step.on b{background:#1678f2;color:#fff}.line{height:2px;background:#dce5f0;flex:1;margin:0 9px}
.card,.right-card{background:#fff;border:1px solid #dfe8f2;border-radius:16px;padding:16px;margin-bottom:12px;box-shadow:0 5px 18px rgba(15,50,90,.035)}.head{display:flex;gap:10px;align-items:center;margin-bottom:10px}.ico{width:34px;height:34px;border-radius:10px;background:#e7f1ff;display:flex;align-items:center;justify-content:center}.section{font-size:.96rem;font-weight:800;color:#172946}.copy{font-size:.66rem;color:#7b8da4}.right-title{font-size:.9rem;font-weight:800;color:#162845;margin-bottom:12px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.metric{border-radius:10px;padding:11px 5px;text-align:center;background:#eef6ff}.metric:nth-child(2){background:#edf9f5}.metric:nth-child(3){background:#f5f0ff}.metric strong{font-size:1rem;color:#096ce8}.metric:nth-child(2) strong{color:#07986f}.metric:nth-child(3) strong{color:#7749d9}.metric small{display:block;color:#647891;font-size:.58rem;margin-top:2px}.info{margin:11px 0;font-size:.66rem;color:#536984}.info strong{display:block;color:#273c59}.how{display:flex;gap:9px;margin:11px 0}.num{flex:0 0 27px;height:27px;border-radius:50%;background:#e3efff;color:#176fe3;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.7rem}.how strong{font-size:.67rem;color:#203650}.how small{display:block;color:#7a8ca3;font-size:.6rem;margin-top:2px}.notice{background:#e7f3ff;color:#1565cf;border-radius:10px;padding:10px;font-size:.61rem;line-height:1.45}.quote{border-radius:15px;padding:17px;background:linear-gradient(145deg,#0a2444,#061427);color:#fff;font-size:.72rem;line-height:1.45}
.result-ok{background:#effdf6;border:1px solid #b8efd5;border-radius:18px;padding:22px}.result-no{background:#fff3f4;border:1px solid #fecbd1;border-radius:18px;padding:22px}.result h2{color:#122640}.footer{text-align:center;color:#8a9ab0;font-size:.64rem;padding:18px}
div[data-testid="stFormSubmitButton"] button{min-height:50px;border-radius:11px;font-weight:800}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df=pd.read_csv("loan_approval_data.csv")
    df.columns=df.columns.str.strip()
    return df

@st.cache_resource
def train_model():
    df=load_data().copy(); target="Loan_Approved"
    y=df[target].astype("string").str.strip().str.lower().map({"yes":1,"approved":1,"1":1,"true":1,"no":0,"rejected":0,"0":0,"false":0})
    valid=y.notna(); df=df.loc[valid].copy(); y=y.loc[valid].astype(int)
    df["DTI_Ratio_sq"]=pd.to_numeric(df["DTI_Ratio"],errors="coerce")**2
    df["Credit_Score_sq"]=pd.to_numeric(df["Credit_Score"],errors="coerce")**2
    X=df.drop(columns=[target,"Applicant_ID","Credit_Score","DTI_Ratio"])
    onehot=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ordinal=["Education_Level"]; numeric=[c for c in X.columns if c not in onehot+ordinal]
    num=Pipeline([("imputer",SimpleImputer(strategy="mean")),("scaler",StandardScaler())])
    cats=sorted(df["Education_Level"].dropna().astype(str).str.strip().unique().tolist())
    edu=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OrdinalEncoder(categories=[cats],handle_unknown="use_encoded_value",unknown_value=-1)),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)),("scaler",StandardScaler())])
    prep=ColumnTransformer([("num",num,numeric),("education",edu,ordinal),("cat",cat,onehot)])
    pipe=Pipeline([("preprocessor",prep),("classifier",LogisticRegression(max_iter=2000,random_state=42))]); pipe.fit(X,y); return pipe

try:
    df=load_data(); model=train_model(); error=None
except Exception as e:
    df=load_data(); model=None; error=str(e)

def options(col):
    vals=sorted(v for v in df[col].dropna().astype(str).str.strip().unique().tolist() if v)
    if not vals: raise ValueError(f"No usable values in {col}")
    return vals

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-row"><div class="brand-icon">💳</div><div><div class="brand-name">CreditWise</div><div class="brand-tag">AI for a Brighter Financial Future</div></div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="online"><span class="dot"></span> AI MODEL ONLINE</div>',unsafe_allow_html=True)
    st.markdown('<div class="nav active">⌂ &nbsp; Loan Assessment</div><div class="nav">▥ &nbsp; Model Performance</div><div class="nav">▤ &nbsp; About Project</div><div class="nav">⚙ &nbsp; How It Works</div><div class="nav">▤ &nbsp; Dataset Insights</div>',unsafe_allow_html=True)
    st.markdown('<div class="side-card"><h3>Smarter<br/>Loan Decisions<br/><span>with AI</span></h3><p>Data-driven insights.<br/>Real impact.</p></div><div style="color:#91a8c2;font-size:.7rem;margin:20px 2px">“Enabling responsible lending through AI.”</div><div class="creator"><small>Built by</small><br/><b>Chandu Sri</b><br/><small>AI/ML Engineer</small></div>',unsafe_allow_html=True)

st.markdown('<div class="top"><div><div class="title">Loan Assessment</div><div class="sub">Fill in the details below to get an AI-powered decision support.</div></div><div class="pill"><b>●</b> Live Model →</div></div>',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Credit<span>Wise</span></h1><h3>AI-Powered Loan Approval Decision Support</h3><p>Quick. Intelligent. Responsible.</p><div class="features"><span><b>✓</b>Secure &amp; Private</span><span><b>ϟ</b>Instant Insights</span><span><b>●</b>Data-Driven</span></div></div>',unsafe_allow_html=True)
st.markdown('<div class="progress"><div class="step on"><b>1</b>Applicant</div><div class="line"></div><div class="step"><b>2</b>Employment</div><div class="line"></div><div class="step"><b>3</b>Financial</div><div class="line"></div><div class="step"><b>4</b>Loan Details</div></div>',unsafe_allow_html=True)

left,right=st.columns([2.65,1],gap="large")
with left:
    if model is None:
        st.error(f"Model could not be loaded: {error}")
    else:
        with st.form("creditwise_form"):
            st.markdown('<div class="card"><div class="head"><div class="ico">👤</div><div><div class="section">Applicant Profile</div><div class="copy">Basic information about the applicant</div></div></div>',unsafe_allow_html=True)
            a,b,c,d=st.columns(4)
            applicant_income=a.number_input("Applicant Income (monthly)",0.0,1000000.0,10000.0,500.0)
            coapplicant_income=b.number_input("Co-applicant Income (monthly)",0.0,1000000.0,5000.0,500.0)
            age=c.number_input("Age",18,100,35,1); dependents=d.number_input("Dependents",0,20,1,1)
            st.markdown('</div>',unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="head"><div class="ico">💼</div><div><div class="section">Employment &amp; Background</div><div class="copy">Professional and personal background details</div></div></div>',unsafe_allow_html=True)
            a,b,c,d=st.columns(4)
            employment=a.selectbox("Employment Status",options("Employment_Status")); marital=b.selectbox("Marital Status",options("Marital_Status")); education=c.selectbox("Education Level",options("Education_Level")); gender=d.selectbox("Gender",options("Gender"))
            a,b=st.columns(2); employer=a.selectbox("Employer Category",options("Employer_Category")); property_area=b.selectbox("Property Area",options("Property_Area")); st.markdown('</div>',unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="head"><div class="ico">🛡️</div><div><div class="section">Credit &amp; Financial Profile</div><div class="copy">Current financial standing and credit history</div></div></div>',unsafe_allow_html=True)
            a,b,c,d=st.columns(4); credit=a.number_input("Credit Score",300,900,700,1); existing=b.number_input("Existing Loans",0,20,1,1); dti=c.number_input("DTI Ratio",0.0,1.0,0.30,0.01,format="%.2f"); savings=d.number_input("Savings (₹)",0.0,10000000.0,50000.0,500.0); st.markdown('</div>',unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="head"><div class="ico">▣</div><div><div class="section">Loan Details</div><div class="copy">Information about the requested loan</div></div></div>',unsafe_allow_html=True)
            a,b,c,d=st.columns(4); collateral=a.number_input("Collateral Value (₹)",0.0,100000000.0,100000.0,1000.0); loan_amount=b.number_input("Loan Amount (₹)",0.0,100000000.0,200000.0,1000.0); terms=sorted(pd.to_numeric(df["Loan_Term"],errors="coerce").dropna().unique().tolist()) or [12,24,36,48,60]; loan_term=c.selectbox("Loan Term (months)",terms); purpose=d.selectbox("Loan Purpose",options("Loan_Purpose")); st.markdown('</div>',unsafe_allow_html=True)
            submitted=st.form_submit_button("✨  Analyze Application  →",use_container_width=True)
        if submitted:
            applicant=pd.DataFrame([{"Applicant_Income":applicant_income,"Coapplicant_Income":coapplicant_income,"Employment_Status":employment,"Age":age,"Marital_Status":marital,"Dependents":dependents,"Credit_Score":credit,"Existing_Loans":existing,"DTI_Ratio":dti,"Savings":savings,"Collateral_Value":collateral,"Loan_Amount":loan_amount,"Loan_Term":loan_term,"Loan_Purpose":purpose,"Property_Area":property_area,"Education_Level":education,"Gender":gender,"Employer_Category":employer}])
            applicant["DTI_Ratio_sq"]=applicant["DTI_Ratio"]**2; applicant["Credit_Score_sq"]=applicant["Credit_Score"]**2; applicant=applicant.drop(columns=["Credit_Score","DTI_Ratio"])
            pred=int(model.predict(applicant)[0]); prob=float(model.predict_proba(applicant)[0,1])
            st.markdown("### Prediction Result"); x,y,z=st.columns(3); x.metric("Credit Score",int(credit)); y.metric("DTI Ratio",f"{dti:.2f}"); z.metric("Estimated Approval Probability",f"{prob*100:.1f}%")
            if pred==1: st.markdown(f'<div class="result-ok"><h2>✅ Loan Approved!</h2><p>CreditWise predicts a positive loan-approval outcome. Estimated approval probability: <strong>{prob*100:.1f}%</strong>.</p></div>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="result-no"><h2>❌ Loan Rejected</h2><p>CreditWise predicts a negative loan-approval outcome. Estimated approval probability: <strong>{prob*100:.1f}%</strong>.</p></div>',unsafe_allow_html=True)
            st.progress(prob,text=f"Estimated approval probability — {prob*100:.1f}%")
            st.caption("Decision-support demonstration only. Final lending decisions require bank verification, underwriting, compliance, fairness review and human judgment.")

with right:
    st.markdown('<div class="right-card"><div class="right-title">▥ &nbsp; Model Performance <span style="float:right;color:#08a875;font-size:.6rem">● Live</span></div><div class="metrics"><div class="metric"><strong>88.00%</strong><small>Accuracy</small></div><div class="metric"><strong>83.61%</strong><small>Recall</small></div><div class="metric"><strong>80.95%</strong><small>F1 Score</small></div></div><div class="info"><strong>Model</strong>Feature-engineered Logistic Regression</div><div class="info"><strong>Dataset</strong>1,000 Loan Applications</div><div class="info"><strong>Target</strong>Loan Status (Yes/No)</div></div><div class="right-card"><div class="right-title">⚙ &nbsp; How CreditWise Works</div><div class="how"><div class="num">1</div><div><strong>You enter applicant details</strong><small>Provide basic, financial and loan information</small></div></div><div class="how"><div class="num">2</div><div><strong>Our ML model analyzes the data</strong><small>Feature engineering + Logistic Regression</small></div></div><div class="how"><div class="num">3</div><div><strong>Get instant prediction</strong><small>Receive approval probability and insights</small></div></div><div class="how"><div class="num">4</div><div><strong>Final decision by loan officers</strong><small>This is a decision support system</small></div></div><div class="notice">ⓘ &nbsp; This is a decision support tool. Final loan approval is subject to bank verification and human review.</div></div><div class="quote">“AI can make lending more inclusive, fair and responsible.”<br/><small>— CreditWise</small></div>',unsafe_allow_html=True)

st.markdown('<div class="footer">CreditWise • AI for Responsible Lending<br/>Built with Python, Scikit-learn &amp; Streamlit</div>',unsafe_allow_html=True)
