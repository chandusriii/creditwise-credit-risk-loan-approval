# Credit Risk & Loan Approval Prediction

> A supervised machine learning project for predicting whether a loan application is approved or rejected using applicant financial, demographic, employment, and credit-related features.

## 🚀 Project Overview

This project builds an end-to-end **binary classification** workflow to predict loan approval outcomes from historical application data.

The workflow covers:

**Data Understanding → EDA → Preprocessing → Encoding → Correlation Analysis → Feature Engineering → Model Training → Evaluation**

The main objective is to compare multiple classification algorithms and identify the model with the strongest overall performance.

## 🎯 Problem Statement

Loan approval can depend on factors such as income, employment, credit score, debt-to-income ratio, loan details, and applicant characteristics. This project explores how supervised machine learning can learn patterns from historical applications and generate a consistent approval prediction.

### Target Variable

- `1` → Loan Approved
- `0` → Loan Rejected

## 📊 Dataset

The dataset contains **1,000 loan applications** with **20 original columns**. Each row represents a loan applicant with personal, financial, employment, and credit-related information.

### Key Features

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier |
| `Applicant_Income` | Applicant income |
| `Coapplicant_Income` | Co-applicant income |
| `Employment_Status` | Employment type |
| `Age` | Applicant age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Existing loan count |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Collateral value |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property location |
| `Education_Level` | Education level |
| `Gender` | Applicant gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target: 1 = Approved, 0 = Rejected |

## 🔍 Exploratory Data Analysis

The analysis examines class distribution, income, credit score, DTI ratio, savings, and relationships between numerical features and loan approval.

### Loan Approval Distribution

- **Rejected:** 722 applications — 72.2%
- **Approved:** 278 applications — 27.8%

### Strongest Numerical Associations

| Feature | Correlation with Loan Approval |
|---|---:|
| `Credit_Score` | +0.451 |
| `DTI_Ratio` | -0.445 |
| `Applicant_Income` | +0.120 |
| `Loan_Amount` | -0.126 |
| `Loan_Term` | -0.087 |

`Credit_Score` had the strongest positive numerical association with loan approval, while `DTI_Ratio` had the strongest negative association among the evaluated numerical features.

## 🧹 Data Preprocessing

### Missing Values

- Numerical features → **Mean imputation**
- Categorical features → **Most-frequent imputation**

### Identifier Removal

`Applicant_ID` was removed because it is an identifier rather than a predictive feature.

### Categorical Encoding

- Label Encoding for `Education_Level` and the target variable
- One-Hot Encoding for nominal categorical variables including employment status, marital status, loan purpose, property area, gender, and employer category

### Feature Scaling

`StandardScaler` was applied before model training.

### Train-Test Split

The data was split into:

- **80% training data**
- **20% testing data**

using `random_state=42`.

## ⚙️ Feature Engineering

Two nonlinear features were created:

```text
DTI_Ratio_sq = DTI_Ratio²
Credit_Score_sq = Credit_Score²
```

For the feature-engineered model comparison, the original `Credit_Score` and `DTI_Ratio` columns were dropped from the final feature matrix.

## 🤖 Models Compared

### Logistic Regression

Used as a primary linear classification model and final model candidate.

### K-Nearest Neighbors

Evaluated with:

```text
n_neighbors = 5
```

### Gaussian Naive Bayes

Used as a probabilistic classification model for comparison.

## 📈 Model Performance

### Before Feature Engineering

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | **86.50%** | 78.33% | 77.05% | 77.69% |
| KNN | 76.00% | 62.75% | 52.46% | 57.14% |
| Gaussian Naive Bayes | **86.50%** | **80.36%** | 73.77% | 76.92% |

### After Feature Engineering

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | **88.00%** | 78.46% | **83.61%** | **80.95%** |
| KNN | 78.50% | 67.31% | 57.38% | 61.95% |
| Gaussian Naive Bayes | 86.00% | **81.13%** | 70.49% | 75.44% |

## 🏆 Final Model

### Feature-Engineered Logistic Regression

The feature-engineered **Logistic Regression** model achieved the strongest overall balance across the evaluated metrics.

| Metric | Score |
|---|---:|
| Accuracy | **88.00%** |
| Precision | **78.46%** |
| Recall | **83.61%** |
| F1 Score | **80.95%** |

Gaussian Naive Bayes achieved the highest precision at **81.13%**, while Logistic Regression delivered the strongest overall balance of accuracy, recall, and F1 score.

## 📌 Feature Engineering Impact

For Logistic Regression:

```text
Accuracy: 86.50% → 88.00%
Recall:   77.05% → 83.61%
F1 Score: 77.69% → 80.95%
```

The engineered squared features improved the overall Logistic Regression results.

## 🧠 Key Insights

- `Credit_Score` showed the strongest positive numerical relationship with loan approval.
- `DTI_Ratio` showed the strongest negative numerical relationship with loan approval among the evaluated numerical features.
- Feature engineering improved Logistic Regression performance.
- Logistic Regression achieved the best overall results among the three tested models.
- Gaussian Naive Bayes achieved the highest precision after feature engineering.

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

## 📁 Project Structure

```text
creditwise-credit-risk-loan-approval/
│
├── data/
│   └── loan_approval_data.csv
│
├── notebooks/
│   └── credit_wise.ipynb
│
├── README.md
└── requirements.txt
```

## ▶️ Run the Project

### Clone the Repository

```bash
git clone https://github.com/chandusriii/creditwise-credit-risk-loan-approval.git
cd creditwise-credit-risk-loan-approval
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `notebooks/credit_wise.ipynb` and run the notebook cells sequentially.

## 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## 🔮 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Class-imbalance handling
- ROC-AUC and Precision-Recall analysis
- Probability-based risk scoring
- Explainable AI techniques
- API deployment
- Interactive prediction dashboard

## ⚠️ Disclaimer

This project is for educational and machine learning demonstration purposes. Real-world lending decisions require appropriate financial, regulatory, fairness, privacy, security, and human-review processes.

## ⭐ Project

**Credit Risk & Loan Approval Prediction**

Built with Python and Scikit-learn to demonstrate supervised classification, data preprocessing, feature engineering, model comparison, and evaluation.

---

**Repository:** https://github.com/chandusriii/creditwise-credit-risk-loan-approval