# Problem Statement

## 1. Business Background

SecureTrust Bank is a mid-sized financial company that offers **personal and home loans** to customers across **urban and rural regions of India**. Every day, hundreds of customers submit loan applications through online and branch channels.

Currently, the bank follows a **manual verification process**. Loan officers review income proofs, employment details, credit history, and other applicant documents before deciding whether an application should be approved or rejected.

As application volumes grow, this manual approach becomes difficult to manage efficiently.

## 2. What Banks Are Facing Today

The current loan-review process creates several business problems.

### Time-consuming verification

Loan officers must manually examine multiple pieces of information for each application. Reviewing hundreds of applications every day makes the process slow and difficult to scale.

### Inconsistent decisions

Manual decisions can vary from one loan officer to another. Similar applicants may not always receive the same outcome when their information is interpreted differently.

### Human bias

The problem statement identifies the existing process as **biased**. Human judgment during manual assessment can influence lending decisions and make it difficult to maintain consistent treatment across applications.

### Good customers can be rejected

A customer who may be suitable for a loan can still be rejected. This creates a direct **loss of potential business** for the bank.

### High-risk customers can be approved

A customer who presents higher lending risk can sometimes be approved. This can expose the bank to **financial losses**.

Therefore, the bank faces two major business risks:

> **Rejecting good customers can lead to lost business, while approving high-risk customers can lead to financial losses.**

## 3. What the Bank Needs

To address these challenges, SecureTrust Bank wants an **intelligent loan approval system powered by Machine Learning**.

The system should analyze historical loan application records, learn patterns from previous customer data, and predict whether a new application should be:

- **Approved (`1`)**
- **Rejected (`0`)**

The prediction should be available **before final human verification** so that the model supports the loan officer's review process.

## 4. What We Need to Build

As the Machine Learning Engineer, the task is to design and develop a supervised machine learning system that can:

1. Analyze historical loan application data.
2. Learn patterns associated with previous approval and rejection decisions.
3. Use applicant financial, employment, credit, demographic, and loan information as model inputs.
4. Predict whether a new loan application is likely to be approved or rejected.
5. Provide a fast and consistent prediction to assist loan officers.
6. Help reduce unnecessary rejection of suitable customers and reduce the risk of approving high-risk customers.

## 5. Machine Learning Problem

This project is a **binary classification problem**.

### Target Variable

```text
Loan_Approved
1 → Approved
0 → Rejected
```

The model learns from historical applications and predicts the approval outcome for a new applicant.

## 6. Applicant Information Used by the System

Each row in the dataset represents a **loan applicant** and contains personal, financial, employment, and credit information.

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant ID |
| `Applicant_Income` | Monthly income of applicant |
| `Coapplicant_Income` | Monthly income of co-applicant |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Age` | Applicant age |
| `Marital_Status` | Married / Single |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of already running loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Loan amount requested |
| `Loan_Term` | Loan duration in months |
| `Loan_Purpose` | Home / Education / Personal / Business |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Education_Level` | Graduate / Postgraduate / Undergraduate |
| `Gender` | Male / Female |
| `Employer_Category` | Govt / Private / Self |
| `Loan_Approved` | Target: `1 = Approved`, `0 = Rejected` |

## 7. Expected System Outcome

The final solution should act as an **intelligent decision-support system** for the bank's loan officers.

A typical flow is:

```text
Applicant submits loan application
              ↓
      Applicant information
              ↓
      Machine Learning Model
              ↓
     Approved / Rejected prediction
              ↓
       Final human verification
```

The purpose is not simply to maximize an accuracy score. The system should provide a useful, consistent prediction that supports faster loan screening while considering the business consequences of both incorrect approvals and incorrect rejections.

## 8. Project Goal

> **Build a Machine Learning-powered loan approval prediction system that learns from historical customer applications and provides a fast, consistent, and explainable Approved/Rejected prediction to support SecureTrust Bank's loan officers before final human verification.**

## 9. Scope of This Project

The project focuses on building and evaluating supervised machine learning classification models using the provided historical loan application dataset. The model output is intended to support the bank's decision-making process; the final lending decision remains subject to human verification.
