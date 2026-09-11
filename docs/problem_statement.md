# Problem Statement

## Background

Loan approval is a binary decision influenced by an applicant's financial profile, credit history, employment information, and loan characteristics. This project uses historical loan application data to build a machine learning model that predicts whether a loan application is likely to be approved or rejected.

## Objective

Build and evaluate supervised binary classification models for predicting the `Loan_Approved` target variable.

The project compares:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

The models are evaluated using accuracy, precision, recall, and F1-score.

## Target Variable

`Loan_Approved` is the binary target:

- `1` — Loan approved
- `0` — Loan rejected

The dataset contains 1,000 applications. The target distribution is 722 rejected applications (72.2%) and 278 approved applications (27.8%), so metrics beyond accuracy are important when comparing models.

## Input Features

The project uses applicant, financial, employment, and loan-related attributes such as:

- Applicant income
- Coapplicant income
- Employment status
- Age
- Marital status
- Dependents
- Credit score
- Existing loans
- DTI ratio
- Savings
- Collateral value
- Loan amount
- Loan term
- Loan purpose
- Property area
- Education level
- Gender
- Employer category

`Applicant_ID` is removed before modeling because it is an identifier rather than a predictive feature.

## Success Criteria

The goal is to identify a model that provides a strong overall balance between precision, recall, F1-score, and accuracy, while examining whether feature engineering improves predictive performance.
