# Data Preprocessing & Feature Engineering

## 1. Dataset Preparation

The dataset contains 1,000 loan applications and 20 original columns.

`Applicant_ID` is dropped because it is an identifier and does not represent a meaningful predictive feature.

## 2. Missing Values

Missing values are handled according to feature type:

- **Numerical features:** mean imputation
- **Categorical features:** most-frequent-value imputation

This keeps all observations available for model training without removing rows because of missing values.

## 3. Encoding Categorical Features

`Education_Level` is label encoded. The categorical variables below are one-hot encoded using `drop="first"` and `handle_unknown="ignore"`:

- Employment_Status
- Marital_Status
- Loan_Purpose
- Property_Area
- Gender
- Employer_Category

The target `Loan_Approved` is also encoded as a binary label.

## 4. Train-Test Split

The data is split into:

- **80% training data**
- **20% test data**
- `random_state=42`

## 5. Feature Scaling

`StandardScaler` is fitted on the training data and then used to transform both the training and test sets. Scaling is particularly important for KNN because it relies on distances between observations.

## 6. Feature Engineering

Two nonlinear features are created:

- `DTI_Ratio_sq = DTI_Ratio ** 2`
- `Credit_Score_sq = Credit_Score ** 2`

The original `Credit_Score` and `DTI_Ratio` features are then dropped from the modeling matrix after the squared versions are created.

The notebook does **not** implement outlier removal or capping; outliers are visualized during exploratory analysis.

## 7. Why Feature Engineering?

The squared features allow the linear Logistic Regression model to capture nonlinear relationships in credit score and debt-to-income ratio. The results show that this improved Logistic Regression performance, especially recall and F1-score.
