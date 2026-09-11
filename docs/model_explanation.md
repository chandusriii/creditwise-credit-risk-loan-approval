# Model Explanations

## 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a binary outcome. It is a strong baseline for this problem because it is relatively simple and interpretable.

In this project, feature engineering improved Logistic Regression from **86.5% to 88.0% accuracy**, while recall increased from **77.05% to 83.61%**.

It became the best overall model in the final comparison.

## 2. K-Nearest Neighbors (KNN)

KNN predicts a new observation based on the labels of nearby training observations. The project uses `n_neighbors=5`.

Because KNN is distance-based, feature scaling is important. KNN produced lower performance than Logistic Regression and Gaussian Naive Bayes on this dataset.

## 3. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes conditional independence between features and models continuous features using Gaussian distributions.

After feature engineering, Gaussian Naive Bayes achieved the **highest precision (81.13%)** among the final models, although its recall and F1-score were below the final Logistic Regression model.

## Why Compare These Three Models?

The three models represent different modeling approaches:

| Model | Approach |
|---|---|
| Logistic Regression | Linear probabilistic classifier |
| KNN | Distance-based classifier |
| Gaussian Naive Bayes | Probabilistic classifier based on feature independence |

Comparing them provides a useful baseline across different assumptions and learning strategies.

## Final Model Choice

Feature-engineered **Logistic Regression** is selected as the final model because it provides the strongest overall balance of accuracy, recall, precision, and F1-score in the notebook results.
