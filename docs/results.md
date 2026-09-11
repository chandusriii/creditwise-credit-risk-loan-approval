# Results & Model Evaluation

## Baseline Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.50% | 78.33% | 77.05% | 77.69% |
| KNN | 76.00% | 62.75% | 52.46% | 57.14% |
| Gaussian Naive Bayes | 86.50% | 80.36% | 73.77% | 76.92% |

## Results After Feature Engineering

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **88.00%** | 78.46% | **83.61%** | **80.95%** |
| KNN | 78.50% | 67.31% | 57.38% | 61.95% |
| Gaussian Naive Bayes | 86.00% | **81.13%** | 70.49% | 75.44% |

## Final Model

The final model is **feature-engineered Logistic Regression**.

### Final performance

- **Accuracy:** 88.00%
- **Precision:** 78.46%
- **Recall:** 83.61%
- **F1-score:** 80.95%

### Confusion Matrix

The final Logistic Regression confusion matrix is:

```text
[[125, 14],
 [ 10, 51]]
```

Using the usual binary classification layout, this corresponds to:

- True Negatives: 125
- False Positives: 14
- False Negatives: 10
- True Positives: 51

## Impact of Feature Engineering

For Logistic Regression:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Accuracy | 86.50% | 88.00% | +1.50 percentage points |
| Recall | 77.05% | 83.61% | +6.56 percentage points |
| F1-Score | 77.69% | 80.95% | +3.26 percentage points |

The biggest improvement is in recall, indicating that the engineered features helped the model identify more of the approved-class observations in the test set.

## Key Findings

1. Logistic Regression achieved the best overall final performance.
2. Feature engineering improved Logistic Regression accuracy, recall, and F1-score.
3. Gaussian Naive Bayes achieved the highest final precision at 81.13%.
4. KNN performed below the other two approaches on the reported metrics.
5. Credit Score and DTI Ratio showed strong relationships with the target during exploratory analysis.

## Interpretation

For this dataset, the feature-engineered Logistic Regression model provides the strongest overall balance of the reported evaluation metrics. Model selection is based on the notebook's test-set results and should not be interpreted as proof of production performance without further validation.
