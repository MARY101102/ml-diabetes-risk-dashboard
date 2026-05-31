# Step 4: Baseline Model Training

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Purpose

This step trains and compares baseline machine learning classification models
using the processed Version 1 dataset.

## Dataset Files Used

| File | Purpose |
|---|---|
| `train_v1.csv` | Used to train models |
| `validation_v1.csv` | Used to compare models |
| `test_v1.csv` | Not used in this step |

## Important Rule

The test dataset was not used in this step. It is reserved for final model
evaluation only.

## Models Trained

| Model | Reason |
|---|---|
| Logistic Regression | Simple baseline model |
| Decision Tree | Interpretable tree-based baseline |
| Random Forest | Strong ensemble baseline |

## Metrics Used

| Metric | Reason |
|---|---|
| Accuracy | Measures overall correctness |
| Precision | Measures correctness of positive predictions |
| Recall | Measures how many actual positive cases were found |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures class separation ability |

## Class Imbalance Note

The dataset is imbalanced. The positive class is much smaller than the negative
class. Therefore, model selection should not be based on accuracy alone.

Main selection focus:

1. F1-score
2. Recall
3. ROC-AUC

## Validation Results

Fill this table after opening:

`ml-training/results/model_training/baseline_model_metrics.csv`

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7281 | 0.3073 | 0.7586 | 0.4374 | 0.8191 |
| Decision Tree | 0.7035 | 0.2905 | 0.7820 | 0.4236 | 0.8139 |
| Random Forest | 0.7397 | 0.3153 | 0.7412 | 0.4424 | 0.8210 |

## Best Baseline Model
Random Forest

`ml-training/results/model_training/best_baseline_model_summary.json`

## Charts Generated

- Logistic Regression confusion matrix
- Logistic Regression ROC curve
- Decision Tree confusion matrix
- Decision Tree ROC curve
- Random Forest confusion matrix
- Random Forest ROC curve
- Baseline model metrics comparison chart

## Responsible Use Note

The model output should be displayed as an estimated diabetes-related risk or
model probability of the positive dataset class. It must not be displayed as a
confirmed medical diagnosis.