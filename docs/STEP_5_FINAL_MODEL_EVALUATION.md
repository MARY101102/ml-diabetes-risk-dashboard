# Step 5: Final Model Evaluation and Model Selection

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Purpose

This step evaluates the selected baseline model on the untouched test dataset.

## Final Selected Model

Random Forest

## Why Random Forest Was Selected

Random Forest was selected from the baseline validation results because it achieved the best F1-score and ROC-AUC among the tested baseline models.

## Dataset Used

| File                | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `train_v1.csv`      | Used earlier for model training             |
| `validation_v1.csv` | Used earlier for model comparison           |
| `test_v1.csv`       | Used in this step for final evaluation only |

## Important Rule

The test dataset was not used during training or model selection. It was used only once for final performance evaluation.

## Final Test Metrics

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.7460 |
| Precision | 0.3218 |
| Recall    | 0.7429 |
| F1-score  | 0.4490 |
| ROC-AUC   | 0.8234 |

## Confusion Matrix

| Confusion Matrix Value |  Count |
| ---------------------- | -----: |
| True Negative          | 24,447 |
| False Positive         |  8,303 |
| False Negative         |  1,363 |
| True Positive          |  3,939 |

## Charts Generated

* Final confusion matrix
* Final ROC curve
* Final precision-recall curve

## Metric Interpretation

Because the dataset is imbalanced, model quality should not be judged by accuracy alone. Recall, F1-score and ROC-AUC are more meaningful for this risk-estimation project.

The final Random Forest model achieved an ROC-AUC score of 0.8234, which shows that the model has good ability to separate the positive class from the negative class.

The recall score of 0.7429 shows that the model correctly identified a high number of actual prediabetes or diabetes records. This is useful for a diabetes risk estimation dashboard because missing possible risk cases can be more serious than producing some false positives.

The precision score of 0.3218 shows that some records predicted as positive were actually negative. However, for a health risk screening project, higher recall is important because the goal is to identify users who may need further checking.

## Responsible Use Note

The model predicts the dataset label for prediabetes or diabetes. The final dashboard must not claim that the user has diabetes. It should display an estimated diabetes-related risk or model probability of the positive dataset class.

## Conclusion

The Random Forest model was selected as the final Version 1 model for the diabetes risk estimation dashboard. It was evaluated using the untouched test dataset and achieved an accuracy of 0.7460, recall of 0.7429, F1-score of 0.4490, and ROC-AUC of 0.8234. These results show that the model is suitable for educational diabetes-related risk estimation, but it should not be used as a medical diagnostic system.
