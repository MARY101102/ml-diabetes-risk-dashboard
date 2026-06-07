# Step 6: Probability Calibration

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Purpose

This step calibrates the final Random Forest model's probability output before
using it in the dashboard.

## Why Calibration Is Needed

The final model can output probabilities, but raw model probabilities may not
always be reliable enough to display as confidence-like values.

Therefore, calibration is used before showing model probability in the user
interface.

## Data Usage Rule

| Dataset | Usage |
|---|---|
| `validation_v1.csv` | Used to fit and select calibration method |
| `test_v1.csv` | Used only for reporting after calibration |
| `train_v1.csv` | Not used directly in this step |

## Calibration Methods Compared

- Uncalibrated probability
- Sigmoid calibration
- Isotonic calibration

## Selection Rule

The calibration method with the lowest validation Brier score was selected.

## Selected Calibration Method

Add selected method here after running:

`ml-training/src/calibrate_model.py`

## Validation Calibration Metrics

Fill this table after opening:

`ml-training/results/calibration/calibration_metrics_v1.json`

| Method | Accuracy | Precision | Recall | F1-score | ROC-AUC | Brier Score | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uncalibrated | Add | Add | Add | Add | Add | Add | Add |
| Sigmoid | Add | Add | Add | Add | Add | Add | Add |
| Isotonic | Add | Add | Add | Add | Add | Add | Add |

## Output Files

- `ml-training/models/calibrated/diabetes_risk_calibrated_model_v1.joblib`
- `ml-training/results/calibration/calibration_metrics_v1.json`
- `ml-training/results/calibration/calibration_report_v1.txt`
- `ml-training/results/calibration/charts/validation_calibration_curve.png`
- `ml-training/results/calibration/charts/test_calibration_curve_reporting_only.png`
- `ml-training/results/calibration/charts/validation_brier_score_comparison.png`

## Dashboard Display Rule

The dashboard should display calibrated probability as:

`Model Probability of Positive Dataset Class`

or:

`Estimated Diabetes-Related Risk`

It must not display the result as a confirmed diagnosis.

## Responsible Use Note

The positive class represents prediabetes or diabetes in the dataset. The final
application is an educational machine learning project and must not provide
medical diagnosis, medical advice or treatment recommendations.