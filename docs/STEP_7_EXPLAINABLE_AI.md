# Step 7: Explainable AI

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Purpose

This step adds explainability to the diabetes risk estimation model.

The goal is to show which features influence the model globally and which
features influenced a sample prediction.

## Method Used

SHAP-based feature explanation was used to explain the tree-based Random Forest
model.

## Global Explanation

Global explanation shows which features are most influential across a sample of
validation records.

Output file:

`ml-training/results/explainability/global_feature_importance_v1.csv`

Chart:

`ml-training/results/explainability/charts/global_feature_importance.png`

## Local Explanation

Local explanation shows which features increased or reduced the model's
estimated probability for one sample prediction.

Output file:

`ml-training/results/explainability/sample_prediction_explanation_v1.json`

Chart:

`ml-training/results/explainability/charts/sample_prediction_explanation.png`

## Dashboard Use

The dashboard can show:

- Estimated risk level
- Calibrated model probability
- Top factors increasing estimated risk
- Top factors reducing estimated risk

## Important Wording

Use:

`This feature increased the model's estimated probability for the positive dataset class.`

Do not use:

`This feature caused diabetes.`

## Responsible Use Note

The explanations describe model behavior only. They do not provide medical
diagnosis, medical advice, or treatment recommendations.