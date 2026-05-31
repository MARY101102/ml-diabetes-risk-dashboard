# Step 3: Data Preprocessing and Feature Decision

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Purpose

This step prepares the raw CDC Diabetes Health Indicators dataset for model
training while keeping the original raw dataset unchanged.

## Raw Dataset Summary

| Item | Value |
|---|---:|
| Rows | 253,680 |
| Columns | 22 |
| Missing Values | 0 |
| Duplicate Rows Detected | 24,206 |

## Target Variable

Target column: `Diabetes_binary`

| Target Value | Meaning |
|---:|---|
| 0 | No diabetes |
| 1 | Prediabetes or diabetes |

## Version 1 Feature Selection

The first model version uses 18 input features.

### Selected Features

- HighBP
- HighChol
- CholCheck
- BMI
- Smoker
- Stroke
- HeartDiseaseorAttack
- PhysActivity
- Fruits
- Veggies
- HvyAlcoholConsump
- AnyHealthcare
- NoDocbcCost
- GenHlth
- MentHlth
- PhysHlth
- DiffWalk
- Age

### Excluded Features

- Sex
- Education
- Income

## Reason for Excluding Features

`Sex`, `Education`, and `Income` were excluded from Version 1 because they are
sensitive demographic or socioeconomic variables. The first public dashboard
version focuses on health and lifestyle indicators.

## Duplicate Row Decision

Duplicate rows were not removed in Version 1.

Reason:

This is a survey-based dataset. Multiple participants may provide identical
answers, so identical rows do not automatically prove data entry errors.

## Missing Value Decision

No missing values were detected, so no imputation was applied.

## Class Imbalance

The dataset is imbalanced.

| Class | Meaning | Percentage |
|---:|---|---:|
| 0 | No diabetes | 86.07% |
| 1 | Prediabetes or diabetes | 13.93% |

Because of this imbalance, stratified train/validation/test splitting was used.

## Dataset Split

| Split | Purpose |
|---|---|
| Train | Used for model training |
| Validation | Used for model comparison and later calibration |
| Test | Used only for final evaluation |

## Output Files

- `ml-training/data/processed/diabetes_modeling_v1.csv`
- `ml-training/data/processed/train_v1.csv`
- `ml-training/data/processed/validation_v1.csv`
- `ml-training/data/processed/test_v1.csv`
- `ml-training/data/processed/feature_schema_v1.json`
- `ml-training/results/preprocessing/split_summary_v1.json`
- `ml-training/results/preprocessing/preprocessing_report_v1.txt`

## Responsible Use Note

The positive class represents a dataset label for prediabetes or diabetes.
The final dashboard must not claim that the user has diabetes. It should show
a model-based estimated risk or model probability of the positive dataset class.