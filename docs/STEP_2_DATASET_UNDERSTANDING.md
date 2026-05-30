# Step 2: Dataset Understanding and Initial EDA

## Project

Machine Learning-Based Diabetes Risk Estimation Dashboard

## Dataset Used

CDC Diabetes Health Indicators Dataset from the UCI Machine Learning Repository.

## Dataset Purpose

This dataset contains health statistics and lifestyle survey indicators used
to study diabetes-related classification outcomes.

## Dataset Shape

| Item | Value |
|---|---:|
| Number of Rows | Add your result here |
| Number of Input Features | Add your result here |
| Number of Total Columns Including Target | Add your result here |

## Target Variable

Target column: `Diabetes_binary`

| Target Value | Meaning |
|---:|---|
| 0 | No diabetes |
| 1 | Prediabetes or diabetes |

## Missing Value Check

| Check | Result |
|---|---:|
| Total Missing Values | Add your result here |

## Duplicate Row Check

| Check | Result |
|---|---:|
| Duplicate Rows Detected | Add your result here |

Important note:

Duplicate rows will not be removed automatically because multiple survey
participants may provide identical responses.

## Target Class Distribution

| Class | Meaning | Count | Percentage |
|---:|---|---:|---:|
| 0 | No diabetes | Add result | Add result |
| 1 | Prediabetes or diabetes | Add result | Add result |

## Initial Observation About Class Balance

Write whether the target is balanced or imbalanced after viewing the target
distribution chart.

## Features Requiring Careful Review

The dataset includes potentially sensitive variables such as:

- Sex
- Education
- Income

These variables will be reviewed carefully before deciding whether they should
be included in the final public-facing prediction form.

## Charts Generated

- Target class distribution chart
- BMI distribution chart
- Missing values chart

## Responsible Use Statement

This dataset will be used only for an educational machine learning portfolio
project. The final dashboard will estimate a model-based dataset risk class
and will not provide medical diagnosis, medical advice or treatment guidance.

Number of rows:  253680
Number of columns: 22 
Total missing values: 0 
Duplicate rows: 24206 
Class 0 count: 218334 
Class 1 count: 35346 