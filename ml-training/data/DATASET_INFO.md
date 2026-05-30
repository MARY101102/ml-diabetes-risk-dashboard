# Dataset Information

## Dataset Name

CDC Diabetes Health Indicators Dataset

## Source

UCI Machine Learning Repository

## Dataset Purpose

The dataset contains health statistics and lifestyle survey indicators used
to study relationships with diabetes-related classification labels.

## Project Usage

This dataset will be used to build a machine learning classification model
for an educational diabetes-related risk estimation dashboard.

## Target Variable

`Diabetes_binary`

| Target Value | Meaning |
|---|---|
| 0 | No diabetes |
| 1 | Prediabetes or diabetes |

## Important Limitation

The positive class represents prediabetes or diabetes according to the
dataset definition. Therefore, this project must not display results as a
confirmed medical diagnosis.

## Raw Data Policy

The original downloaded CSV file in `data/raw/` must never be manually edited.
All cleaned or transformed datasets will be stored separately in
`data/processed/`.