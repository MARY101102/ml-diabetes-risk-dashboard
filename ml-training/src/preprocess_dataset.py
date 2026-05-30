"""
Step 3: Data Preprocessing and Feature Decision

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Purpose:
- Select responsible Version 1 features
- Keep raw data unchanged
- Create processed dataset
- Create stratified train, validation and test splits
- Create feature schema for future frontend forms
- Generate preprocessing report

Important:
This script does not train a machine learning model.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

ML_TRAINING_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    ML_TRAINING_DIR
    / "data"
    / "raw"
    / "cdc_diabetes_health_indicators_raw.csv"
)

PROCESSED_DATA_DIR = ML_TRAINING_DIR / "data" / "processed"
RESULTS_DIR = ML_TRAINING_DIR / "results"
PREPROCESSING_RESULTS_DIR = RESULTS_DIR / "preprocessing"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Diabetes_binary"


# ---------------------------------------------------------------------
# Version 1 Feature Selection
# ---------------------------------------------------------------------

# These features are used in the first public-facing dashboard version.
# Sex, Education and Income are excluded in Version 1 because they are
# sensitive demographic/socioeconomic variables.
SELECTED_FEATURES_V1 = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Age",
]

EXCLUDED_FEATURES_V1 = [
    "Sex",
    "Education",
    "Income",
]

BINARY_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
]

RANGE_RULES = {
    "Diabetes_binary": {"min": 0, "max": 1},
    "BMI": {"min": 1, "max": 100},
    "GenHlth": {"min": 1, "max": 5},
    "MentHlth": {"min": 0, "max": 30},
    "PhysHlth": {"min": 0, "max": 30},
    "Age": {"min": 1, "max": 13},
}


# ---------------------------------------------------------------------
# Feature Schema for Future Web Form
# ---------------------------------------------------------------------

FEATURE_SCHEMA_V1 = {
    "project": "Machine Learning-Based Diabetes Risk Estimation Dashboard",
    "version": "diabetes_features_v1",
    "target": {
        "name": "Diabetes_binary",
        "meaning": {
            "0": "No diabetes",
            "1": "Prediabetes or diabetes",
        },
        "important_note": (
            "The positive class represents the dataset label for "
            "prediabetes or diabetes. It must not be displayed as a "
            "confirmed medical diagnosis."
        ),
    },
    "excluded_features_v1": {
        "features": EXCLUDED_FEATURES_V1,
        "reason": (
            "Excluded from the first public-facing version because they are "
            "sensitive demographic or socioeconomic variables."
        ),
    },
    "features": [
        {
            "name": "HighBP",
            "label": "High Blood Pressure",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "HighChol",
            "label": "High Cholesterol",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "CholCheck",
            "label": "Cholesterol Check in Last 5 Years",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "BMI",
            "label": "Body Mass Index",
            "type": "number",
            "allowed_range": {"min": 1, "max": 100},
        },
        {
            "name": "Smoker",
            "label": "Smoking History",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "Stroke",
            "label": "Stroke History",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "HeartDiseaseorAttack",
            "label": "Heart Disease or Heart Attack History",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "PhysActivity",
            "label": "Physical Activity in Last 30 Days",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "Fruits",
            "label": "Consumes Fruit Regularly",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "Veggies",
            "label": "Consumes Vegetables Regularly",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "HvyAlcoholConsump",
            "label": "Heavy Alcohol Consumption Indicator",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "AnyHealthcare",
            "label": "Has Healthcare Coverage",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "NoDocbcCost",
            "label": "Could Not See Doctor Because of Cost",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "GenHlth",
            "label": "General Health Rating",
            "type": "ordinal",
            "allowed_values": {
                "1": "Excellent",
                "2": "Very good",
                "3": "Good",
                "4": "Fair",
                "5": "Poor",
            },
        },
        {
            "name": "MentHlth",
            "label": "Poor Mental Health Days in Last 30 Days",
            "type": "number",
            "allowed_range": {"min": 0, "max": 30},
        },
        {
            "name": "PhysHlth",
            "label": "Poor Physical Health Days in Last 30 Days",
            "type": "number",
            "allowed_range": {"min": 0, "max": 30},
        },
        {
            "name": "DiffWalk",
            "label": "Difficulty Walking or Climbing Stairs",
            "type": "binary",
            "allowed_values": {"0": "No", "1": "Yes"},
        },
        {
            "name": "Age",
            "label": "Age Group",
            "type": "ordinal",
            "allowed_values": {
                "1": "18-24",
                "2": "25-29",
                "3": "30-34",
                "4": "35-39",
                "5": "40-44",
                "6": "45-49",
                "7": "50-54",
                "8": "55-59",
                "9": "60-64",
                "10": "65-69",
                "11": "70-74",
                "12": "75-79",
                "13": "80 or older",
            },
        },
    ],
}


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def load_raw_dataset() -> pd.DataFrame:
    """Load the original raw dataset."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_FILE}\n"
            "Run Step 1 download_dataset.py first."
        )

    return pd.read_csv(RAW_DATA_FILE)


def validate_required_columns(dataset: pd.DataFrame) -> None:
    """Ensure all required columns are available."""

    required_columns = [TARGET_COLUMN] + SELECTED_FEATURES_V1 + EXCLUDED_FEATURES_V1
    missing_columns = [col for col in required_columns if col not in dataset.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_binary_columns(dataset: pd.DataFrame) -> dict:
    """Check whether binary columns only contain 0 and 1."""

    validation_results = {}

    for column in [TARGET_COLUMN] + BINARY_FEATURES:
        unique_values = sorted(dataset[column].dropna().unique().tolist())
        is_valid = set(unique_values).issubset({0, 1})

        validation_results[column] = {
            "unique_values": unique_values,
            "is_valid_binary": is_valid,
        }

        if not is_valid:
            raise ValueError(
                f"Column {column} contains values other than 0 and 1: "
                f"{unique_values}"
            )

    return validation_results


def validate_range_rules(dataset: pd.DataFrame) -> dict:
    """Check whether selected numeric columns are inside expected ranges."""

    validation_results = {}

    for column, rule in RANGE_RULES.items():
        actual_min = dataset[column].min()
        actual_max = dataset[column].max()

        is_valid = actual_min >= rule["min"] and actual_max <= rule["max"]

        validation_results[column] = {
            "expected_min": rule["min"],
            "expected_max": rule["max"],
            "actual_min": float(actual_min),
            "actual_max": float(actual_max),
            "is_valid_range": bool(is_valid),
        }

        if not is_valid:
            raise ValueError(
                f"Column {column} is outside expected range. "
                f"Expected {rule}, got min={actual_min}, max={actual_max}"
            )

    return validation_results


def create_class_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create class distribution table for the target column."""

    counts = dataset[TARGET_COLUMN].value_counts().sort_index()
    percentages = (
        dataset[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    return pd.DataFrame(
        {
            "target_value": counts.index,
            "meaning": [
                "No diabetes" if value == 0 else "Prediabetes or diabetes"
                for value in counts.index
            ],
            "count": counts.values,
            "percentage": percentages.values,
        }
    )


def save_json(data: dict, output_file: Path) -> None:
    """Save dictionary as readable JSON."""

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def write_preprocessing_report(
    raw_dataset: pd.DataFrame,
    processed_dataset: pd.DataFrame,
    duplicate_count: int,
    missing_count: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    binary_validation: dict,
    range_validation: dict,
) -> None:
    """Create a readable preprocessing report."""

    report_file = PREPROCESSING_RESULTS_DIR / "preprocessing_report_v1.txt"

    raw_distribution = create_class_distribution(raw_dataset)
    processed_distribution = create_class_distribution(processed_dataset)
    train_distribution = create_class_distribution(train_df)
    validation_distribution = create_class_distribution(validation_df)
    test_distribution = create_class_distribution(test_df)

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("STEP 3 PREPROCESSING REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write("Project:\n")
        report.write("Machine Learning-Based Diabetes Risk Estimation Dashboard\n\n")

        report.write("Raw Dataset Summary:\n")
        report.write(f"- Raw rows: {raw_dataset.shape[0]}\n")
        report.write(f"- Raw columns: {raw_dataset.shape[1]}\n")
        report.write(f"- Total missing values: {missing_count}\n")
        report.write(f"- Duplicate rows detected: {duplicate_count}\n\n")

        report.write("Target Variable:\n")
        report.write("- Diabetes_binary\n")
        report.write("- 0 = No diabetes\n")
        report.write("- 1 = Prediabetes or diabetes\n\n")

        report.write("Important Responsible Use Note:\n")
        report.write(
            "- The positive class should not be displayed as a confirmed "
            "medical diagnosis.\n"
        )
        report.write(
            "- The final dashboard should say estimated diabetes-related "
            "risk or model probability of positive dataset class.\n\n"
        )

        report.write("Feature Decision for Version 1:\n")
        report.write(f"- Selected input features: {len(SELECTED_FEATURES_V1)}\n")
        for feature in SELECTED_FEATURES_V1:
            report.write(f"  - {feature}\n")

        report.write("\nExcluded Features for Version 1:\n")
        for feature in EXCLUDED_FEATURES_V1:
            report.write(f"  - {feature}\n")

        report.write(
            "\nReason for exclusion:\n"
            "- Sex, Education and Income are excluded from the first "
            "public-facing version because they are sensitive demographic "
            "or socioeconomic variables.\n\n"
        )

        report.write("Duplicate Row Decision:\n")
        report.write(
            "- Duplicate rows are kept in Version 1 because this is a "
            "survey-based dataset and multiple participants may provide "
            "identical answers.\n\n"
        )

        report.write("Missing Value Decision:\n")
        report.write(
            "- No missing values were detected, so no imputation was applied.\n\n"
        )

        report.write("Class Imbalance Observation:\n")
        report.write(raw_distribution.to_string(index=False))
        report.write("\n\n")
        report.write(
            "- The dataset is imbalanced because the positive class is much "
            "smaller than the negative class.\n"
        )
        report.write(
            "- Stratified train/validation/test splitting was used to preserve "
            "class proportions.\n"
        )
        report.write(
            "- Oversampling or undersampling was not applied in Step 3.\n\n"
        )

        report.write("Processed Dataset Summary:\n")
        report.write(f"- Processed rows: {processed_dataset.shape[0]}\n")
        report.write(f"- Processed columns: {processed_dataset.shape[1]}\n")
        report.write("- Processed columns include target + selected features.\n\n")

        report.write("Processed Target Distribution:\n")
        report.write(processed_distribution.to_string(index=False))
        report.write("\n\n")

        report.write("Train/Validation/Test Split:\n")
        report.write(f"- Train rows: {train_df.shape[0]}\n")
        report.write(f"- Validation rows: {validation_df.shape[0]}\n")
        report.write(f"- Test rows: {test_df.shape[0]}\n\n")

        report.write("Train Target Distribution:\n")
        report.write(train_distribution.to_string(index=False))
        report.write("\n\n")

        report.write("Validation Target Distribution:\n")
        report.write(validation_distribution.to_string(index=False))
        report.write("\n\n")

        report.write("Test Target Distribution:\n")
        report.write(test_distribution.to_string(index=False))
        report.write("\n\n")

        report.write("Binary Column Validation:\n")
        report.write(json.dumps(binary_validation, indent=4))
        report.write("\n\n")

        report.write("Range Validation:\n")
        report.write(json.dumps(range_validation, indent=4))
        report.write("\n\n")

        report.write("Output Files Created:\n")
        report.write("- ml-training/data/processed/diabetes_modeling_v1.csv\n")
        report.write("- ml-training/data/processed/train_v1.csv\n")
        report.write("- ml-training/data/processed/validation_v1.csv\n")
        report.write("- ml-training/data/processed/test_v1.csv\n")
        report.write("- ml-training/data/processed/feature_schema_v1.json\n")
        report.write("- ml-training/results/preprocessing/preprocessing_report_v1.txt\n")

    print(f"Created preprocessing report: {report_file}")


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main() -> None:
    """Run Step 3 preprocessing workflow."""

    print("=" * 70)
    print("STEP 3: DATA PREPROCESSING AND FEATURE DECISION")
    print("=" * 70)

    print("\n1. Loading raw dataset")
    raw_dataset = load_raw_dataset()
    print(f"Raw dataset shape: {raw_dataset.shape}")

    print("\n2. Validating required columns")
    validate_required_columns(raw_dataset)
    print("All required columns are present.")

    print("\n3. Checking missing values")
    missing_count = int(raw_dataset.isna().sum().sum())
    print(f"Total missing values: {missing_count}")

    print("\n4. Checking duplicate rows")
    duplicate_count = int(raw_dataset.duplicated().sum())
    print(f"Duplicate rows detected: {duplicate_count}")
    print("Decision: duplicates are kept for Version 1.")

    print("\n5. Validating binary columns")
    binary_validation = validate_binary_columns(raw_dataset)
    print("Binary column validation passed.")

    print("\n6. Validating numeric and ordinal ranges")
    range_validation = validate_range_rules(raw_dataset)
    print("Range validation passed.")

    print("\n7. Creating processed Version 1 dataset")
    processed_columns = [TARGET_COLUMN] + SELECTED_FEATURES_V1
    processed_dataset = raw_dataset[processed_columns].copy()

    processed_dataset_file = PROCESSED_DATA_DIR / "diabetes_modeling_v1.csv"
    processed_dataset.to_csv(processed_dataset_file, index=False)

    print(f"Processed dataset saved: {processed_dataset_file}")
    print(f"Processed dataset shape: {processed_dataset.shape}")

    print("\n8. Creating stratified train, validation and test splits")

    train_df, temp_df = train_test_split(
        processed_dataset,
        test_size=0.30,
        random_state=42,
        stratify=processed_dataset[TARGET_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[TARGET_COLUMN],
    )

    train_file = PROCESSED_DATA_DIR / "train_v1.csv"
    validation_file = PROCESSED_DATA_DIR / "validation_v1.csv"
    test_file = PROCESSED_DATA_DIR / "test_v1.csv"

    train_df.to_csv(train_file, index=False)
    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Train file saved: {train_file}")
    print(f"Validation file saved: {validation_file}")
    print(f"Test file saved: {test_file}")

    print("\nSplit sizes:")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {validation_df.shape}")
    print(f"Test: {test_df.shape}")

    print("\n9. Saving feature schema for future frontend form")
    feature_schema_file = PROCESSED_DATA_DIR / "feature_schema_v1.json"
    save_json(FEATURE_SCHEMA_V1, feature_schema_file)
    print(f"Feature schema saved: {feature_schema_file}")

    print("\n10. Saving split distribution summary")
    split_summary = {
        "target_column": TARGET_COLUMN,
        "selected_features_v1": SELECTED_FEATURES_V1,
        "excluded_features_v1": EXCLUDED_FEATURES_V1,
        "raw_shape": list(raw_dataset.shape),
        "processed_shape": list(processed_dataset.shape),
        "train_shape": list(train_df.shape),
        "validation_shape": list(validation_df.shape),
        "test_shape": list(test_df.shape),
        "duplicate_rows_kept": duplicate_count,
        "missing_values": missing_count,
        "raw_target_distribution": create_class_distribution(raw_dataset).to_dict(
            orient="records"
        ),
        "train_target_distribution": create_class_distribution(train_df).to_dict(
            orient="records"
        ),
        "validation_target_distribution": create_class_distribution(
            validation_df
        ).to_dict(orient="records"),
        "test_target_distribution": create_class_distribution(test_df).to_dict(
            orient="records"
        ),
    }

    split_summary_file = PREPROCESSING_RESULTS_DIR / "split_summary_v1.json"
    save_json(split_summary, split_summary_file)
    print(f"Split summary saved: {split_summary_file}")

    print("\n11. Writing preprocessing report")
    write_preprocessing_report(
        raw_dataset=raw_dataset,
        processed_dataset=processed_dataset,
        duplicate_count=duplicate_count,
        missing_count=missing_count,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        binary_validation=binary_validation,
        range_validation=range_validation,
    )

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")
    print("- ml-training/data/processed/diabetes_modeling_v1.csv")
    print("- ml-training/data/processed/train_v1.csv")
    print("- ml-training/data/processed/validation_v1.csv")
    print("- ml-training/data/processed/test_v1.csv")
    print("- ml-training/data/processed/feature_schema_v1.json")
    print("- ml-training/results/preprocessing/split_summary_v1.json")
    print("- ml-training/results/preprocessing/preprocessing_report_v1.txt")


if __name__ == "__main__":
    main()