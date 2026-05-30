"""
Step 2: Dataset Understanding and Initial Exploratory Data Analysis

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Purpose:
- Inspect the original raw dataset
- Understand the target variable
- Detect missing values and duplicate rows
- Check class balance
- Generate basic charts and summary files

Important:
This script does not clean, modify, or train on the dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

EDA_RESULTS_DIR = ML_TRAINING_DIR / "results" / "eda"
CHARTS_DIR = ML_TRAINING_DIR / "results" / "charts"

EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Diabetes_binary"


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the original raw dataset without modifying it."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file was not found at: {RAW_DATA_FILE}\n"
            "Run download_dataset.py from Step 1 first."
        )

    dataset = pd.read_csv(RAW_DATA_FILE)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found.\n"
            f"Available columns: {dataset.columns.tolist()}"
        )

    return dataset


def create_column_profile(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table describing each column."""

    profile_rows = []

    for column in dataset.columns:
        profile_rows.append(
            {
                "column_name": column,
                "data_type": str(dataset[column].dtype),
                "non_null_count": int(dataset[column].notna().sum()),
                "missing_count": int(dataset[column].isna().sum()),
                "unique_values": int(dataset[column].nunique()),
                "minimum_value": dataset[column].min(),
                "maximum_value": dataset[column].max(),
            }
        )

    return pd.DataFrame(profile_rows)


def create_target_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculate counts and percentages for the target variable."""

    counts = dataset[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    percentages = (
        dataset[TARGET_COLUMN]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        .mul(100)
        .round(2)
    )

    target_distribution = pd.DataFrame(
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

    return target_distribution


def create_target_chart(target_distribution: pd.DataFrame) -> None:
    """Create a bar chart showing target class distribution."""

    chart_file = CHARTS_DIR / "target_class_distribution.png"

    labels = target_distribution["meaning"]
    counts = target_distribution["count"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.title("Target Class Distribution")
    plt.xlabel("Dataset Target Class")
    plt.ylabel("Number of Participants")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    plt.close()

    print(f"Created chart: {chart_file}")


def create_bmi_histogram(dataset: pd.DataFrame) -> None:
    """Create a basic BMI distribution histogram."""

    if "BMI" not in dataset.columns:
        print("BMI column not found. Skipping BMI histogram.")
        return

    chart_file = CHARTS_DIR / "bmi_distribution.png"

    plt.figure(figsize=(8, 5))
    plt.hist(dataset["BMI"].dropna(), bins=30)
    plt.title("BMI Distribution in Dataset")
    plt.xlabel("BMI")
    plt.ylabel("Number of Participants")
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    plt.close()

    print(f"Created chart: {chart_file}")


def create_missing_values_chart(dataset: pd.DataFrame) -> None:
    """Create a chart showing missing values per column."""

    missing_values = dataset.isna().sum()
    chart_file = CHARTS_DIR / "missing_values_by_column.png"

    plt.figure(figsize=(12, 5))
    plt.bar(missing_values.index, missing_values.values)
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Value Count")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    plt.close()

    print(f"Created chart: {chart_file}")


def write_summary_report(
    dataset: pd.DataFrame,
    target_distribution: pd.DataFrame,
    duplicate_count: int,
) -> None:
    """Write the main dataset understanding report as a text file."""

    summary_file = EDA_RESULTS_DIR / "dataset_summary.txt"

    missing_total = int(dataset.isna().sum().sum())
    positive_class_percentage = float(
        target_distribution.loc[
            target_distribution["target_value"] == 1, "percentage"
        ].iloc[0]
    )

    imbalance_message = (
        "The target classes appear imbalanced."
        if positive_class_percentage < 40
        else "The target classes appear relatively balanced."
    )

    with open(summary_file, "w", encoding="utf-8") as report:
        report.write("DATASET UNDERSTANDING SUMMARY\n")
        report.write("=" * 60 + "\n\n")

        report.write("Project:\n")
        report.write(
            "Machine Learning-Based Diabetes Risk Estimation Dashboard\n\n"
        )

        report.write("Dataset Overview:\n")
        report.write(f"- Rows: {dataset.shape[0]}\n")
        report.write(f"- Total columns including target: {dataset.shape[1]}\n")
        report.write(f"- Input feature columns: {dataset.shape[1] - 1}\n")
        report.write(f"- Target column: {TARGET_COLUMN}\n\n")

        report.write("Target Meaning:\n")
        report.write("- 0 = No diabetes\n")
        report.write("- 1 = Prediabetes or diabetes\n\n")

        report.write("Target Distribution:\n")
        report.write(target_distribution.to_string(index=False))
        report.write("\n\n")

        report.write("Data Quality Checks:\n")
        report.write(f"- Total missing values detected: {missing_total}\n")
        report.write(f"- Duplicate rows detected: {duplicate_count}\n\n")

        report.write("Initial Observation:\n")
        report.write(f"- {imbalance_message}\n")
        report.write(
            "- Class imbalance must be considered during model evaluation.\n"
        )
        report.write(
            "- Accuracy alone will not be sufficient for selecting the model.\n\n"
        )

        report.write("Responsible Use Note:\n")
        report.write(
            "- This dataset supports educational risk classification only.\n"
        )
        report.write(
            "- The positive target class represents prediabetes or diabetes.\n"
        )
        report.write(
            "- The dashboard must not claim to diagnose a user.\n"
        )

    print(f"Created report: {summary_file}")


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main() -> None:
    """Run all Step 2 dataset understanding checks."""

    print("=" * 70)
    print("STEP 2: DATASET UNDERSTANDING AND INITIAL EDA")
    print("=" * 70)

    dataset = load_dataset()

    print("\n1. Dataset loaded successfully")
    print(f"Dataset path: {RAW_DATA_FILE}")

    print("\n2. Dataset shape")
    print(f"Rows: {dataset.shape[0]}")
    print(f"Columns: {dataset.shape[1]}")

    print("\n3. Column names")
    for column in dataset.columns:
        print(f"- {column}")

    print("\n4. First 5 rows")
    print(dataset.head())

    print("\n5. Data types")
    print(dataset.dtypes)

    print("\n6. Missing values")
    missing_values = dataset.isna().sum()
    print(missing_values)
    print(f"Total missing values: {int(missing_values.sum())}")

    print("\n7. Duplicate rows")
    duplicate_count = int(dataset.duplicated().sum())
    print(f"Duplicate rows detected: {duplicate_count}")

    print("\n8. Target variable distribution")
    target_distribution = create_target_distribution(dataset)
    print(target_distribution.to_string(index=False))

    print("\n9. Basic numerical statistics")
    print(dataset.describe().transpose())

    print("\n10. Saving column profile")
    column_profile = create_column_profile(dataset)
    column_profile_file = EDA_RESULTS_DIR / "column_profile.csv"
    column_profile.to_csv(column_profile_file, index=False)
    print(f"Created file: {column_profile_file}")

    print("\n11. Saving target distribution")
    target_distribution_file = EDA_RESULTS_DIR / "target_distribution.csv"
    target_distribution.to_csv(target_distribution_file, index=False)
    print(f"Created file: {target_distribution_file}")

    print("\n12. Creating charts")
    create_target_chart(target_distribution)
    create_bmi_histogram(dataset)
    create_missing_values_chart(dataset)

    print("\n13. Writing dataset summary report")
    write_summary_report(dataset, target_distribution, duplicate_count)

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files:")
    print("- ml-training/results/eda/dataset_summary.txt")
    print("- ml-training/results/eda/column_profile.csv")
    print("- ml-training/results/eda/target_distribution.csv")
    print("- ml-training/results/charts/target_class_distribution.png")
    print("- ml-training/results/charts/bmi_distribution.png")
    print("- ml-training/results/charts/missing_values_by_column.png")


if __name__ == "__main__":
    main()