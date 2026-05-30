"""
Download and store the CDC Diabetes Health Indicators dataset.

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Important:
This dataset is used for educational machine learning development only.
The target class represents a dataset label, not a medical diagnosis.
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_save_dataset() -> None:
    """Download the official UCI dataset and save a local raw CSV copy."""

    # Locate the project's raw data directory.
    current_file = Path(__file__).resolve()
    raw_data_dir = current_file.parent.parent / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_data_dir / "cdc_diabetes_health_indicators_raw.csv"

    print("Downloading CDC Diabetes Health Indicators dataset from UCI...")

    # Official UCI dataset ID: 891
    dataset = fetch_ucirepo(id=891)

    features = dataset.data.features
    targets = dataset.data.targets

    # Combine features and target into one DataFrame.
    full_dataset = pd.concat([targets, features], axis=1)

    # Save original retrieved data without cleaning or modifying it.
    full_dataset.to_csv(output_file, index=False)

    print("\nDataset downloaded successfully.")
    print(f"Saved file: {output_file}")
    print(f"Rows: {full_dataset.shape[0]}")
    print(f"Columns: {full_dataset.shape[1]}")
    print("\nFirst 5 rows:")
    print(full_dataset.head())

    print("\nColumn names:")
    print(full_dataset.columns.tolist())

    print("\nTarget column distribution:")
    print(full_dataset["Diabetes_binary"].value_counts(dropna=False))


if __name__ == "__main__":
    download_and_save_dataset()