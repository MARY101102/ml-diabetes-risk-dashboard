"""
Step 6: Probability Calibration

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Purpose:
- Calibrate the final Random Forest model's probability output
- Compare uncalibrated, sigmoid-calibrated and isotonic-calibrated probabilities
- Select calibration method using validation data only
- Save dashboard-ready calibrated model package

Important:
- Calibration is fitted using validation_v1.csv.
- Test data is used only for reporting after calibration.
- Do not tune calibration based on test results.
"""

from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

ML_TRAINING_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = ML_TRAINING_DIR / "data" / "processed"

FINAL_MODEL_FILE = (
    ML_TRAINING_DIR
    / "models"
    / "final"
    / "diabetes_risk_model_v1.joblib"
)

CALIBRATED_MODELS_DIR = ML_TRAINING_DIR / "models" / "calibrated"

CALIBRATION_RESULTS_DIR = ML_TRAINING_DIR / "results" / "calibration"
CALIBRATION_CHARTS_DIR = CALIBRATION_RESULTS_DIR / "charts"

CALIBRATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = PROCESSED_DATA_DIR / "validation_v1.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_v1.csv"

TARGET_COLUMN = "Diabetes_binary"

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


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def split_features_and_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into X features and y target."""

    X = dataset[SELECTED_FEATURES_V1]
    y = dataset[TARGET_COLUMN]

    return X, y


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip probabilities to avoid log loss errors at exactly 0 or 1."""

    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def fit_sigmoid_calibrator(
    validation_probabilities: np.ndarray,
    y_validation: pd.Series,
) -> LogisticRegression:
    """
    Fit sigmoid/Platt-style calibration.

    The input is the model's predicted probability from validation data.
    """

    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(validation_probabilities.reshape(-1, 1), y_validation)

    return calibrator


def fit_isotonic_calibrator(
    validation_probabilities: np.ndarray,
    y_validation: pd.Series,
) -> IsotonicRegression:
    """Fit isotonic calibration."""

    calibrator = IsotonicRegression(
        y_min=0,
        y_max=1,
        out_of_bounds="clip",
    )

    calibrator.fit(validation_probabilities, y_validation)

    return calibrator


def apply_calibrator(
    probabilities: np.ndarray,
    calibrator,
    method: str,
) -> np.ndarray:
    """Apply selected calibration method to raw probabilities."""

    if method == "uncalibrated":
        return probabilities

    if method == "sigmoid":
        calibrated_probabilities = calibrator.predict_proba(
            probabilities.reshape(-1, 1)
        )[:, 1]
        return calibrated_probabilities

    if method == "isotonic":
        calibrated_probabilities = calibrator.predict(probabilities)
        return calibrated_probabilities

    raise ValueError(f"Unknown calibration method: {method}")


def calculate_probability_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    method: str,
    threshold: float = 0.5,
) -> dict:
    """Calculate classification and probability quality metrics."""

    probabilities = clip_probabilities(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "method": method,
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
        "brier_score": round(brier_score_loss(y_true, probabilities), 4),
        "log_loss": round(log_loss(y_true, probabilities), 4),
    }


def save_json(data: dict, output_file: Path) -> None:
    """Save dictionary as readable JSON."""

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_calibration_curve_chart(
    y_true: pd.Series,
    probability_sets: dict,
    output_file: Path,
    title: str,
) -> None:
    """Save calibration curve comparison chart."""

    plt.figure(figsize=(8, 6))

    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for method, probabilities in probability_sets.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="uniform",
        )

        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            label=method,
        )

    plt.title(title)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved calibration curve chart: {output_file}")


def save_brier_score_chart(metrics: list[dict], output_file: Path, title: str) -> None:
    """Save Brier score comparison chart."""

    methods = [item["method"] for item in metrics]
    scores = [item["brier_score"] for item in metrics]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, scores)
    plt.title(title)
    plt.xlabel("Calibration Method")
    plt.ylabel("Brier Score - Lower is Better")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved Brier score chart: {output_file}")


def save_probability_distribution_chart(
    uncalibrated_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    output_file: Path,
) -> None:
    """Save before vs after probability distribution chart."""

    plt.figure(figsize=(9, 5))
    plt.hist(
        uncalibrated_probabilities,
        bins=30,
        alpha=0.5,
        label="Uncalibrated",
    )
    plt.hist(
        calibrated_probabilities,
        bins=30,
        alpha=0.5,
        label="Calibrated",
    )
    plt.title("Predicted Probability Distribution Before vs After Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Number of Records")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved probability distribution chart: {output_file}")


def save_confusion_matrix_chart(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_file: Path,
    title: str,
    threshold: float = 0.5,
) -> None:
    """Save confusion matrix using calibrated probabilities."""

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No diabetes", "Prediabetes/Diabetes"],
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved confusion matrix chart: {output_file}")


def write_calibration_report(
    selected_method: str,
    validation_metrics: list[dict],
    test_metrics: list[dict],
) -> None:
    """Write Step 6 calibration report."""

    report_file = CALIBRATION_RESULTS_DIR / "calibration_report_v1.txt"

    selected_validation_metric = next(
        item for item in validation_metrics if item["method"] == selected_method
    )

    selected_test_metric = next(
        item for item in test_metrics if item["method"] == selected_method
    )

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("STEP 6 PROBABILITY CALIBRATION REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write("Project:\n")
        report.write("Machine Learning-Based Diabetes Risk Estimation Dashboard\n\n")

        report.write("Purpose:\n")
        report.write(
            "- Calibrate model probability output before showing it in the dashboard.\n\n"
        )

        report.write("Important Data Usage Rule:\n")
        report.write("- Validation data was used to fit and select calibration method.\n")
        report.write("- Test data was used only for reporting after calibration.\n")
        report.write("- Test data was not used to tune the calibration method.\n\n")

        report.write("Calibration Methods Compared:\n")
        report.write("- Uncalibrated baseline probability\n")
        report.write("- Sigmoid calibration\n")
        report.write("- Isotonic calibration\n\n")

        report.write("Validation Metrics:\n")
        report.write(pd.DataFrame(validation_metrics).to_string(index=False))
        report.write("\n\n")

        report.write("Selected Calibration Method:\n")
        report.write(f"- {selected_method}\n\n")

        report.write("Selection Rule:\n")
        report.write(
            "- The selected calibration method is the one with the lowest "
            "validation Brier score.\n\n"
        )

        report.write("Selected Method Validation Metrics:\n")
        for key, value in selected_validation_metric.items():
            report.write(f"- {key}: {value}\n")

        report.write("\nSelected Method Test Metrics:\n")
        for key, value in selected_test_metric.items():
            report.write(f"- {key}: {value}\n")

        report.write("\nDashboard Use:\n")
        report.write(
            "- Use calibrated probability for dashboard display.\n"
        )
        report.write(
            "- Display as model probability of positive dataset class, not as diagnosis.\n\n"
        )

        report.write("Responsible Use Note:\n")
        report.write(
            "- The model estimates the dataset label for prediabetes or diabetes.\n"
        )
        report.write(
            "- The dashboard must not say the user has diabetes.\n"
        )
        report.write(
            "- The dashboard should say estimated diabetes-related risk.\n"
        )

    print(f"Saved calibration report: {report_file}")


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main() -> None:
    """Run probability calibration workflow."""

    print("=" * 70)
    print("STEP 6: PROBABILITY CALIBRATION")
    print("=" * 70)

    print("\n1. Loading final model")
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final model not found: {FINAL_MODEL_FILE}\n"
            "Run Step 5 evaluate_final_model.py first."
        )

    base_model = joblib.load(FINAL_MODEL_FILE)
    print(f"Loaded final model: {FINAL_MODEL_FILE}")

    print("\n2. Loading validation and test datasets")
    validation_df = load_dataset(VALIDATION_FILE)
    test_df = load_dataset(TEST_FILE)

    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape: {test_df.shape}")

    X_validation, y_validation = split_features_and_target(validation_df)
    X_test, y_test = split_features_and_target(test_df)

    print("\n3. Getting uncalibrated model probabilities")
    validation_uncalibrated_probs = base_model.predict_proba(X_validation)[:, 1]
    test_uncalibrated_probs = base_model.predict_proba(X_test)[:, 1]

    print("\n4. Fitting calibration methods using validation set only")
    sigmoid_calibrator = fit_sigmoid_calibrator(
        validation_uncalibrated_probs,
        y_validation,
    )

    isotonic_calibrator = fit_isotonic_calibrator(
        validation_uncalibrated_probs,
        y_validation,
    )

    print("Calibration methods fitted successfully.")

    print("\n5. Creating calibrated probabilities")

    validation_probability_sets = {
        "uncalibrated": validation_uncalibrated_probs,
        "sigmoid": apply_calibrator(
            validation_uncalibrated_probs,
            sigmoid_calibrator,
            "sigmoid",
        ),
        "isotonic": apply_calibrator(
            validation_uncalibrated_probs,
            isotonic_calibrator,
            "isotonic",
        ),
    }

    test_probability_sets = {
        "uncalibrated": test_uncalibrated_probs,
        "sigmoid": apply_calibrator(
            test_uncalibrated_probs,
            sigmoid_calibrator,
            "sigmoid",
        ),
        "isotonic": apply_calibrator(
            test_uncalibrated_probs,
            isotonic_calibrator,
            "isotonic",
        ),
    }

    print("\n6. Calculating validation calibration metrics")
    validation_metrics = [
        calculate_probability_metrics(
            y_validation,
            probabilities,
            method,
        )
        for method, probabilities in validation_probability_sets.items()
    ]

    validation_metrics_df = pd.DataFrame(validation_metrics)
    print(validation_metrics_df.to_string(index=False))

    print("\n7. Selecting best calibration method using validation Brier score")

    sorted_validation_metrics = validation_metrics_df.sort_values(
        by=["brier_score", "log_loss"],
        ascending=True,
    )

    selected_method = sorted_validation_metrics.iloc[0]["method"]
    print(f"Selected calibration method: {selected_method}")

    if selected_method == "uncalibrated":
        selected_calibrator = None
    elif selected_method == "sigmoid":
        selected_calibrator = sigmoid_calibrator
    elif selected_method == "isotonic":
        selected_calibrator = isotonic_calibrator
    else:
        raise ValueError(f"Unknown selected method: {selected_method}")

    print("\n8. Calculating test metrics for reporting only")
    test_metrics = [
        calculate_probability_metrics(
            y_test,
            probabilities,
            method,
        )
        for method, probabilities in test_probability_sets.items()
    ]

    test_metrics_df = pd.DataFrame(test_metrics)
    print(test_metrics_df.to_string(index=False))

    print("\n9. Saving calibrated model package")

    calibrated_model_package = {
        "project": "Machine Learning-Based Diabetes Risk Estimation Dashboard",
        "model_version": "diabetes_risk_calibrated_model_v1",
        "base_model": base_model,
        "selected_features": SELECTED_FEATURES_V1,
        "target_column": TARGET_COLUMN,
        "target_meaning": {
            "0": "No diabetes",
            "1": "Prediabetes or diabetes",
        },
        "calibration_method": selected_method,
        "calibrator": selected_calibrator,
        "threshold": 0.5,
        "responsible_use_note": (
            "This model estimates the dataset label for prediabetes or diabetes. "
            "It is not a medical diagnosis."
        ),
    }

    calibrated_model_file = (
        CALIBRATED_MODELS_DIR
        / "diabetes_risk_calibrated_model_v1.joblib"
    )

    joblib.dump(calibrated_model_package, calibrated_model_file)
    print(f"Saved calibrated model package: {calibrated_model_file}")

    print("\n10. Saving metrics and metadata")

    calibration_results = {
        "selected_calibration_method": selected_method,
        "selection_metric": "lowest validation brier_score",
        "validation_metrics": validation_metrics,
        "test_metrics_for_reporting_only": test_metrics,
        "important_note": (
            "Validation data was used for calibration selection. "
            "Test data was used only for reporting after calibration."
        ),
    }

    save_json(
        calibration_results,
        CALIBRATION_RESULTS_DIR / "calibration_metrics_v1.json",
    )

    print("\n11. Saving charts")

    save_calibration_curve_chart(
        y_validation,
        validation_probability_sets,
        CALIBRATION_CHARTS_DIR / "validation_calibration_curve.png",
        "Validation Calibration Curve",
    )

    save_calibration_curve_chart(
        y_test,
        test_probability_sets,
        CALIBRATION_CHARTS_DIR / "test_calibration_curve_reporting_only.png",
        "Test Calibration Curve - Reporting Only",
    )

    save_brier_score_chart(
        validation_metrics,
        CALIBRATION_CHARTS_DIR / "validation_brier_score_comparison.png",
        "Validation Brier Score Comparison",
    )

    selected_validation_probs = validation_probability_sets[selected_method]
    selected_test_probs = test_probability_sets[selected_method]

    save_probability_distribution_chart(
        validation_uncalibrated_probs,
        selected_validation_probs,
        CALIBRATION_CHARTS_DIR / "validation_probability_distribution_before_after.png",
    )

    save_confusion_matrix_chart(
        y_test,
        selected_test_probs,
        CALIBRATION_CHARTS_DIR / "calibrated_test_confusion_matrix_reporting_only.png",
        "Calibrated Model Confusion Matrix - Test Set Reporting Only",
    )

    print("\n12. Writing calibration report")
    write_calibration_report(
        selected_method=selected_method,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    print("\n" + "=" * 70)
    print("STEP 6 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated calibrated model:")
    print("- ml-training/models/calibrated/diabetes_risk_calibrated_model_v1.joblib")

    print("\nGenerated calibration results:")
    print("- ml-training/results/calibration/calibration_metrics_v1.json")
    print("- ml-training/results/calibration/calibration_report_v1.txt")
    print("- ml-training/results/calibration/charts/validation_calibration_curve.png")
    print("- ml-training/results/calibration/charts/test_calibration_curve_reporting_only.png")
    print("- ml-training/results/calibration/charts/validation_brier_score_comparison.png")
    print("- ml-training/results/calibration/charts/validation_probability_distribution_before_after.png")
    print("- ml-training/results/calibration/charts/calibrated_test_confusion_matrix_reporting_only.png")


if __name__ == "__main__":
    main()