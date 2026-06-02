"""
Step 5: Final Model Evaluation and Model Selection

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Purpose:
- Load the best baseline model selected from validation results
- Evaluate it on the untouched test dataset
- Save final model, metrics, charts and evaluation report

Important:
- The test set is used only for final evaluation.
- Do not tune or retrain the model using the test set.
"""

from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

ML_TRAINING_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = ML_TRAINING_DIR / "data" / "processed"

BASELINE_MODELS_DIR = ML_TRAINING_DIR / "models" / "baseline"
FINAL_MODELS_DIR = ML_TRAINING_DIR / "models" / "final"

MODEL_TRAINING_RESULTS_DIR = ML_TRAINING_DIR / "results" / "model_training"
FINAL_RESULTS_DIR = ML_TRAINING_DIR / "results" / "final_evaluation"
FINAL_CHARTS_DIR = FINAL_RESULTS_DIR / "charts"

FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = PROCESSED_DATA_DIR / "test_v1.csv"
BEST_BASELINE_SUMMARY_FILE = (
    MODEL_TRAINING_RESULTS_DIR / "best_baseline_model_summary.json"
)

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

def load_best_model_name() -> str:
    """Load best baseline model name selected during Step 4."""

    if not BEST_BASELINE_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Best baseline summary file not found: {BEST_BASELINE_SUMMARY_FILE}\n"
            "Run Step 4 train_baseline_models.py first."
        )

    with open(BEST_BASELINE_SUMMARY_FILE, "r", encoding="utf-8") as file:
        summary = json.load(file)

    best_model_name = summary["best_baseline_model"]

    return best_model_name


def get_baseline_model_file(best_model_name: str) -> Path:
    """Return model file path for selected baseline model."""

    model_file = BASELINE_MODELS_DIR / f"{best_model_name}_baseline.joblib"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Selected baseline model file not found: {model_file}"
        )

    return model_file


def load_test_data() -> pd.DataFrame:
    """Load untouched test dataset."""

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}\n"
            "Run Step 3 preprocess_dataset.py first."
        )

    return pd.read_csv(TEST_FILE)


def split_features_and_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into X features and y target."""

    X = dataset[SELECTED_FEATURES_V1]
    y = dataset[TARGET_COLUMN]

    return X, y


def calculate_final_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """Calculate final test-set metrics."""

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "evaluation_dataset": "test_v1.csv",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_probability), 4),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No diabetes", "Prediabetes or diabetes"],
        zero_division=0,
        output_dict=True,
    )

    metrics["classification_report"] = report

    matrix = confusion_matrix(y_test, y_pred)

    metrics["confusion_matrix"] = {
        "true_negative": int(matrix[0][0]),
        "false_positive": int(matrix[0][1]),
        "false_negative": int(matrix[1][0]),
        "true_positive": int(matrix[1][1]),
    }

    return metrics


def save_json(data: dict, output_file: Path) -> None:
    """Save dictionary as readable JSON."""

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_final_confusion_matrix_chart(model, X_test, y_test) -> None:
    """Save final confusion matrix chart."""

    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No diabetes", "Prediabetes/Diabetes"],
    )

    display.plot(values_format="d")
    plt.title("Final Model Confusion Matrix - Test Set")
    plt.tight_layout()

    output_file = FINAL_CHARTS_DIR / "final_confusion_matrix.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved final confusion matrix chart: {output_file}")


def save_final_roc_curve_chart(model, X_test, y_test) -> None:
    """Save final ROC curve chart."""

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Final Model ROC Curve - Test Set")
    plt.tight_layout()

    output_file = FINAL_CHARTS_DIR / "final_roc_curve.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved final ROC curve chart: {output_file}")


def save_final_precision_recall_curve_chart(model, X_test, y_test) -> None:
    """Save final precision-recall curve chart."""

    y_probability = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_probability)

    display = PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
    )

    display.plot()
    plt.title("Final Model Precision-Recall Curve - Test Set")
    plt.tight_layout()

    output_file = FINAL_CHARTS_DIR / "final_precision_recall_curve.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved final precision-recall curve chart: {output_file}")


def write_final_evaluation_report(
    model_name: str,
    test_df: pd.DataFrame,
    metrics: dict,
) -> None:
    """Write final model evaluation report."""

    report_file = FINAL_RESULTS_DIR / "final_model_evaluation_report.txt"

    confusion = metrics["confusion_matrix"]

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("STEP 5 FINAL MODEL EVALUATION REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write("Project:\n")
        report.write("Machine Learning-Based Diabetes Risk Estimation Dashboard\n\n")

        report.write("Final Selected Model:\n")
        report.write(f"- {model_name}\n\n")

        report.write("Evaluation Dataset:\n")
        report.write("- test_v1.csv\n")
        report.write(f"- Test shape: {test_df.shape}\n\n")

        report.write("Important Evaluation Rule:\n")
        report.write(
            "- The test dataset was used only after selecting the best "
            "baseline model from validation results.\n"
        )
        report.write(
            "- The test dataset was not used for training or model tuning.\n\n"
        )

        report.write("Final Test Metrics:\n")
        report.write(f"- Accuracy : {metrics['accuracy']}\n")
        report.write(f"- Precision: {metrics['precision']}\n")
        report.write(f"- Recall   : {metrics['recall']}\n")
        report.write(f"- F1-score : {metrics['f1_score']}\n")
        report.write(f"- ROC-AUC  : {metrics['roc_auc']}\n\n")

        report.write("Confusion Matrix Values:\n")
        report.write(f"- True Negative : {confusion['true_negative']}\n")
        report.write(f"- False Positive: {confusion['false_positive']}\n")
        report.write(f"- False Negative: {confusion['false_negative']}\n")
        report.write(f"- True Positive : {confusion['true_positive']}\n\n")

        report.write("Metric Interpretation:\n")
        report.write(
            "- Accuracy is not the only selection metric because the dataset "
            "is imbalanced.\n"
        )
        report.write(
            "- Recall, F1-score and ROC-AUC are more useful for understanding "
            "performance on the positive class.\n"
        )
        report.write(
            "- False negatives are important to monitor because they represent "
            "positive-class records predicted as negative.\n\n"
        )

        report.write("Responsible Use Note:\n")
        report.write(
            "- The model predicts the dataset label for prediabetes or diabetes.\n"
        )
        report.write(
            "- The dashboard must not display the output as a confirmed medical "
            "diagnosis.\n"
        )
        report.write(
            "- The result should be shown as an estimated diabetes-related risk "
            "or model probability of the positive dataset class.\n\n"
        )

        report.write("Next Step:\n")
        report.write(
            "- Step 6 should calibrate predicted probabilities before showing "
            "confidence-like values in the dashboard.\n"
        )

    print(f"Saved final evaluation report: {report_file}")


def save_final_model_metadata(model_name: str, metrics: dict) -> None:
    """Save metadata for the final selected model."""

    metadata = {
        "project": "Machine Learning-Based Diabetes Risk Estimation Dashboard",
        "model_version": "diabetes_risk_model_v1",
        "selected_model": model_name,
        "target_column": TARGET_COLUMN,
        "target_meaning": {
            "0": "No diabetes",
            "1": "Prediabetes or diabetes",
        },
        "selected_features": SELECTED_FEATURES_V1,
        "evaluation_dataset": "test_v1.csv",
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        },
        "responsible_use_note": (
            "This model is for educational risk estimation only. "
            "It must not be used as a medical diagnostic system."
        ),
    }

    metadata_file = FINAL_RESULTS_DIR / "final_model_metadata.json"
    save_json(metadata, metadata_file)

    print(f"Saved final model metadata: {metadata_file}")


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main() -> None:
    """Run final model evaluation."""

    print("=" * 70)
    print("STEP 5: FINAL MODEL EVALUATION")
    print("=" * 70)

    print("\n1. Loading best baseline model name")
    best_model_name = load_best_model_name()
    print(f"Best baseline model from Step 4: {best_model_name}")

    print("\n2. Loading selected baseline model")
    baseline_model_file = get_baseline_model_file(best_model_name)
    model = joblib.load(baseline_model_file)
    print(f"Loaded model: {baseline_model_file}")

    print("\n3. Loading untouched test dataset")
    test_df = load_test_data()
    print(f"Test dataset shape: {test_df.shape}")

    print("\n4. Splitting test features and target")
    X_test, y_test = split_features_and_target(test_df)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\n5. Evaluating final model on test set")
    final_metrics = calculate_final_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=best_model_name,
    )

    print("Final test metrics:")
    print(f"Accuracy : {final_metrics['accuracy']}")
    print(f"Precision: {final_metrics['precision']}")
    print(f"Recall   : {final_metrics['recall']}")
    print(f"F1-score : {final_metrics['f1_score']}")
    print(f"ROC-AUC  : {final_metrics['roc_auc']}")

    print("\nConfusion matrix:")
    for key, value in final_metrics["confusion_matrix"].items():
        print(f"{key}: {value}")

    print("\n6. Saving final selected model")
    final_model_file = FINAL_MODELS_DIR / "diabetes_risk_model_v1.joblib"
    shutil.copy2(baseline_model_file, final_model_file)
    print(f"Saved final model: {final_model_file}")

    print("\n7. Saving final metrics")
    final_metrics_file = FINAL_RESULTS_DIR / "final_model_metrics.json"
    save_json(final_metrics, final_metrics_file)
    print(f"Saved final metrics: {final_metrics_file}")

    print("\n8. Saving final charts")
    save_final_confusion_matrix_chart(model, X_test, y_test)
    save_final_roc_curve_chart(model, X_test, y_test)
    save_final_precision_recall_curve_chart(model, X_test, y_test)

    print("\n9. Saving final evaluation report")
    write_final_evaluation_report(
        model_name=best_model_name,
        test_df=test_df,
        metrics=final_metrics,
    )

    print("\n10. Saving final model metadata")
    save_final_model_metadata(
        model_name=best_model_name,
        metrics=final_metrics,
    )

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated final model file:")
    print("- ml-training/models/final/diabetes_risk_model_v1.joblib")

    print("\nGenerated final evaluation files:")
    print("- ml-training/results/final_evaluation/final_model_metrics.json")
    print("- ml-training/results/final_evaluation/final_model_metadata.json")
    print("- ml-training/results/final_evaluation/final_model_evaluation_report.txt")
    print("- ml-training/results/final_evaluation/charts/final_confusion_matrix.png")
    print("- ml-training/results/final_evaluation/charts/final_roc_curve.png")
    print("- ml-training/results/final_evaluation/charts/final_precision_recall_curve.png")


if __name__ == "__main__":
    main()