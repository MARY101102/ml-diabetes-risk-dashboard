"""
Step 4: Baseline Model Training

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Purpose:
- Train baseline classification models
- Evaluate models on the validation set
- Compare Accuracy, Precision, Recall, F1-score and ROC-AUC
- Save trained model files, metrics and charts

Important:
- This script uses train_v1.csv and validation_v1.csv only.
- test_v1.csv is NOT used in this step.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

ML_TRAINING_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = ML_TRAINING_DIR / "data" / "processed"
MODELS_DIR = ML_TRAINING_DIR / "models" / "baseline"
RESULTS_DIR = ML_TRAINING_DIR / "results" / "model_training"
CHARTS_DIR = RESULTS_DIR / "charts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED_DATA_DIR / "train_v1.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation_v1.csv"

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

# Logistic Regression benefits from scaling.
NUMERIC_FEATURES_TO_SCALE = [
    "BMI",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
]


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def load_train_and_validation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    return train_df, validation_df


def split_features_and_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into X features and y target."""

    X = dataset[SELECTED_FEATURES_V1]
    y = dataset[TARGET_COLUMN]

    return X, y


def build_models() -> dict:
    """Create baseline model pipelines."""

    scaler_preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), NUMERIC_FEATURES_TO_SCALE),
        ],
        remainder="passthrough",
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", scaler_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=8,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Evaluate one trained model on the validation dataset."""

    y_pred = model.predict(X_validation)
    y_probability = model.predict_proba(X_validation)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_validation, y_pred), 4),
        "precision": round(
            precision_score(y_validation, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_validation, y_pred, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_validation, y_pred, zero_division=0), 4
        ),
        "roc_auc": round(roc_auc_score(y_validation, y_probability), 4),
    }

    report = classification_report(
        y_validation,
        y_pred,
        target_names=["No diabetes", "Prediabetes or diabetes"],
        zero_division=0,
        output_dict=True,
    )

    metrics["classification_report"] = report

    return metrics


def save_confusion_matrix_chart(
    model_name: str,
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    """Save confusion matrix chart for a model."""

    y_pred = model.predict(X_validation)
    matrix = confusion_matrix(y_validation, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No diabetes", "Prediabetes/Diabetes"],
    )

    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    plt.tight_layout()

    output_file = CHARTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved confusion matrix chart: {output_file}")


def save_roc_curve_chart(
    model_name: str,
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    """Save ROC curve chart for a model."""

    RocCurveDisplay.from_estimator(model, X_validation, y_validation)
    plt.title(f"ROC Curve - {model_name.replace('_', ' ').title()}")
    plt.tight_layout()

    output_file = CHARTS_DIR / f"{model_name}_roc_curve.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved ROC curve chart: {output_file}")


def save_metrics_comparison_chart(metrics_df: pd.DataFrame) -> None:
    """Save comparison chart for main validation metrics."""

    chart_file = CHARTS_DIR / "baseline_model_metrics_comparison.png"

    chart_data = metrics_df.set_index("model_name")[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]

    chart_data.plot(kind="bar", figsize=(10, 6))
    plt.title("Baseline Model Validation Metrics Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    plt.close()

    print(f"Saved model comparison chart: {chart_file}")


def save_json(data: dict, output_file: Path) -> None:
    """Save dictionary as readable JSON."""

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def write_training_report(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    """Write a text report summarizing Step 4."""

    report_file = RESULTS_DIR / "baseline_training_report.txt"

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("STEP 4 BASELINE MODEL TRAINING REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write("Project:\n")
        report.write("Machine Learning-Based Diabetes Risk Estimation Dashboard\n\n")

        report.write("Important Note:\n")
        report.write(
            "- This step used the train and validation datasets only.\n"
        )
        report.write(
            "- The test dataset was not used for model selection.\n\n"
        )

        report.write("Dataset Split Used:\n")
        report.write(f"- Train shape: {train_df.shape}\n")
        report.write(f"- Validation shape: {validation_df.shape}\n\n")

        report.write("Models Trained:\n")
        report.write("- Logistic Regression\n")
        report.write("- Decision Tree\n")
        report.write("- Random Forest\n\n")

        report.write("Validation Metrics:\n")
        report.write(metrics_df.to_string(index=False))
        report.write("\n\n")

        report.write("Current Best Baseline Model:\n")
        report.write(f"- {best_model_name}\n\n")

        report.write("Model Selection Rule:\n")
        report.write(
            "- Because the dataset is imbalanced, the model should not be "
            "selected using accuracy alone.\n"
        )
        report.write(
            "- Recall, F1-score and ROC-AUC are more important for this "
            "risk-estimation project.\n\n"
        )

        report.write("Responsible Use Note:\n")
        report.write(
            "- The output should be presented as an estimated diabetes-related "
            "risk or model probability of the positive dataset class.\n"
        )
        report.write(
            "- It must not be presented as a confirmed medical diagnosis.\n"
        )

    print(f"Saved training report: {report_file}")


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main() -> None:
    """Train and evaluate baseline models."""

    print("=" * 70)
    print("STEP 4: BASELINE MODEL TRAINING")
    print("=" * 70)

    print("\n1. Loading train and validation datasets")
    train_df, validation_df = load_train_and_validation_data()

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    print("\n2. Splitting features and target")
    X_train, y_train = split_features_and_target(train_df)
    X_validation, y_validation = split_features_and_target(validation_df)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_validation shape: {X_validation.shape}")
    print(f"y_validation shape: {y_validation.shape}")

    print("\n3. Building baseline models")
    models = build_models()

    all_metrics = []
    full_metrics = {}

    for model_name, model in models.items():
        print("\n" + "-" * 70)
        print(f"Training model: {model_name}")
        print("-" * 70)

        model.fit(X_train, y_train)
        print(f"Training completed: {model_name}")

        model_file = MODELS_DIR / f"{model_name}_baseline.joblib"
        joblib.dump(model, model_file)
        print(f"Saved model: {model_file}")

        metrics = evaluate_model(
            model_name=model_name,
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        all_metrics.append(
            {
                "model_name": metrics["model_name"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
            }
        )

        full_metrics[model_name] = metrics

        print("Validation metrics:")
        print(f"Accuracy : {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall   : {metrics['recall']}")
        print(f"F1-score : {metrics['f1_score']}")
        print(f"ROC-AUC  : {metrics['roc_auc']}")

        save_confusion_matrix_chart(
            model_name=model_name,
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        save_roc_curve_chart(
            model_name=model_name,
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    print("\n4. Saving metrics")
    metrics_df = pd.DataFrame(all_metrics)

    metrics_csv_file = RESULTS_DIR / "baseline_model_metrics.csv"
    metrics_json_file = RESULTS_DIR / "baseline_model_metrics_full.json"

    metrics_df.to_csv(metrics_csv_file, index=False)
    save_json(full_metrics, metrics_json_file)

    print(f"Saved metrics CSV: {metrics_csv_file}")
    print(f"Saved full metrics JSON: {metrics_json_file}")

    print("\n5. Saving metrics comparison chart")
    save_metrics_comparison_chart(metrics_df)

    print("\n6. Selecting current best baseline model")

    # Main selection priority:
    # 1. F1-score
    # 2. Recall
    # 3. ROC-AUC
    #
    # This is only a baseline selection. Final model selection will be refined later.
    sorted_metrics = metrics_df.sort_values(
        by=["f1_score", "recall", "roc_auc"],
        ascending=False,
    )

    best_model_name = sorted_metrics.iloc[0]["model_name"]
    print(f"Current best baseline model: {best_model_name}")

    best_model_summary_file = RESULTS_DIR / "best_baseline_model_summary.json"
    save_json(
        {
            "best_baseline_model": best_model_name,
            "selection_priority": [
                "f1_score",
                "recall",
                "roc_auc",
            ],
            "note": (
                "This is the best baseline model based on validation metrics. "
                "The test dataset has not been used yet."
            ),
        },
        best_model_summary_file,
    )

    print(f"Saved best model summary: {best_model_summary_file}")

    print("\n7. Writing training report")
    write_training_report(
        train_df=train_df,
        validation_df=validation_df,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
    )

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated model files:")
    print("- ml-training/models/baseline/logistic_regression_baseline.joblib")
    print("- ml-training/models/baseline/decision_tree_baseline.joblib")
    print("- ml-training/models/baseline/random_forest_baseline.joblib")

    print("\nGenerated result files:")
    print("- ml-training/results/model_training/baseline_model_metrics.csv")
    print("- ml-training/results/model_training/baseline_model_metrics_full.json")
    print("- ml-training/results/model_training/best_baseline_model_summary.json")
    print("- ml-training/results/model_training/baseline_training_report.txt")
    print("- ml-training/results/model_training/charts/")


if __name__ == "__main__":
    main()