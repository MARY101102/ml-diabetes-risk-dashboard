"""
Step 7: Explainable AI

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard

Purpose:
- Explain the final calibrated diabetes risk model
- Generate global feature importance
- Generate one sample prediction explanation
- Save dashboard-ready explanation JSON
- Save charts and explanation report

Important:
- This step does not train or tune the model.
- This step explains the already selected/calibrated model.
- Results must be shown as estimated risk factors, not medical diagnosis.
"""

from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

ML_TRAINING_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = ML_TRAINING_DIR / "data" / "processed"

CALIBRATED_MODEL_FILE = (
    ML_TRAINING_DIR
    / "models"
    / "calibrated"
    / "diabetes_risk_calibrated_model_v1.joblib"
)

FINAL_MODEL_FILE = (
    ML_TRAINING_DIR
    / "models"
    / "final"
    / "diabetes_risk_model_v1.joblib"
)

EXPLAINABILITY_RESULTS_DIR = ML_TRAINING_DIR / "results" / "explainability"
EXPLAINABILITY_CHARTS_DIR = EXPLAINABILITY_RESULTS_DIR / "charts"

EXPLAINABILITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPLAINABILITY_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------
# Human-Friendly Feature Labels
# ---------------------------------------------------------------------

FEATURE_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check in last 5 years",
    "BMI": "Body Mass Index",
    "Smoker": "Smoking history",
    "Stroke": "Stroke history",
    "HeartDiseaseorAttack": "Heart disease or heart attack history",
    "PhysActivity": "Physical activity in last 30 days",
    "Fruits": "Fruit consumption",
    "Veggies": "Vegetable consumption",
    "HvyAlcoholConsump": "Heavy alcohol consumption indicator",
    "AnyHealthcare": "Healthcare coverage",
    "NoDocbcCost": "Could not see doctor because of cost",
    "GenHlth": "General health rating",
    "MentHlth": "Poor mental health days",
    "PhysHlth": "Poor physical health days",
    "DiffWalk": "Difficulty walking or climbing stairs",
    "Age": "Age group",
}


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def load_validation_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load validation dataset for explanation generation."""

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}\n"
            "Run Step 3 preprocess_dataset.py first."
        )

    validation_df = pd.read_csv(VALIDATION_FILE)

    X_validation = validation_df[SELECTED_FEATURES_V1]
    y_validation = validation_df[TARGET_COLUMN]

    return X_validation, y_validation


def load_model_package() -> dict:
    """
    Load calibrated model package if available.
    If calibrated model package is missing, load final model as fallback.
    """

    if CALIBRATED_MODEL_FILE.exists():
        print(f"Loading calibrated model package: {CALIBRATED_MODEL_FILE}")
        package = joblib.load(CALIBRATED_MODEL_FILE)
        return package

    if FINAL_MODEL_FILE.exists():
        print("Calibrated model package not found.")
        print(f"Loading final uncalibrated model instead: {FINAL_MODEL_FILE}")
        model = joblib.load(FINAL_MODEL_FILE)

        return {
            "project": "Machine Learning-Based Diabetes Risk Estimation Dashboard",
            "model_version": "diabetes_risk_model_v1_uncalibrated",
            "base_model": model,
            "calibration_method": "uncalibrated",
            "calibrator": None,
            "threshold": 0.5,
            "selected_features": SELECTED_FEATURES_V1,
            "target_column": TARGET_COLUMN,
        }

    raise FileNotFoundError(
        "No calibrated or final model found. Run Step 5 and Step 6 first."
    )


def extract_tree_model(base_model):
    """
    Extract Random Forest model from the saved sklearn Pipeline.

    In this project, the Random Forest pipeline has a step named 'model'.
    """

    if hasattr(base_model, "named_steps") and "model" in base_model.named_steps:
        return base_model.named_steps["model"]

    return base_model


def get_positive_class_shap_values(shap_values):
    """
    Handle different SHAP output formats.

    Possible formats:
    - list: [class_0_values, class_1_values]
    - ndarray with shape (rows, features, classes)
    - ndarray with shape (rows, features)
    """

    if isinstance(shap_values, list):
        return shap_values[1]

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        return shap_array[:, :, 1]

    return shap_array


def get_calibrated_probability(package: dict, X_row: pd.DataFrame) -> float:
    """Calculate calibrated probability for one row."""

    base_model = package["base_model"]
    raw_probability = base_model.predict_proba(X_row)[:, 1][0]

    method = package.get("calibration_method", "uncalibrated")
    calibrator = package.get("calibrator")

    if method == "uncalibrated" or calibrator is None:
        return float(raw_probability)

    if method == "sigmoid":
        calibrated_probability = calibrator.predict_proba(
            np.array([[raw_probability]])
        )[:, 1][0]
        return float(calibrated_probability)

    if method == "isotonic":
        calibrated_probability = calibrator.predict(
            np.array([raw_probability])
        )[0]
        return float(calibrated_probability)

    return float(raw_probability)


def map_probability_to_risk(probability: float) -> str:
    """Map probability into dashboard risk label."""

    if probability < 0.40:
        return "Low estimated risk"

    if probability < 0.70:
        return "Medium estimated risk"

    return "High estimated risk"


def create_global_feature_importance(
    X_sample: pd.DataFrame,
    shap_values_positive_class: np.ndarray,
) -> pd.DataFrame:
    """Create global SHAP feature importance table."""

    mean_abs_shap = np.abs(shap_values_positive_class).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "feature_label": [
                FEATURE_LABELS.get(feature, feature)
                for feature in X_sample.columns
            ],
            "mean_absolute_shap_value": mean_abs_shap,
        }
    )

    importance_df = importance_df.sort_values(
        by="mean_absolute_shap_value",
        ascending=False,
    )

    return importance_df


def save_global_feature_importance_chart(
    importance_df: pd.DataFrame,
) -> None:
    """Save global feature importance bar chart."""

    top_features = importance_df.head(12).sort_values(
        by="mean_absolute_shap_value",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature_label"],
        top_features["mean_absolute_shap_value"],
    )
    plt.title("Top Global Feature Importance - SHAP")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_file = EXPLAINABILITY_CHARTS_DIR / "global_feature_importance.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved global feature importance chart: {output_file}")


def create_sample_local_explanation(
    X_sample: pd.DataFrame,
    shap_values_positive_class: np.ndarray,
    package: dict,
) -> dict:
    """
    Create one dashboard-ready local explanation.

    We choose the row with the highest calibrated probability from the sample.
    """

    probabilities = []

    for index in range(len(X_sample)):
        row = X_sample.iloc[[index]]
        probability = get_calibrated_probability(package, row)
        probabilities.append(probability)

    probabilities = np.array(probabilities)
    selected_position = int(np.argmax(probabilities))

    selected_row = X_sample.iloc[[selected_position]]
    selected_probability = float(probabilities[selected_position])
    selected_risk_label = map_probability_to_risk(selected_probability)

    selected_shap_values = shap_values_positive_class[selected_position]

    local_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "feature_label": [
                FEATURE_LABELS.get(feature, feature)
                for feature in X_sample.columns
            ],
            "input_value": selected_row.iloc[0].values,
            "shap_value": selected_shap_values,
            "absolute_shap_value": np.abs(selected_shap_values),
        }
    )

    local_df = local_df.sort_values(
        by="absolute_shap_value",
        ascending=False,
    )

    top_positive = local_df[local_df["shap_value"] > 0].head(5)
    top_negative = local_df[local_df["shap_value"] < 0].head(5)

    explanation = {
        "selected_sample_position": selected_position,
        "calibrated_probability": round(selected_probability, 4),
        "display_probability": f"{selected_probability * 100:.1f}%",
        "risk_label": selected_risk_label,
        "important_note": (
            "This is a model explanation for the positive dataset class "
            "representing prediabetes or diabetes. It is not a medical diagnosis."
        ),
        "top_factors_increasing_estimated_risk": [
            {
                "feature": row["feature"],
                "label": row["feature_label"],
                "input_value": float(row["input_value"]),
                "shap_value": round(float(row["shap_value"]), 6),
                "explanation": (
                    f"{row['feature_label']} increased the model's estimated "
                    "probability for the positive dataset class."
                ),
            }
            for _, row in top_positive.iterrows()
        ],
        "top_factors_reducing_estimated_risk": [
            {
                "feature": row["feature"],
                "label": row["feature_label"],
                "input_value": float(row["input_value"]),
                "shap_value": round(float(row["shap_value"]), 6),
                "explanation": (
                    f"{row['feature_label']} reduced the model's estimated "
                    "probability for the positive dataset class."
                ),
            }
            for _, row in top_negative.iterrows()
        ],
        "all_local_feature_effects": local_df.to_dict(orient="records"),
    }

    return explanation


def save_local_explanation_chart(explanation: dict) -> None:
    """Save local explanation chart for one sample."""

    local_effects = pd.DataFrame(explanation["all_local_feature_effects"])
    top_effects = local_effects.head(10).sort_values(
        by="absolute_shap_value",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_effects["feature_label"],
        top_effects["shap_value"],
    )
    plt.title("Sample Prediction Explanation - SHAP Values")
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_file = EXPLAINABILITY_CHARTS_DIR / "sample_prediction_explanation.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved sample prediction explanation chart: {output_file}")


def save_json(data: dict, output_file: Path) -> None:
    """Save dictionary as readable JSON."""

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def write_explainability_report(
    package: dict,
    importance_df: pd.DataFrame,
    sample_explanation: dict,
) -> None:
    """Write Step 7 explainability report."""

    report_file = EXPLAINABILITY_RESULTS_DIR / "explainability_report_v1.txt"

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("STEP 7 EXPLAINABLE AI REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write("Project:\n")
        report.write("Machine Learning-Based Diabetes Risk Estimation Dashboard\n\n")

        report.write("Purpose:\n")
        report.write(
            "- Explain the model globally and locally using SHAP values.\n\n"
        )

        report.write("Model Package:\n")
        report.write(f"- Version: {package.get('model_version')}\n")
        report.write(
            f"- Calibration method: {package.get('calibration_method')}\n\n"
        )

        report.write("Global Explanation:\n")
        report.write(
            "- Global feature importance shows which features generally "
            "influence the model most across sampled validation records.\n\n"
        )

        report.write("Top Global Features:\n")
        top_features = importance_df.head(10)

        for _, row in top_features.iterrows():
            report.write(
                f"- {row['feature_label']}: "
                f"{row['mean_absolute_shap_value']:.6f}\n"
            )

        report.write("\nSample Local Explanation:\n")
        report.write(
            f"- Risk label: {sample_explanation['risk_label']}\n"
        )
        report.write(
            f"- Calibrated probability: "
            f"{sample_explanation['display_probability']}\n\n"
        )

        report.write("Top Factors Increasing Estimated Risk:\n")

        for item in sample_explanation[
            "top_factors_increasing_estimated_risk"
        ]:
            report.write(
                f"- {item['label']} "
                f"(value={item['input_value']}, "
                f"SHAP={item['shap_value']})\n"
            )

        report.write("\nTop Factors Reducing Estimated Risk:\n")

        for item in sample_explanation[
            "top_factors_reducing_estimated_risk"
        ]:
            report.write(
                f"- {item['label']} "
                f"(value={item['input_value']}, "
                f"SHAP={item['shap_value']})\n"
            )

        report.write("\nDashboard Use:\n")
        report.write(
            "- The dashboard can show the top increasing and reducing factors "
            "for each prediction.\n"
        )
        report.write(
            "- These explanations describe model behavior, not medical causes.\n\n"
        )

        report.write("Responsible Use Note:\n")
        report.write(
            "- Do not say these factors prove the user has diabetes.\n"
        )
        report.write(
            "- Say these factors influenced the model's estimated risk.\n"
        )
        report.write(
            "- This system is for educational portfolio use, not diagnosis.\n"
        )

    print(f"Saved explainability report: {report_file}")


# ---------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------

def main() -> None:
    """Run explainable AI workflow."""

    print("=" * 70)
    print("STEP 7: EXPLAINABLE AI")
    print("=" * 70)

    print("\n1. Loading validation data")
    X_validation, y_validation = load_validation_data()
    print(f"Validation features shape: {X_validation.shape}")
    print(f"Validation target shape: {y_validation.shape}")

    print("\n2. Loading model package")
    package = load_model_package()
    base_model = package["base_model"]
    tree_model = extract_tree_model(base_model)

    print(f"Loaded model version: {package.get('model_version')}")
    print(f"Calibration method: {package.get('calibration_method')}")

    print("\n3. Creating validation sample for explanation")

    # Use a sample to keep SHAP calculation fast.
    # random_state makes the output reproducible.
    sample_size = min(1000, len(X_validation))

    X_sample = X_validation.sample(
        n=sample_size,
        random_state=42,
    )

    print(f"SHAP sample shape: {X_sample.shape}")

    print("\n4. Building SHAP TreeExplainer")
    explainer = shap.TreeExplainer(tree_model)

    print("\n5. Calculating SHAP values")
    shap_values = explainer.shap_values(X_sample)
    shap_values_positive_class = get_positive_class_shap_values(shap_values)

    print(f"SHAP values shape: {shap_values_positive_class.shape}")

    print("\n6. Creating global feature importance")
    importance_df = create_global_feature_importance(
        X_sample=X_sample,
        shap_values_positive_class=shap_values_positive_class,
    )

    importance_file = (
        EXPLAINABILITY_RESULTS_DIR
        / "global_feature_importance_v1.csv"
    )

    importance_df.to_csv(importance_file, index=False)
    print(f"Saved global feature importance CSV: {importance_file}")

    save_global_feature_importance_chart(importance_df)

    print("\n7. Creating sample local explanation")
    sample_explanation = create_sample_local_explanation(
        X_sample=X_sample,
        shap_values_positive_class=shap_values_positive_class,
        package=package,
    )

    sample_explanation_file = (
        EXPLAINABILITY_RESULTS_DIR
        / "sample_prediction_explanation_v1.json"
    )

    save_json(sample_explanation, sample_explanation_file)
    print(f"Saved sample explanation JSON: {sample_explanation_file}")

    save_local_explanation_chart(sample_explanation)

    print("\n8. Creating dashboard explanation schema")

    dashboard_schema = {
        "result_fields": {
            "risk_label": "Low / Medium / High estimated risk",
            "calibrated_probability": "Model probability of positive dataset class",
            "top_factors_increasing_estimated_risk": (
                "Features that increased model probability"
            ),
            "top_factors_reducing_estimated_risk": (
                "Features that reduced model probability"
            ),
            "important_note": (
                "Explanation describes model behavior, not medical diagnosis."
            ),
        },
        "example_response": {
            "risk_label": sample_explanation["risk_label"],
            "display_probability": sample_explanation["display_probability"],
            "top_factors_increasing_estimated_risk": sample_explanation[
                "top_factors_increasing_estimated_risk"
            ][:3],
            "top_factors_reducing_estimated_risk": sample_explanation[
                "top_factors_reducing_estimated_risk"
            ][:3],
            "disclaimer": (
                "This is an educational machine learning risk estimate. "
                "It is not a medical diagnosis."
            ),
        },
    }

    dashboard_schema_file = (
        EXPLAINABILITY_RESULTS_DIR
        / "dashboard_explanation_schema_v1.json"
    )

    save_json(dashboard_schema, dashboard_schema_file)
    print(f"Saved dashboard explanation schema: {dashboard_schema_file}")

    print("\n9. Writing explainability report")
    write_explainability_report(
        package=package,
        importance_df=importance_df,
        sample_explanation=sample_explanation,
    )

    print("\n" + "=" * 70)
    print("STEP 7 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated explainability files:")
    print("- ml-training/results/explainability/global_feature_importance_v1.csv")
    print("- ml-training/results/explainability/sample_prediction_explanation_v1.json")
    print("- ml-training/results/explainability/dashboard_explanation_schema_v1.json")
    print("- ml-training/results/explainability/explainability_report_v1.txt")

    print("\nGenerated explainability charts:")
    print("- ml-training/results/explainability/charts/global_feature_importance.png")
    print("- ml-training/results/explainability/charts/sample_prediction_explanation.png")


if __name__ == "__main__":
    main()