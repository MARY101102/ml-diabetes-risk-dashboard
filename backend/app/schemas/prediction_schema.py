"""
Prediction request and response schemas.

Project:
Machine Learning-Based Diabetes Risk Estimation Dashboard
"""

from pydantic import BaseModel, Field
from typing import List


class DiabetesPredictionRequest(BaseModel):
    assessment_id: int | None = None
    HighBP: int = Field(..., ge=0, le=1, description="High blood pressure: 0=No, 1=Yes")
    HighChol: int = Field(..., ge=0, le=1, description="High cholesterol: 0=No, 1=Yes")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol check in last 5 years: 0=No, 1=Yes")
    BMI: float = Field(..., ge=1, le=100, description="Body Mass Index")
    Smoker: int = Field(..., ge=0, le=1, description="Smoking history: 0=No, 1=Yes")
    Stroke: int = Field(..., ge=0, le=1, description="Stroke history: 0=No, 1=Yes")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Heart disease or attack history: 0=No, 1=Yes")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in last 30 days: 0=No, 1=Yes")
    Fruits: int = Field(..., ge=0, le=1, description="Fruit consumption: 0=No, 1=Yes")
    Veggies: int = Field(..., ge=0, le=1, description="Vegetable consumption: 0=No, 1=Yes")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption: 0=No, 1=Yes")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Healthcare coverage: 0=No, 1=Yes")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Could not see doctor because of cost: 0=No, 1=Yes")
    GenHlth: int = Field(..., ge=1, le=5, description="General health: 1=Excellent, 5=Poor")
    MentHlth: int = Field(..., ge=0, le=30, description="Poor mental health days in last 30 days")
    PhysHlth: int = Field(..., ge=0, le=30, description="Poor physical health days in last 30 days")
    DiffWalk: int = Field(..., ge=0, le=1, description="Difficulty walking: 0=No, 1=Yes")
    Age: int = Field(..., ge=1, le=13, description="Age group: 1=18-24, 13=80 or older")

    model_config = {
        "json_schema_extra": {
            "example": {
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "BMI": 32,
                "Smoker": 0,
                "Stroke": 0,
                "HeartDiseaseorAttack": 0,
                "PhysActivity": 1,
                "Fruits": 1,
                "Veggies": 1,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 0,
                "GenHlth": 3,
                "MentHlth": 5,
                "PhysHlth": 3,
                "DiffWalk": 0,
                "Age": 8
            }
        }
    }


class ExplanationFactor(BaseModel):
    feature: str
    label: str
    input_value: float
    shap_value: float
    explanation: str


class DiabetesPredictionResponse(BaseModel):
    estimated_risk: str
    calibrated_probability: float
    display_probability: str
    raw_model_probability: float
    calibration_method: str
    top_factors_increasing_estimated_risk: List[ExplanationFactor]
    top_factors_reducing_estimated_risk: List[ExplanationFactor]
    disclaimer: str