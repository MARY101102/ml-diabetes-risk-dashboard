from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PredictionHistoryResponse(BaseModel):
    id: int
    disease_type: str
    input_values: dict[str, Any]
    prediction_response: dict[str, Any]
    estimated_risk: str
    calibrated_probability: float
    display_probability: str
    raw_model_probability: float
    calibration_method: str
    created_at: datetime

    model_config = {
        "from_attributes": True
    }