from typing import List

from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.database import get_db
from backend.app.dependencies.auth_dependency import get_current_user
from backend.app.models.prediction_history import PredictionHistory
from backend.app.models.user import User
from backend.app.schemas.history_schema import PredictionHistoryResponse
from backend.app.schemas.prediction_schema import (
    DiabetesPredictionRequest,
    DiabetesPredictionResponse,
)
from backend.app.services.diabetes_prediction_service import diabetes_prediction_service


router = APIRouter(
    prefix="/api/predictions",
    tags=["Predictions"],
)


@router.post(
    "/diabetes",
    response_model=DiabetesPredictionResponse,
    summary="Estimate diabetes-related risk",
)
def predict_diabetes_risk(
    request: DiabetesPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiabetesPredictionResponse:
    result = diabetes_prediction_service.predict(request.model_dump())

    history_record = PredictionHistory(
        user_id=current_user.id,
        disease_type="diabetes",
        input_values=request.model_dump(),
        prediction_response=result,
        estimated_risk=result["estimated_risk"],
        calibrated_probability=result["calibrated_probability"],
        display_probability=result["display_probability"],
        raw_model_probability=result["raw_model_probability"],
        calibration_method=result["calibration_method"],
    )

    db.add(history_record)
    db.commit()
    db.refresh(history_record)

    result["assessment_id"] = history_record.id

    return DiabetesPredictionResponse(**result)


@router.get(
    "/history",
    response_model=List[PredictionHistoryResponse],
    summary="Get logged-in user's prediction history",
)
def get_prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PredictionHistory]:
    history = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )

    return history


@router.get(
    "/history/{history_id}",
    response_model=PredictionHistoryResponse,
    summary="Get one prediction history record",
)
def get_prediction_history_by_id(
    history_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PredictionHistory:
    history_record = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.id == history_id,
            PredictionHistory.user_id == current_user.id,
        )
        .first()
    )

    if history_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction history record not found",
        )

    return history_record