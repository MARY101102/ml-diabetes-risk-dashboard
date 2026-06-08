from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from backend.app.database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    disease_type = Column(String(100), default="diabetes", nullable=False)

    input_values = Column(JSON, nullable=False)
    prediction_response = Column(JSON, nullable=False)

    estimated_risk = Column(String(100), nullable=False)
    calibrated_probability = Column(Float, nullable=False)
    display_probability = Column(String(50), nullable=False)
    raw_model_probability = Column(Float, nullable=False)
    calibration_method = Column(String(50), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")