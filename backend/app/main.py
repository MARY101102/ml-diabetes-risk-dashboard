from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.database import Base, engine
from backend.app.models.user import User
from backend.app.models.prediction_history import PredictionHistory
from backend.app.routes.auth_routes import router as auth_router
from backend.app.routes.prediction_routes import router as prediction_router


app = FastAPI(
    title="Diabetes Risk Estimation API",
    description=(
        "Backend API for the Machine Learning-Based Diabetes Risk "
        "Estimation Dashboard. This API provides educational ML-based "
        "risk estimation only and does not provide medical diagnosis."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def create_database_tables() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/", tags=["Health"])
def root() -> dict:
    return {
        "message": "Diabetes Risk Estimation API is running",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "diabetes-risk-estimation-api",
    }


app.include_router(auth_router)
app.include_router(prediction_router)