import os
import logging
from contextlib import asynccontextmanager
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from src.models.evaluator import ModelEvaluator
import yaml

from src.models.preprocessor import DataPreprocessor
from src.models.classifier import FitnessClassifier

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

preprocessor: DataPreprocessor = None
model: FitnessClassifier = None


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint."""

    age: float = Field(..., ge=18, le=100)
    weight_kg: float = Field(..., ge=30, le=200)
    heart_rate: float = Field(..., ge=40, le=200)
    sleep_hours: float = Field(..., ge=0, le=24)
    activity_days_week: int = Field(..., ge=0, le=7)
    smoking: str = Field(..., pattern="^(yes|no)$")
    nutrition_quality: str = Field(..., pattern="^(low|medium|high)$")

    @field_validator("smoking", "nutrition_quality")
    @classmethod
    def lowercase(cls, v):
        return v.lower()


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""

    is_fit: int
    probability: float
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global preprocessor, model
    from src.models.evaluator import ModelEvaluator
    import yaml

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Generate synthetic data for demo
    df = generate_synthetic_data(200)

    # Правильно извлекаем настройки модели
    model_config = config["model"]["improved"]
    model_type = model_config["type"]  # "random_forest"
    model_params = model_config.get("params", {})  # {n_estimators: 100, ...}

    preprocessor = DataPreprocessor(
        numeric_features=config["data"]["numeric_features"],
        categorical_features=config["data"]["categorical_features"],
    )
    X = preprocessor.fit_transform(df)
    y = df[config["data"]["target_column"]].values

    # Train model with correct unpacking
    model = FitnessClassifier(model_type=model_type, **model_params)
    model.fit(X, y)
    model.save(os.getenv("MODEL_PATH", "artifacts/fitness_model.pkl"))

    logger.info("Service started with trained model")
    yield
    logger.info("Service shutting down")


app = FastAPI(title="Fitness Classifier", lifespan=lifespan)


def generate_synthetic_data(n: int = 200) -> pd.DataFrame:
    """Generate minimal synthetic data for demo startup."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, n),
            "weight_kg": np.random.normal(70, 15, n).clip(40, 150),
            "heart_rate": np.random.normal(70, 15, n).clip(50, 120),
            "sleep_hours": np.random.normal(7, 1.5, n).clip(4, 10),
            "activity_days_week": np.random.randint(0, 8, n),
            "smoking": np.random.choice(["yes", "no"], n),
            "nutrition_quality": np.random.choice(["low", "medium", "high"], n),
        }
    )

    score = (
        (df["activity_days_week"] >= 4) * 2
        + (df["sleep_hours"] >= 7)
        + (df["nutrition_quality"] == "high")
        + (df["smoking"] == "no")
        - (df["weight_kg"] > 90)
    )
    df["is_fit"] = ((score >= 3) & (np.random.random(n) > 0.15)).astype(int)
    return df


@app.get("/health")
async def health_check():
    """Liveness probe endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict fitness status for a person."""
    if model is None or preprocessor is None:
        raise HTTPException(503, "Model not loaded")

    input_df = pd.DataFrame([request.model_dump()])

    try:
        X = preprocessor.transform(input_df)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(400, f"Prediction failed: {str(e)}")

    return PredictionResponse(
        is_fit=int(pred),
        probability=round(float(proba), 3),
        message="Fit!" if pred == 1 else "Consider lifestyle changes",
    )
