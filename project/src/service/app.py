import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import yaml

from src.models.preprocessor import DataPreprocessor
from src.models.classifier import FitnessClassifier

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

preprocessor: DataPreprocessor = None
model: FitnessClassifier = None

ALLOWED_NUMERIC = ["age", "height_cm", "weight_kg", "sleep_hours", "activity_index"]
ALLOWED_CATEGORICAL = ["smokes", "gender"]
TARGET_COLUMN = "is_fit"
project_root = Path(__file__).resolve().parents[2]


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    height_cm: int = Field(..., ge=150, le=200)
    weight_kg: float = Field(..., ge=30, le=250)
    sleep_hours: float = Field(..., ge=4, le=12)
    activity_index: float = Field(..., ge=1.0, le=5.0)
    smokes: str = Field(..., pattern="^(yes|no)$")
    gender: str = Field(..., pattern="^(M|F|m|f)$")

    @field_validator("smokes", "gender")
    @classmethod
    def to_lowercase(cls, v: str) -> str:
        return v.lower()


class PredictionResponse(BaseModel):
    is_fit: int
    probability: float
    message: str


def preprocess_real_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "smokes" in df.columns:
        df["smokes"] = (
            df["smokes"]
            .astype(str)
            .str.lower()
            .map(lambda x: 1 if x in ["1", "yes", "true"] else 0)
        )

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (
            pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        )
        df = df.dropna(subset=[TARGET_COLUMN])

    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, model

    data_path = project_root / "data" / "fitness_dataset.csv"

    if not data_path.exists():
        logger.error(
            f"Real dataset not found at {data_path}. Please download it first."
        )
        yield
        return

    df = pd.read_csv(data_path)
    df = preprocess_real_dataset(df)

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]["improved"]
    model_type = model_config["type"]
    model_params = model_config.get("params", {})

    preprocessor = DataPreprocessor(
        numeric_features=ALLOWED_NUMERIC,
        categorical_features=ALLOWED_CATEGORICAL,
        target_column=TARGET_COLUMN,
    )
    X = preprocessor.fit_transform(df)
    y = df[TARGET_COLUMN].values

    model = FitnessClassifier(model_type=model_type, **model_params)
    model.fit(X, y)
    model.save(os.getenv("MODEL_PATH", "artifacts/fitness_model.pkl"))

    logger.info("Service started with trained model")
    yield
    logger.info("Service shutting down")


app = FastAPI(title="Fitness Classifier", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(503, "Model not loaded")

    smokes_val = 1 if request.smokes == "yes" else 0

    input_df = pd.DataFrame(
        [
            {
                "age": request.age,
                "height_cm": request.height_cm,
                "weight_kg": request.weight_kg,
                "sleep_hours": request.sleep_hours,
                "activity_index": request.activity_index,
                "smokes": smokes_val,
                "gender": request.gender,
            }
        ]
    )

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


@app.get("/features")
async def feature_descriptions():
    return {
        "age": "Возраст в годах",
        "height_cm": "Рост в сантиметрах",
        "weight_kg": "Вес в килограммах",
        "sleep_hours": "Среднее количество часов сна в сутки",
        "activity_index": "Индекс физической активности (1.0-5.0), где 1 — сидячий образ жизни, 5 — очень активный",
        "smokes": "Курение: 'yes'/'no' или '1'/'0'",
        "gender": "Пол: 'M' (мужской) или 'F' (женский)",
        "target": "is_fit: 1 — человек в хорошей физической форме, 0 — нет",
    }
