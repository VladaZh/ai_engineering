import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib

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

    model_path = Path(os.getenv("MODEL_PATH", "artifacts/fitness_model.pkl"))
    preprocessor_path = Path(os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"))

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run train.py first!")
        yield
        return

    if not preprocessor_path.exists():
        logger.error(f"Preprocessor not found at {preprocessor_path}. Run train.py first!")
        yield
        return

    logger.info(f"Loading model from {model_path}")
    model = FitnessClassifier().load(str(model_path))
    logger.info(f"Model loaded | type={type(model.model).__name__}")

    logger.info(f"Loading preprocessor from {preprocessor_path}")
    preprocessor = joblib.load(str(preprocessor_path))

    if not preprocessor._is_fitted:
        logger.error("Loaded preprocessor is not fitted!")
        yield
        return

    logger.info(f"Preprocessor loaded | is_fitted={preprocessor._is_fitted}")
    logger.info("Service started successfully")

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
