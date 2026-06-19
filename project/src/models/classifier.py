import joblib
import logging
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FitnessClassifier:
    """Wrapper for fitness classification models."""

    MODEL_REGISTRY = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
    }

    def __init__(self, model_type: str = "logistic_regression", **model_params):
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_type}")
        self.model_type = model_type
        self.model: Optional[ClassifierMixin] = self.MODEL_REGISTRY[model_type](
            **model_params
        )
        self._is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FitnessClassifier":
        """Train the model."""
        self.model.fit(X, y)
        self._is_trained = True
        logger.info(f"Trained {self.model_type} on {len(y)} samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")
        return self.model.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> "FitnessClassifier":
        """Load model from disk."""
        self.model = joblib.load(path)
        self._is_trained = True
        logger.info(f"Model loaded from {path}")
        return self
