import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.service.app import app
from src.models.preprocessor import DataPreprocessor
from src.models.classifier import FitnessClassifier


@pytest.fixture
def sample_df():
    """Create small synthetic dataframe for tests with correct columns."""
    return pd.DataFrame(
        {
            "age": [25, 45, 60],
            "height_cm": [170, 180, 165],
            "weight_kg": [65, 80, 95],
            "sleep_hours": [8.0, 6.0, 5.0],
            "activity_index": [4.5, 3.0, 1.5],
            "smokes": ["no", "yes", "no"],
            "gender": ["F", "M", "F"],
            "is_fit": [1, 0, 0],
        }
    )


@pytest.fixture
def preprocessor(sample_df):
    """Fitted preprocessor fixture."""
    pp = DataPreprocessor(
        numeric_features=[
            "age",
            "height_cm",
            "weight_kg",
            "sleep_hours",
            "activity_index",
        ],
        categorical_features=["smokes", "gender"],
    )
    pp.fit(sample_df)
    return pp


@pytest.fixture
def trained_classifier(preprocessor, sample_df):
    """Trained classifier fixture."""
    X = preprocessor.transform(sample_df)
    y = sample_df["is_fit"].values
    clf = FitnessClassifier(model_type="logistic_regression", max_iter=100)
    clf.fit(X, y)
    return clf


@pytest.fixture
def mock_model():
    """Mock sklearn model for unit tests."""
    mock = Mock()
    mock.predict.return_value = np.array([1, 0, 1])
    mock.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    return mock


@pytest.fixture
def client(mock_model):
    """TestClient with mocked model dependencies."""
    mock_prep = Mock()
    mock_prep.transform.return_value = np.array([[0.1, 0.2, 0.3]])

    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    @asynccontextmanager
    async def mock_lifespan(app):
        from src.service import app as app_module
        app_module.model = mock_model
        app_module.preprocessor = mock_prep
        yield

    app.router.lifespan_context = mock_lifespan

    with TestClient(app) as c:
        yield c
