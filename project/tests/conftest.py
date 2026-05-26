# tests/conftest.py
import sys
from pathlib import Path
import pytest
import pandas as pd

# Добавляем корень проекта в sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Теперь импорты работают
from src.data.preprocessor import DataPreprocessor
from src.models.classifier import FitnessClassifier


@pytest.fixture
def sample_df():
    """Create small synthetic dataframe for tests."""
    return pd.DataFrame(
        {
            "age": [25, 45, 60],
            "weight_kg": [65, 80, 95],
            "heart_rate": [60, 75, 85],
            "sleep_hours": [8, 6, 5],
            "activity_days_week": [5, 3, 1],
            "smoking": ["no", "yes", "no"],
            "nutrition_quality": ["high", "medium", "low"],
            "is_fit": [1, 0, 0],
        }
    )


@pytest.fixture
def preprocessor(sample_df):
    """Fitted preprocessor fixture."""
    pp = DataPreprocessor(
        numeric_features=[
            "age",
            "weight_kg",
            "heart_rate",
            "sleep_hours",
            "activity_days_week",
        ],
        categorical_features=["smoking", "nutrition_quality"],
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
