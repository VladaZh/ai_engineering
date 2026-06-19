import sys
import logging
import json
from pathlib import Path
import joblib

import pandas as pd

import matplotlib

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_project_root() -> Path:
    current = Path.cwd()
    for p in [current] + list(current.parents):
        if (p / "src").is_dir() and (p / "requirements.txt").is_file():
            return p
    return current


project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.preprocessor import DataPreprocessor
from src.models.evaluator import ModelEvaluator
from src.models.classifier import FitnessClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_NUMERIC = ["age", "height_cm", "weight_kg", "sleep_hours", "activity_index"]
ALLOWED_CATEGORICAL = ["smokes", "gender"]
TARGET_COLUMN = "is_fit"
METRIC_FOR_SELECTION = "roc_auc"  # Критерий выбора лучшей модели
RANDOM_STATE = 42


def train_pipeline(output_dir: Path | None = None) -> int:
    """Обучение, сравнение и сохранение лучшей модели."""
    artifacts_dir = output_dir or (project_root / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting training pipeline")

    df = pd.read_csv(project_root / "data" / "fitness_dataset.csv")

    numeric_features = [f for f in ALLOWED_NUMERIC if f in df.columns]
    categorical_features = [f for f in ALLOWED_CATEGORICAL if f in df.columns]

    preprocessor = DataPreprocessor(
        numeric_features, categorical_features, TARGET_COLUMN
    )
    X = preprocessor.fit_transform(df)
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Split: Train={len(X_train)}, Test={len(X_test)}")

    # Обучение Baseline
    logger.info("Training Baseline: Logistic Regression")
    baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    y_proba_base = baseline.predict_proba(X_test)
    metrics_base = ModelEvaluator.evaluate(y_test, y_pred_base, y_proba_base)
    logger.info(f"Baseline metrics: {metrics_base}")

    # Обучение Improved
    logger.info("Training Improved: Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    metrics_rf = ModelEvaluator.evaluate(y_test, y_pred_rf, y_proba_rf)
    logger.info(f"RF metrics: {metrics_rf}")

    # Выбор лучшей модели
    base_score = metrics_base.get(METRIC_FOR_SELECTION, 0)
    rf_score = metrics_rf.get(METRIC_FOR_SELECTION, 0)

    if rf_score > base_score:
        best_type = "random_forest"
        best_metrics = metrics_rf
        logger.info(f"Selected Random Forest ({METRIC_FOR_SELECTION}={rf_score:.4f})")
    else:
        best_type = "logistic_regression"
        best_metrics = metrics_base
        logger.info(
            f"Selected Logistic Regression ({METRIC_FOR_SELECTION}={base_score:.4f})"
        )

    logger.info("Training final model on full dataset...")
    if best_type == "random_forest":
        clf = FitnessClassifier(
            model_type="random_forest",
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
        )
    else:
        clf = FitnessClassifier(
            model_type="logistic_regression", max_iter=1000, random_state=RANDOM_STATE
        )

    clf.fit(X, y)
    model_path = artifacts_dir / "fitness_model.pkl"
    clf.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    preprocessor_path = artifacts_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    # Сохранение метрик в JSON
    metrics_path = artifacts_dir / "training_metrics.json"
    report = {
        "best_model": best_type,
        "selection_metric": METRIC_FOR_SELECTION,
        "baseline": {k: float(v) for k, v in metrics_base.items()},
        "improved": {k: float(v) for k, v in metrics_rf.items()},
        "selected_metrics": {k: float(v) for k, v in best_metrics.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(train_pipeline())
