"""Model evaluation utilities."""

import logging
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute and log classification metrics."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_proba is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        logger.info(f"Metrics: {metrics}")
        return metrics

    @staticmethod
    def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print confusion matrix to logs."""
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
