import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """Encapsulates data cleaning and transformation logic."""

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        target_column: str = "is_fit",
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self._preprocessor = self._build_preprocessor()
        self._is_fitted = False

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build sklearn ColumnTransformer pipeline."""
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            [
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessor on training data."""
        self._preprocessor.fit(df[self.numeric_features + self.categorical_features])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data to model-ready array."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        return self._preprocessor.transform(
            df[self.numeric_features + self.categorical_features]
        )

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience method: fit + transform."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """Return transformed feature names."""
        if not self._is_fitted:
            return []
        names = []
        if hasattr(self._preprocessor.named_transformers_["num"], "named_steps"):
            names.extend(self.numeric_features)
        if hasattr(self._preprocessor.named_transformers_["cat"], "named_steps"):
            encoder = self._preprocessor.named_transformers_["cat"]["encoder"]
            names.extend(
                encoder.get_feature_names_out(self.categorical_features).tolist()
            )
        return names
