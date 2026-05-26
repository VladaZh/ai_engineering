import pytest
import numpy as np
from src.models.preprocessor import DataPreprocessor


def test_preprocessor_fit_transform(sample_df, preprocessor):
    """Test that fit_transform returns array of correct shape."""
    X = preprocessor.fit_transform(sample_df)
    assert X.shape[0] == len(sample_df)
    assert X.ndim == 2


def test_preprocessor_not_fitted_raises(sample_df):
    """Test that transform before fit raises error."""
    pp = DataPreprocessor(numeric_features=["age"], categorical_features=[])
    with pytest.raises(RuntimeError):
        pp.transform(sample_df)


def test_preprocessor_handles_missing(sample_df):
    """Test imputation of missing values."""
    sample_df_missing = sample_df.copy()
    sample_df_missing.loc[0, "age"] = np.nan
    pp = DataPreprocessor(numeric_features=["age"], categorical_features=[])
    X = pp.fit_transform(sample_df_missing)
    assert not np.isnan(X).any()
