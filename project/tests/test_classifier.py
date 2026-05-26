import pytest
from src.models.classifier import FitnessClassifier


def test_classifier_predict_requires_fit(preprocessor, sample_df):
    """Test that predict before fit raises error."""
    clf = FitnessClassifier()
    X = preprocessor.transform(sample_df)
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_classifier_save_load(tmp_path, trained_classifier, sample_df, preprocessor):
    """Test model persistence."""
    model_path = tmp_path / "model.pkl"
    trained_classifier.save(model_path)

    new_clf = FitnessClassifier().load(model_path)
    X = preprocessor.transform(sample_df)
    preds = new_clf.predict(X)
    assert len(preds) == len(sample_df)
    assert set(preds).issubset({0, 1})
