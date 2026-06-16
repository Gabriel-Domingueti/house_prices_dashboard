import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.ml.model import (
    build_pipeline,
    prepare_data,
    train,
    predict,
    get_feature_importance,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


# --- Fixture ---

@pytest.fixture
def sample_df():
    """
    DataFrame com dados suficientes para treinar e testar.
    Precisa de mais linhas que os outros testes porque o train_test_split
    exige pelo menos algumas amostras em cada split.
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "GrLivArea":    np.random.randint(800, 2500, n),
        "GarageArea":   np.random.randint(200, 800, n),
        "TotalBsmtSF":  np.random.randint(500, 1500, n),
        "1stFlrSF":     np.random.randint(500, 1500, n),
        "FullBath":     np.random.randint(1, 4, n),
        "TotRmsAbvGrd": np.random.randint(4, 10, n),
        "YearBuilt":    np.random.randint(1950, 2020, n),
        "YearRemodAdd": np.random.randint(1970, 2020, n),
        "GarageCars":   np.random.randint(1, 4, n),
        "OverallQual":  np.random.randint(3, 10, n),
        "SalePrice":    np.random.randint(100000, 400000, n),
    })

    # Converte OverallQual para string (igual ao clean_data real)
    df["OverallQual"] = df["OverallQual"].astype(str)
    return df


# --- build_pipeline ---

def test_build_pipeline_has_preprocessor_and_model():
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

# --- prepare_data ---

def test_prepared_data_returns_correct_shapes(sample_df):
    X, y = prepare_data(sample_df)
    assert X.shape[0] == len(sample_df)
    assert len(y) == len(sample_df)

def test_prepare_data_X_has_correct_columns(sample_df):
    X, _ = prepare_data(sample_df)
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert list(X.columns) == expected


def test_prepare_data_y_is_log_transformed(sample_df):
    _, y = prepare_data(sample_df)
    # log1p(100000) ≈ 11.5 — valores muito menores que os preços originais
    assert y.max() < 20


# --- train ---

def test_train_returns_pipeline_and_metrics(sample_df):
    pipeline, metrics = train(sample_df)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(metrics, dict)


def test_train_metrics_have_expected_keys(sample_df):
    _, metrics = train(sample_df)
    for key in ["r2", "rmse", "mae"]:
        assert key in metrics


def test_train_r2_is_valid_range(sample_df):
    _, metrics = train(sample_df)
    assert -1 <= metrics["r2"] <= 1


def test_train_rmse_is_positive(sample_df):
    _, metrics = train(sample_df)
    assert metrics["rmse"] > 0


# --- predict ---

def test_predict_returns_float(sample_df):
    pipeline, _ = train(sample_df)
    input_data = {
        "GrLivArea": 1500, "GarageArea": 400, "TotalBsmtSF": 800,
        "1stFlrSF": 800, "FullBath": 2, "TotRmsAbvGrd": 7,
        "YearBuilt": 2000, "YearRemodAdd": 2005, "GarageCars": 2,
        "OverallQual": "7",
    }
    result = predict(pipeline, input_data)
    assert isinstance(result, float)


def test_predict_returns_positive_price(sample_df):
    pipeline, _ = train(sample_df)
    input_data = {
        "GrLivArea": 1500, "GarageArea": 400, "TotalBsmtSF": 800,
        "1stFlrSF": 800, "FullBath": 2, "TotRmsAbvGrd": 7,
        "YearBuilt": 2000, "YearRemodAdd": 2005, "GarageCars": 2,
        "OverallQual": "7",
    }
    result = predict(pipeline, input_data)
    assert result > 0


# --- get_feature_importance ---

def test_get_feature_importance_returns_series(sample_df):
    pipeline, _ = train(sample_df)
    importance = get_feature_importance(pipeline)
    assert isinstance(importance, pd.Series)


def test_get_feature_importance_sorted_descending(sample_df):
    pipeline, _ = train(sample_df)
    importance = get_feature_importance(pipeline)
    values = importance.tolist()
    assert values == sorted(values, reverse=True)


def test_get_feature_importance_has_all_features(sample_df):
    pipeline, _ = train(sample_df)
    importance = get_feature_importance(pipeline)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(importance.index) == expected