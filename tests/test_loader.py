import pytest 
import pandas as pd
import numpy as np
from src.data.loader import (
    load_raw,
    get_numeric_columns,
    get_categorical_columns,
    clean_data,
)

# Fixtures: dados de teste reutilizáveis

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "price": [100, 200, 300, np.nan],
        "area": [50, 80, 120, 90],
        "category": ["A", "B", np.nan, "A"],
    })

@pytest.fixture
def df_with_empty_column(sample_df):
    sample_df["mostly_null"] = [np.nan, np.nan, np.nan, 1.0]
    return sample_df

# Testes de get_numeric_columns 

def test_get_numeric_columns_returns_only_numbers(sample_df):
    result = get_numeric_columns(sample_df)
    assert "price" in result
    assert "area" in result
    assert "category" not in result

# Testes de get_categorical_columns

def test_get_categorical_columns_returns_only_strings(sample_df):
    result = get_categorical_columns(sample_df)
    assert "category" in result
    assert "price" not in result

# Testes de clean_data

def test_clean_data_fills_numeric_nulls(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["price"].isna().sum() == 0

def test_clean_data_fills_categorical_nulls(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["category"].isna().sum() == 0
    assert "Unknown" in cleaned["category"].values

def test_clean_data_removes_mostly_null_columns(df_with_empty_column):
    cleaned = clean_data(df_with_empty_column)
    assert "mostly_null" not in cleaned.columns

def test_clean_data_does_not_motify_original(sample_df):
    original_nulls = sample_df["price"].isna().sum()
    clean_data(sample_df)
    assert sample_df["price"].isna().sum() == original_nulls

# Teste de load_raw

def test_load_raw_raises_if_file_missing(tmp_path, monkeypatch):
    import src.data.loader as loader_module
    monkeypatch.setattr(loader_module, "RAW_DATA_PATH", tmp_path / "nao_existe.csv")
    with pytest.raises(FileNotFoundError):
        load_raw()