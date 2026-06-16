import pytest
import pandas as pd
import numpy as np
from src.analysis.eda import (
    summary_stats,
    missing_values_report,
    correlation_with_target,
    price_distribution_stats,
    categorical_value_counts,
)


# --- Fixtures ---

@pytest.fixture
def sample_df():
    """DataFrame com características similares ao House Prices."""
    return pd.DataFrame({
        "SalePrice": [100000, 150000, 200000, 250000, 300000],
        "GrLivArea": [800,    1000,   1200,   1500,   1800],
        "YearBuilt": [1990,   1985,   2000,   2010,   2015],
        "Neighborhood": ["A", "B", "A", "C", "B"],
        "GarageType":   ["Att", "Det", "Att", np.nan, "Att"],
    })


@pytest.fixture
def df_with_missing(sample_df):
    sample_df.loc[0, "SalePrice"] = np.nan
    sample_df.loc[1, "GrLivArea"] = np.nan
    return sample_df


# --- summary_stats ---

def test_summary_stats_returns_expected_columns(sample_df):
    result = summary_stats(sample_df)
    for col in ["mean", "median", "std", "min", "max", "skew"]:
        assert col in result.columns


def test_summary_stats_only_numeric(sample_df):
    result = summary_stats(sample_df)
    assert "Neighborhood" not in result.index
    assert "SalePrice" in result.index


# --- missing_values_report ---

def test_missing_values_report_finds_missing(df_with_missing):
    report = missing_values_report(df_with_missing)
    assert "GarageType" in report.index


def test_missing_values_report_excludes_complete_columns(sample_df):
    # sample_df tem NaN só em GarageType
    report = missing_values_report(sample_df)
    assert "SalePrice" not in report.index
    assert "GrLivArea" not in report.index


def test_missing_values_report_sorted_descending(df_with_missing):
    report = missing_values_report(df_with_missing)
    pcts = report["missing_pct"].tolist()
    assert pcts == sorted(pcts, reverse=True)


# --- correlation_with_target ---

def test_correlation_with_target_returns_series(sample_df):
    result = correlation_with_target(sample_df, target="SalePrice")
    assert isinstance(result, pd.Series)


def test_correlation_with_target_excludes_target_itself(sample_df):
    result = correlation_with_target(sample_df, target="SalePrice")
    assert "SalePrice" not in result.index


def test_correlation_with_target_raises_if_target_missing(sample_df):
    with pytest.raises(ValueError, match="não encontrada"):
        correlation_with_target(sample_df, target="ColunaNaoExiste")


def test_correlation_with_target_respects_top_n(sample_df):
    result = correlation_with_target(sample_df, target="SalePrice", top_n=2)
    assert len(result) <= 2


# --- price_distribution_stats ---

def test_price_distribution_stats_returns_expected_keys(sample_df):
    result = price_distribution_stats(sample_df)
    for key in ["mean", "median", "skew", "kurtosis", "log_skew"]:
        assert key in result


def test_price_distribution_stats_raises_if_target_missing(sample_df):
    with pytest.raises(ValueError, match="não encontrada"):
        price_distribution_stats(sample_df, target="ColunaNaoExiste")


# --- categorical_value_counts ---

def test_categorical_value_counts_returns_series(sample_df):
    result = categorical_value_counts(sample_df, column="Neighborhood")
    assert isinstance(result, pd.Series)


def test_categorical_value_counts_respects_top_n(sample_df):
    result = categorical_value_counts(sample_df, column="Neighborhood", top_n=2)
    assert len(result) <= 2


def test_categorical_value_counts_raises_if_column_missing(sample_df):
    with pytest.raises(ValueError, match="não encontrada"):
        categorical_value_counts(sample_df, column="ColunaNaoExiste")