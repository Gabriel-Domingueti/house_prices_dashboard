import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.visualization.charts import (
    plot_price_distribution,
    plot_area_vs_price,
    plot_correlation_heatmap,
    plot_price_by_quality,
)


# --- Fixture ---

@pytest.fixture
def sample_df():
    """DataFrame mínimo com as colunas que os gráficos precisam."""
    return pd.DataFrame({
        "SalePrice":   [100000, 150000, 200000, 250000, 300000, 180000, 220000],
        "GrLivArea":   [800,    1000,   1200,   1500,   1800,   1100,   1300],
        "OverallQual": [5,      6,      7,      8,      9,      6,      7],
        "GarageArea":  [300,    400,    500,    600,    700,    450,    520],
        "YearBuilt":   [1990,   1995,   2000,   2005,   2010,   1998,   2003],
    })


# --- plot_price_distribution ---

def test_plot_price_distribution_returns_figure(sample_df):
    fig = plot_price_distribution(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_price_distribution_log_scale_returns_figure(sample_df):
    fig = plot_price_distribution(sample_df, log_scale=True)
    assert isinstance(fig, go.Figure)


def test_plot_price_distribution_has_title(sample_df):
    fig = plot_price_distribution(sample_df)
    assert "Preço" in fig.layout.title.text


# --- plot_area_vs_price ---

def test_plot_area_vs_price_returns_figure(sample_df):
    fig = plot_area_vs_price(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_area_vs_price_has_data(sample_df):
    fig = plot_area_vs_price(sample_df)
    assert len(fig.data) > 0


def test_plot_area_vs_price_custom_color_col(sample_df):
    fig = plot_area_vs_price(sample_df, color_col="YearBuilt")
    assert isinstance(fig, go.Figure)


# --- plot_correlation_heatmap ---

def test_plot_correlation_heatmap_returns_figure(sample_df):
    fig = plot_correlation_heatmap(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_correlation_heatmap_respects_top_n(sample_df):
    fig = plot_correlation_heatmap(sample_df, top_n=2)
    # A matriz inclui as top_n features + SalePrice = top_n + 1 colunas
    n_cols = len(fig.data[0].x)
    assert n_cols <= 4  # top_n=2 + SalePrice, mais alguma margem


# --- plot_price_by_quality ---

def test_plot_price_by_quality_returns_figure(sample_df):
    fig = plot_price_by_quality(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_price_by_quality_has_data(sample_df):
    fig = plot_price_by_quality(sample_df)
    assert len(fig.data) > 0


def test_plot_price_by_quality_has_title(sample_df):
    fig = plot_price_by_quality(sample_df)
    assert "Qualidade" in fig.layout.title.text