import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.analysis.eda import FEATURE_NAMES_PT

def plot_price_distribution(df: pd.DataFrame, log_scale: bool = False) -> go.Figure:
    """
    Histograma da distribuição do SalePrice.
    
    log_scale=True aplica log1p nos valores antes de plotar —
    útil para visualizar a distribuição real sem o efeito dos outliers.
    """
    values = np.log1p(df["SalePrice"]) if log_scale else df["SalePrice"]
    x_label = "log(SalePrice)" if log_scale else "SalePrice (USD)"

    fig = px.histogram(
        x=values,
        nbins=50,
        labels={"x": x_label},
        title="Distribuição do Preço de Venda",
    )

    fig.update_traces(marker_color="#636EFA", marker_line_width=0.5)
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Quantidade de imóveis",
        bargap=0.05,
    )

    return fig


def plot_area_vs_price(df: pd.DataFrame, color_col: str = "OverallQual") -> go.Figure:
    """
    Scatter plot de GrLivArea vs SalePrice.

    color_col permite colorir os pontos por uma terceira variável —
    por padrão OverallQual, o preditor mais forte do dataset.
    Isso revela padrões que um scatter simples esconderia.
    """
    fig = px.scatter(
        df,
        x="GrLivArea",
        y="SalePrice",
        color=color_col,
        opacity=0.6,
        title="Área Habitável vs Preço de Venda",
        labels={
            "GrLivArea": "Área habitável (pés²)",
            "SalePrice": "Preço de venda (USD)",
            color_col: color_col,
        },
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        xaxis_title="Área habitável (pés²)",
        yaxis_title="Preço de venda (USD)",
    )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Heatmap das top_n features mais correlacionadas com SalePrice.

    Usamos go.Heatmap (graph_objects) em vez de px porque precisamos
    de controle fino sobre a matriz — anotações, ordem das colunas,
    e colorscale divergente centrada em zero.
    """
    df_numeric = df.select_dtypes(include="number")
    correlations = df_numeric.corr()["SalePrice"].drop("SalePrice")
    top_features = correlations.abs().sort_values(ascending=False).head(top_n).index.tolist()

    # Matriz de correlação apenas das top features + SalePrice
    cols = top_features + ["SalePrice"]
    corr_matrix = df_numeric[cols].corr()

    labels_pt = [FEATURE_NAMES_PT.get(c, c) for c in corr_matrix.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=labels_pt,
            y=labels_pt,
            colorscale="RdBu",
            zmid=0,           # centraliza o colorscale em 0
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title=f"Correlação entre as top {top_n} features",
        xaxis={"tickangle": -45},
        height=500,
    )

    return fig


def plot_price_by_quality(df: pd.DataFrame) -> go.Figure:
    """
    Box plot do SalePrice agrupado por OverallQual.

    Box plot mostra mediana, quartis e outliers — muito mais informativo
    que uma média simples. Aqui vemos como a qualidade (1-10)
    impacta não só o preço médio, mas a variação dentro de cada nota.
    """
    fig = px.box(
        df,
        x="OverallQual",
        y="SalePrice",
        title="Distribuição de Preço por Qualidade Geral",
        labels={
            "OverallQual": "Qualidade geral (1-10)",
            "SalePrice": "Preço de venda (USD)",
        },
        color="OverallQual",
        category_orders={"OverallQual": [str(i) for i in range(1, 11)]},
    )

    fig.update_layout(
        xaxis_title="Qualidade geral (1-10)",
        yaxis_title="Preço de venda (USD)",
        showlegend=False,
    )

    return fig

def plot_feature_importance(pipeline) -> go.Figure:
    """
    Gráfico de barras horizontais com a importância de cada feature.
    Barras horizontais são mais legíveis que verticais quando os
    nomes das features são longos.
    """

    from src.ml.model import get_feature_importance

    importance = get_feature_importance(pipeline)

    labels_pt = [FEATURE_NAMES_PT.get(f, f) for f in importance.index]

    fig = px.bar(
        x=importance.values,
        y=labels_pt,
        orientation="h",
        title="Importância das Features",
        labels={"x": "Importância", "y": "Feature"},
        color=importance.values,
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
    )

    return fig