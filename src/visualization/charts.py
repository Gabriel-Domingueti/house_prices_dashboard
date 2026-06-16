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
    x_label = "log(Preço de Venda)" if log_scale else "Preço de Venda (USD)"
    titulo = "Distribuição do Preço de Venda — escala logarítmica" if log_scale else "Distribuição do Preço de Venda"

    fig = px.histogram(
        x=values,
        nbins=50,
        labels={"x": x_label},
        title=titulo,
    )

    fig.update_traces(
        marker_color="#636EFA",
        marker_line_color="#4046c4",
        marker_line_width=0.8,
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Quantidade de imóveis",
        bargap=0.05,
        shapes=[{
            "type": "line",
            "x0": df["SalePrice"].median() if not log_scale else np.log1p(df["SalePrice"].median()),
            "x1": df["SalePrice"].median() if not log_scale else np.log1p(df["SalePrice"].median()),
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#ef4444", "width": 1.5, "dash": "dash"},
        }],
        annotations=[{
            "x": df["SalePrice"].median() if not log_scale else np.log1p(df["SalePrice"].median()),
            "y": 1,
            "yref": "paper",
            "text": f"Mediana: ${df['SalePrice'].median():,.0f}",
            "showarrow": False,
            "xanchor": "left",
            "xshift": 6,
            "font": {"size": 11, "color": "#ef4444"},
        }],
    )

    return fig


def plot_area_vs_price(df: pd.DataFrame, color_col: str = "OverallQual") -> go.Figure:
    """
    Scatter plot de GrLivArea vs SalePrice.

    color_col permite colorir os pontos por uma terceira variável —
    por padrão OverallQual, o preditor mais forte do dataset.
    Isso revela padrões que um scatter simples esconderia.
    """

    nome_pt = FEATURE_NAMES_PT.get(color_col, color_col)

    category_orders = {}
    if color_col == "OverallQual":
        category_orders = {"OverallQual": [str(i) for i in range(1, 11)]}

    fig = px.scatter(
        df,
        x="GrLivArea",
        y="SalePrice",
        color=color_col,
        opacity=0.6,
        size_max=8,
        title=f"Área Habitável vs Preço de Venda - colorido por {nome_pt}",
        labels={
            "GrLivArea": "Área habitável (pés²)",
            "SalePrice": "Preço de venda (USD)",
            color_col:  nome_pt,
        },
        color_continuous_scale="Viridis",
        category_orders=category_orders,
    )

    fig.update_traces(marker=dict(size=7))

    fig.update_layout(
        xaxis_title="Área habitável (pés²)",
        yaxis_title="Preço de venda (USD)",
        coloraxis_colorbar=dict(
            title=nome_pt,
        ),
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
    values = corr_matrix.values

    text_colors = [
        ["white" if abs(v) > 0.5 else "black" for v in row]
        for row in values
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
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

    annotations = []
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            annotations.append({
                "x": labels_pt[j],
                "y": labels_pt[i],
                "text": f"{val:.2f}",
                "font": {"color": text_colors[i][j], "size": 10},
                "showarrow": False,
            })

    fig.update_layout(
        title=f"Correlação entre as top {top_n} features mais relevantes",
        xaxis={"tickangle": -45},
        height=500,
        annotations=annotations,
    )

    fig.update_traces(text=None, texttemplate=None)

    return fig


def plot_price_by_quality(df: pd.DataFrame) -> go.Figure:
    """
    Box plot do SalePrice agrupado por OverallQual.

    Box plot mostra mediana, quartis e outliers — muito mais informativo
    que uma média simples. Aqui vemos como a qualidade (1-10)
    impacta não só o preço médio, mas a variação dentro de cada nota.
    """
    quality_labels = {
        "1":  "1 - Muito Ruim",
        "2":  "2 - Ruim",
        "3":  "3 - Regular",
        "4":  "4 - Abaixo da Média",
        "5":  "5 - Média",
        "6":  "6 - Acima da Média",
        "7":  "7 - Boa",
        "8":  "8 - Muito Boa",
        "9":  "9 - Excelente",
        "10": "10 - Superior",
    }

    df_plot = df.copy()
    df_plot["OverallQual"] = df_plot["OverallQual"].astype(str)
    df_plot["Qualidade"] = df_plot["OverallQual"].map(quality_labels)

    fig = px.box(
        df_plot,
        x="Qualidade",
        y="SalePrice",
        title="Preço de Venda por Qualidade de Construção",
        labels={
            "Qualidade": "",
            "SalePrice": "Preço de venda (USD)",
        },
        color="Qualidade",
        category_orders={"Qualidade": list(quality_labels.values())},
    )

    fig.update_layout(
        yaxis_title="Preço de venda (USD)",
        showlegend=False,
        xaxis_tickangle=30,
    )

    return fig

def plot_feature_importance(pipeline) -> go.Figure:
    """
    Gráfico de barras horizontais com a importância de cada feature.
    Valores exibidos diretamente nas barras para leitura imediata.
    """

    from src.ml.model import get_feature_importance

    importance = get_feature_importance(pipeline)
    labels_pt = [FEATURE_NAMES_PT.get(f, f) for f in importance.index]
    values = importance.values

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=values,
        y=labels_pt,
        orientation="h",
        marker_color=values,
        marker_colorscale="Viridis",
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))

    fig.update_traces(cliponaxis=False)

    fig.update_layout(
        title="O que mais influencia o preço?",
        xaxis=dict(
            title="Importância relativa do modelo",
            tickformat=".0%",
            range=[0, max(values) * 1.25],
        ),
        yaxis=dict(
            title="",
            categoryorder="total ascending",
            automargin=True,
        ),
        height=420,
        margin=dict(l=10, r=60, t=50, b=40),
        showlegend=False,
    )

    return fig