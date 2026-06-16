import pandas as pd
from dash import html, dcc
from src.analysis.eda import correlation_with_target, FEATURE_NAMES_PT
from src.visualization.charts import (
    plot_price_distribution,
    plot_area_vs_price,
    plot_correlation_heatmap,
    plot_price_by_quality,
    plot_feature_importance,
)


def create_layout(df: pd.DataFrame, pipeline, metrics: dict) -> html.Div:
    """
    Monta a estrutura completa do dashboard.
    Recebe o DataFrame já limpo e retorna o layout completo.
    Nenhuma lógica de negócio aqui — só estrutura visual.
    """
    top_correlations = correlation_with_target(df).index.tolist()

    return html.Div(
        className="container",
        children=[

            # Cabeçalho
            html.Div(className="header", children=[
                html.H1("🏠 House Prices Dashboard"),
                html.P("Análise exploratória do dataset House Prices — Kaggle"),
            ]),

            # Gráficos — linha 1
            html.Div(className="charts-row", children=[

                html.Div(className="chart-card", children=[
                    html.Div(style={"padding": "12px 12px 0 12px"}, children=[
                        html.Label("Escala", style={"fontSize": "11px", "fontWeight": "600",
                                                    "textTransform": "uppercase", "color": "#6c757d"}),
                        dcc.RadioItems(
                            id="price-scale",
                            options=[
                                {"label": "Normal", "value": "normal"},
                                {"label": "Logarítmica", "value": "log"},
                            ],
                            value="normal",
                            inline=True,
                            style={"fontSize": "13px"},
                        ),
                    ]),
                    dcc.Graph(id="price-distribution", figure=plot_price_distribution(df)),
                ]),

                html.Div(className="chart-card", children=[
                    dcc.Graph(id="price-by-quality", figure=plot_price_by_quality(df)),
                ]),

            ]),

            # Gráficos — linha 2
            html.Div(className="charts-row", children=[

                html.Div(className="chart-card", children=[
                    html.Div(style={"padding": "12px 12px 0 12px"}, children=[
                        html.Label("Colorir por", style={"fontSize": "11px", "fontWeight": "600",
                                                        "textTransform": "uppercase", "color": "#6c757d",
                                                        "marginBottom": "4px", "display": "block"}),
                        dcc.Dropdown(
                            id="scatter-color",
                            options=[
                                {"label": FEATURE_NAMES_PT.get(col, col), "value": col}
                                for col in top_correlations
                            ],
                            value="OverallQual",
                            clearable=False,
                        ),
                    ]),
                    dcc.Graph(id="area-vs-price", figure=plot_area_vs_price(df)),
                ]),

                html.Div(className="chart-card", children=[
                    html.Div(style={"padding": "12px 12px 0 12px"}, children=[
                        html.Label("Número de features", style={"fontSize": "11px", "fontWeight": "600",
                                                                "textTransform": "uppercase", "color": "#6c757d"}),
                        dcc.Slider(
                            id="heatmap-topn",
                            min=5,
                            max=15,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(5, 16)},
                        ),
                    ]),
                    dcc.Graph(id="correlation-heatmap", figure=plot_correlation_heatmap(df)),
                ]),

            ]),

            # Seção ML
            html.Div(className="header", style={"marginTop": "32px"}, children=[
                html.H2("🤖 Previsão de Preço"),
                html.P(f"Modelo: Gradient Boosting  |  R²: {metrics['r2']}  |  RMSE: ${metrics['rmse']:,.0f}  |  MAE: ${metrics['mae']:,.0f}"),
            ]),

            html.Div(className="charts-row", children=[

                # Painel de inputs
                html.Div(className="chart-card", style={"padding": "24px"}, children=[
                    html.H3("Características do imóvel", style={"marginBottom": "20px", "fontSize": "16px"}),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[

                        html.Div([
                            html.Label("Área habitável (pés²)"),
                            dcc.Input(id="input-grlivarea", type="number", value=1500, min=300, max=5000, step=50,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("Qualidade geral (1-10)"),
                            dcc.Slider(id="input-overallqual", min=1, max=10, step=1, value=7,
                                       marks={i: str(i) for i in range(1, 11)}),
                        ]),

                        html.Div([
                            html.Label("Ano de construção"),
                            dcc.Input(id="input-yearbuilt", type="number", value=2000, min=1870, max=2024, step=1,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("Área do porão (pés²)"),
                            dcc.Input(id="input-totalbsmtsf", type="number", value=800, min=0, max=3000, step=50,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("Área 1º andar (pés²)"),
                            dcc.Input(id="input-1stflrsf", type="number", value=800, min=300, max=3000, step=50,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px", "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("Carros na garagem"),
                            dcc.Slider(id="input-garagecars", min=0, max=4, step=1, value=2,
                                       marks={i: str(i) for i in range(5)}),
                        ]),

                    ]),

                    # Resultado da previsão
                    html.Div(id="prediction-output", style={
                        "marginTop": "24px",
                        "padding": "20px",
                        "background": "#f0fdf4",
                        "borderRadius": "10px",
                        "border": "1px solid #bbf7d0",
                        "textAlign": "center",
                    }),
                ]),

                # Gráfico de importância das features
                html.Div(className="chart-card", children=[
                    dcc.Graph(figure=plot_feature_importance(pipeline)),
                ]),

            ]),
        ]
    )