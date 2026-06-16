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


def create_layout(df: pd.DataFrame, pipeline, metrics: dict, top_correlations: list) -> html.Div:
    """
    Monta a estrutura completa do dashboard.
    Recebe o DataFrame já limpo e retorna o layout completo.
    top_correlations é calculado antes da conversão de OverallQual para string.
    Nenhuma lógica de negócio aqui — só estrutura visual.
    """

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
                        html.Label("Escala do histograma", style={"fontSize": "11px", "fontWeight": "600",
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
                                if col != "GrLivArea"
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
                # Painel de inputs
                html.Div(className="chart-card", style={"padding": "24px"}, children=[
                    html.H3("Simule o preço de um imóvel", style={"marginBottom": "4px", "fontSize": "16px"}),
                    html.P("Ajuste as características abaixo e veja a previsão atualizar em tempo real.",
                           style={"fontSize": "13px", "color": "#6c757d", "marginBottom": "20px"}),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[

                        html.Div([
                            html.Label("🏠 Área habitável", style={"fontWeight": "600"}),
                            html.P("Espaço interno excluindo porão (334 – 5.642 pés²)",
                                   style={"fontSize": "11px", "color": "#6c757d", "margin": "2px 0 6px"}),
                            dcc.Input(id="input-grlivarea", type="number", value=1500,
                                      min=334, max=5642, step=50,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px",
                                             "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("⭐ Qualidade de construção", style={"fontWeight": "600"}),
                            html.P("1 = Muito ruim  →  10 = Acabamento superior",
                                   style={"fontSize": "11px", "color": "#6c757d", "margin": "2px 0 6px"}),
                            dcc.Slider(id="input-overallqual", min=1, max=10, step=1, value=7,
                                       marks={1: "1", 3: "Regular", 5: "Média", 7: "Boa", 10: "10"},
                                       tooltip={"placement": "bottom", "always_visible": True}),
                        ]),

                        html.Div([
                            html.Label("📅 Ano de construção", style={"fontWeight": "600"}),
                            html.P("Imóveis no dataset: 1872 – 2010",
                                   style={"fontSize": "11px", "color": "#6c757d", "margin": "2px 0 6px"}),
                            dcc.Input(id="input-yearbuilt", type="number", value=2000,
                                      min=1872, max=2010, step=1,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px",
                                             "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("⬇️ Área do porão", style={"fontWeight": "600"}),
                            html.P("0 se não tiver porão (0 – 6.110 pés²)",
                                   style={"fontSize": "11px", "color": "#6c757d", "margin": "2px 0 6px"}),
                            dcc.Input(id="input-totalbsmtsf", type="number", value=800,
                                      min=0, max=6110, step=50,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px",
                                             "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("📐 Área do 1º andar", style={"fontWeight": "600"}),
                            html.P("Geralmente igual ou maior que a área habitável (334 – 4.692 pés²)",
                                   style={"fontSize": "11px", "color": "#6c757d", "margin": "2px 0 6px"}),
                            dcc.Input(id="input-1stflrsf", type="number", value=800,
                                      min=334, max=4692, step=50,
                                      style={"width": "100%", "padding": "8px", "borderRadius": "6px",
                                             "border": "1px solid #dee2e6"}),
                        ]),

                        html.Div([
                            html.Label("🚗 Vagas na garagem", style={"fontWeight": "600"}),
                            html.P("Capacidade da garagem em número de carros",
                                   style={"fontSize": "11px", "color": "#6c757d", "margin": "2px 0 6px"}),
                            dcc.Slider(id="input-garagecars", min=0, max=4, step=1, value=2,
                                       marks={0: "Sem garagem", 1: "1", 2: "2", 3: "3", 4: "4"},
                                       tooltip={"placement": "bottom", "always_visible": True}),
                        ]),

                    ]),

                    # Resultado
                    html.Div(id="prediction-output", style={
                        "marginTop": "24px",
                        "padding": "20px",
                        "background": "#f0fdf4",
                        "borderRadius": "10px",
                        "border": "1px solid #bbf7d0",
                        "textAlign": "center",
                    }),

                    # Aviso sobre unidades
                    html.P("💡 Áreas em pés² (1 m² ≈ 10,76 pés²). Um imóvel de 150 m² equivale a ~1.614 pés².",
                           style={"fontSize": "11px", "color": "#6c757d", "marginTop": "12px",
                                  "padding": "8px", "background": "#f8f9fa", "borderRadius": "6px"}),
                ]),

                # Gráfico de importância das features
                html.Div(className="chart-card", children=[
                    dcc.Graph(figure=plot_feature_importance(pipeline)),
                ]),

            ]),
        ]
    )