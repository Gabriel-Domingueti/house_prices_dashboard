import pandas as pd
import dash
from dash import Input, Output, html
from src.ml.model import predict, NUMERIC_FEATURES
from src.visualization.charts import (
    plot_area_vs_price,
    plot_correlation_heatmap,
    plot_price_by_quality,
    plot_price_distribution
)

def register_callbacks(app: dash.Dash, df: pd.DataFrame, pipeline, metrics: dict) -> None:
    """
    Registra todos os callbacks do dashboard.

    Cada callback conecta um controle (Input) a um gráfico (Output).
    O Dash chama essas funções automaticamente quando o usuário interage.
    """

    @app.callback(
        Output("price-distribution", "figure"),
        Input("price-scale", "value"),
    )
    def update_price_distribution(scale: str):
        log_scale = scale =="log"
        return plot_price_distribution(df, log_scale=log_scale)
    
    @app.callback(
        Output("area-vs-price", "figure"),
        Input("scatter-color", "value"),
    )
    def update_ares_vs_price(color_col: str):
        return plot_area_vs_price(df, color_col=color_col)
    
    @app.callback(
        Output("correlation-heatmap", "figure"),
        Input("heatmap-topn", "value"),
    )
    def update_correlation_heatmap(top_n: int):
        return plot_correlation_heatmap(df, top_n=top_n)

    @app.callback(
        Output("prediction-output", "children"),
        Input("input-grlivarea",   "value"),
        Input("input-overallqual", "value"),
        Input("input-yearbuilt",   "value"),
        Input("input-totalbsmtsf", "value"),
        Input("input-1stflrsf",    "value"),
        Input("input-garagecars",  "value"),
    )
    def update_prediction(grlivarea, overallqual, yearbuilt, totalbsmtsf, firstflrsf, garagecars):
        if any(v is None for v in [grlivarea, overallqual, yearbuilt, totalbsmtsf, firstflrsf, garagecars]):
            return html.P("Preencha todos os campos para ver a previsão.")

        input_data = {
            "GrLivArea":    grlivarea,
            "GarageArea":   400,        # valor médio fixo
            "TotalBsmtSF":  totalbsmtsf,
            "1stFlrSF":     firstflrsf,
            "FullBath":     2,          # valor médio fixo
            "TotRmsAbvGrd": 7,          # valor médio fixo
            "YearBuilt":    yearbuilt,
            "YearRemodAdd": yearbuilt,  # assume sem reforma
            "GarageCars":   garagecars,
            "OverallQual":  str(overallqual),
        }

        price = predict(pipeline, input_data)

        return [
            html.P("Preço estimado", style={"color": "#6c757d", "fontSize": "14px", "margin": "0"}),
            html.H2(f"${price:,.0f}", style={"color": "#16a34a", "fontSize": "36px", "margin": "4px 0"}),
            html.P(f"± ${metrics['mae']:,.0f} (MAE do modelo)", style={"color": "#6c757d", "fontSize": "12px", "margin": "0"}),
        ]