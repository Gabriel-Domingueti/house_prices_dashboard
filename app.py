import dash
from src.data.loader import load_raw, clean_data
from src.dashboard import layout, callbacks
from src.ml.model import train

df = clean_data(load_raw())
df["OverallQual"] = df["OverallQual"].astype(str)

pipeline, metrics = train(df)
print(f"Modelo treinado — R²: {metrics['r2']} | RMSE: ${metrics['rmse']:,.0f}")

app = dash.Dash(
    __name__,
    title="House Prices Dashboard",
    suppress_callback_exceptions=True,
)

app.layout = layout.create_layout(df, pipeline, metrics)

callbacks.register_callbacks(app, df, pipeline, metrics)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)