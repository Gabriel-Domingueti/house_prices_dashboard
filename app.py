import dash
import os
from src.data.loader import load_raw, clean_data
from src.dashboard import layout, callbacks
from src.ml.model import train
from src.analysis.eda import correlation_with_target

df = clean_data(load_raw())

top_correlations = correlation_with_target(df).index.tolist() 

df["OverallQual"] = df["OverallQual"].astype(str)

pipeline, metrics = train(df)

app = dash.Dash(
    __name__,
    title="House Prices Dashboard",
    suppress_callback_exceptions=True,
)

app.layout = layout.create_layout(df, pipeline, metrics, top_correlations)

callbacks.register_callbacks(app, df, pipeline, metrics)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)