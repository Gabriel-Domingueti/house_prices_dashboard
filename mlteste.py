from src.data.loader import load_raw, clean_data
from src.ml.model import train, get_feature_importance

df = clean_data(load_raw())

# OverallQual precisa ser string (igual ao plot_price_by_quality)
df["OverallQual"] = df["OverallQual"].astype(str)

pipeline, metrics = train(df)

print("=== Métricas ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\n=== Importância das features ===")
print(get_feature_importance(pipeline))