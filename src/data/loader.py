import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/train.csv")

def load_raw() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="object").columns.tolist()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove colunas muito vazias
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)

    # Preenche os valores faltantes por tipo
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df