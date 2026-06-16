import pandas as pd
import numpy as np

FEATURE_NAMES_PT = {
    "SalePrice":     "Preço de Venda",
    "OverallQual":   "Qualidade Geral",
    "GrLivArea":     "Área Habitável",
    "GarageCars":    "Vagas na Garagem",
    "GarageArea":    "Área da Garagem",
    "TotalBsmtSF":   "Área do Porão",
    "1stFlrSF":      "Área 1º Andar",
    "FullBath":      "Banheiros Completos",
    "TotRmsAbvGrd":  "Total de Cômodos",
    "YearBuilt":     "Ano de Construção",
    "YearRemodAdd":  "Ano de Reforma",
    "MasVnrArea":    "Área de Alvenaria",
    "BsmtFinSF1":    "Porão Acabado",
    "LotArea":       "Área do Terreno",
    "LotFrontage":   "Frente do Terreno",
}

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna estatísticas descritivas apenas das colunas numéricas:
    média, mediana, desvio padrão, min, max e skewness.
    
    Skewness (assimetria) indica se a distribuição é equilibrada:
      - próximo de 0: distribuição simétrica
      - positivo alto: cauda longa à direita (ex: preços de imóveis)
      - negativo alto: cauda longa à esquerda
    """
    df_numeric = df.select_dtypes(include="number")

    stats = pd.DataFrame({
        "mean": df_numeric.mean(),
        "median": df_numeric.median(),
        "std": df_numeric.std(),
        "min": df_numeric.min(),
        "max": df_numeric.max(),
        "skew": df_numeric.skew(),
    })

    return stats.round(2)

def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna um relatório de valores ausentes por coluna, ordenado
    do mais crítico para o menos crítico.
    Inclui contagem absoluta e percentual.
    """
    total= len(df)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / total * 100).round(2)

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    })

    report = report[report["missing_count"] > 0]

    return report.sort_values("missing_pct", ascending=False)

def correlation_with_target(
        df: pd.DataFrame,
        target: str = "SalePrice",
        top_n: int = 10,
) -> pd.Series:
    """
    Retorna as top_n colunas numéricas com maior correlação
    (positiva ou negativa) com a variável alvo.
    
    Correlação de Pearson vai de -1 a 1:
      -  1.0: correlação positiva perfeita
      - -1.0: correlação negativa perfeita
      -  0.0: nenhuma correlação linear
    """
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada no DataFrame.")
    
    df_numeric = df.select_dtypes(include="number")
    correlations = df_numeric.corr()[target].drop(target)

    return correlations.abs().sort_values(ascending=False).head(top_n)

def price_distribution_stats(
        df: pd.DataFrame,
        target: str = "SalePrice",
) -> dict:
    """
    Retorna estatísticas específicas da distribuição do preço:
    útil para decidir se aplicar transformação logarítmica.
    
    Imóveis costumam ter distribuição com skew positivo alto —
    alguns poucos imóveis caríssimos puxam a média pra cima.
    Aplicar log() deixa a distribuição mais simétrica e melhora
    a performance de modelos de regressão linear.
    """
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada no DataFrame.")

    col = df[target].dropna()

    return {
        "mean": round(col.mean(), 2),
        "median": round(col.median(), 2),
        "skew": round(col.skew(), 2),
        "kurtosis": round(col.kurtosis(), 2),
        "log_skew": round(np.log1p(col).skew(), 2),
    } 

def categorical_value_counts(
        df: pd.DataFrame,
        column: str,
        top_n: int = 10,
) -> pd.Series:
    """
    Retorna a frequência dos top_n valores de uma coluna categórica.
    Útil para entender quais categorias dominam os dados.
    """
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame.")

    return df[column].value_counts().head(top_n) 