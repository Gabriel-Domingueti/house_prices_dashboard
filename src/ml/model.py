import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Features selecionadas com base nas correlações do EDA
NUMERIC_FEATURES = [
    "GrLivArea", "GarageArea", "TotalBsmtSF",
    "1stFlrSF", "FullBath", "TotRmsAbvGrd",
    "YearBuilt", "YearRemodAdd", "GarageCars",
] 

CATEGORICAL_FEATURES = [
    "OverallQual",
]

TARGET = "SalePrice"

def build_pipeline() -> Pipeline:
    """
    Monta o pipeline de pré-processamento + modelo.

    ColumnTransformer aplica transformações diferentes por tipo de coluna:
    - Numéricas: StandardScaler (média 0, desvio padrão 1)
    - Categóricas ordinais: OrdinalEncoder (1-10 vira 0-9 internamente)

    GradientBoosting foi escolhido por ser robusto a outliers e
    não exigir que as features sejam perfeitamente normais.
    """

    numeric_transformer = StandardScaler()
    
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )),
    ])

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features e target, aplicando log1p no target.
    Retorna X com as features selecionadas e y em escala logarítmica.
    """
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features].copy()
    y = np.log1p(df[TARGET])  # log(SalePrice) para corrigir o skew
    return X, y

def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """
    Treina o pipeline e retorna o modelo treinado + métricas de avaliação.

    Usa 80% dos dados para treino e 20% para teste.
    As métricas de teste refletem performance em dados nunca vistos.
    """

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline =  build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)

    metrics = {
        "r2": round(r2_score(y_test_real, y_pred_real), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test_real, y_pred_real)), 2),
        "mae": round(mean_absolute_error(y_test_real, y_pred_real), 2),
    }

    return pipeline, metrics

def predict(pipeline: Pipeline, input_data: dict) -> float:
    """
    Faz uma previsão para um imóvel com as features fornecidas.
    Retorna o preço previsto em dólares (já convertido do log).

    input_data deve conter as mesmas features de NUMERIC_FEATURES
    e CATEGORICAL_FEATURES.
    """

    X = pd.DataFrame([input_data])
    log_price = pipeline.predict(X)[0]

    return round(np.expm1(log_price), 2)

def get_feature_importance(pipeline: Pipeline) -> pd.Series:
    """
    Retorna a importância de cada feature no modelo treinado.
    Só funciona após o pipeline ter sido treinado (fit).
    """

    model = pipeline.named_steps["model"]
    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    )

    return importance.sort_values(ascending=False)