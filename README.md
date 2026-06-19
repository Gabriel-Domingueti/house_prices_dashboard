# 🏠 House Prices Dashboard

Dashboard interativo de análise exploratória e previsão de preços de imóveis, construído com **Dash**, **Plotly**, **Pandas** e **scikit-learn**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-2.17-blue?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/testes-45%20passed-brightgreen?logo=pytest)
![Deploy](https://img.shields.io/badge/deploy-Render-46E3B7?logo=render&logoColor=white)

### 🔗 [Acesse o dashboard ao vivo](https://house-prices-dashboard.onrender.com)

> ⚠️ Hospedado no plano gratuito do Render — o app "dorme" após 15 minutos sem acesso. Se a primeira tela demorar a carregar, aguarde de 30 a 50 segundos enquanto o serviço reinicia.

---

## 📊 Sobre o projeto

Este projeto utiliza o dataset [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) do Kaggle para construir um dashboard completo que permite:

- Explorar a distribuição dos preços de venda em escala normal e logarítmica
- Analisar correlações entre as features e o preço
- Visualizar como a qualidade geral impacta o valor dos imóveis
- Prever o preço de um imóvel em tempo real com um modelo de Gradient Boosting (R² 0.88)

---

## 🖥️ Screenshots

| EDA | Previsão |
|-----|---------|
| Histograma, box plot, scatter e heatmap interativos | Painel de inputs com previsão em tempo real |

---

## 🗂️ Estrutura do projeto

```
house_prices_dashboard/
│
├── data/
│   ├── raw/               # CSV original do Kaggle (versionado)
│   └── processed/         # Dados processados
│
├── src/
│   ├── data/
│   │   └── loader.py      # Carregamento e limpeza dos dados
│   │
│   ├── analysis/
│   │   └── eda.py         # Funções de análise exploratória
│   │
│   ├── visualization/
│   │   └── charts.py      # Gráficos Plotly
│   │
│   ├── ml/
│   │   └── model.py       # Pipeline sklearn + métricas
│   │
│   └── dashboard/
│       ├── layout.py      # Estrutura HTML do dashboard
│       └── callbacks.py   # Interatividade
│
├── tests/                 # Espelha a estrutura de src/
│   ├── test_loader.py
│   ├── test_eda.py
│   ├── test_charts.py
│   └── test_model.py
│
├── assets/
│   └── style.css          # Estilos do dashboard
│
├── app.py                 # Ponto de entrada
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Como rodar

### Pré-requisitos

- [Docker](https://www.docker.com/) instalado, **ou**
- Python 3.11+ com [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 1. Clone o repositório

```bash
git clone https://github.com/Gabriel-Domingueti/house_prices_dashboard.git
cd house_prices_dashboard
```

O dataset (`train.csv`) já vem versionado no repositório em `data/raw/`, então não é necessário baixar nada manualmente.

### 2. Rode com Docker

```bash
docker compose up --build
```

Acesse em [http://localhost:8050](http://localhost:8050)

### 3. Ou rode localmente com Miniconda

```bash
pip install -r requirements.txt
python app.py
```

---

## 🧪 Testes

```bash
# Rodar todos os testes com cobertura
pytest tests/ -v --cov=src --cov-report=term-missing
```

O projeto mantém cobertura de testes em todos os módulos principais:

| Módulo | Testes | Cobertura |
|--------|--------|-----------|
| `loader.py` | 7 | 95% |
| `eda.py` | 14 | 100% |
| `charts.py` | 11 | 100% |
| `model.py` | 13 | 100% |

---

## 🤖 Modelo de Machine Learning

| Métrica | Valor |
|---------|-------|
| Algoritmo | Gradient Boosting Regressor |
| R² | 0.88 |
| RMSE | $30,312 |
| MAE | $18,838 |

**Decisões de projeto:**
- O target `SalePrice` é transformado com `log1p` antes do treino — o skew cai de 1.88 para 0.12, o que melhora significativamente a performance da regressão
- `ColumnTransformer` aplica `StandardScaler` nas features numéricas e `OrdinalEncoder` na qualidade geral
- O pipeline evita data leakage ao aprender as transformações apenas no conjunto de treino

---

## 📦 Tecnologias

| Biblioteca | Uso |
|------------|-----|
| `dash` | Framework do dashboard |
| `plotly` | Gráficos interativos |
| `pandas` | Manipulação dos dados |
| `numpy` | Transformações numéricas |
| `scikit-learn` | Pipeline de ML |
| `pytest` | Testes automatizados |
| `Docker` | Containerização |

---

## ☁️ Deploy

O dashboard está hospedado gratuitamente no [Render](https://render.com), usando o `Dockerfile` do projeto diretamente — sem necessidade de configuração adicional além de tornar a porta dinâmica (`os.environ.get("PORT")`) para compatibilidade com a plataforma.

---

## 📄 Licença

MIT