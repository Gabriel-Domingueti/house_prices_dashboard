FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia e instala dependências primeiro (cache inteligente do Docker)
# Se você não mudar o requirements.txt, essa camada fica em cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Porta que o Dash usa por padrão
EXPOSE 8050

# Comando padrão ao iniciar o container
CMD ["python", "app.py"]
