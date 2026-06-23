FROM python:3.11-slim

# Install system dependencies (solo quelle utili per numpy/pandas/sklearn)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Variabili d'ambiente essenziali
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# (opzionale ma utile per ML parallelo)
ENV LOKY_MAX_CPU_COUNT=4

# Copia tutto il progetto
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Porta Flask
EXPOSE 5000

# Avvio container
CMD ["python", "app.py"]