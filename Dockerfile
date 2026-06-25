FROM python:3.11

WORKDIR /app

# dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia tutto il progetto (incluso il modello)
COPY . .

# crea cartella modelli (se non esiste già)
RUN mkdir -p /app/models

# porta Flask
EXPOSE 5000

# avvio app
CMD ["python", "app.py"]