FROM python:3.11-slim

WORKDIR /app

# Copie tous les fichiers nécessaires
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 3001
