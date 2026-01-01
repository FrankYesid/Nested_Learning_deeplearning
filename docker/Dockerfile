# Dockerfile para la API de predicción de churn
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY src/ ./src/
COPY data/ ./data/

# Crear directorios necesarios
RUN mkdir -p models mlruns

# Exponer puerto de la API
EXPOSE 8000

# Variable de entorno para MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Comando para ejecutar la API
CMD ["python", "-m", "uvicorn", "src.infrastructure.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

