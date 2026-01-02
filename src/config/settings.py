"""
Configuración del proyecto.
"""
import os
from pathlib import Path


class Settings:
    """Configuración centralizada del proyecto."""
    
    # Rutas del proyecto
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    MLRUNS_DIR = BASE_DIR / "mlruns"
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLRUNS_DIR", "churn_prediction_nested_cv")
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Modelo
    MODEL_NAME = "churn_deep_learning_model"
    MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
    
    # Dataset
    DATASET_PATH = DATA_DIR / "churn_data.csv"
    
    @classmethod
    def ensure_directories(cls):
        """Asegura que los directorios necesarios existan."""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.MLRUNS_DIR.mkdir(exist_ok=True)


# Instancia global de configuración
settings = Settings()

