"""
Repositorio para persistencia y carga de modelos.
"""
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import settings
from src.infrastructure.mlflow.mlflow_tracking import MLflowTracking


class ModelRepository:
    """
    Repositorio que gestiona la persistencia y carga de modelos desde MLflow.
    """
    
    def __init__(self):
        """Inicializa el repositorio."""
        self.mlflow_tracking = MLflowTracking()
        self.client = MlflowClient(settings.MLFLOW_TRACKING_URI)
        self._cached_model = None
        self._cached_model_uri = None
    
    def load_latest_model(self, model_name: str = None, stage: str = None) -> Optional[object]:
        """
        Carga el modelo más reciente desde MLflow Model Registry.
        
        Args:
            model_name: Nombre del modelo (por defecto usa settings.MODEL_NAME)
            stage: Etapa del modelo (por defecto usa settings.MODEL_STAGE)
            
        Returns:
            Modelo cargado o None si no se encuentra
        """
        model_name = model_name or settings.MODEL_NAME
        stage = stage or settings.MODEL_STAGE
        
        try:
            model_uri = self.mlflow_tracking.get_model(model_name, stage)
            
            # Cachear modelo si es el mismo
            if model_uri != self._cached_model_uri:
                self._cached_model = self.mlflow_tracking.load_model(model_uri, model_type="keras")
                self._cached_model_uri = model_uri
            
            return self._cached_model
        except Exception as e:
            print(f"Error al cargar modelo desde MLflow: {e}")
            return None
    
    def get_model_info(self, model_name: str = None, stage: str = None) -> Optional[dict]:
        """
        Obtiene información del modelo desde MLflow.
        
        Args:
            model_name: Nombre del modelo
            stage: Etapa del modelo
            
        Returns:
            Diccionario con información del modelo o None
        """
        model_name = model_name or settings.MODEL_NAME
        stage = stage or settings.MODEL_STAGE
        
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model_version = self.client.get_latest_versions(model_name, stages=[stage])[0]
            
            return {
                "name": model_name,
                "version": model_version.version,
                "stage": stage,
                "run_id": model_version.run_id,
                "uri": model_uri
            }
        except Exception as e:
            print(f"Error al obtener información del modelo: {e}")
            return None
    
    def clear_cache(self):
        """Limpia el caché del modelo."""
        self._cached_model = None
        self._cached_model_uri = None

