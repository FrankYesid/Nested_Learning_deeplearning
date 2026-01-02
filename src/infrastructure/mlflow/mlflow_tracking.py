"""
Integración con MLflow para tracking de experimentos y modelos.
"""
import mlflow
import mlflow.keras
import mlflow.sklearn
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from src.config.settings import settings


class MLflowTracking:
    """
    Clase para gestionar el tracking de experimentos y modelos en MLflow.
    """
    
    def __init__(self):
        """Inicializa la conexión con MLflow."""
        self._initialized = False
        self._tracking_uri = settings.MLFLOW_TRACKING_URI
        self._experiment_name = settings.MLFLOW_EXPERIMENT_NAME
    
    def _ensure_initialized(self):
        """Inicializa MLflow solo cuando se necesita (lazy initialization)."""
        if not self._initialized:
            try:
                mlflow.set_tracking_uri(self._tracking_uri)
                # No intentar set_experiment aquí, se hará cuando se necesite
                self._initialized = True
            except Exception as e:
                print(f"⚠️ Advertencia: No se pudo inicializar MLflow: {e}")
                print(f"   MLflow Tracking URI: {self._tracking_uri}")
                print(f"   Asegúrate de que el servidor MLflow esté corriendo.")
                raise
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Inicia una nueva ejecución en MLflow.
        
        Args:
            run_name: Nombre de la ejecución
            nested: Si True, crea una ejecución anidada
        """
        self._ensure_initialized()
        # Configurar experimento solo cuando se inicia un run
        try:
            mlflow.set_experiment(self._experiment_name)
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo configurar experimento: {e}")
            # Continuar de todas formas, MLflow puede crear el experimento automáticamente
        return mlflow.start_run(run_name=run_name, nested=nested)
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Registra parámetros en MLflow.
        
        Args:
            params: Diccionario con parámetros a registrar
        """
        self._ensure_initialized()
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Registra métricas en MLflow.
        
        Args:
            metrics: Diccionario con métricas a registrar
            step: Paso/iteración (opcional)
        """
        self._ensure_initialized()
        if step is not None:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metrics(metrics)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        model_type: str = "keras"
    ):
        """
        Registra un modelo en MLflow.
        
        Args:
            model: Modelo a registrar (Keras o sklearn)
            artifact_path: Ruta del artefacto
            registered_model_name: Nombre del modelo en el registro
            model_type: Tipo de modelo ("keras" o "sklearn")
        """
        self._ensure_initialized()
        if model_type == "keras":
            mlflow.keras.log_model(model, artifact_path=artifact_path)
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Registrar en Model Registry si se especifica nombre
        if registered_model_name:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
                registered_model_name
            )
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Registra artefactos (archivos) en MLflow.
        
        Args:
            local_path: Ruta local del archivo o directorio
            artifact_path: Ruta del artefacto en MLflow (opcional)
        """
        self._ensure_initialized()
        mlflow.log_artifacts(local_path, artifact_path)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Registra un único artefacto en MLflow.
        
        Args:
            local_path: Ruta local del archivo
            artifact_path: Ruta del artefacto en MLflow (opcional)
        """
        self._ensure_initialized()
        mlflow.log_artifact(local_path, artifact_path)
    
    def load_model(self, model_uri: str, model_type: str = "keras"):
        """
        Carga un modelo desde MLflow.
        
        Args:
            model_uri: URI del modelo (puede ser runs:/, models:/, etc.)
            model_type: Tipo de modelo ("keras" o "sklearn")
            
        Returns:
            Modelo cargado
        """
        self._ensure_initialized()
        if model_type == "keras":
            return mlflow.keras.load_model(model_uri)
        elif model_type == "sklearn":
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    def register_model_version(
        self,
        model_uri: str,
        registered_model_name: str,
        stage: str = "Production"
    ):
        """
        Registra una versión del modelo en el Model Registry.
        
        Args:
            model_uri: URI del modelo
            registered_model_name: Nombre del modelo registrado
            stage: Etapa del modelo (Staging, Production, Archived)
        """
        self._ensure_initialized()
        mlflow.register_model(model_uri, registered_model_name)
        
        # Transicionar a la etapa especificada
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(registered_model_name, stages=[])[0].version
        client.transition_model_version_stage(
            registered_model_name,
            latest_version,
            stage
        )
    
    def get_model(self, model_name: str, stage: str = "Production"):
        """
        Obtiene un modelo del Model Registry.
        
        Args:
            model_name: Nombre del modelo registrado
            stage: Etapa del modelo
            
        Returns:
            URI del modelo
        """
        model_uri = f"models:/{model_name}/{stage}"
        return model_uri
    
    def end_run(self):
        """Finaliza la ejecución actual en MLflow."""
        if self._initialized:
            mlflow.end_run()

