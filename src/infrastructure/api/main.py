"""
API REST con FastAPI para predicción de churn.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
from typing import Dict, Any

from src.application.dto.prediction_request import PredictionRequest, PredictionResponse
from src.domain.services.preprocessing_service import PreprocessingService
from src.infrastructure.persistence.model_repository import ModelRepository
from src.config.settings import settings

# Inicializar aplicación FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API para predicción de churn de clientes usando Deep Learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes (lazy initialization para evitar errores al importar)
model_repository = None
preprocessing_service = None


def get_model_repository():
    """Obtiene el repositorio de modelos (lazy initialization)."""
    global model_repository
    if model_repository is None:
        try:
            model_repository = ModelRepository()
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo inicializar ModelRepository: {e}")
            print("   La API funcionará pero no podrá cargar modelos desde MLflow.")
            print("   Asegúrate de que MLflow esté corriendo en http://localhost:5000")
            # Crear un objeto dummy para evitar errores
            model_repository = None
    return model_repository


def load_preprocessing_service():
    """Carga el preprocessing service desde archivo guardado."""
    global preprocessing_service
    try:
        import joblib
        from pathlib import Path
        
        preprocessing_path = Path("models/preprocessing_service.pkl")
        if preprocessing_path.exists():
            preprocessing_service = joblib.load(preprocessing_path)
            print("✅ Preprocessing service cargado desde archivo.")
        else:
            print("⚠️ Preprocessing service no encontrado. Inicializando nuevo (sin encoders entrenados).")
            preprocessing_service = PreprocessingService()
    except Exception as e:
        print(f"⚠️ Error al cargar preprocessing service: {e}. Inicializando nuevo.")
        preprocessing_service = PreprocessingService()


@app.on_event("startup")
async def startup_event():
    """Carga el modelo y preprocessing service al iniciar la aplicación."""
    print("🚀 Iniciando API de Predicción de Churn...")
    
    # Cargar preprocessing service
    load_preprocessing_service()
    
    # Intentar cargar modelo desde MLflow (opcional)
    repo = get_model_repository()
    if repo is not None:
        print("📦 Intentando cargar modelo desde MLflow...")
        try:
            model = repo.load_latest_model()
            if model is None:
                print("⚠️ ADVERTENCIA: No se pudo cargar el modelo desde MLflow.")
                print("   La API funcionará pero las predicciones no estarán disponibles.")
                print("   Para cargar un modelo:")
                print("   1. Asegúrate de que MLflow esté corriendo: mlflow server --host 0.0.0.0 --port 5000")
                print("   2. Entrena y registra el modelo usando el notebook de entrenamiento")
            else:
                print("✅ Modelo cargado exitosamente desde MLflow.")
        except Exception as e:
            print(f"⚠️ Error al cargar modelo: {e}")
            print("   La API funcionará pero las predicciones no estarán disponibles.")
    else:
        print("⚠️ ModelRepository no disponible. MLflow no está corriendo.")
        print("   Para usar la API con modelos:")
        print("   1. Inicia MLflow: mlflow server --host 0.0.0.0 --port 5000")
        print("   2. Reinicia la API")
    
    print("✅ API iniciada. Endpoints disponibles en http://localhost:8000")


@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    repo = get_model_repository()
    model_loaded = False
    mlflow_available = repo is not None
    
    if repo is not None:
        try:
            model = repo.load_latest_model()
            model_loaded = model is not None
        except Exception:
            model_loaded = False
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "mlflow_available": mlflow_available,
        "mlflow_uri": settings.MLFLOW_TRACKING_URI
    }


@app.get("/model/info")
async def get_model_info():
    """Obtiene información del modelo actual."""
    repo = get_model_repository()
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="MLflow no está disponible. Inicia el servidor MLflow primero."
        )
    
    info = repo.get_model_info()
    if info is None:
        raise HTTPException(
            status_code=404,
            detail="Modelo no encontrado en MLflow. Entrena y registra el modelo primero."
        )
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest) -> PredictionResponse:
    """
    Realiza una predicción de churn para un cliente.
    
    Args:
        request: Datos del cliente para predicción
        
    Returns:
        Predicción de churn con probabilidad
    """
    try:
        # Obtener repositorio
        repo = get_model_repository()
        if repo is None:
            raise HTTPException(
                status_code=503,
                detail="MLflow no está disponible. Inicia el servidor MLflow primero: mlflow server --host 0.0.0.0 --port 5000"
            )
        
        # Cargar modelo
        model = repo.load_latest_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. Por favor, entrena y registra el modelo primero usando el notebook de entrenamiento."
            )
        
        # Preprocesar datos de entrada
        input_data = {
            "tenure": request.tenure,
            "phone_service": request.phone_service,
            "contract": request.contract,
            "paperless_billing": request.paperless_billing,
            "payment_method": request.payment_method,
            "monthly_charges": request.monthly_charges,
            "total_charges": request.total_charges
        }
        
        # Verificar que el preprocessing service esté cargado
        if preprocessing_service is None:
            load_preprocessing_service()
        
        # Preprocesar datos de entrada
        X = preprocessing_service.preprocess_single_prediction(input_data)
        
        # Realizar predicción
        prediction_proba = model.predict(X, verbose=0)[0][0]
        churn_prediction = "Yes" if prediction_proba > 0.5 else "No"
        
        return PredictionResponse(
            churn_probability=float(prediction_proba),
            churn_prediction=churn_prediction,
            customer_id=request.customer_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


# Servir archivos estáticos del frontend
try:
    app.mount("/static", StaticFiles(directory="src/infrastructure/web/frontend"), name="static")
except Exception:
    pass


@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """Sirve la página del frontend."""
    try:
        with open("src/infrastructure/web/frontend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend no encontrado</h1>",
            status_code=404
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)

