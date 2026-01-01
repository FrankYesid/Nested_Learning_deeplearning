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

# Inicializar componentes
model_repository = ModelRepository()
preprocessing_service = PreprocessingService()


@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la aplicación."""
    print("Cargando modelo desde MLflow...")
    model = model_repository.load_latest_model()
    if model is None:
        print("ADVERTENCIA: No se pudo cargar el modelo. Asegúrate de que el modelo esté registrado en MLflow.")
    else:
        print("Modelo cargado exitosamente.")
    
    # Cargar el preprocessing service (necesita los encoders entrenados)
    # En producción, estos deberían guardarse junto con el modelo
    print("NOTA: El preprocessing service necesita ser inicializado con datos de entrenamiento.")


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
    model = model_repository.load_latest_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model/info")
async def get_model_info():
    """Obtiene información del modelo actual."""
    info = model_repository.get_model_info()
    if info is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado en MLflow")
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
        # Cargar modelo
        model = model_repository.load_latest_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. Por favor, entrena y registra el modelo primero."
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
        
        # NOTA: En producción, el preprocessing_service debería cargarse desde un artefacto guardado
        # Por ahora, asumimos que ya está entrenado
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

