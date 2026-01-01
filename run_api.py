#!/usr/bin/env python
"""
Script de ayuda para ejecutar la API de predicción de churn.
"""
import uvicorn
from src.config.settings import settings

if __name__ == "__main__":
    print("="*60)
    print("🚀 Iniciando API de Predicción de Churn")
    print("="*60)
    print(f"Host: {settings.API_HOST}")
    print(f"Port: {settings.API_PORT}")
    print(f"MLflow URI: {settings.MLFLOW_TRACKING_URI}")
    print("="*60)
    print("\n📚 Documentación disponible en:")
    print(f"   - Swagger UI: http://localhost:{settings.API_PORT}/docs")
    print(f"   - ReDoc: http://localhost:{settings.API_PORT}/redoc")
    print(f"   - Frontend: http://localhost:{settings.API_PORT}/frontend")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "src.infrastructure.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )

