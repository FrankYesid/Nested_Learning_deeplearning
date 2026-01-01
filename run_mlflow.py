#!/usr/bin/env python
"""
Script de ayuda para ejecutar el servidor MLflow.
"""
import subprocess
import sys
from pathlib import Path
from src.config.settings import settings

if __name__ == "__main__":
    print("="*60)
    print("📊 Iniciando MLflow Tracking Server")
    print("="*60)
    print(f"URI: {settings.MLFLOW_TRACKING_URI}")
    print(f"Artifact Root: {settings.MLRUNS_DIR}")
    print("="*60)
    print("\n🌐 MLflow UI disponible en:")
    print(f"   {settings.MLFLOW_TRACKING_URI}")
    print("\n" + "="*60 + "\n")
    
    # Asegurar que el directorio existe
    settings.ensure_directories()
    
    # Ejecutar MLflow server
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", f"file:{settings.MLRUNS_DIR}",
        "--default-artifact-root", f"file:{settings.MLRUNS_DIR}"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 MLflow server detenido.")

