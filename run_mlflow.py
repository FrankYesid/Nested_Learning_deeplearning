#!/usr/bin/env python
"""
Script para ejecutar el servidor de MLflow.
"""

# -------------------------------------------------------
# 1. Configurar runtime ANTES de cualquier otro import
# -------------------------------------------------------
from src.config.runtime import configure_runtime
configure_runtime()

# -------------------------------------------------------
# 2. Imports estándar
# -------------------------------------------------------
import subprocess
import sys

# -------------------------------------------------------
# 3. Configuración del proyecto
# -------------------------------------------------------
from src.config.settings import settings


def run_mlflow_server():
    print("=" * 60)
    print("📊 Iniciando MLflow Tracking Server")
    print("=" * 60)
    print(f"Tracking URI : {settings.MLFLOW_TRACKING_URI}")
    print(f"Artifact Root: {settings.MLRUNS_DIR}")
    print("=" * 60)
    print("\n🌐 MLflow UI disponible en:")
    print(f"   {settings.MLFLOW_TRACKING_URI}")
    print("\n" + "=" * 60 + "\n")

    # Asegurar directorios
    settings.ensure_directories()

    # Comando MLflow
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", f"file:{settings.MLRUNS_DIR}",
        "--default-artifact-root", f"file:{settings.MLRUNS_DIR}"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 MLflow server detenido.")
    except subprocess.CalledProcessError as e:
        print("\n❌ Error al iniciar MLflow:", e)


if __name__ == "__main__":
    run_mlflow_server()
