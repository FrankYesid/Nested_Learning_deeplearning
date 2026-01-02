"""
Configuración global de runtime (logs y warnings).
Debe ejecutarse antes de importar MLflow.
"""
import os
import warnings
import logging

def configure_runtime():
    # TensorFlow / C++ logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Suprimir warnings específicos
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources.*",
        category=UserWarning
    )

    warnings.filterwarnings(
        "ignore",
        message=".*schema_extra.*",
        category=UserWarning
    )

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning
    )

    # Logging general
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
