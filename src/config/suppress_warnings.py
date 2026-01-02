# """
# Script para suprimir warnings comunes que no afectan la funcionalidad.
# """
# import warnings

# # Suprimir warnings de MLflow sobre pkg_resources
# warnings.filterwarnings('ignore', message='.*pkg_resources.*', category=UserWarning)
# warnings.filterwarnings('ignore', message='.*pkg_resources.*', module='mlflow')

# # Suprimir warnings de Pydantic sobre schema_extra (si vienen de otras librerías)
# warnings.filterwarnings('ignore', message='.*schema_extra.*', category=UserWarning)
# warnings.filterwarnings('ignore', message='.*json_schema_extra.*', category=UserWarning)

# # Suprimir warnings de TensorFlow sobre oneDNN (opcional)
# warnings.filterwarnings('ignore', message='.*oneDNN.*', category=UserWarning)

# # Suprimir warnings de deprecación de TensorFlow (opcional)
# warnings.filterwarnings('ignore', message='.*deprecated.*', category=UserWarning, module='tensorflow')

# print("✓ Warnings suprimidos (no afectan la funcionalidad)")



