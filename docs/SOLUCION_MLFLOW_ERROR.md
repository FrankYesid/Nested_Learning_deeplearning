# 🔧 Solución: Error de Conexión con MLflow

## ❌ Error

```
ConnectionRefusedError: [WinError 10061] No se puede establecer una conexión ya que el equipo de destino denegó expresamente dicha conexión
MlflowException: API request to http://localhost:5000/api/2.0/mlflow/experiments/get-by-name failed
```

## 🔍 Causa

La API intenta conectarse a MLflow en `http://localhost:5000` pero el servidor MLflow **no está corriendo**.

## ✅ Solución

### Opción 1: Iniciar MLflow antes de la API (Recomendado)

**Terminal 1 - MLflow:**
```powershell
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Iniciar MLflow Tracking Server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns --default-artifact-root file:./mlruns
```

**Terminal 2 - API:**
```powershell
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Iniciar API
python -m uvicorn src.infrastructure.api.main:app --host 0.0.0.0 --port 8000
```

### Opción 2: Usar Scripts de Ayuda

**Iniciar MLflow:**
```powershell
python run_mlflow.py
```

**Iniciar API:**
```powershell
python run_api.py
```

### Opción 3: Usar Docker Compose (Todo en uno)

```powershell
cd docker
docker-compose up --build
```

Esto inicia tanto MLflow como la API automáticamente.

## 🔄 Cambios Realizados

He modificado el código para que:

1. **Inicialización Lazy**: MLflow solo se inicializa cuando se necesita, no al importar el módulo
2. **Manejo de Errores**: La API puede iniciar sin MLflow, mostrando advertencias
3. **Mensajes Claros**: Indicaciones sobre qué hacer si MLflow no está disponible

## 📋 Verificación

### 1. Verificar que MLflow esté corriendo

Abre en tu navegador: http://localhost:5000

Deberías ver la interfaz de MLflow.

### 2. Verificar que la API esté funcionando

Abre en tu navegador: http://localhost:8000/health

Deberías ver:
```json
{
  "status": "healthy",
  "model_loaded": false,
  "mlflow_available": true,
  "mlflow_uri": "http://localhost:5000"
}
```

### 3. Verificar endpoints

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/model/info

## 🚀 Flujo de Trabajo Completo

### 1. Entrenar Modelo (Primera vez)

```powershell
# Iniciar MLflow
mlflow server --host 0.0.0.0 --port 5000

# En otra terminal, ejecutar notebook de entrenamiento
jupyter notebook notebooks/02_Nested_Learning_Training.ipynb
```

### 2. Iniciar Servicios para Producción

```powershell
# Terminal 1: MLflow
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2: API
python -m uvicorn src.infrastructure.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Usar la API

- **Frontend**: http://localhost:8000/frontend
- **API Docs**: http://localhost:8000/docs
- **Predicción**: POST http://localhost:8000/predict

## ⚠️ Notas Importantes

1. **MLflow debe estar corriendo** antes de entrenar modelos o usar la API con modelos
2. **El modelo debe estar registrado** en MLflow Model Registry antes de que la API pueda cargarlo
3. **El preprocessing service** debe estar guardado en `models/preprocessing_service.pkl`

## 🐛 Troubleshooting

### Error: "MLflow no está disponible"

**Solución**: Inicia MLflow primero:
```powershell
mlflow server --host 0.0.0.0 --port 5000
```

### Error: "Modelo no encontrado"

**Solución**: Entrena y registra el modelo:
1. Ejecuta el notebook `02_Nested_Learning_Training.ipynb`
2. Asegúrate de que el modelo se registre en MLflow Model Registry

### Error: "Puerto 5000 ya en uso"

**Solución**: 
- Cierra el proceso que usa el puerto 5000
- O cambia el puerto en `settings.py` y reinicia MLflow

---

**✅ Con estos cambios, la API puede iniciar sin MLflow y mostrará mensajes claros sobre qué hacer.**

