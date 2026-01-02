# Proyecto de Predicción de Churn con Deep Learning y Nested Cross Validation

Sistema completo de Machine Learning para predicción de churn de clientes usando Deep Learning, implementando Nested Cross Validation, arquitectura hexagonal, MLflow, API REST y frontend web.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [MLflow](#mlflow)
- [API REST](#api-rest)
- [Frontend](#frontend)
- [Docker](#docker)
- [Contribución](#contribución)

## 🎯 Descripción

Este proyecto implementa un sistema end-to-end para predecir el churn (abandono) de clientes utilizando:

- **Deep Learning**: Redes neuronales con TensorFlow/Keras
- **Nested Cross Validation**: Validación cruzada anidada para selección de hiperparámetros y evaluación robusta
- **Arquitectura Hexagonal**: Separación clara entre dominio, aplicación e infraestructura
- **MLflow**: Tracking de experimentos, versionado de modelos y Model Registry
- **FastAPI**: API REST para servir predicciones
- **Frontend Web**: Interfaz de usuario para realizar predicciones
- **Docker**: Contenedorización completa del sistema

## 🏗️ Arquitectura

El proyecto sigue una **Arquitectura Hexagonal (Ports & Adapters)**:

```
┌─────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │  MLflow  │  │   API     │  │Frontend  │  │Persistence│
│  └──────────┘  └──────────┘  └──────────┘  └───────┘ │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                   APPLICATION                           │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │  Use Cases   │  │     DTOs     │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                      DOMAIN                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Entities  │  │  Models  │  │ Services │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

### Capas:

1. **Domain**: Lógica de negocio pura, sin dependencias externas
   - Entidades: Representación de datos
   - Modelos: Modelos de ML
   - Servicios: Lógica de preprocesamiento

2. **Application**: Casos de uso y orquestación
   - Use Cases: Lógica de aplicación
   - DTOs: Objetos de transferencia de datos

3. **Infrastructure**: Implementaciones concretas
   - MLflow: Tracking y registro de modelos
   - API: FastAPI endpoints
   - Persistence: Repositorio de modelos
   - Web: Frontend HTML/CSS/JS

## 📊 Dataset

El dataset `churn_data.csv` contiene las siguientes columnas:

- `customerID`: Identificador único del cliente
- `tenure`: Meses de permanencia
- `PhoneService`: Servicio telefónico (Yes/No)
- `Contract`: Tipo de contrato (Month-to-month/One year/Two year)
- `PaperlessBilling`: Facturación sin papel (Yes/No)
- `PaymentMethod`: Método de pago
- `MonthlyCharges`: Cargos mensuales
- `TotalCharges`: Cargos totales
- `Churn`: Variable objetivo (Yes/No)

## 📁 Estructura del Proyecto

```
project-churn-nested-learning/
│
├── data/
│   └── churn_data.csv              # Dataset
│
├── notebooks/
│   ├── 01_EDA_Churn.ipynb         # Análisis exploratorio
│   └── 02_Nested_Learning_Training.ipynb  # Entrenamiento
│
├── src/
│   ├── domain/
│   │   ├── entities/
│   │   │   └── churn_entity.py
│   │   ├── models/
│   │   │   └── deep_learning_model.py
│   │   └── services/
│   │       └── preprocessing_service.py
│   │
│   ├── application/
│   │   ├── use_cases/
│   │   │   └── train_model_use_case.py
│   │   └── dto/
│   │       └── prediction_request.py
│   │
│   ├── infrastructure/
│   │   ├── mlflow/
│   │   │   └── mlflow_tracking.py
│   │   ├── api/
│   │   │   └── main.py
│   │   ├── persistence/
│   │   │   └── model_repository.py
│   │   └── web/
│   │       └── frontend/
│   │           ├── index.html
│   │           ├── styles.css
│   │           └── app.js
│   │
│   └── config/
│       └── settings.py
│
├── mlruns/                         # MLflow tracking (generado)
├── models/                         # Modelos guardados (generado)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

## 🔧 Requisitos

- **Python 3.10 o 3.11** (⚠️ NO compatible con Python 3.12+ debido a TensorFlow 2.15.0)
- Docker y Docker Compose (opcional, para despliegue)
- MLflow Tracking Server

> 📌 **IMPORTANTE**: TensorFlow 2.15.0 solo soporta Python 3.9, 3.10 y 3.11.  
> Ver [VERSION_PYTHON.md](VERSION_PYTHON.md) para más detalles sobre compatibilidad.

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd Nested_Learning_deeplearning
```

### 2. Verificar versión de Python

```bash
# Verificar que tienes Python 3.10 o 3.11
python --version

# Si tienes Python 3.12+, necesitas instalar Python 3.10 o 3.11
# Descarga desde: https://www.python.org/downloads/
```

### 3. Crear entorno virtual

**Con UV (Recomendado):**
```bash
# Crear entorno con Python 3.10
uv venv --python 3.10

# O con Python 3.11
uv venv --python 3.11

# Activar entorno (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activar entorno (macOS/Linux)
source .venv/bin/activate
```

**Método tradicional:**
```bash
# Asegúrate de usar Python 3.10 o 3.11
python3.10 -m venv venv  # o python3.11
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
# Con UV
uv pip install -r requirements.txt

# O con pip tradicional
pip install -r requirements.txt
```

### 5. Configurar MLflow (opcional)

Si quieres usar un servidor MLflow externo, configura la variable de entorno:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🚀 Uso

### 1. Análisis Exploratorio de Datos

```bash
jupyter notebook notebooks/01_EDA_Churn.ipynb
```

Ejecuta todas las celdas para realizar el análisis exploratorio.

### 2. Entrenamiento del Modelo

```bash
jupyter notebook notebooks/02_Nested_Learning_Training.ipynb
```

Este notebook:
- Carga y preprocesa los datos
- Realiza Nested Cross Validation
- Entrena múltiples modelos con diferentes hiperparámetros
- Registra todo en MLflow
- Guarda el mejor modelo en MLflow Model Registry

**Nota**: El entrenamiento puede tardar varios minutos dependiendo de tu hardware.

### 3. Iniciar MLflow Tracking Server

```bash
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns
```

Accede a la UI en: http://localhost:5000

### 4. Iniciar la API

```bash
cd src/infrastructure/api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

O desde la raíz:

```bash
python -m uvicorn src.infrastructure.api.main:app --host 0.0.0.0 --port 8000
```

La API estará disponible en: http://localhost:8000

### 5. Acceder al Frontend

Abre en tu navegador:
- http://localhost:8000/frontend
- O directamente: http://localhost:8000/static/index.html

## 📈 MLflow

### Acceder a MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Luego abre: http://localhost:5000

### Ver Modelos Registrados

1. Ve a la pestaña "Models" en MLflow UI
2. Busca el modelo: `churn_deep_learning_model`
3. Verás todas las versiones y sus stages

### Cargar Modelo desde MLflow

El código de la API carga automáticamente el modelo desde MLflow Model Registry.

## 🔌 API REST

### Endpoints

#### `GET /`
Información general de la API.

#### `GET /health`
Health check del servicio.

#### `GET /model/info`
Información del modelo actual cargado.

#### `POST /predict`
Realiza una predicción de churn.

**Request Body:**
```json
{
  "tenure": 12,
  "phone_service": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 70.5,
  "total_charges": 846.0,
  "customer_id": "1234-ABCDE"
}
```

**Response:**
```json
{
  "churn_probability": 0.75,
  "churn_prediction": "Yes",
  "customer_id": "1234-ABCDE"
}
```

### Documentación Interactiva

FastAPI proporciona documentación automática:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🎨 Frontend

El frontend es una aplicación web simple que permite:
- Ingresar datos del cliente
- Realizar predicciones en tiempo real
- Visualizar probabilidades de churn
- Ver resultados con feedback visual

Accede en: http://localhost:8000/frontend

## 🐳 Docker

### Construir y ejecutar con Docker Compose

```bash
cd docker
docker-compose up --build
```

Esto iniciará:
- **MLflow Tracking Server** en http://localhost:5000
- **API FastAPI** en http://localhost:8000

### Comandos útiles

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up

# Iniciar en segundo plano
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

### Construir imagen manualmente

```bash
docker build -f docker/Dockerfile -t churn-api:latest .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://localhost:5000 churn-api:latest
```

## 🔬 Nested Cross Validation

El proyecto implementa **Nested Cross Validation** para:

1. **Outer CV (K=5)**: Evalúa el rendimiento final del modelo
2. **Inner CV (K=3)**: Selecciona los mejores hiperparámetros

Esto evita el sobreajuste y proporciona una evaluación más robusta del modelo.

## 📝 Notas Importantes

1. **Preprocessing Service**: Debe ser entrenado antes de usar la API. Se guarda durante el entrenamiento.

2. **MLflow**: Asegúrate de que el servidor MLflow esté corriendo antes de iniciar la API.

3. **Modelo**: El modelo debe estar registrado en MLflow Model Registry con stage "Production".

4. **Datos**: El dataset debe estar en `data/churn_data.csv`.

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 👥 Autores

Equipo Senior:
- Arquitecto de Software
- Ingeniero de Machine Learning
- Ingeniero MLOps
- Desarrollador Backend
- Desarrollador Frontend

## 🙏 Agradecimientos

- TensorFlow/Keras por el framework de Deep Learning
- MLflow por el tracking de experimentos
- FastAPI por el framework web moderno
- La comunidad de código abierto

---

**¡Listo para producción!** 🚀

