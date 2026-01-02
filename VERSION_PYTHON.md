# 🐍 Versión de Python Requerida

## ⚠️ IMPORTANTE: Compatibilidad de Python

Este proyecto **requiere Python 3.10 o 3.11**.

**TensorFlow 2.15.0 NO es compatible con Python 3.12+**

## 🔍 Verificar tu Versión de Python

```powershell
# Ver versión actual
python --version

# O con uv
uv python list
```

## ✅ Versiones Compatibles

- ✅ **Python 3.10** (Recomendado)
- ✅ **Python 3.11** (Recomendado)
- ❌ **Python 3.12** (NO compatible con TensorFlow 2.15.0)
- ❌ **Python 3.13** (NO compatible)

## 🚀 Solución: Usar Python 3.10 o 3.11

### Opción 1: Con UV (Recomendado)

```powershell
# Listar versiones de Python disponibles
uv python list

# Crear entorno virtual con Python 3.10
uv venv --python 3.10

# O con Python 3.11
uv venv --python 3.11

# Instalar dependencias
uv pip install -r requirements.txt
```

### Opción 2: Instalar Python 3.10/3.11

Si no tienes Python 3.10 o 3.11:

1. **Descargar Python 3.10 o 3.11:**
   - https://www.python.org/downloads/
   - Selecciona Python 3.10.11 o Python 3.11.7

2. **Instalar y verificar:**
   ```powershell
   python3.10 --version
   # o
   python3.11 --version
   ```

3. **Crear entorno con versión específica:**
   ```powershell
   # Con Python 3.10
   python3.10 -m venv .venv
   .venv\Scripts\Activate.ps1
   
   # O con UV
   uv venv --python 3.10
   ```

### Opción 3: Usar Conda

```powershell
# Crear entorno con Python 3.10
conda create -n churn-env python=3.10
conda activate churn-env

# Instalar dependencias
pip install -r requirements.txt
```

## 🔧 Actualizar Requirements.txt

Si necesitas usar Python 3.12, tendrías que actualizar TensorFlow:

```txt
# Para Python 3.12, usar TensorFlow 2.16+ (cuando esté disponible)
# O usar versiones más recientes:
tensorflow>=2.16.0
```

**NOTA**: TensorFlow 2.16+ aún no está disponible. Se recomienda usar Python 3.10 o 3.11.

## 📋 Verificar Compatibilidad

```powershell
# Verificar versión de Python
python --version

# Verificar que TensorFlow puede instalarse
uv pip install tensorflow==2.15.0 --dry-run
```

## 🎯 Resumen

1. **Usa Python 3.10 o 3.11**
2. **Crea entorno virtual con versión específica:**
   ```powershell
   uv venv --python 3.10
   ```
3. **Instala dependencias:**
   ```powershell
   uv pip install -r requirements.txt
   ```

---

**💡 Recomendación**: Usa Python 3.10 para máxima compatibilidad con todas las dependencias.

