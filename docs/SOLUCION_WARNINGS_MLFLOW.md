# 🔧 Solución: Warnings al ejecutar `mlflow --version`

## ⚠️ Warnings Observados

Cuando ejecutas `mlflow --version`, puedes ver estos warnings:

```
UserWarning: pkg_resources is deprecated as an API...
UserWarning: Valid config keys have changed in V2:
* 'schema_extra' has been renamed to 'json_schema_extra'
```

## ✅ Estado Actual

**IMPORTANTE**: Estos son **warnings informativos**, NO son errores. MLflow funciona correctamente.

El comando `mlflow --version` muestra:
```
mlflow, version 2.9.2
```

Esto confirma que MLflow está instalado y funcionando.

## 🔍 Explicación de los Warnings

### 1. Warning de `pkg_resources`

**Origen**: Viene de MLflow 2.9.2, no de tu código.

**Causa**: MLflow usa `pkg_resources` que está deprecado en setuptools.

**Solución**: 
- Este warning viene de MLflow, no podemos corregirlo directamente
- Se solucionará cuando MLflow actualice su código
- No afecta la funcionalidad

**Para suprimir el warning** (opcional):
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')
```

### 2. Warning de `schema_extra` en Pydantic

**Estado**: ✅ **YA CORREGIDO** en `src/application/dto/prediction_request.py`

**Si aún aparece**:
- Puede venir de otras librerías que usan Pydantic v1
- O de código en caché de Python

**Solución**:
```powershell
# Limpiar caché de Python
python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
python -Bc "import pathlib; [pathlib.Path(p).rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
```

## 🚀 Verificación

### Verificar que MLflow funciona:

```powershell
# Ver versión
mlflow --version

# Iniciar servidor (debería funcionar sin problemas)
mlflow server --host 0.0.0.0 --port 5000
```

### Verificar que no hay errores reales:

```powershell
# Probar importación
python -c "import mlflow; print('MLflow OK')"

# Probar tracking
python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns'); print('Tracking OK')"
```

## 📋 Soluciones Opcionales

### Opción 1: Suprimir warnings en scripts

Crear un archivo `suppress_warnings.py`:

```python
import warnings

# Suprimir warnings de MLflow
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

# Suprimir warnings de Pydantic (si persisten)
warnings.filterwarnings('ignore', message='.*schema_extra.*')
```

Usar en scripts:
```python
import suppress_warnings  # Al inicio del script
import mlflow
# ... resto del código
```

### Opción 2: Actualizar MLflow (cuando esté disponible)

```powershell
pip install --upgrade mlflow
```

**Nota**: MLflow 2.9.2 es la versión actual. El warning se solucionará en futuras versiones.

### Opción 3: Pin setuptools (temporal)

```powershell
pip install "setuptools<81"
```

**Advertencia**: Esto puede afectar otras dependencias. Solo si es absolutamente necesario.

## ✅ Conclusión

- ✅ MLflow funciona correctamente
- ✅ Los warnings son informativos, no críticos
- ✅ El código del proyecto ya está corregido
- ✅ Puedes usar MLflow sin problemas

**Recomendación**: Ignora estos warnings por ahora. No afectan la funcionalidad del proyecto.

---

**💡 Tip**: Si quieres ver solo la salida sin warnings, puedes redirigir stderr:

```powershell
# Windows PowerShell
mlflow --version 2>$null

# O filtrar warnings
mlflow --version 2>&1 | Where-Object { $_ -notmatch 'UserWarning' }
```


