# 🔧 Solución: ModuleNotFoundError: No module named 'src'

## ❌ Error

```
ModuleNotFoundError: No module named 'src'
```

Este error ocurre cuando el notebook no puede encontrar el módulo `src` porque el directorio raíz del proyecto no está en el PATH de Python.

## ✅ Solución

El notebook ya está corregido. Si aún tienes problemas, sigue estos pasos:

### Opción 1: Ejecutar desde el directorio raíz

1. **Abre Jupyter desde el directorio raíz del proyecto:**
   ```powershell
   # Navegar al directorio raíz
   cd D:\GitHub\Nested_Learning_deeplearning
   
   # Activar entorno virtual
   .venv\Scripts\Activate.ps1
   
   # Iniciar Jupyter
   jupyter notebook
   ```

2. **Abrir el notebook desde Jupyter:**
   - Navega a `notebooks/02_Nested_Learning_Training.ipynb`
   - Ejecuta las celdas

### Opción 2: Verificar que el PATH esté configurado

El notebook ahora incluye código que automáticamente agrega el directorio raíz al PATH. Si aún tienes problemas:

1. **Verifica que estás ejecutando desde el directorio correcto:**
   ```python
   # En una celda del notebook, ejecuta:
   import os
   from pathlib import Path
   print(f"Directorio actual: {Path.cwd()}")
   print(f"¿Estamos en notebooks?: {Path.cwd().name == 'notebooks'}")
   ```

2. **Si el problema persiste, agrega manualmente:**
   ```python
   import sys
   from pathlib import Path
   
   # Agregar directorio raíz manualmente
   project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
   if str(project_root) not in sys.path:
       sys.path.insert(0, str(project_root))
   
   print(f"PATH actualizado: {project_root}")
   ```

### Opción 3: Instalar el proyecto como paquete

Si quieres una solución permanente, instala el proyecto:

```powershell
# Desde el directorio raíz del proyecto
cd D:\GitHub\Nested_Learning_deeplearning

# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Instalar en modo desarrollo
pip install -e .
```

Esto requiere un `setup.py` o `pyproject.toml` (ya existe `pyproject.toml`).

## 🔍 Verificación

Para verificar que todo funciona:

```python
# En una celda del notebook
import sys
print("Directorios en PATH:")
for p in sys.path[:5]:  # Primeros 5
    print(f"  - {p}")

# Intentar importar
try:
    from src.config.settings import settings
    print("✓ Importación exitosa!")
except ImportError as e:
    print(f"✗ Error de importación: {e}")
```

## 📝 Notas

- El notebook está en `notebooks/02_Nested_Learning_Training.ipynb`
- El código fuente está en `src/`
- El directorio raíz debe estar en `sys.path` para que `from src...` funcione
- El código corregido detecta automáticamente si estás en `notebooks/` o en la raíz

## 🚀 Solución Rápida

Si solo quieres que funcione rápido, ejecuta esto en la primera celda del notebook:

```python
import sys
from pathlib import Path

# Agregar raíz del proyecto al PATH
root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(root))
print(f"✓ PATH configurado: {root}")
```

---

**El notebook ya está corregido con esta solución.** Solo asegúrate de ejecutarlo desde Jupyter iniciado en el directorio raíz del proyecto.

