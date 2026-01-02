# Script de configuración del proyecto
# Verifica la versión de Python y configura el entorno

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configurando Proyecto de Churn Prediction" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar versión de Python
Write-Host "Verificando version de Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python no encontrado" -ForegroundColor Red
    Write-Host "Por favor, instala Python 3.10 o 3.11 desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Python encontrado: $pythonVersion" -ForegroundColor Green

# Verificar que sea Python 3.10 o 3.11
if ($pythonVersion -match "Python 3\.(10|11)") {
    Write-Host "Version de Python compatible: $pythonVersion" -ForegroundColor Green
} elseif ($pythonVersion -match "Python 3\.12") {
    Write-Host "ADVERTENCIA: Python 3.12 detectado" -ForegroundColor Red
    Write-Host "TensorFlow 2.15.0 NO es compatible con Python 3.12" -ForegroundColor Red
    Write-Host ""
    Write-Host "Soluciones:" -ForegroundColor Yellow
    Write-Host "1. Instalar Python 3.10 o 3.11" -ForegroundColor Gray
    Write-Host "2. Usar UV para crear entorno con version especifica:" -ForegroundColor Gray
    Write-Host "   uv venv --python 3.10" -ForegroundColor Cyan
    Write-Host ""
    $continuar = Read-Host "Deseas continuar de todas formas? (s/N)"
    if ($continuar -ne "s" -and $continuar -ne "S") {
        exit 1
    }
} elseif ($pythonVersion -match "Python 3\.(9|10|11)") {
    Write-Host "Version de Python compatible: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "ADVERTENCIA: Version de Python no verificada: $pythonVersion" -ForegroundColor Yellow
    Write-Host "Se recomienda Python 3.10 o 3.11" -ForegroundColor Yellow
}

Write-Host ""

# Verificar si UV está disponible
Write-Host "Verificando UV..." -ForegroundColor Yellow
$uvPath = "$env:USERPROFILE\.local\bin"
if (Test-Path "$uvPath\uv.exe") {
    $env:Path = "$uvPath;$env:Path"
    $uvVersion = uv --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "UV encontrado: $uvVersion" -ForegroundColor Green
        $usarUV = $true
    } else {
        $usarUV = $false
    }
} else {
    Write-Host "UV no encontrado. Usando pip tradicional." -ForegroundColor Yellow
    $usarUV = $false
}

Write-Host ""

# Crear entorno virtual
Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
if ($usarUV) {
    # Intentar crear con Python 3.10 si está disponible
    $python310 = Get-Command python3.10 -ErrorAction SilentlyContinue
    if ($python310) {
        Write-Host "Creando entorno con Python 3.10 usando UV..." -ForegroundColor Cyan
        uv venv --python 3.10
    } else {
        Write-Host "Creando entorno virtual con UV..." -ForegroundColor Cyan
        uv venv
    }
} else {
    Write-Host "Creando entorno virtual con venv..." -ForegroundColor Cyan
    python -m venv .venv
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR al crear entorno virtual" -ForegroundColor Red
    exit 1
}

Write-Host "Entorno virtual creado exitosamente" -ForegroundColor Green
Write-Host ""

# Activar entorno
Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
    Write-Host "Entorno activado" -ForegroundColor Green
} else {
    Write-Host "ADVERTENCIA: No se pudo activar el entorno automaticamente" -ForegroundColor Yellow
    Write-Host "Activa manualmente con: .venv\Scripts\Activate.ps1" -ForegroundColor Gray
}

Write-Host ""

# Instalar dependencias
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
if ($usarUV) {
    Write-Host "Usando UV para instalar dependencias..." -ForegroundColor Cyan
    uv pip install -r requirements.txt
} else {
    Write-Host "Usando pip para instalar dependencias..." -ForegroundColor Cyan
    pip install -r requirements.txt
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR al instalar dependencias" -ForegroundColor Red
    Write-Host "Verifica que tengas Python 3.10 o 3.11 instalado" -ForegroundColor Yellow
    exit 1
}

Write-Host "Dependencias instaladas exitosamente" -ForegroundColor Green
Write-Host ""

# Resumen
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuracion completada exitosamente!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Activar entorno virtual (si no esta activado):" -ForegroundColor White
Write-Host "   .venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Ejecutar EDA:" -ForegroundColor White
Write-Host "   jupyter notebook notebooks/01_EDA_Churn.ipynb" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Entrenar modelo:" -ForegroundColor White
Write-Host "   jupyter notebook notebooks/02_Nested_Learning_Training.ipynb" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Iniciar MLflow:" -ForegroundColor White
Write-Host "   mlflow server --host 0.0.0.0 --port 5000" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Iniciar API:" -ForegroundColor White
Write-Host "   python run_api.py" -ForegroundColor Gray
Write-Host ""

