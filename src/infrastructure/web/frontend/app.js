// Configuración de la API
const API_BASE_URL = window.location.origin;

// Elementos del DOM
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resultContainer = document.getElementById('resultContainer');
const errorContainer = document.getElementById('errorContainer');
const loadingContainer = document.getElementById('loadingContainer');
const probabilityValue = document.getElementById('probabilityValue');
const predictionValue = document.getElementById('predictionValue');
const progressBar = document.getElementById('progressBar');
const errorMessage = document.getElementById('errorMessage');

// Manejar envío del formulario
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Ocultar resultados anteriores
    hideAllContainers();
    showLoading();
    
    // Obtener datos del formulario
    const formData = {
        tenure: parseInt(document.getElementById('tenure').value),
        phone_service: document.getElementById('phone_service').value,
        contract: document.getElementById('contract').value,
        paperless_billing: document.getElementById('paperless_billing').value,
        payment_method: document.getElementById('payment_method').value,
        monthly_charges: parseFloat(document.getElementById('monthly_charges').value),
        total_charges: parseFloat(document.getElementById('total_charges').value),
        customer_id: document.getElementById('customer_id').value || null
    };
    
    try {
        // Realizar petición a la API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Error en la predicción');
        }
        
        const result = await response.json();
        
        // Mostrar resultado
        displayResult(result);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

function displayResult(result) {
    const probability = result.churn_probability;
    const prediction = result.churn_prediction;
    
    // Actualizar probabilidad
    probabilityValue.textContent = `${(probability * 100).toFixed(2)}%`;
    
    // Actualizar predicción
    predictionValue.textContent = prediction;
    predictionValue.className = `prediction-value ${prediction.toLowerCase() === 'yes' ? 'yes' : 'no'}`;
    
    // Actualizar barra de progreso
    progressBar.style.width = `${probability * 100}%`;
    
    // Mostrar contenedor de resultados
    resultContainer.classList.remove('hidden');
    
    // Scroll suave al resultado
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorContainer.classList.remove('hidden');
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showLoading() {
    loadingContainer.classList.remove('hidden');
    submitBtn.disabled = true;
}

function hideLoading() {
    loadingContainer.classList.add('hidden');
    submitBtn.disabled = false;
}

function hideAllContainers() {
    resultContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');
}

// Validación en tiempo real
document.getElementById('monthly_charges').addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    if (value < 0) {
        e.target.setCustomValidity('Los cargos mensuales no pueden ser negativos');
    } else {
        e.target.setCustomValidity('');
    }
});

document.getElementById('total_charges').addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    if (value < 0) {
        e.target.setCustomValidity('Los cargos totales no pueden ser negativos');
    } else {
        e.target.setCustomValidity('');
    }
});

// Cargar información del modelo al iniciar
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        if (response.ok) {
            const info = await response.json();
            console.log('Modelo cargado:', info);
        }
    } catch (error) {
        console.warn('No se pudo cargar información del modelo:', error);
    }
}

// Ejecutar al cargar la página
window.addEventListener('load', loadModelInfo);


