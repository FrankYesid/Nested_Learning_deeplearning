"""
DTOs (Data Transfer Objects) para las peticiones de predicción.
"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    DTO para la petición de predicción de churn.
    """
    tenure: int = Field(..., ge=0, description="Meses de permanencia del cliente")
    phone_service: str = Field(..., description="Servicio telefónico (Yes/No)")
    contract: str = Field(..., description="Tipo de contrato (Month-to-month/One year/Two year)")
    paperless_billing: str = Field(..., description="Facturación sin papel (Yes/No)")
    payment_method: str = Field(..., description="Método de pago")
    monthly_charges: float = Field(..., ge=0, description="Cargos mensuales")
    total_charges: float = Field(..., ge=0, description="Cargos totales")
    customer_id: Optional[str] = Field(None, description="ID del cliente (opcional)")
    
    class Config:
        schema_extra = {
            "example": {
                "tenure": 12,
                "phone_service": "Yes",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 70.5,
                "total_charges": 846.0,
                "customer_id": "1234-ABCDE"
            }
        }


class PredictionResponse(BaseModel):
    """
    DTO para la respuesta de predicción de churn.
    """
    churn_probability: float = Field(..., ge=0, le=1, description="Probabilidad de churn (0-1)")
    churn_prediction: str = Field(..., description="Predicción (Yes/No)")
    customer_id: Optional[str] = Field(None, description="ID del cliente")
    
    class Config:
        schema_extra = {
            "example": {
                "churn_probability": 0.75,
                "churn_prediction": "Yes",
                "customer_id": "1234-ABCDE"
            }
        }

