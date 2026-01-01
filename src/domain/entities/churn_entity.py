"""
Entidad de dominio para representar un cliente y su información de churn.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChurnEntity:
    """
    Entidad que representa un cliente con sus características para predicción de churn.
    """
    customer_id: Optional[str] = None
    tenure: int = 0
    phone_service: str = "No"
    contract: str = "Month-to-month"
    paperless_billing: str = "No"
    payment_method: str = "Electronic check"
    monthly_charges: float = 0.0
    total_charges: float = 0.0
    churn: Optional[str] = None  # "Yes" o "No" para datos históricos
    
    def to_dict(self) -> dict:
        """Convierte la entidad a diccionario."""
        return {
            "customer_id": self.customer_id,
            "tenure": self.tenure,
            "phone_service": self.phone_service,
            "contract": self.contract,
            "paperless_billing": self.paperless_billing,
            "payment_method": self.payment_method,
            "monthly_charges": self.monthly_charges,
            "total_charges": self.total_charges,
            "churn": self.churn
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChurnEntity":
        """Crea una entidad desde un diccionario."""
        return cls(
            customer_id=data.get("customer_id") or data.get("customerID"),
            tenure=int(data.get("tenure", 0)),
            phone_service=data.get("phone_service") or data.get("PhoneService", "No"),
            contract=data.get("contract") or data.get("Contract", "Month-to-month"),
            paperless_billing=data.get("paperless_billing") or data.get("PaperlessBilling", "No"),
            payment_method=data.get("payment_method") or data.get("PaymentMethod", "Electronic check"),
            monthly_charges=float(data.get("monthly_charges") or data.get("MonthlyCharges", 0.0)),
            total_charges=float(data.get("total_charges") or data.get("TotalCharges", 0.0)),
            churn=data.get("churn") or data.get("Churn")
        )

