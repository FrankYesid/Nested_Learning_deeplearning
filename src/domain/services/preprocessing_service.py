"""
Servicio de dominio para preprocesamiento de datos de churn.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class PreprocessingService:
    """
    Servicio que encapsula la lógica de preprocesamiento de datos.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = []
        self.categorical_columns = ['PhoneService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el dataset eliminando valores nulos y convirtiendo tipos.
        
        Args:
            df: DataFrame con los datos crudos
            
        Returns:
            DataFrame limpio
        """
        df_clean = df.copy()
        
        # Convertir TotalCharges a numérico, reemplazando espacios vacíos con 0
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        
        # Eliminar filas con valores nulos críticos
        df_clean = df_clean.dropna(subset=['tenure', 'MonthlyCharges'])
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica variables categóricas usando Label Encoding.
        
        Args:
            df: DataFrame con variables categóricas
            fit: Si True, ajusta los encoders; si False, solo transforma
            
        Returns:
            DataFrame con variables codificadas
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    if col in self.label_encoders:
                        # Manejar valores nuevos no vistos durante el entrenamiento
                        unique_values = df_encoded[col].unique()
                        known_values = set(self.label_encoders[col].classes_)
                        for val in unique_values:
                            if val not in known_values:
                                df_encoded.loc[df_encoded[col] == val, col] = self.label_encoders[col].classes_[0]
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Prepara las características para el modelo.
        
        Args:
            df: DataFrame con datos preprocesados
            fit: Si True, ajusta el scaler; si False, solo transforma
            
        Returns:
            Array numpy con características preparadas
        """
        # Seleccionar columnas numéricas y categóricas codificadas
        feature_cols = self.numerical_columns + self.categorical_columns
        X = df[feature_cols].values
        
        # Escalar características
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_columns = feature_cols
        return X_scaled
    
    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepara la variable objetivo.
        
        Args:
            df: DataFrame con la columna Churn
            
        Returns:
            Array numpy con la variable objetivo codificada (1 para "Yes", 0 para "No")
        """
        y = (df['Churn'] == 'Yes').astype(int).values
        return y
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            df: DataFrame crudo
            fit: Si True, ajusta los transformadores; si False, solo transforma
            
        Returns:
            Tupla (X, y) con características y objetivo preparados
        """
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical(df_clean, fit=fit)
        X = self.prepare_features(df_encoded, fit=fit)
        y = self.prepare_target(df_clean)
        
        return X, y
    
    def preprocess_single_prediction(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocesa un solo registro para predicción.
        
        Args:
            data: Diccionario con los datos del cliente
            
        Returns:
            Array numpy con características preparadas
        """
        # Crear DataFrame de una sola fila
        df = pd.DataFrame([{
            'tenure': data.get('tenure', 0),
            'PhoneService': data.get('phone_service', 'No'),
            'Contract': data.get('contract', 'Month-to-month'),
            'PaperlessBilling': data.get('paperless_billing', 'No'),
            'PaymentMethod': data.get('payment_method', 'Electronic check'),
            'MonthlyCharges': data.get('monthly_charges', 0.0),
            'TotalCharges': data.get('total_charges', 0.0)
        }])
        
        # Limpiar y codificar
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical(df_clean, fit=False)
        X = self.prepare_features(df_encoded, fit=False)
        
        return X
    
    def get_feature_names(self) -> list:
        """Retorna los nombres de las características."""
        return self.feature_columns if self.feature_columns else (self.numerical_columns + self.categorical_columns)

