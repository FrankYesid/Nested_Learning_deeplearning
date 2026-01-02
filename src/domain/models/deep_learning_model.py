"""
Modelo de Deep Learning para predicción de churn.
"""
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential


class DeepLearningModel:
    """
    Modelo de red neuronal para clasificación de churn.
    """
    
    def __init__(self, input_dim: int, hyperparameters: Dict[str, Any] = None):
        """
        Inicializa el modelo.
        
        Args:
            input_dim: Dimensión de las características de entrada
            hyperparameters: Diccionario con hiperparámetros del modelo
        """
        self.input_dim = input_dim
        self.hyperparameters = hyperparameters or self._default_hyperparameters()
        self.model = None
        
    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Retorna hiperparámetros por defecto."""
        return {
            'hidden_layers': 2,
            'units_per_layer': [64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'activation': 'relu',
            'optimizer': 'adam'
        }
    
    def build_model(self) -> keras.Model:
        """
        Construye la arquitectura del modelo.
        
        Returns:
            Modelo de Keras compilado
        """
        hp = self.hyperparameters
        
        model = Sequential()
        
        # Input layer
        model.add(layers.Dense(
            hp['units_per_layer'][0],
            activation=hp['activation'],
            input_shape=(self.input_dim,),
        ))
        model.add(layers.Dropout(hp['dropout_rate']))
        
        # Hidden layers
        for i in range(1, min(hp['hidden_layers'], len(hp['units_per_layer']))):
            model.add(layers.Dense(
                hp['units_per_layer'][i],
                activation=hp['activation'],
            ))
            model.add(layers.Dropout(hp['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = self._get_optimizer(hp['optimizer'], hp['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def _get_optimizer(self, optimizer_name: str, learning_rate: float):
        """Retorna el optimizador configurado."""
        optimizers = {
            'adam': keras.optimizers.Adam(learning_rate=learning_rate),
            'sgd': keras.optimizers.SGD(learning_rate=learning_rate),
            'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate)
        }
        return optimizers.get(optimizer_name.lower(), optimizers['adam'])
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ):
        """
        Entrena el modelo.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            verbose: Nivel de verbosidad
            
        Returns:
            Historial de entrenamiento
        """
        if self.model is None:
            self.build_model()
        
        hp = self.hyperparameters
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=hp['batch_size'],
            epochs=hp['epochs'],
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X: Características
            
        Returns:
            Probabilidades de churn
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo.
        
        Args:
            X: Características
            y: Etiquetas verdaderas
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado o cargado.")
        
        # Evaluate model
        results = self.model.evaluate(X, y, verbose=0)
        
        # Get metric names from model
        metric_names = self.model.metrics_names
        
        # Create dictionary with metric names and values
        metrics = dict(zip(metric_names, results))
        
        # Add additional metrics with sklearn for consistency
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = (self.predict(X) > 0.5).astype(int).flatten()
        
        # Use sklearn for precision and recall (more reliable)
        metrics['precision'] = float(precision_score(y, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y, y_pred, zero_division=0))
        
        return metrics
    
    def save_model(self, filepath: str):
        """Guarda el modelo en disco."""
        if self.model is None:
            raise ValueError("No hay modelo para guardar.")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Carga el modelo desde disco."""
        self.model = keras.models.load_model(filepath)
