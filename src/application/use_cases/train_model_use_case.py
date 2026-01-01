"""
Caso de uso para entrenar el modelo de churn con Nested Cross Validation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.domain.models.deep_learning_model import DeepLearningModel
from src.domain.services.preprocessing_service import PreprocessingService


class TrainModelUseCase:
    """
    Caso de uso que orquesta el entrenamiento del modelo con Nested Cross Validation.
    """
    
    def __init__(self, preprocessing_service: PreprocessingService):
        """
        Inicializa el caso de uso.
        
        Args:
            preprocessing_service: Servicio de preprocesamiento
        """
        self.preprocessing_service = preprocessing_service
    
    def nested_cross_validation(
        self,
        df: pd.DataFrame,
        outer_k: int = 5,
        inner_k: int = 3,
        hyperparameter_grid: List[Dict[str, Any]] = None,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Realiza Nested Cross Validation para entrenamiento y evaluación del modelo.
        
        Args:
            df: DataFrame con los datos
            outer_k: Número de folds para el CV externo
            inner_k: Número de folds para el CV interno (selección de hiperparámetros)
            hyperparameter_grid: Lista de diccionarios con combinaciones de hiperparámetros
            random_state: Semilla para reproducibilidad
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        if hyperparameter_grid is None:
            hyperparameter_grid = self._default_hyperparameter_grid()
        
        # Preprocesar datos una vez
        X, y = self.preprocessing_service.preprocess_pipeline(df, fit=True)
        
        # Nested Cross Validation
        outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        
        outer_fold_results = []
        best_hyperparams = None
        best_outer_score = -np.inf
        best_model = None
        
        print(f"Iniciando Nested Cross Validation: {outer_k} folds externos, {inner_k} folds internos")
        
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            print(f"\n{'='*60}")
            print(f"Fold Externo {outer_fold + 1}/{outer_k}")
            print(f"{'='*60}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV para selección de hiperparámetros
            best_inner_score = -np.inf
            best_inner_hyperparams = None
            
            for hyperparams in hyperparameter_grid:
                inner_scores = []
                
                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer)):
                    X_train_inner = X_train_outer[inner_train_idx]
                    X_val_inner = X_train_outer[inner_val_idx]
                    y_train_inner = y_train_outer[inner_train_idx]
                    y_val_inner = y_train_outer[inner_val_idx]
                    
                    # Entrenar modelo con estos hiperparámetros
                    model = DeepLearningModel(
                        input_dim=X_train_inner.shape[1],
                        hyperparameters=hyperparams
                    )
                    model.build_model()
                    
                    history = model.train(
                        X_train_inner,
                        y_train_inner,
                        X_val_inner,
                        y_val_inner,
                        verbose=0
                    )
                    
                    # Evaluar en validación
                    val_metrics = model.evaluate(X_val_inner, y_val_inner)
                    inner_scores.append(val_metrics['f1_score'])
                
                # Promedio de scores en inner CV
                mean_inner_score = np.mean(inner_scores)
                
                if mean_inner_score > best_inner_score:
                    best_inner_score = mean_inner_score
                    best_inner_hyperparams = hyperparams
            
            # Entrenar modelo final con mejores hiperparámetros en todo el conjunto de entrenamiento externo
            print(f"\nMejores hiperparámetros encontrados: {best_inner_hyperparams}")
            print(f"Score promedio en CV interno: {best_inner_score:.4f}")
            
            final_model = DeepLearningModel(
                input_dim=X_train_outer.shape[1],
                hyperparameters=best_inner_hyperparams
            )
            final_model.build_model()
            
            # Dividir entrenamiento externo en train/val para early stopping
            from sklearn.model_selection import train_test_split
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train_outer, y_train_outer, test_size=0.2, random_state=random_state
            )
            
            history = final_model.train(
                X_train_final,
                y_train_final,
                X_val_final,
                y_val_final,
                verbose=1
            )
            
            # Evaluar en conjunto de prueba externo
            test_metrics = final_model.evaluate(X_test_outer, y_test_outer)
            
            outer_fold_results.append({
                'fold': outer_fold + 1,
                'hyperparameters': best_inner_hyperparams,
                'test_metrics': test_metrics
            })
            
            print(f"\nMétricas en conjunto de prueba externo:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Actualizar mejor modelo global
            if test_metrics['f1_score'] > best_outer_score:
                best_outer_score = test_metrics['f1_score']
                best_hyperparams = best_inner_hyperparams
                best_model = final_model
        
        # Calcular métricas promedio
        avg_metrics = self._calculate_average_metrics(outer_fold_results)
        
        results = {
            'nested_cv_results': outer_fold_results,
            'average_metrics': avg_metrics,
            'best_hyperparameters': best_hyperparams,
            'best_model': best_model,
            'preprocessing_service': self.preprocessing_service
        }
        
        print(f"\n{'='*60}")
        print("RESUMEN FINAL - Nested Cross Validation")
        print(f"{'='*60}")
        print(f"Métricas promedio en {outer_k} folds externos:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\nMejores hiperparámetros: {best_hyperparams}")
        
        return results
    
    def _default_hyperparameter_grid(self) -> List[Dict[str, Any]]:
        """Retorna una grilla de hiperparámetros por defecto."""
        return [
            {
                'hidden_layers': 2,
                'units_per_layer': [64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'activation': 'relu',
                'optimizer': 'adam'
            },
            {
                'hidden_layers': 2,
                'units_per_layer': [128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'activation': 'relu',
                'optimizer': 'adam'
            },
            {
                'hidden_layers': 3,
                'units_per_layer': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.0005,
                'batch_size': 64,
                'epochs': 50,
                'activation': 'relu',
                'optimizer': 'adam'
            }
        ]
    
    def _calculate_average_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula el promedio de métricas a través de los folds."""
        metrics_list = [result['test_metrics'] for result in fold_results]
        
        avg_metrics = {}
        for metric_name in metrics_list[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in metrics_list])
        
        return avg_metrics

