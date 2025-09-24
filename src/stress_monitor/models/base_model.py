from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from typing import Any, Dict
from pydantic import BaseModel

class ModelResult(BaseModel):
    """Pydantic model for storing model results"""
    model_name: str
    accuracy: float
    parameters: Dict[str, Any]

class BaseStressModel(ABC):
    """Abstract base class for stress prediction models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipeline = None
        self.is_trained = False
    
    @abstractmethod
    def build_pipeline(self, **kwargs) -> Pipeline:
        """Build the model pipeline"""
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate model accuracy"""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        return self.pipeline.score(X, y)
    
    def get_result(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
        """Get model results"""
        accuracy = self.score(X_test, y_test)
        return ModelResult(
            model_name=self.model_name,
            accuracy=accuracy,
            parameters=self.get_parameters()
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.pipeline, 'get_params'):
            return self.pipeline.get_params()
        return {}