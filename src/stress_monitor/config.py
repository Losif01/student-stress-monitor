from pydantic_settings import BaseSettings
from typing import Dict, Any
from enum import Enum

class DataSource(str, Enum):
    STRESS_LEVEL = "stress_level"
    STRESS_DATA = "stress_data"

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    TWEEBIE_REGRESSOR = "tweedie_regressor"
    SGD_CLASSIFIER = "sgd_classifier"
    CATBOOST = "catboost"
    MLP_CLASSIFIER = "mlp_classifier"

class AppConfig(BaseSettings):
    # Data paths
    data_directory: str = "data"
    stress_level_file: str = "StressLevelDataset.csv"
    stress_data_file: str = "Stress_Dataset.csv"
    
    # Model parameters
    test_size: float = 0.2
    random_state: int = 104
    
    # Model specific parameters
    model_params: Dict[str, Dict[str, Any]] = {
        ModelType.GRADIENT_BOOSTING: {"n_estimators": 100, "max_depth": 30},
        ModelType.RANDOM_FOREST: {"n_estimators": 100, "max_depth": 30},
        ModelType.TWEEBIE_REGRESSOR: {"power": 1, "alpha": 0.5, "link": "log"},
        ModelType.SGD_CLASSIFIER: {"max_iter": 1000, "tol": 1e-3},
        ModelType.CATBOOST: {
            "iterations": 500, 
            "depth": 6, 
            "learning_rate": 0.05,
        },
        ModelType.MLP_CLASSIFIER: {"max_iter": 300, "verbose": False}
    }
    
    class Config:
        env_file = ".env"

config = AppConfig()
