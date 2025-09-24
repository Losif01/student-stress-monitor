from sklearn.linear_model import LogisticRegression, TweedieRegressor, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from typing import Dict, Type
from .base_model import BaseStressModel
from ..config import config, ModelType
from ..data.data_preprocessor import DataPreprocessor

class LogisticRegressionModel(BaseStressModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("minmax").steps[0][1],
            LogisticRegression(**kwargs)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class GradientBoostingModel(BaseStressModel):
    def __init__(self):
        super().__init__("Gradient Boosting")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        params = config.model_params[ModelType.GRADIENT_BOOSTING].copy()
        params.update(kwargs)
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("minmax").steps[0][1],
            GradientBoostingClassifier(**params)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class RandomForestModel(BaseStressModel):
    def __init__(self):
        super().__init__("Random Forest")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        params = config.model_params[ModelType.RANDOM_FOREST].copy()
        params.update(kwargs)
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("minmax").steps[0][1],
            RandomForestClassifier(**params)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class TweedieRegressorModel(BaseStressModel):
    def __init__(self):
        super().__init__("Tweedie Regressor")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        params = config.model_params[ModelType.TWEEBIE_REGRESSOR].copy()
        params.update(kwargs)
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("minmax").steps[0][1],
            TweedieRegressor(**params)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class SGDClassifierModel(BaseStressModel):
    def __init__(self):
        super().__init__("SGD Classifier")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        params = config.model_params[ModelType.SGD_CLASSIFIER].copy()
        params.update(kwargs)
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("standard").steps[0][1],
            SGDClassifier(**params)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class CatBoostModel(BaseStressModel):
    def __init__(self):
        super().__init__("CatBoost")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        params = config.model_params[ModelType.CATBOOST].copy()
        params.update(kwargs)
        # CatBoost has built-in handling for categorical features, so we might not need scaling
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("standard").steps[0][1],
            CatBoostClassifier(**params, verbose=False)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class MLPClassifierModel(BaseStressModel):
    def __init__(self):
        super().__init__("MLP Classifier")
        self.preprocessor = DataPreprocessor()
    
    def build_pipeline(self, **kwargs):
        params = config.model_params[ModelType.MLP_CLASSIFIER].copy()
        params.update(kwargs)
        # MLP typically benefits from StandardScaler
        return make_pipeline(
            self.preprocessor.create_preprocessing_pipeline("standard").steps[0][1],
            MLPClassifier(**params)
        )
    
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        return self.pipeline.predict(X)

class ModelFactory:
    """Factory class for creating stress prediction models"""
    
    _models: Dict[ModelType, Type[BaseStressModel]] = {
        ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
        ModelType.GRADIENT_BOOSTING: GradientBoostingModel,
        ModelType.RANDOM_FOREST: RandomForestModel,
        ModelType.TWEEBIE_REGRESSOR: TweedieRegressorModel,
        ModelType.SGD_CLASSIFIER: SGDClassifierModel,
        ModelType.CATBOOST: CatBoostModel,
        ModelType.MLP_CLASSIFIER: MLPClassifierModel
    }
    
    @classmethod
    def create_model(cls, model_type: ModelType) -> BaseStressModel:
        """Create a model instance based on type"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type]()
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types"""
        return list(cls._models.keys())