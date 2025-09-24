import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Optional
from pydantic import validate_arguments
from ..config import config

class DataPreprocessor:
    """Data preprocessing and feature engineering class"""
    
    def __init__(self):
        self.categorical_columns = [
            "mental_health_history", "headache", "blood_pressure", 
            "sleep_quality", "breathing_problem", "noise_level", 
            "living_conditions", "safety", "basic_needs", 
            "academic_performance", "study_load", 
            "teacher_student_relationship", "future_career_concerns", 
            "social_support", "peer_pressure", "extracurricular_activities", 
            "bullying"
        ]
    
    # Remove @validate_arguments decorator for DataFrame methods
    def prepare_features_target(self, df: pd.DataFrame, target_column: str = "stress_level") -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return X, y
    
    # Remove @validate_arguments decorator for DataFrame methods
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: Optional[float] = None) -> Tuple:
        """Split data into train and test sets"""
        test_size = test_size or config.test_size
        return train_test_split(X, y, test_size=test_size, random_state=config.random_state)
    
    def create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical columns"""
        return pd.get_dummies(df, columns=self.categorical_columns, dtype=float)
    
    # Keep @validate_arguments for simple parameter validation
    @validate_arguments
    def create_preprocessing_pipeline(self, scaler_type: str = "minmax") -> Pipeline:
        """Create preprocessing pipeline"""
        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        
        return Pipeline([('scaler', scaler)])