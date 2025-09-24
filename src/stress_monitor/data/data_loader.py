import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from pydantic import validate_arguments
from ..config import config, DataSource

class DataLoader:
    """Data loader class for stress monitoring datasets"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(config.data_directory)
        
    @validate_arguments
    def load_dataset(self, data_source: DataSource) -> pd.DataFrame:
        """Load specified dataset"""
        file_map = {
            DataSource.STRESS_LEVEL: config.stress_level_file,
            DataSource.STRESS_DATA: config.stress_data_file
        }
        
        file_path = self.data_dir / file_map[data_source]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        return pd.read_csv(file_path)
    
    def load_all_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both datasets"""
        stresslvl_data = self.load_dataset(DataSource.STRESS_LEVEL)
        stress_data = self.load_dataset(DataSource.STRESS_DATA)
        return stresslvl_data, stress_data
    
    def get_dataset_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive dataset information report"""
        report = pd.DataFrame(index=df.columns)
        report['dtype'] = df.dtypes
        report['number_of_nulls'] = df.isnull().sum()
        report['ratio_of_nulls'] = df.isnull().sum() / df.shape[0]
        report['number_of_uniques'] = df.nunique()
        report['ratio_of_uniques'] = df.nunique() / df.shape[0]
        return report