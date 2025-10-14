"""
Clinical Data Loader Module

This module provides functionality to load and preprocess clinical datasets
for the topic modeling framework.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from loguru import logger


class ClinicalDataLoader:
    """
    Loads and preprocesses clinical datasets for topic modeling.
    
    Supports various file formats and provides data validation,
    missing value handling, and train/test splitting functionality.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.data = None
        self.feature_columns = []
        self.target_column = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def load_data(self, 
                  file_path: str,
                  file_type: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load clinical dataset from file.
        
        Args:
            file_path: Path to the dataset file
            file_type: File type ('csv', 'excel', 'json'). Auto-detected if None
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Auto-detect file type if not specified
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip('.')
        
        try:
            if file_type in ['csv']:
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_type in ['xlsx', 'xls', 'excel']:
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_type in ['json']:
                self.data = pd.read_json(file_path, **kwargs)
            elif file_type in ['parquet']:
                self.data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            logger.info(f"Data loaded successfully: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def validate_data(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate the loaded dataset.
        
        Args:
            data: DataFrame to validate. Uses self.data if None
            
        Returns:
            Dictionary with validation results
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available for validation")
        
        validation_results = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'duplicates': data.duplicated().sum(),
            'columns': list(data.columns)
        }
        
        # Check for required columns from config
        if 'data' in self.config:
            required_cols = []
            
            if 'target_column' in self.config['data']:
                required_cols.append(self.config['data']['target_column'])
            if 'patient_id_column' in self.config['data']:
                required_cols.append(self.config['data']['patient_id_column'])
            if 'clinical_topics' in self.config and 'feature_columns' in self.config['clinical_topics']:
                required_cols.extend(self.config['clinical_topics']['feature_columns'])
            
            missing_cols = [col for col in required_cols if col not in data.columns]
            validation_results['missing_required_columns'] = missing_cols
        
        logger.info(f"Data validation completed: {validation_results['shape']}")
        return validation_results
    
    def preprocess_data(self, 
                       data: Optional[pd.DataFrame] = None,
                       handle_missing: str = 'drop',
                       remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Preprocess the clinical dataset.
        
        Args:
            data: DataFrame to preprocess. Uses self.data if None
            handle_missing: How to handle missing values ('drop', 'fill', 'ignore')
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            Preprocessed DataFrame
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
            
        if data is None:
            raise ValueError("No data available for preprocessing")
        
        original_shape = data.shape
        
        # Remove duplicates
        if remove_duplicates:
            data = data.drop_duplicates()
            logger.info(f"Removed {original_shape[0] - data.shape[0]} duplicate rows")
        
        # Handle missing values
        if handle_missing == 'drop':
            data = data.dropna()
            logger.info(f"Dropped rows with missing values: {original_shape[0] - data.shape[0]} rows removed")
        elif handle_missing == 'fill':
            # Fill numerical columns with median, categorical with mode
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col] = data[col].fillna(data[col].median())
                else:
                    data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown')
            logger.info("Filled missing values")
        
        # Validate data types and convert if necessary
        data = self._convert_data_types(data)
        
        self.data = data
        logger.info(f"Data preprocessing completed: {original_shape} -> {data.shape}")
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types based on configuration or heuristics."""
        # Convert target column to appropriate type
        if 'data' in self.config and 'target_column' in self.config['data']:
            target_col = self.config['data']['target_column']
            if target_col in data.columns:
                # Ensure target is integer for classification
                if data[target_col].dtype == 'float64':
                    data[target_col] = data[target_col].astype(int)
        
        return data
    
    def split_data(self, 
                   data: Optional[pd.DataFrame] = None,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            data: DataFrame to split. Uses self.data if None
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split based on target
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available for splitting")
        
        # Get parameters from config if available
        if 'data' in self.config:
            test_size = self.config['data'].get('test_size', test_size)
            random_state = self.config['data'].get('random_state', random_state)
        
        # Determine stratification column
        stratify_col = None
        if stratify and 'data' in self.config and 'target_column' in self.config['data']:
            target_col = self.config['data']['target_column']
            if target_col in data.columns:
                stratify_col = data[target_col]
        
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        logger.info(f"Data split: Train={train_data.shape}, Test={test_data.shape}")
        return train_data, test_data
    
    def get_feature_columns(self, data: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Get list of feature columns based on configuration.
        
        Args:
            data: DataFrame to extract features from
            
        Returns:
            List of feature column names
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available")
        
        # Get feature columns from config
        if 'clinical_topics' in self.config and 'feature_columns' in self.config['clinical_topics']:
            feature_cols = self.config['clinical_topics']['feature_columns']
            # Validate that columns exist in data
            available_cols = [col for col in feature_cols if col in data.columns]
            if len(available_cols) != len(feature_cols):
                missing = set(feature_cols) - set(available_cols)
                logger.warning(f"Some feature columns not found in data: {missing}")
            return available_cols
        else:
            # Auto-detect numerical columns as features
            exclude_cols = []
            if 'data' in self.config:
                if 'target_column' in self.config['data']:
                    exclude_cols.append(self.config['data']['target_column'])
                if 'patient_id_column' in self.config['data']:
                    exclude_cols.append(self.config['data']['patient_id_column'])
            
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numerical_cols if col not in exclude_cols]
            
            logger.info(f"Auto-detected {len(feature_cols)} feature