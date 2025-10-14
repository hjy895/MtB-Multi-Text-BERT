"""
Clinical Topic Modeling Framework - Main Module

This module provides the main framework class that orchestrates all components
for clinical topic modeling, feature extraction, and model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import yaml
import warnings
from loguru import logger

# Import framework components
from .data.data_loader import ClinicalDataLoader
from .data.topic_extractor import ClinicalTopicExtractor
from .models.bert_classifier import BERTTopicClassifier, MultilevelBERTClassifier
from .models.classical_models import create_classical_models
from .models.ensemble_models import create_ensemble_models
from .evaluation.evaluator import ModelEvaluator, BenchmarkEvaluator

warnings.filterwarnings('ignore')


class ClinicalTopicModelingFramework:
    """
    Main framework class for clinical topic modeling and prediction.
    
    Orchestrates data loading, topic extraction, model training, and evaluation
    for clinical prediction tasks using multi-level text extraction and BERT-based
    modeling approaches.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the clinical topic modeling framework.
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (alternative to config_path)
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.data_loader = ClinicalDataLoader(config_path if config_path else None)
        self.topic_extractor = ClinicalTopicExtractor(self.config)
        self.evaluator = ModelEvaluator(
            cv_folds=self.config.get('evaluation', {}).get('cv_folds', 5),
            random_state=self.config.get('reproducibility', {}).get('seed', 42)
        )
        self.benchmark_evaluator = BenchmarkEvaluator(self.config)
        
        # Data storage
        self.data = None
        self.clinical_texts = None
        self.labels = None
        self.models = {}
        self.results = {}
        
        logger.info("Clinical Topic Modeling Framework initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'target_column': 'target_label',
                'test_size': 0.2,
                'random_state': 42
            },
            'clinical_topics': {
                'enable': True,
                'feature_columns': [],
                'topic_definitions': {}
            },
            'feature_extraction': {
                'levels': ['word', 'sentence', 'paragraph'],
                'word_extraction_sizes': [5, 10, 15, 20],
                'sentence_extraction_sizes': [1, 2, 3],
                'paragraph_extraction_sizes': [0, 1, 2]
            },
            'models': {
                'bert': {
                    'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    'max_length': 128
                }
            },
            'evaluation': {
                'task_type': 'binary',
                'cv_folds': 5,
                'random_state': 42
            }
        }
    
    def load_data(self, 
                  file_path: Optional[str] = None,
                  data: Optional[pd.DataFrame] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load clinical dataset.
        
        Args:
            file_path: Path to dataset file
            data: DataFrame to use directly
            **kwargs: Additional arguments for data loading
            
        Returns:
            Loaded and preprocessed DataFrame
        """
        logger.info("Loading clinical data...")
        
        if data is not None:
            self.data = data.copy()
        elif file_path:
            self.data = self.data_loader.load_data(file_path, **kwargs)
        elif 'data' in self.config and 'input_file' in self.config['data']:
            self.data = self.data_loader.load_data(self.config['data']['input_file'], **kwargs)
        else:
            raise ValueError("No data source specified")
        
        # Validate and preprocess data
        validation_results = self.data_loader.validate_data(self.data)
        logger.info(f"Data loaded: {validation_results['shape']}")
        
        # Preprocess data
        self.data = self.data_loader.preprocess_data(self.data)
        
        return self.data
    
    def extract_clinical_topics(self, 
                              data: Optional[pd.DataFrame] = None,
                              create_labels: bool = True) -> pd.Series:
        """
        Extract clinical topics from numerical data.
        
        Args:
            data: DataFrame to extract topics from (uses self.data if None)
            create_labels: Whether to create labels based on topics
            
        Returns:
            Series with clinical text representations
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data available for topic extraction")
        
        logger.info("Extracting clinical topics...")
        
        # Extract clinical topics
        self.clinical_texts = self.topic_extractor.create_clinical_text(
            data, use_multiprocessing=True
        )
        
        # Create labels if requested and no target column exists
        if create_labels:
            try:
                target_col = self.data_loader.get_target_column()
                if target_col in data.columns:
                    self.labels = data[target_col]
                    logger.info("Using existing target column for labels")
                else:
                    raise ValueError("Target column not found")
            except ValueError:
                # Create labels based on topics
                self.labels = self.topic_extractor.create_topic_labels(
                    data, self.clinical_texts, label_strategy='count_based'
                )
                logger.info("Created labels based on topic counts")
        
        # Get topic statistics
        topic_stats = self.topic_extractor.get_topic_statistics(self.clinical_texts)
        logger.info(f"Topic extraction completed. {topic_stats['patients_with_topics']} patients with topics")
        
        return self.clinical_texts
    
    def create_multilevel_extractions(self, 
                                    clinical_texts: Optional[pd.Series] = None) -> Dict[str, Dict]:
        """
        Create multi-level text extractions for different granularities.
        
        Args:
            clinical_texts: Clinical text data (uses self.clinical_texts if None)
            
        Returns:
            Dictionary with multi-level extractions
        """
        if clinical_texts is None:
            clinical_texts = self.clinical_texts
        
        if clinical_texts is None:
            raise ValueError("No clinical texts available for extraction")
        
        logger.info("Creating multi-level text extractions...")
        
        extractions = {}
        
        # Get extraction parameters from config
        extraction_config = self.config.get('feature_extraction', {})
        
        # Word-level extractions
        word_sizes = extraction_config.get('word_extraction_sizes', [5, 10, 15, 20])
        word_extractions = {}
        for size in word_sizes:
            word_extractions[size] = self.topic_extractor.extract_text_features(
                clinical_texts, 'word', size
            )
        extractions['word'] = word_extractions
        
        # Sentence-level extractions
        sentence_sizes = extraction_config.get('sentence_extraction_sizes', [1, 2, 3])
        sentence_extractions = {}
        for size in sentence_sizes:
            sentence_extractions[size] = self.topic_extractor.extract_text_features(
                clinical_texts, 'sentence', size
            )
        extractions['sentence'] = sentence_extractions
        
        # Paragraph-level extractions
        paragraph_sizes = extraction_config.get('paragraph_extraction_sizes', [0, 1, 2])
        paragraph_extractions = {}
        for size in paragraph_sizes:
            paragraph_extractions[size] = self.topic_extractor.extract_text_features(
                clinical_texts, 'paragraph', size
            )
        extractions['paragraph'] = paragraph_extractions
        
        # Add baseline (full text)
        baseline_text = clinical_texts.copy()
        for extraction_type in extractions:
            extractions[extraction_type]['Baseline'] = baseline_text
        
        logger.info("Multi-level extractions created")
        return extractions
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create all models for evaluation.
        
        Returns:
            Dictionary of initialized models
        """
        logger.info("Creating models...")
        
        models = {}
        
        # Classical models
        classical_models = create_classical_models()
        models.update(classical_models)
        
        # Ensemble models
        ensemble_models = create_ensemble_models()
        models.update(ensemble_models)
        
        # BERT-based models
        bert_config = self.config.get('models', {}).get('bert', {})
        model_name = bert_config.get('model_name', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        max_length = bert_config.get('max_length', 128)
        
        # BERT + Logistic Regression
        models['BERT-LR'] = BERTTopicClassifier(
            bert_model_name=model_name,
            classifier_type='logistic_regression',
            max_length=max_length
        )
        
        # BERT + SVM
        models['BERT-SVM'] = BERTTopicClassifier(
            bert_model_name=model_name,
            classifier_type='svm',
            max_length=max_length
        )
        
        # BERT + Ensemble
        models['BERT-Ensemble'] = BERTTopicClassifier(
            bert_model_name=model_name,
            classifier_type='ensemble',
            max_length=max_length
        )
        
        # Multi-level BERT classifier
        extraction_config = self.config.get('feature_extraction', {})
        models['BERT-Multilevel'] = MultilevelBERTClassifier(
            bert_model_name=model_name,
            word_extraction_sizes=extraction_config.get('word_extraction_sizes', [5, 10, 15]),
            sentence_extraction_sizes=extraction_config.get('sentence_extraction_sizes', [1, 2, 3]),
            paragraph_extraction_sizes=extraction_config.get('paragraph_extraction_sizes', [0, 1, 2])
        )
        
        self.models = models
        logger.info(f"Created {len(models)} models")
        return models
    
    def run_evaluation(self,
                      models: Optional[Dict] = None,
                      extraction_data: Optional[Dict] = None,
                      labels: Optional[Union[List, np.ndarray, pd.Series]] = None,
                      task_type: str = 'binary') -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all models and extraction methods.
        
        Args:
            models: Dictionary of models to evaluate (uses self.models if None)
            extraction_data: Multi-level extraction data
            labels: Target labels (uses self.labels if None)
            task_type: Type of classification task ('binary', 'multiclass')
            
        Returns:
            Comprehensive evaluation results
        """
        if models is None:
            models = self.models if self.models else self.create_models()
        
        if labels is None:
            labels = self.labels
        
        if labels is None:
            raise ValueError("No labels available for evaluation")
        
        if extraction_data is None:
            extraction_data = self.create_multilevel_extractions()
        
        logger.info(f"Running {task_type} classification evaluation...")
        
        # Run comprehensive benchmark
        results = self.benchmark_evaluator.run_comprehensive_benchmark(
            models, extraction_data, labels, task_type
        )
        
        self.results = results
        logger.info("Evaluation completed")
        
        return results
    
    def run_binary_classification(self,
                                 data: Optional[pd.DataFrame] = None,
                                 target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete binary classification pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Binary classification results
        """
        logger.info("Running binary classification pipeline...")
        
        # Load data if provided
        if data is not None:
            self.load_data(data=data)
        
        # Set target column
        if target_column:
            self.config['data']['target_column'] = target_column
        
        # Extract topics
        self.extract_clinical_topics()
        
        # Create extractions
        extraction_data = self.create_multilevel_extractions()
        
        # Create models
        models = self.create_models()
        
        # Run evaluation
        results = self.run_evaluation(models, extraction_data, task_type='binary')
        
        logger.info("Binary classification pipeline completed")
        return results
    
    def run_multiclass_classification(self,
                                    data: Optional[pd.DataFrame] = None,
                                    target_column: Optional[str] = None,
                                    num_classes: int = 3) -> Dict[str, Any]:
        """
        Run complete multiclass classification pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            num_classes: Number of classes
            
        Returns:
            Multiclass classification results
        """
        logger.info(f"Running {num_classes}-class classification pipeline...")
        
        # Update config for multiclass
        self.config['evaluation']['task_type'] = 'multiclass'
        self.config['evaluation']['num_classes'] = num_classes
        
        # Load data if provided
        if data is not None:
            self.load_data(data=data)
        
        # Set target column
        if target_column:
            self.config['data']['target_column'] = target_column
        
        # Extract topics
        self.extract_clinical_topics()
        
        # Create extractions
        extraction_data = self.create_multilevel_extractions()
        
        # Create models
        models = self.create_models()
        
        # Run evaluation
        results = self.run_evaluation(models, extraction_data, task_type='multiclass')
        
        logger.info("Multiclass classification pipeline completed")
        return results
    
    def get_best_model(self, metric: str = 'score') -> Dict[str, Any]:
        """
        Get the best performing model configuration.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Best model information
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return {}
        
        best_overall = self.results.get('benchmark_summary', {}).get('best_overall', {})
        
        if best_overall:
            logger.info(f"Best model: {best_overall['model']} "
                       f"({best_overall['extraction_type']}, size {best_overall['extraction_size']}) "
                       f"- Score: {best_overall['score']:.4f}")
        
        return best_overall
    
    def generate_report(self, save_path: Optional[str] = None) ->