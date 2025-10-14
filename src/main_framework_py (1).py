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
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        if not self.results:
            logger.warning("No evaluation results available for report generation")
            return "No evaluation results available"
        
        report = self.benchmark_evaluator.generate_benchmark_report(save_path)
        
        if save_path:
            logger.info(f"Report saved to {save_path}")
        
        return report
    
    def save_results(self, save_dir: str):
        """
        Save all results to directory.
        
        Args:
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save extraction results
        if 'extraction_results' in self.results:
            for extraction_type, results_df in self.results['extraction_results'].items():
                if not results_df.empty:
                    file_path = save_path / f"{extraction_type}_results.csv"
                    results_df.to_csv(file_path)
        
        # Save benchmark summary
        if 'benchmark_summary' in self.results:
            summary_path = save_path / "benchmark_summary.yaml"
            with open(summary_path, 'w') as f:
                yaml.dump(self.results['benchmark_summary'], f)
        
        # Save comprehensive report
        report_path = save_path / "evaluation_report.txt"
        self.generate_report(str(report_path))
        
        # Save configuration
        config_path = save_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Results saved to {save_dir}")
    
    def load_results(self, save_dir: str):
        """
        Load previously saved results.
        
        Args:
            save_dir: Directory containing saved results
        """
        save_path = Path(save_dir)
        
        if not save_path.exists():
            logger.error(f"Save directory not found: {save_dir}")
            return
        
        # Load extraction results
        extraction_results = {}
        for extraction_type in ['word', 'sentence', 'paragraph']:
            file_path = save_path / f"{extraction_type}_results.csv"
            if file_path.exists():
                extraction_results[extraction_type] = pd.read_csv(file_path, index_col=0)
        
        # Load benchmark summary
        summary_path = save_path / "benchmark_summary.yaml"
        benchmark_summary = {}
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                benchmark_summary = yaml.safe_load(f)
        
        self.results = {
            'extraction_results': extraction_results,
            'benchmark_summary': benchmark_summary
        }
        
        logger.info(f"Results loaded from {save_dir}")
    
    def plot_results(self, 
                    save_dir: Optional[str] = None,
                    show_plots: bool = True):
        """
        Generate visualization plots for results.
        
        Args:
            save_dir: Directory to save plots
            show_plots: Whether to display plots
        """
        if not self.results or 'extraction_results' not in self.results:
            logger.warning("No results available for plotting")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Extraction method comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (extraction_type, results_df) in enumerate(self.results['extraction_results'].items()):
            if results_df.empty:
                continue
                
            ax = axes[i]
            
            # Get top 10 models
            best_scores = results_df.max(axis=1).nlargest(10)
            
            # Create heatmap
            plot_data = results_df.loc[best_scores.index]
            sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title(f'{extraction_type.title()} Extraction Results')
            ax.set_xlabel('Models')
            ax.set_ylabel('Extraction Size')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_path / 'extraction_comparison.png', dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Plot 2: Best model comparison across extraction types
        plt.figure(figsize=(12, 8))
        
        extraction_best = {}
        for extraction_type, results_df in self.results['extraction_results'].items():
            if not results_df.empty:
                best_score = results_df.max().max()
                best_model = results_df.max().idxmax()
                extraction_best[extraction_type] = best_score
        
        if extraction_best:
            plt.bar(extraction_best.keys(), extraction_best.values())
            plt.title('Best Performance by Extraction Method')
            plt.xlabel('Extraction Method')
            plt.ylabel('Best Accuracy Score')
            plt.ylim(0, 1)
            
            # Add value labels
            for i, (method, score) in enumerate(extraction_best.items()):
                plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_path / 'best_by_extraction.png', dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Plot 3: Model category comparison
        if 'best_models' in self.results:
            category_scores = {}
            for category, info in self.results['best_models'].items():
                if info and 'score' in info:
                    category_scores[category] = info['score']
            
            if category_scores:
                plt.figure(figsize=(10, 6))
                plt.bar(category_scores.keys(), category_scores.values(), 
                       color=['skyblue', 'lightcoral', 'lightgreen'])
                plt.title('Best Performance by Model Category')
                plt.xlabel('Model Category')
                plt.ylabel('Best Accuracy Score')
                plt.ylim(0, 1)
                
                # Add value labels
                for i, (category, score) in enumerate(category_scores.items()):
                    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                if save_dir:
                    plt.savefig(save_path / 'category_comparison.png', dpi=300, bbox_inches='tight')
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
        
        logger.info("Plots generated successfully")
    
    def add_custom_model(self, name: str, model: Any):
        """
        Add a custom model to the framework.
        
        Args:
            name: Name for the model
            model: Sklearn-compatible model instance
        """
        if self.models is None:
            self.models = {}
        
        self.models[name] = model
        logger.info(f"Added custom model: {name}")
    
    def add_custom_topic_definition(self, 
                                  topic_name: str,
                                  column: str,
                                  operator: str,
                                  threshold: Union[float, int]):
        """
        Add a custom topic definition.
        
        Args:
            topic_name: Name of the clinical topic
            column: Column name to evaluate
            operator: Comparison operator
            threshold: Threshold value
        """
        self.topic_extractor.add_topic_definition(topic_name, column, operator, threshold)
        logger.info(f"Added custom topic definition: {topic_name}")
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get a summary of model performance across all extraction methods.
        
        Returns:
            DataFrame with performance summary
        """
        if not self.results or 'extraction_results' not in self.results:
            logger.warning("No results available")
            return pd.DataFrame()
        
        summary_data = []
        
        for extraction_type, results_df in self.results['extraction_results'].items():
            if results_df.empty:
                continue
            
            for model in results_df.columns:
                best_score = results_df[model].max()
                best_size = results_df[model].idxmax()
                
                summary_data.append({
                    'model': model,
                    'extraction_type': extraction_type,
                    'best_score': best_score,
                    'best_extraction_size': best_size
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            # Find overall best for each model
            model_best = summary_df.groupby('model')['best_score'].max().reset_index()
            model_best = model_best.sort_values('best_score', ascending=False)
            
            return model_best
        
        return pd.DataFrame()
    
    def export_results_to_excel(self, file_path: str):
        """
        Export all results to an Excel file with multiple sheets.
        
        Args:
            file_path: Path to save Excel file
        """
        if not self.results:
            logger.warning("No results to export")
            return
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Export extraction results
            if 'extraction_results' in self.results:
                for extraction_type, results_df in self.results['extraction_results'].items():
                    if not results_df.empty:
                        results_df.to_excel(writer, sheet_name=f'{extraction_type}_results')
            
            # Export performance summary
            summary_df = self.get_model_performance_summary()
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name='performance_summary', index=False)
            
            # Export best models
            if 'best_models' in self.results:
                best_models_df = pd.DataFrame(self.results['best_models']).T
                if not best_models_df.empty:
                    best_models_df.to_excel(writer, sheet_name='best_models')
            
            # Export benchmark summary
            if 'benchmark_summary' in self.results:
                # Convert benchmark summary to DataFrame format
                summary_items = []
                for key, value in self.results['benchmark_summary'].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            summary_items.append({
                                'category': key,
                                'metric': subkey,
                                'value': str(subvalue)
                            })
                    else:
                        summary_items.append({
                            'category': key,
                            'metric': 'value',
                            'value': str(value)
                        })
                
                if summary_items:
                    benchmark_df = pd.DataFrame(summary_items)
                    benchmark_df.to_excel(writer, sheet_name='benchmark_summary', index=False)
        
        logger.info(f"Results exported to {file_path}")
    
    def reset_framework(self):
        """Reset the framework to initial state."""
        self.data = None
        self.clinical_texts = None
        self.labels = None
        self.models = {}
        self.results = {}
        
        # Reinitialize components
        self.evaluator.clear_results()
        
        logger.info("Framework reset to initial state")


def create_framework_from_config(config_path: str) -> ClinicalTopicModelingFramework:
    """
    Create framework instance from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Initialized framework instance
    """
    return ClinicalTopicModelingFramework(config_path=config_path)


def run_quick_evaluation(data_path: str, 
                        target_column: str,
                        config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a quick evaluation with minimal setup.
    
    Args:
        data_path: Path to dataset file
        target_column: Name of target column
        config_path: Optional configuration file path
        
    Returns:
        Evaluation results
    """
    # Create framework
    framework = ClinicalTopicModelingFramework(config_path=config_path)
    
    # Load data
    framework.load_data(data_path)
    
    # Update target column
    framework.config['data']['target_column'] = target_column
    
    # Run binary classification
    results = framework.run_binary_classification()
    
    return results