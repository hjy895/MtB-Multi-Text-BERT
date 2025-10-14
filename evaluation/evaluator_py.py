"""
Model Evaluation Module

This module provides comprehensive evaluation functionality for clinical
topic modeling framework, including cross-validation, metrics computation,
and results analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation for clinical topic modeling.
    
    Provides cross-validation, metrics computation, statistical analysis,
    and visualization capabilities for model comparison.
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 random_state: int = 42,
                 scoring: Optional[List[str]] = None):
        """
        Initialize model evaluator.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            scoring: List of scoring metrics to compute
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring or ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Cross-validation splitter
        self.cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
        
        # Results storage
        self.results = {}
        self.detailed_results = {}
    
    def evaluate_model(self,
                      model: BaseEstimator,
                      X: Union[List, np.ndarray, pd.Series],
                      y: Union[List, np.ndarray, pd.Series],
                      model_name: str,
                      use_vectorization: bool = True) -> Dict[str, float]:
        """
        Evaluate a single model using cross-validation.
        
        Args:
            model: Sklearn-compatible model to evaluate
            X: Input features or texts
            y: Target labels
            model_name: Name identifier for the model
            use_vectorization: Whether to apply CountVectorizer to text data
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            # Prepare data
            X_processed = self._prepare_features(X, use_vectorization)
            y_processed = np.array(y)
            
            # Perform cross-validation
            cv_results = self._perform_cross_validation(model, X_processed, y_processed)
            
            # Compute additional metrics
            additional_metrics = self._compute_additional_metrics(
                model, X_processed, y_processed
            )
            
            # Combine results
            results = {**cv_results, **additional_metrics}
            
            # Store results
            self.results[model_name] = results
            
            logger.info(f"Model {model_name} - Accuracy: {results.get('accuracy_mean', 0):.4f} ± {results.get('accuracy_std', 0):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            # Return default results
            default_results = {f"{metric}_mean": 0.0 for metric in self.scoring}
            default_results.update({f"{metric}_std": 0.0 for metric in self.scoring})
            return default_results
    
    def evaluate_multiple_models(self,
                                models: Dict[str, BaseEstimator],
                                X: Union[List, np.ndarray, pd.Series],
                                y: Union[List, np.ndarray, pd.Series],
                                use_vectorization: bool = True) -> pd.DataFrame:
        """
        Evaluate multiple models and return comparison results.
        
        Args:
            models: Dictionary of {model_name: model} pairs
            X: Input features or texts
            y: Target labels
            use_vectorization: Whether to apply CountVectorizer
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Evaluating {len(models)} models...")
        
        all_results = {}
        
        for model_name, model in models.items():
            results = self.evaluate_model(
                model, X, y, model_name, use_vectorization
            )
            all_results[model_name] = results
        
        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame(all_results).T
        
        # Sort by accuracy (descending)
        if 'accuracy_mean' in results_df.columns:
            results_df = results_df.sort_values('accuracy_mean', ascending=False)
        
        logger.info("Multi-model evaluation completed")
        return results_df
    
    def evaluate_extraction_methods(self,
                                  models: Dict[str, BaseEstimator],
                                  extraction_data: Dict[str, Dict],
                                  y: Union[List, np.ndarray, pd.Series],
                                  extraction_types: List[str] = ['word', 'sentence', 'paragraph']) -> Dict[str, pd.DataFrame]:
        """
        Evaluate models across different text extraction methods.
        
        Args:
            models: Dictionary of models to evaluate
            extraction_data: Nested dict with extraction methods and sizes
            y: Target labels
            extraction_types: Types of extraction to evaluate
            
        Returns:
            Dictionary of results DataFrames for each extraction type
        """
        logger.info("Evaluating extraction methods...")
        
        all_extraction_results = {}
        
        for extraction_type in extraction_types:
            if extraction_type not in extraction_data:
                continue
                
            logger.info(f"Evaluating {extraction_type} extraction...")
            
            extraction_results = {}
            type_data = extraction_data[extraction_type]
            
            for extraction_size, texts in type_data.items():
                logger.info(f"Processing {extraction_type} extraction, size {extraction_size}")
                
                size_results = {}
                for model_name, model in models.items():
                    try:
                        results = self.evaluate_model(
                            model, texts, y, f"{model_name}_{extraction_type}_{extraction_size}"
                        )
                        size_results[model_name] = results.get('accuracy_mean', 0.0)
                    except Exception as e:
                        logger.warning(f"Error with {model_name} on {extraction_type}_{extraction_size}: {e}")
                        size_results[model_name] = 0.0
                
                extraction_results[extraction_size] = size_results
            
            # Convert to DataFrame
            if extraction_results:
                results_df = pd.DataFrame(extraction_results).T
                results_df.index.name = 'extraction_size'
                all_extraction_results[extraction_type] = results_df
        
        logger.info("Extraction method evaluation completed")
        return all_extraction_results
    
    def _prepare_features(self, X, use_vectorization: bool):
        """Prepare features for model training."""
        if use_vectorization and self._is_text_data(X):
            # Apply count vectorization for text data
            vectorizer = CountVectorizer(binary=True, max_features=5000)
            X_vectorized = vectorizer.fit_transform(X)
            return X_vectorized
        else:
            # Return as-is for non-text data or BERT models
            return X
    
    def _is_text_data(self, X) -> bool:
        """Check if input data is text data."""
        if isinstance(X, (list, np.ndarray, pd.Series)):
            # Check first few elements
            sample = X[:5] if len(X) > 5 else X
            return all(isinstance(item, str) for item in sample)
        return False
    
    def _perform_cross_validation(self, model, X, y) -> Dict[str, float]:
        """Perform cross-validation and return metric statistics."""
        cv_results = {}
        
        # Standard cross-validation metrics
        for metric in self.scoring:
            try:
                scores = cross_val_score(
                    model, X, y, cv=self.cv, scoring=metric, n_jobs=-1
                )
                cv_results[f"{metric}_mean"] = np.mean(scores)
                cv_results[f"{metric}_std"] = np.std(scores)
                cv_results[f"{metric}_scores"] = scores.tolist()
                
            except Exception as e:
                logger.warning(f"Could not compute {metric}: {e}")
                cv_results[f"{metric}_mean"] = 0.0
                cv_results[f"{metric}_std"] = 0.0
        
        return cv_results
    
    def _compute_additional_metrics(self, model, X, y) -> Dict[str, Any]:
        """Compute additional metrics not available in cross_val_score."""
        try:
            # Fit model on full data for additional analysis
            model.fit(X, y)
            y_pred = model.predict(X)
            
            additional_metrics = {
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
            
            # Add probability-based metrics if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                if y_proba.shape[1] == 2:  # Binary classification
                    additional_metrics['roc_auc_full'] = roc_auc_score(y, y_proba[:, 1])
                    additional_metrics['pr_auc_full'] = average_precision_score(y, y_proba[:, 1])
            
            return additional_metrics
            
        except Exception as e:
            logger.warning(f"Error computing additional metrics: {e}")
            return {}
    
    def get_best_models(self, 
                       top_k: int = 5,
                       metric: str = 'accuracy_mean') -> pd.DataFrame:
        """
        Get top-k best performing models.
        
        Args:
            top_k: Number of top models to return
            metric: Metric to use for ranking
            
        Returns:
            DataFrame with top models
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results).T
        
        if metric not in results_df.columns:
            logger.warning(f"Metric {metric} not found. Using first available metric.")
            metric = results_df.columns[0]
        
        top_models = results_df.nlargest(top_k, metric)
        
        return top_models
    
    def compare_models(self, 
                      model_names: Optional[List[str]] = None,
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare models across multiple metrics.
        
        Args:
            model_names: List of model names to compare (None for all)
            metrics: List of metrics to include (None for all available)
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results).T
        
        # Filter models if specified
        if model_names:
            available_models = [name for name in model_names if name in results_df.index]
            if available_models:
                results_df = results_df.loc[available_models]
        
        # Filter metrics if specified
        if metrics:
            available_metrics = [col for col in results_df.columns 
                               if any(metric in col for metric in metrics)]
            if available_metrics:
                results_df = results_df[available_metrics]
        
        return results_df
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all evaluations.
        
        Returns:
            Dictionary with summary statistics and insights
        """
        if not self.results:
            return {"error": "No evaluation results available"}
        
        results_df = pd.DataFrame(self.results).T
        
        # Basic statistics
        summary = {
            "total_models_evaluated": len(results_df),
            "metrics_computed": [col.replace('_mean', '') for col in results_df.columns if '_mean' in col],
            "best_overall_model": None,
            "performance_statistics": {},
            "model_rankings": {}
        }
        
        # Find best model
        if 'accuracy_mean' in results_df.columns:
            best_model_idx = results_df['accuracy_mean'].idxmax()
            summary["best_overall_model"] = {
                "name": best_model_idx,
                "accuracy": results_df.loc[best_model_idx, 'accuracy_mean'],
                "std": results_df.loc[best_model_idx, 'accuracy_std'] if 'accuracy_std' in results_df.columns else None
            }
        
        # Performance statistics for each metric
        for col in results_df.columns:
            if '_mean' in col:
                metric = col.replace('_mean', '')
                summary["performance_statistics"][metric] = {
                    "mean": results_df[col].mean(),
                    "std": results_df[col].std(),
                    "min": results_df[col].min(),
                    "max": results_df[col].max(),
                    "range": results_df[col].max() - results_df[col].min()
                }
        
        # Model rankings for key metrics
        key_metrics = ['accuracy_mean', 'f1_mean', 'roc_auc_mean']
        for metric in key_metrics:
            if metric in results_df.columns:
                ranking = results_df[metric].sort_values(ascending=False)
                summary["model_rankings"][metric.replace('_mean', '')] = ranking.to_dict()
        
        return summary
    
    def plot_model_comparison(self,
                            metric: str = 'accuracy_mean',
                            top_k: int = 10,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None):
        """
        Plot model comparison visualization.
        
        Args:
            metric: Metric to plot
            top_k: Number of top models to include
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No evaluation results available for plotting")
            return
        
        results_df = pd.DataFrame(self.results).T
        
        if metric not in results_df.columns:
            logger.warning(f"Metric {metric} not found")
            return
        
        # Get top-k models
        top_models = results_df.nlargest(top_k, metric)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Bar plot
        bars = plt.bar(range(len(top_models)), top_models[metric])
        
        # Add error bars if std is available
        std_col = metric.replace('_mean', '_std')
        if std_col in top_models.columns:
            plt.errorbar(range(len(top_models)), top_models[metric], 
                        yerr=top_models[std_col], fmt='none', color='black', capsize=5)
        
        # Customize plot
        plt.xlabel('Models')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(range(len(top_models)), top_models.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (idx, value) in enumerate(top_models[metric].items()):
            plt.text(i, value + 0.001, f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_metric_distribution(self,
                               metrics: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None):
        """
        Plot distribution of metrics across all models.
        
        Args:
            metrics: List of metrics to plot (None for all)
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No evaluation results available for plotting")
            return
        
        results_df = pd.DataFrame(self.results).T
        
        # Select metrics to plot
        if metrics is None:
            metric_cols = [col for col in results_df.columns if '_mean' in col]
        else:
            metric_cols = [f"{m}_mean" for m in metrics if f"{m}_mean" in results_df.columns]
        
        if not metric_cols:
            logger.warning("No valid metrics found for plotting")
            return
        
        # Create subplots
        n_metrics = len(metric_cols)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metric_cols):
            if i < len(axes):
                ax = axes[i]
                
                # Histogram
                ax.hist(results_df[metric], bins=15, alpha=0.7, edgecolor='black')
                ax.set_title(metric.replace('_mean', '').replace('_', ' ').title())
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                
                # Add mean line
                mean_val = results_df[metric].mean()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                ax.legend()
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, 
                      file_path: str,
                      format: str = 'csv'):
        """
        Export evaluation results to file.
        
        Args:
            file_path: Path to save the results
            format: Export format ('csv', 'excel', 'json')
        """
        if not self.results:
            logger.warning("No evaluation results to export")
            return
        
        results_df = pd.DataFrame(self.results).T
        
        try:
            if format.lower() == 'csv':
                results_df.to_csv(file_path)
            elif format.lower() in ['excel', 'xlsx']:
                results_df.to_excel(file_path)
            elif format.lower() == 'json':
                results_df.to_json(file_path, indent=2)