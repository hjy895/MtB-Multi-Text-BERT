"""
Classical Machine Learning Models Module

This module provides classical machine learning models for clinical
topic modeling and classification tasks.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from loguru import logger


def create_classical_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create a comprehensive set of classical machine learning models.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of classical models
    """
    models = {
        # Linear Models
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='liblinear'
        ),
        
        'RidgeClassifier': RidgeClassifier(
            random_state=random_state
        ),
        
        'SGDClassifier': SGDClassifier(
            max_iter=1000,
            random_state=random_state,
            loss='hinge'
        ),
        
        # Support Vector Machines
        'SVC': SVC(
            kernel='linear',
            probability=True,
            random_state=random_state,
            C=1.0
        ),
        
        'LinearSVC': LinearSVC(
            random_state=random_state,
            C=1.0,
            max_iter=2000
        ),
        
        # Naive Bayes
        'MultinomialNB': MultinomialNB(
            alpha=1.0
        ),
        
        'BernoulliNB': BernoulliNB(
            alpha=1.0
        ),
        
        'ComplementNB': ComplementNB(
            alpha=1.0
        ),
        
        # Tree-based
        'DecisionTreeClassifier': DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        
        # Nearest Neighbors
        'KNeighborsClassifier': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        
        # Neural Networks
        'MLPClassifier': MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        ),
        
        # Discriminant Analysis
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis()
    }
    
    logger.info(f"Created {len(models)} classical models")
    return models


def create_optimized_classical_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create optimized versions of classical models with better hyperparameters.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of optimized classical models
    """
    models = {
        # Optimized Linear Models
        'OptimizedLogisticRegression': LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver='liblinear',
            C=0.1,
            penalty='l2'
        ),
        
        'OptimizedSVC': SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state,
            C=10.0,
            gamma='scale'
        ),
        
        'OptimizedDecisionTree': DecisionTreeClassifier(
            random_state=random_state,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion='gini'
        ),
        
        'OptimizedMLP': MLPClassifier(
            hidden_layer_sizes=(200, 100),
            max_iter=1000,
            random_state=random_state,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.15,
            alpha=0.001
        ),
        
        'OptimizedKNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='manhattan'
        ),
        
        'OptimizedMultinomialNB': MultinomialNB(
            alpha=0.5,
            fit_prior=True
        )
    }
    
    logger.info(f"Created {len(models)} optimized classical models")
    return models


def get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available classical models.
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        'LogisticRegression': {
            'category': 'Linear',
            'description': 'Linear classifier using logistic regression',
            'pros': ['Fast training', 'Interpretable', 'Probabilistic output'],
            'cons': ['Assumes linear separability', 'Sensitive to outliers'],
            'best_for': ['Binary classification', 'Baseline model', 'Feature selection']
        },
        
        'SVC': {
            'category': 'Support Vector Machine',
            'description': 'Support Vector Classifier with kernel trick',
            'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
            'cons': ['Slow on large datasets', 'Sensitive to feature scaling'],
            'best_for': ['High-dimensional data', 'Non-linear problems', 'Small to medium datasets']
        },
        
        'RandomForestClassifier': {
            'category': 'Ensemble',
            'description': 'Ensemble of decision trees with voting',
            'pros': ['Handles overfitting', 'Feature importance', 'Works with missing values'],
            'cons': ['Can overfit with noisy data', 'Less interpretable'],
            'best_for': ['Tabular data', 'Feature selection', 'Robust predictions']
        },
        
        'MultinomialNB': {
            'category': 'Naive Bayes',
            'description': 'Naive Bayes classifier for multinomial models',
            'pros': ['Fast training and prediction', 'Good with small datasets', 'Handles multi-class well'],
            'cons': ['Strong independence assumption', 'Requires smoothing'],
            'best_for': ['Text classification', 'Sparse data', 'Quick baselines']
        },
        
        'DecisionTreeClassifier': {
            'category': 'Tree-based',
            'description': 'Decision tree classifier',
            'pros': ['Highly interpretable', 'Handles non-linear relationships', 'No feature scaling needed'],
            'cons': ['Prone to overfitting', 'Unstable', 'Biased towards features with many levels'],
            'best_for': ['Interpretable models', 'Rule extraction', 'Mixed data types']
        },
        
        'KNeighborsClassifier': {
            'category': 'Instance-based',
            'description': 'K-nearest neighbors classifier',
            'pros': ['Simple concept', 'No assumptions about data', 'Works well locally'],
            'cons': ['Computationally expensive', 'Sensitive to irrelevant features', 'Memory intensive'],
            'best_for': ['Small datasets', 'Irregular decision boundaries', 'Recommendation systems']
        },
        
        'MLPClassifier': {
            'category': 'Neural Network',
            'description': 'Multi-layer perceptron classifier',
            'pros': ['Can model complex patterns', 'Flexible architecture', 'Universal approximator'],
            'cons': ['Requires parameter tuning', 'Black box', 'Prone to overfitting'],
            'best_for': ['Complex patterns', 'Large datasets', 'Non-linear problems']
        }
    }
    
    return model_info


def create_clinical_specialized_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create models specifically tuned for clinical data characteristics.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of clinical-specialized models
    """
    models = {
        # Clinical Logistic Regression - conservative approach for interpretability
        'ClinicalLogisticRegression': LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver='liblinear',
            C=1.0,  # Less regularization for clinical interpretability
            penalty='l2',
            class_weight='balanced'  # Handle imbalanced clinical data
        ),
        
        # Clinical SVM - tuned for sparse clinical features
        'ClinicalSVM': SVC(
            kernel='linear',  # Linear for interpretability
            probability=True,
            random_state=random_state,
            C=0.1,  # More regularization for clinical stability
            class_weight='balanced',
            gamma='scale'
        ),
        
        # Clinical Naive Bayes - good for clinical text features
        'ClinicalNaiveBayes': MultinomialNB(
            alpha=0.1,  # Less smoothing for clinical binary features
            fit_prior=True
        ),
        
        # Clinical Decision Tree - limited depth for interpretability
        'ClinicalDecisionTree': DecisionTreeClassifier(
            random_state=random_state,
            max_depth=5,  # Shallow for clinical interpretability
            min_samples_split=20,  # Conservative splits
            min_samples_leaf=10,   # Ensure statistical significance
            criterion='gini',
            class_weight='balanced'
        ),
        
        # Clinical MLP - simple architecture for clinical stability
        'ClinicalMLP': MLPClassifier(
            hidden_layer_sizes=(50,),  # Single hidden layer
            max_iter=1000,
            random_state=random_state,
            learning_rate='constant',
            early_stopping=True,
            validation_fraction=0.2,
            alpha=0.01,  # More regularization
            solver='lbfgs'  # Good for small datasets
        )
    }
    
    logger.info(f"Created {len(models)} clinical-specialized models")
    return models


def evaluate_model_suitability(data_characteristics: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate suitability of different classical models based on data characteristics.
    
    Args:
        data_characteristics: Dictionary with data properties
            - n_samples: Number of samples
            - n_features: Number of features
            - sparsity: Fraction of zero values
            - class_balance: Ratio of minority to majority class
            - feature_types: List of feature types
            
    Returns:
        Dictionary with suitability scores (0-1) for each model
    """
    n_samples = data_characteristics.get('n_samples', 1000)
    n_features = data_characteristics.get('n_features', 100)
    sparsity = data_characteristics.get('sparsity', 0.0)
    class_balance = data_characteristics.get('class_balance', 0.5)
    
    suitability_scores = {}
    
    # Logistic Regression
    lr_score = 0.8  # Generally good baseline
    if n_samples < 1000:
        lr_score -= 0.1
    if class_balance < 0.3:
        lr_score -= 0.1
    suitability_scores['LogisticRegression'] = max(0, min(1, lr_score))
    
    # SVM
    svm_score = 0.7
    if n_samples > 5000:
        svm_score -= 0.2  # SVM doesn't scale well
    if n_features > n_samples:
        svm_score += 0.1  # Good for high-dimensional data
    if sparsity > 0.5:
        svm_score += 0.1  # Handles sparse data well
    suitability_scores['SVC'] = max(0, min(1, svm_score))
    
    # Naive Bayes
    nb_score = 0.6
    if sparsity > 0.3:
        nb_score += 0.2  # Excellent for sparse data
    if n_samples < 500:
        nb_score += 0.1  # Good for small datasets
    if class_balance < 0.2:
        nb_score -= 0.2  # Sensitive to imbalance
    suitability_scores['MultinomialNB'] = max(0, min(1, nb_score))
    
    # Decision Tree
    dt_score = 0.6
    if n_features < 50:
        dt_score += 0.1  # Better with fewer features
    if class_balance < 0.3:
        dt_score -= 0.1  # Sensitive to imbalance
    suitability_scores['DecisionTreeClassifier'] = max(0, min(1, dt_score))
    
    # K-Nearest Neighbors
    knn_score = 0.5
    if n_samples < 1000:
        knn_score += 0.2  # Better for smaller datasets
    if n_features > 100:
        knn_score -= 0.2  # Curse of dimensionality
    if sparsity > 0.5:
        knn_score -= 0.2  # Poor with sparse data
    suitability_scores['KNeighborsClassifier'] = max(0, min(1, knn_score))
    
    # MLP
    mlp_score = 0.4  # Generally needs more tuning
    if n_samples > 2000:
        mlp_score += 0.3  # Better with more data
    if n_features > 50:
        mlp_score += 0.2  # Can handle complex features
    if class_balance < 0.2:
        mlp_score -= 0.2  # Sensitive to imbalance
    suitability_scores['MLPClassifier'] = max(0, min(1, mlp_score))
    
    return suitability_scores


def get_hyperparameter_suggestions(model_name: str, 
                                 data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get hyperparameter suggestions for a specific model based on data characteristics.
    
    Args:
        model_name: Name of the model
        data_characteristics: Dictionary with data properties
        
    Returns:
        Dictionary with suggested hyperparameters
    """
    n_samples = data_characteristics.get('n_samples', 1000)
    n_features = data_characteristics.get('n_features', 100)
    class_balance = data_characteristics.get('class_balance', 0.5)
    
    suggestions = {}
    
    if model_name == 'LogisticRegression':
        # Adjust regularization based on dataset size
        if n_samples < 500:
            suggestions['C'] = 0.1  # More regularization
        elif n_samples > 5000:
            suggestions['C'] = 10.0  # Less regularization
        else:
            suggestions['C'] = 1.0
        
        # Handle class imbalance
        if class_balance < 0.3:
            suggestions['class_weight'] = 'balanced'
        
        suggestions['max_iter'] = min(2000, max(1000, n_samples // 10))
    
    elif model_name == 'SVC':
        # Adjust C based on dataset characteristics
        if n_features > n_samples:
            suggestions['C'] = 0.1  # More regularization for high-dim data
        else:
            suggestions['C'] = 1.0
        
        # Kernel selection
        if n_samples < 1000:
            suggestions['kernel'] = 'linear'  # Faster for small datasets
        else:
            suggestions['kernel'] = 'rbf'
        
        # Handle class imbalance
        if class_balance < 0.3:
            suggestions['class_weight'] = 'balanced'
    
    elif model_name == 'DecisionTreeClassifier':
        # Adjust complexity based on sample size
        if n_samples < 500:
            suggestions['max_depth'] = 5
            suggestions['min_samples_split'] = 20
        elif n_samples > 5000:
            suggestions['max_depth'] = 15
            suggestions['min_samples_split'] = 10
        else:
            suggestions['max_depth'] = 10
            suggestions['min_samples_split'] = 15
        
        # Handle class imbalance
        if class_balance < 0.3:
            suggestions['class_weight'] = 'balanced'
    
    elif model_name == 'MLPClassifier':
        # Adjust architecture based on data size
        if n_samples < 1000:
            suggestions['hidden_layer_sizes'] = (50,)
        elif n_samples > 5000:
            suggestions['hidden_layer_sizes'] = (200, 100)
        else:
            suggestions['hidden_layer_sizes'] = (100,)
        
        # Adjust regularization
        if n_features > n_samples:
            suggestions['alpha'] = 0.01  # More regularization
        else:
            suggestions['alpha'] = 0.001
        
        suggestions['max_iter'] = min(1000, max(500, n_samples // 5))
    
    return suggestions