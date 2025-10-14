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