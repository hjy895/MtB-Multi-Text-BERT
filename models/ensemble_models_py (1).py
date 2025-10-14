"""
Ensemble Machine Learning Models Module

This module provides ensemble machine learning models for clinical
topic modeling and classification tasks.
"""

from typing import Dict, Any, Optional
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from loguru import logger

# Optional external ensemble libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install with: pip install catboost")


def create_ensemble_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create a comprehensive set of ensemble machine learning models.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of ensemble models
    """
    models = {}
    
    # Standard Ensemble Models
    models.update({
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        
        'GradientBoostingClassifier': GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4
        ),
        
        'ExtraTreesClassifier': ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        
        'AdaBoostClassifier': AdaBoostClassifier(
            n_estimators=50,
            random_state=random_state,
            learning_rate=1.0,
            algorithm='SAMME.R'
        ),
        
        'BaggingClassifier': BaggingClassifier(
            n_estimators=50,
            random_state=random_state,
            max_samples=0.8,
            max_features=0.8,
            n_jobs=-1
        )
    })
    
    # External Gradient Boosting Models (if available)
    if XGBOOST_AVAILABLE:
        models['XGBClassifier'] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            verbosity=0
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LGBMClassifier'] = LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1
        )
    
    if CATBOOST_AVAILABLE:
        models['CatBoostClassifier'] = CatBoostClassifier(
            iterations=100,
            random_state=random_state,
            learning_rate=0.1,
            depth=6,
            verbose=False
        )
    
    # Voting Classifiers
    voting_models = _create_voting_classifiers(random_state)
    models.update(voting_models)
    
    logger.info(f"Created {len(models)} ensemble models")
    return models


def _create_voting_classifiers(random_state: int = 42) -> Dict[str, VotingClassifier]:
    """Create voting classifier combinations."""
    
    # Base classifiers for voting
    base_lr = LogisticRegression(max_iter=1000, random_state=random_state)
    base_rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    base_svm = SVC(probability=True, random_state=random_state)
    base_nb = MultinomialNB()
    base_gb = GradientBoostingClassifier(n_estimators=50, random_state=random_state)
    
    voting_classifiers = {
        'VotingClassifier_Basic': VotingClassifier(
            estimators=[
                ('lr', base_lr),
                ('rf', base_rf),
                ('svm', base_svm)
            ],
            voting='soft'
        ),
        
        'VotingClassifier_Diverse': VotingClassifier(
            estimators=[
                ('lr', base_lr),
                ('rf', base_rf),
                ('nb', base_nb),
                ('gb', base_gb)
            ],
            voting='soft'
        ),
        
        'VotingClassifier_TreeBased': VotingClassifier(
            estimators=[
                ('rf', base_rf),
                ('et', ExtraTreesClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)),
                ('gb', base_gb)
            ],
            voting='soft'
        )
    }
    
    return voting_classifiers


def create_optimized_ensemble_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create optimized ensemble models with better hyperparameters.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of optimized ensemble models
    """
    models = {
        'OptimizedRandomForest': RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1
        ),
        
        'OptimizedGradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            random_state=random_state,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            max_features='sqrt'
        ),
        
        'OptimizedExtraTrees': ExtraTreesClassifier(
            n_estimators=200,
            random_state=random_state,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1
        ),
        
        'OptimizedAdaBoost': AdaBoostClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.5,
            algorithm='SAMME.R',
            base_estimator=DecisionTreeClassifier(max_depth=3, random_state=random_state)
        )
    }
    
    # Optimized external models (if available)
    if XGBOOST_AVAILABLE:
        models['OptimizedXGBoost'] = XGBClassifier(
            n_estimators=200,
            random_state=random_state,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            eval_metric='logloss',
            verbosity=0
        )
    
    if LIGHTGBM_AVAILABLE:
        models['OptimizedLightGBM'] = LGBMClassifier(
            n_estimators=200,
            random_state=random_state,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=-1
        )
    
    logger.info(f"Created {len(models)} optimized ensemble models")
    return models


def create_clinical_ensemble_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create ensemble models specifically tuned for clinical data.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of clinical-tuned ensemble models
    """
    models = {
        # Conservative Random Forest for clinical interpretability
        'ClinicalRandomForest': RandomForestClassifier(
            n_estimators=50,  # Fewer trees for faster interpretation
            random_state=random_state,
            max_depth=8,      # Shallow trees for interpretability
            min_samples_split=20,  # Conservative splits
            min_samples_leaf=10,   # Ensure statistical significance
            max_features='sqrt',
            class_weight='balanced',  # Handle clinical data imbalance
            n_jobs=-1
        ),
        
        # Gentle Gradient Boosting for clinical stability
        'ClinicalGradientBoosting': GradientBoostingClassifier(
            n_estimators=50,
            random_state=random_state,
            learning_rate=0.05,  # Slow learning for stability
            max_depth=4,         # Shallow for interpretability
            min_samples_split=30,
            min_samples_leaf=15,
            subsample=0.8
        ),
        
        # Clinical Voting Classifier - combines interpretable models
        'ClinicalVotingClassifier': VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')),
                ('nb', MultinomialNB(alpha=0.1)),
                ('rf', RandomForestClassifier(n_estimators=30, max_depth=6, random_state=random_state, class_weight='balanced', n_jobs=-1))
            ],
            voting='soft'
        ),
        
        # Clinical Bagging with Decision Trees
        'ClinicalBagging': BaggingClassifier(
            base_estimator=DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=random_state,
                class_weight='balanced'
            ),
            n_estimators=30,
            random_state=random_state,
            max_samples=0.8,
            max_features=0.8,
            n_jobs=-1
        )
    }
    
    # Clinical XGBoost if available
    if XGBOOST_AVAILABLE:
        models['ClinicalXGBoost'] = XGBClassifier(
            n_estimators=50,
            random_state=random_state,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,      # Regularization for clinical stability
            reg_lambda=0.1,
            eval_metric='logloss',
            verbosity=0
        )
    
    logger.info(f"Created {len(models)} clinical ensemble models")
    return models


def create_stacking_ensembles(random_state: int = 42) -> Dict[str, Any]:
    """
    Create stacking ensemble models.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of stacking ensemble models
    """
    from sklearn.ensemble import StackingClassifier
    
    # Base models for stacking
    base_models = [
        ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)),
        ('svm', SVC(probability=True, random_state=random_state)),
        ('nb', MultinomialNB()),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=random_state))
    ]
    
    models = {
        'StackingClassifier_LR': StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            cv=5,
            n_jobs=-1
        ),
        
        'StackingClassifier_RF': StackingClassifier(
            estimators=base_models,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1),
            cv=5,
            n_jobs=-1
        )
    }
    
    logger.info(f"Created {len(models)} stacking ensemble models")
    return models


def get_ensemble_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available ensemble models.
    
    Returns:
        Dictionary with ensemble model information
    """
    model_info = {
        'RandomForestClassifier': {
            'category': 'Bagging Ensemble',
            'description': 'Ensemble of decision trees using bootstrap aggregating',
            'pros': ['Reduces overfitting', 'Feature importance', 'Parallel training'],
            'cons': ['Less interpretable', 'Memory intensive', 'Can overfit with noise'],
            'best_for': ['Tabular data', 'Feature selection', 'Baseline ensemble']
        },
        
        'GradientBoostingClassifier': {
            'category': 'Boosting Ensemble',
            'description': 'Sequential ensemble that corrects previous model errors',
            'pros': ['High accuracy', 'Handles missing values', 'Feature interactions'],
            'cons': ['Prone to overfitting', 'Sequential training', 'Hyperparameter sensitive'],
            'best_for': ['Structured data', 'Competitions', 'High accuracy requirements']
        },
        
        'XGBClassifier': {
            'category': 'Gradient Boosting',
            'description': 'Optimized gradient boosting with regularization',
            'pros': ['State-of-the-art performance', 'Built-in regularization', 'Efficient'],
            'cons': ['Many hyperparameters', 'Can overfit', 'Requires tuning'],
            'best_for': ['Competitions', 'Structured data', 'High performance needs']
        },
        
        'LGBMClassifier': {
            'category': 'Gradient Boosting',
            'description': 'Light gradient boosting machine',
            'pros': ['Fast training', 'Memory efficient', 'Good accuracy'],
            'cons': ['Can overfit small datasets', 'Sensitive to parameters'],
            'best_for': ['Large datasets', 'Fast training', 'Memory constraints']
        },
        
        'VotingClassifier': {
            'category': 'Voting Ensemble',
            'description': 'Combines predictions from multiple diverse models',
            'pros': ['Simple concept', 'Reduces variance', 'Robust predictions'],
            'cons': ['All models must perform well', 'Computationally expensive'],
            'best_for': ['Model combination', 'Robust predictions', 'Diverse base models']
        },
        
        'StackingClassifier': {
            'category': 'Stacking Ensemble',
            'description': 'Uses meta-learner to combine base model predictions',
            'pros': ['Can learn complex combinations', 'Often high performance'],
            'cons': ['Complex to tune', 'Prone to overfitting', 'Computationally expensive'],
            'best_for': ['Competition settings', 'Maximum performance', 'Advanced ensembling']
        }
    }
    
    return model_info


def get_ensemble_hyperparameter_suggestions(model_name: str,
                                          data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get hyperparameter suggestions for ensemble models based on data characteristics.
    
    Args:
        model_name: Name of the ensemble model
        data_characteristics: Dictionary with data properties
        
    Returns:
        Dictionary with suggested hyperparameters
    """
    n_samples = data_characteristics.get('n_samples', 1000)
    n_features = data_characteristics.get('n_features', 100)
    class_balance = data_characteristics.get('class_balance', 0.5)
    
    suggestions = {}
    
    if model_name == 'RandomForestClassifier':
        # Adjust number of estimators based on dataset size
        if n_samples < 1000:
            suggestions['n_estimators'] = 50
        elif n_samples > 10000:
            suggestions['n_estimators'] = 200
        else:
            suggestions['max_features'] = 'sqrt'
        
        # Handle class imbalance
        if class_balance < 0.3:
            suggestions['class_weight'] = 'balanced'
        
        # Adjust complexity based on sample size
        if n_samples < 500:
            suggestions['max_depth'] = 5
            suggestions['min_samples_split'] = 20
        else:
            suggestions['max_depth'] = 10
            suggestions['min_samples_split'] = 10
    
    elif model_name == 'GradientBoostingClassifier':
        # Adjust learning rate and n_estimators inversely
        if n_samples < 1000:
            suggestions['n_estimators'] = 50
            suggestions['learning_rate'] = 0.1
        elif n_samples > 5000:
            suggestions['n_estimators'] = 200
            suggestions['learning_rate'] = 0.05
        else:
            suggestions['n_estimators'] = 100
            suggestions['learning_rate'] = 0.1
        
        # Adjust max_depth
        if n_features < 20:
            suggestions['max_depth'] = 3
        else:
            suggestions['max_depth'] = 6
        
        # Conservative settings for small datasets
        if n_samples < 1000:
            suggestions['subsample'] = 1.0
            suggestions['min_samples_split'] = 20
        else:
            suggestions['subsample'] = 0.8
            suggestions['min_samples_split'] = 10
    
    elif model_name == 'XGBClassifier' and XGBOOST_AVAILABLE:
        # Adjust based on dataset size
        if n_samples < 1000:
            suggestions['n_estimators'] = 50
            suggestions['learning_rate'] = 0.1
            suggestions['max_depth'] = 3
        elif n_samples > 10000:
            suggestions['n_estimators'] = 300
            suggestions['learning_rate'] = 0.03
            suggestions['max_depth'] = 8
        else:
            suggestions['n_estimators'] = 100
            suggestions['learning_rate'] = 0.1
            suggestions['max_depth'] = 6
        
        # Regularization based on feature count
        if n_features > n_samples:
            suggestions['reg_alpha'] = 0.1
            suggestions['reg_lambda'] = 0.1
        
        suggestions['subsample'] = 0.8
        suggestions['colsample_bytree'] = 0.8
    
    elif model_name == 'LGBMClassifier' and LIGHTGBM_AVAILABLE:
        # Similar to XGBoost but with LightGBM-specific parameters
        if n_samples < 1000:
            suggestions['n_estimators'] = 50
            suggestions['learning_rate'] = 0.1
            suggestions['max_depth'] = 3
            suggestions['num_leaves'] = 15
        else:
            suggestions['n_estimators'] = 100
            suggestions['learning_rate'] = 0.1
            suggestions['max_depth'] = 6
            suggestions['num_leaves'] = 31
        
        suggestions['subsample'] = 0.8
        suggestions['colsample_bytree'] = 0.8
    
    return suggestions


def create_ensemble_pipeline(base_models: Optional[Dict[str, Any]] = None,
                           meta_learner: Optional[Any] = None,
                           ensemble_type: str = 'voting',
                           random_state: int = 42) -> Any:
    """
    Create a custom ensemble pipeline.
    
    Args:
        base_models: Dictionary of base models to ensemble
        meta_learner: Meta-learner for stacking (if ensemble_type is 'stacking')
        ensemble_type: Type of ensemble ('voting', 'stacking', 'bagging')
        random_state: Random seed for reproducibility
        
    Returns:
        Configured ensemble model
    """
    if base_models is None:
        # Default base models
        base_models = {
            'lr': LogisticRegression(max_iter=1000, random_state=random_state),
            'rf': RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1),
            'svm': SVC(probability=True, random_state=random_state),
            'nb': MultinomialNB()
        }
    
    if ensemble_type == 'voting':
        return VotingClassifier(
            estimators=list(base_models.items()),
            voting='soft'
        )
    
    elif ensemble_type == 'stacking':
        from sklearn.ensemble import StackingClassifier
        
        if meta_learner is None:
            meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
        
        return StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
    
    elif ensemble_type == 'bagging':
        # Use the first model as base estimator for bagging
        base_estimator = list(base_models.values())[0]
        
        return BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=len(base_models) * 10,  # Scale with number of base models
            random_state=random_state,
            n_jobs=-1
        )
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def evaluate_ensemble_diversity(models: Dict[str, Any], 
                              X_train, y_train,
                              X_val, y_val) -> Dict[str, float]:
    """
    Evaluate diversity metrics for ensemble models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training labels  
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary with diversity metrics
    """
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # Get predictions from all models
    predictions = {}
    accuracies = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            predictions[name] = pred
            accuracies[name] = accuracy_score(y_val, pred)
        except Exception as e:
            logger.warning(f"Could not evaluate model {name}: {e}")
            continue
    
    if len(predictions) < 2:
        return {'error': 'Need at least 2 models for diversity calculation'}
    
    # Calculate pairwise disagreement
    model_names = list(predictions.keys())
    disagreements = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            disagreement = np.mean(predictions[model1] != predictions[model2])
            disagreements.append(disagreement)
    
    # Calculate Q-statistic (Yule's Q)
    q_statistics = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            pred1, pred2 = predictions[model1], predictions[model2]
            
            # Create confusion matrix for pair
            n11 = np.sum((pred1 == y_val) & (pred2 == y_val))  # Both correct
            n10 = np.sum((pred1 == y_val) & (pred2 != y_val))  # Only model1 correct
            n01 = np.sum((pred1 != y_val) & (pred2 == y_val))  # Only model2 correct
            n00 = np.sum((pred1 != y_val) & (pred2 != y_val))  # Both incorrect
            
            # Q-statistic
            if (n11 * n00 + n01 * n10) != 0:
                q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                q_statistics.append(abs(q))
    
    diversity_metrics = {
        'average_disagreement': np.mean(disagreements) if disagreements else 0.0,
        'max_disagreement': np.max(disagreements) if disagreements else 0.0,
        'min_disagreement': np.min(disagreements) if disagreements else 0.0,
        'average_q_statistic': np.mean(q_statistics) if q_statistics else 0.0,
        'individual_accuracies': accuracies,
        'accuracy_std': np.std(list(accuracies.values())) if accuracies else 0.0
    }
    
    return diversity_metrics


def get_ensemble_recommendations(data_characteristics: Dict[str, Any]) -> Dict[str, str]:
    """
    Get ensemble model recommendations based on data characteristics.
    
    Args:
        data_characteristics: Dictionary with data properties
        
    Returns:
        Dictionary with recommendations
    """
    n_samples = data_characteristics.get('n_samples', 1000)
    n_features = data_characteristics.get('n_features', 100)
    class_balance = data_characteristics.get('class_balance', 0.5)
    sparsity = data_characteristics.get('sparsity', 0.0)
    
    recommendations = {}
    
    # Primary recommendation
    if n_samples < 1000:
        recommendations['primary'] = 'RandomForestClassifier'
        recommendations['reason'] = 'Random Forest works well with small datasets and provides good baseline performance'
    elif n_samples > 10000 and LIGHTGBM_AVAILABLE:
        recommendations['primary'] = 'LGBMClassifier'
        recommendations['reason'] = 'LightGBM is efficient for large datasets'
    elif XGBOOST_AVAILABLE:
        recommendations['primary'] = 'XGBClassifier'
        recommendations['reason'] = 'XGBoost provides excellent performance for medium-sized datasets'
    else:
        recommendations['primary'] = 'GradientBoostingClassifier'
        recommendations['reason'] = 'Gradient Boosting provides good performance when XGBoost is not available'
    
    # Secondary recommendation
    if class_balance < 0.3:
        recommendations['secondary'] = 'VotingClassifier'
        recommendations['secondary_reason'] = 'Voting ensemble can help with imbalanced data by combining diverse models'
    elif n_features > n_samples:
        recommendations['secondary'] = 'RandomForestClassifier'
        recommendations['secondary_reason'] = 'Random Forest handles high-dimensional data well'
    else:
        recommendations['secondary'] = 'ExtraTreesClassifier'
        recommendations['secondary_reason'] = 'Extra Trees provides good alternative with more randomness'
    
    # Clinical recommendation
    recommendations['clinical'] = 'ClinicalRandomForest'
    recommendations['clinical_reason'] = 'Designed specifically for clinical data with interpretability and stability in mind'
    
    return recommendations
            suggestions['n_estimators'] = 100
        
        # Adjust max_features
        if n_features < 10:
            suggestions['max_features'] = None
        else: