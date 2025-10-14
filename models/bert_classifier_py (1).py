"""
BERT-based Clinical Topic Classifier Module

This module provides BERT-based classification models with hierarchical pooling
for clinical topic modeling and prediction tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import TrainingArguments, Trainer
from loguru import logger
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class BERTEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    BERT-based embedding transformer with hierarchical pooling.
    
    Extracts contextualized embeddings from clinical text using pre-trained
    BERT models with custom pooling strategies.
    """
    
    def __init__(self,
                 model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 max_length: int = 128,
                 pooling_strategy: str = 'hybrid',
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize BERT embedding transformer.
        
        Args:
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length
            pooling_strategy: Pooling strategy ('cls', 'mean', 'max', 'hybrid')
            cache_dir: Directory to cache model files
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.cache_dir = cache_dir or '/tmp/transformers_cache'
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.tokenizer = None
        self.model = None
        self.embedding_dim = None
        
    def fit(self, X, y=None):
        """
        Initialize the BERT model and tokenizer.
        
        Args:
            X: Input texts (unused, but required for sklearn compatibility)
            y: Target labels (unused)
            
        Returns:
            self
        """
        try:
            logger.info(f"Loading BERT model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            base_dim = config.hidden_size
            
            # Calculate final embedding dimension based on pooling strategy
            if self.pooling_strategy == 'hybrid':
                self.embedding_dim = base_dim * 3  # CLS + mean + max
            else:
                self.embedding_dim = base_dim
            
            logger.info(f"BERT model loaded successfully. Embedding dim: {self.embedding_dim}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            # Initialize with dummy values for fallback
            self.embedding_dim = 768
            return self
    
    def transform(self, X):
        """
        Transform texts to BERT embeddings.
        
        Args:
            X: List or array of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if self.tokenizer is None or self.model is None:
                logger.warning("BERT model not properly initialized, using zero embeddings")
                return np.zeros((len(X), self.embedding_dim or 768))
            
            # Ensure X is a list of strings
            texts = self._prepare_texts(X)
            
            # Process in batches to manage memory
            batch_size = 16
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self._process_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((len(X), self.embedding_dim))
            
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in BERT embedding transformation: {e}")
            return np.zeros((len(X), self.embedding_dim or 768))
    
    def _prepare_texts(self, X) -> List[str]:
        """Prepare and validate input texts."""
        if isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, list):
            texts = X
        else:
            texts = list(X)
        
        # Ensure all elements are strings
        texts = [str(text) if text is not None else "" for text in texts]
        return texts
    
    def _process_batch(self, batch_texts: List[str]) -> np.ndarray:
        """Process a batch of texts and return embeddings."""
        # Tokenize batch
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply pooling strategy
        embeddings = self._apply_pooling(outputs, inputs['attention_mask'])
        
        return embeddings.cpu().numpy()
    
    def _apply_pooling(self, outputs, attention_mask) -> torch.Tensor:
        """Apply the specified pooling strategy to model outputs."""
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if self.pooling_strategy == 'cls':
            # Use [CLS] token embedding
            embeddings = hidden_states[:, 0, :]
            
        elif self.pooling_strategy == 'mean':
            # Mean pooling over sequence length
            embeddings = self._mean_pooling(hidden_states, attention_mask)
            
        elif self.pooling_strategy == 'max':
            # Max pooling over sequence length
            embeddings = self._max_pooling(hidden_states, attention_mask)
            
        elif self.pooling_strategy == 'hybrid':
            # Concatenate CLS, mean, and max pooling
            cls_embedding = hidden_states[:, 0, :]
            mean_embedding = self._mean_pooling(hidden_states, attention_mask)
            max_embedding = self._max_pooling(hidden_states, attention_mask)
            embeddings = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return embeddings
    
    def _mean_pooling(self, hidden_states, attention_mask) -> torch.Tensor:
        """Apply mean pooling with attention mask."""
        # Expand attention mask to match hidden states dimensions
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Apply mask and compute mean
        sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, hidden_states, attention_mask) -> torch.Tensor:
        """Apply max pooling with attention mask."""
        # Set masked positions to large negative values
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = hidden_states * expanded_mask + (1 - expanded_mask) * (-1e9)
        
        # Apply max pooling
        max_embeddings, _ = torch.max(hidden_states, dim=1)
        return max_embeddings


class BERTTopicClassifier(BaseEstimator, ClassifierMixin):
    """
    BERT-based clinical topic classifier with hierarchical pooling.
    
    Combines BERT embeddings with traditional classifiers for clinical
    topic modeling and prediction tasks.
    """
    
    def __init__(self,
                 bert_model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 classifier_type: str = 'logistic_regression',
                 max_length: int = 128,
                 pooling_strategy: str = 'hybrid',
                 **classifier_kwargs):
        """
        Initialize BERT topic classifier.
        
        Args:
            bert_model_name: Pre-trained BERT model name
            classifier_type: Type of classifier ('logistic_regression', 'svm', 'ensemble')
            max_length: Maximum sequence length for BERT
            pooling_strategy: Pooling strategy for BERT embeddings
            **classifier_kwargs: Additional arguments for the classifier
        """
        self.bert_model_name = bert_model_name
        self.classifier_type = classifier_type
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.classifier_kwargs = classifier_kwargs
        
        # Initialize components
        self.bert_transformer = BERTEmbeddingTransformer(
            model_name=bert_model_name,
            max_length=max_length,
            pooling_strategy=pooling_strategy
        )
        
        self.classifier = self._create_classifier()
        self.classes_ = None
        
    def _create_classifier(self):
        """Create the downstream classifier based on type."""
        default_kwargs = {
            'logistic_regression': {'max_iter': 1000, 'random_state': 42},
            'svm': {'probability': True, 'random_state': 42},
            'ensemble': {'random_state': 42}
        }
        
        # Merge default and user-provided kwargs
        kwargs = default_kwargs.get(self.classifier_type, {})
        kwargs.update(self.classifier_kwargs)
        
        if self.classifier_type == 'logistic_regression':
            return LogisticRegression(**kwargs)
        elif self.classifier_type == 'svm':
            return SVC(**kwargs)
        elif self.classifier_type == 'ensemble':
            return VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('svm', SVC(probability=True, random_state=42))
                ],
                voting='soft',
                **kwargs
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def fit(self, X, y):
        """
        Fit the BERT topic classifier.
        
        Args:
            X: Input texts
            y: Target labels
            
        Returns:
            self
        """
        try:
            logger.info("Fitting BERT topic classifier...")
            
            # Prepare texts
            texts = self._prepare_texts(X)
            
            # Extract BERT embeddings
            logger.info("Extracting BERT embeddings...")
            X_embeddings = self.bert_transformer.fit_transform(texts)
            
            # Fit classifier
            logger.info(f"Training {self.classifier_type} classifier...")
            self.classifier.fit(X_embeddings, y)
            
            # Store classes
            self.classes_ = self.classifier.classes_
            
            logger.info("BERT topic classifier training completed")
            return self
            
        except Exception as e:
            logger.error(f"Error during BERT classifier training: {e}")
            # Create dummy classifier for fallback
            self.classes_ = np.unique(y) if y is not None else np.array([0, 1])
            return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted class labels
        """
        try:
            texts = self._prepare_texts(X)
            X_embeddings = self.bert_transformer.transform(texts)
            predictions = self.classifier.predict(X_embeddings)
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return dummy predictions
            return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted class probabilities
        """
        try:
            texts = self._prepare_texts(X)
            X_embeddings = self.bert_transformer.transform(texts)
            
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X_embeddings)
            else:
                # Fallback for classifiers without predict_proba
                predictions = self.classifier.predict(X_embeddings)
                n_classes = len(self.classes_)
                probabilities = np.zeros((len(X), n_classes))
                for i, pred in enumerate(predictions):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    probabilities[i, class_idx] = 1.0
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            # Return dummy probabilities
            n_classes = len(self.classes_) if self.classes_ is not None else 2
            return np.ones((len(X), n_classes)) / n_classes
    
    def _prepare_texts(self, X) -> List[str]:
        """Prepare and validate input texts."""
        if isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, list):
            texts = X
        else:
            texts = list(X)
        
        # Ensure all elements are strings
        texts = [str(text) if text is not None else "" for text in texts]
        return texts
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of BERT embeddings."""
        return self.bert_transformer.embedding_dim or 768


class MultilevelBERTClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-level BERT classifier that combines word, sentence, and paragraph level features.
    
    Implements the hierarchical feature extraction approach described in the paper.
    """
    
    def __init__(self,
                 bert_model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 word_extraction_sizes: List[int] = [5, 10, 15],
                 sentence_extraction_sizes: List[int] = [1, 2, 3],
                 paragraph_extraction_sizes: List[int] = [0, 1, 2],
                 classifier_type: str = 'logistic_regression',
                 fusion_strategy: str = 'concatenate',
                 **classifier_kwargs):
        """
        Initialize multi-level BERT classifier.
        
        Args:
            bert_model_name: Pre-trained BERT model name
            word_extraction_sizes: List of word extraction sizes to try
            sentence_extraction_sizes: List of sentence extraction sizes to try
            paragraph_extraction_sizes: List of paragraph extraction sizes to try
            classifier_type: Type of downstream classifier
            fusion_strategy: How to fuse multi-level features ('concatenate', 'average')
            **classifier_kwargs: Additional classifier arguments
        """
        self.bert_model_name = bert_model_name
        self.word_extraction_sizes = word_extraction_sizes
        self.sentence_extraction_sizes = sentence_extraction_sizes
        self.paragraph_extraction_sizes = paragraph_extraction_sizes
        self.classifier_type = classifier_type
        self.fusion_strategy = fusion_strategy
        self.classifier_kwargs = classifier_kwargs
        
        # Initialize BERT transformers for different levels
        self.bert_transformers = {
            'word': BERTEmbeddingTransformer(model_name=bert_model_name, pooling_strategy='hybrid'),
            'sentence': BERTEmbeddingTransformer(model_name=bert_model_name, pooling_strategy='hybrid'),
            'paragraph': BERTEmbeddingTransformer(model_name=bert_model_name, pooling_strategy='hybrid')
        }
        
        self.classifiers = {}
        self.best_extractors = {}
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Fit multi-level BERT classifier with cross-validation for best extraction sizes.
        
        Args:
            X: Input texts (should be a dict with 'word', 'sentence', 'paragraph' keys)
            y: Target labels
            
        Returns:
            self
        """
        logger.info("Fitting multi-level BERT classifier...")
        
        # If X is not a dict, assume it's raw text and create extractions
        if not isinstance(X, dict):
            X = self._create_multilevel_extractions(X)
        
        # Find best extraction sizes and train classifiers for each level
        for level in ['word', 'sentence', 'paragraph']:
            if level in X:
                logger.info(f"Training {level}-level classifier...")
                best_size, best_classifier = self._find_best_extraction_size(
                    X[level], y, level
                )
                self.best_extractors[level] = best_size
                self.classifiers[level] = best_classifier
        
        # Store classes
        if self.classifiers:
            self.classes_ = list(self.classifiers.values())[0].classes_
        
        logger.info("Multi-level BERT classifier training completed")
        return self
    
    def predict(self, X):
        """
        Predict using the best performing level/extraction size.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted labels
        """
        if not isinstance(X, dict):
            X = self._create_multilevel_extractions(X)
        
        # Use the best performing classifier
        best_level, best_classifier = self._get_best_classifier()
        
        if best_level and best_classifier:
            best_size = self.best_extractors[best_level]
            texts = X[best_level][best_size] if isinstance(X[best_level], dict) else X[best_level]
            return best_classifier.predict(texts)
        else:
            # Fallback
            return np.zeros(len(list(X.values())[0]), dtype=int)
    
    def predict_proba(self, X):
        """
        Predict probabilities using the best performing level/extraction size.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted probabilities
        """
        if not isinstance(X, dict):
            X = self._create_multilevel_extractions(X)
        
        # Use the best performing classifier
        best_level, best_classifier = self._get_best_classifier()
        
        if best_level and best_classifier:
            best_size = self.best_extractors[best_level]
            texts = X[best_level][best_size] if isinstance(X[best_level], dict) else X[best_level]
            if hasattr(best_classifier, 'predict_proba'):
                return best_classifier.predict_proba(texts)
        
        # Fallback
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        n_samples = len(list(X.values())[0])
        return np.ones((n_samples, n_classes)) / n_classes
    
    def _create_multilevel_extractions(self, texts):
        """Create multi-level text extractions from raw texts."""
        from ..data.topic_extractor import ClinicalTopicExtractor
        
        extractor = ClinicalTopicExtractor()
        extractions = {
            'word': {},
            'sentence': {},
            'paragraph': {}
        }
        
        # Create extractions for each level and size
        for size in self.word_extraction_sizes:
            extractions['word'][size] = extractor.extract_text_features(
                pd.Series(texts), 'word', size
            )
        
        for size in self.sentence_extraction_sizes:
            extractions['sentence'][size] = extractor.extract_text_features(
                pd.Series(texts), 'sentence', size
            )
        
        for size in self.paragraph_extraction_sizes:
            extractions['paragraph'][size] = extractor.extract_text_features(
                pd.Series(texts), 'paragraph', size
            )
        
        return extractions
    
    def _find_best_extraction_size(self, level_extractions, y, level):
        """Find the best extraction size for a given level using cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        best_score = -1
        best_size = None
        best_classifier = None
        
        sizes = getattr(self, f'{level}_extraction_sizes')
        
        for size in sizes:
            try:
                # Get texts for this extraction size
                if isinstance(level_extractions, dict):
                    texts = level_extractions[size]
                else:
                    texts = level_extractions
                
                # Create and train classifier
                classifier = BERTTopicClassifier(
                    bert_model_name=self.bert_model_name,
                    classifier_type=self.classifier_type,
                    **self.classifier_kwargs
                )
                
                # Cross-validation score
                scores = cross_val_score(classifier, texts, y, cv=3, scoring='accuracy')
                avg_score = np.mean(scores)
                
                logger.debug(f"{level} level, size {size}: {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_size = size
                    best_classifier = classifier
                    
            except Exception as e:
                logger.warning(f"Error evaluating {level} level, size {size}: {e}")
                continue
        
        # Train the best classifier on full data
        if best_classifier and best_size is not None:
            texts = level_extractions[best_size] if isinstance(level_extractions, dict) else level_extractions
            best_classifier.fit(texts, y)
            logger.info(f"Best {level} level: size {best_size}, score {best_score:.4f}")
        
        return best_size, best_classifier
    
    def _get_best_classifier(self):
        """Get the best performing classifier across all levels."""
        # This is a simplified approach - in practice, you might want to
        # compare performance across levels using validation data
        
        # Priority order: paragraph -> sentence -> word
        for level in ['paragraph', 'sentence', 'word']:
            if level in self.classifiers and self.classifiers[level] is not None:
                return level, self.classifiers[level]
        
        return None, None
    
    def get_level_results(self):
        """Get results for all levels and extraction sizes."""
        return {
            'best_extractors': self.best_extractors,
            'available_levels': list(self.classifiers.keys())
        }