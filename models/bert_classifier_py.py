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