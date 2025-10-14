"""
Clinical Topic Extractor Module

This module provides functionality to extract clinical topics from numerical
clinical data and convert them into interpretable textual representations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union
import multiprocessing
from functools import partial
from loguru import logger
import yaml


class ClinicalTopicExtractor:
    """
    Extracts clinical topics from numerical clinical data.
    
    Converts continuous clinical variables into categorical tokens
    that represent clinically meaningful conditions or states.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the topic extractor.
        
        Args:
            config: Configuration dictionary for topic extraction
        """
        self.config = config or {}
        self.topic_definitions = {}
        self.custom_extractor = None
        self._load_topic_definitions()
    
    def _load_topic_definitions(self):
        """Load topic definitions from configuration."""
        if 'clinical_topics' in self.config and 'topic_definitions' in self.config['clinical_topics']:
            self.topic_definitions = self.config['clinical_topics']['topic_definitions']
            logger.info(f"Loaded {len(self.topic_definitions)} topic definitions")
        else:
            logger.warning("No topic definitions found in configuration")
    
    def set_custom_extractor(self, extractor_func: Callable):
        """
        Set a custom topic extraction function.
        
        Args:
            extractor_func: Function that takes a row dict and returns clinical text
        """
        self.custom_extractor = extractor_func
        logger.info("Custom topic extractor registered")
    
    def create_clinical_text(self, 
                           data: pd.DataFrame,
                           use_multiprocessing: bool = True,
                           n_jobs: Optional[int] = None) -> pd.Series:
        """
        Create clinical text features from numerical data.
        
        Args:
            data: DataFrame with clinical measurements
            use_multiprocessing: Whether to use multiprocessing for extraction
            n_jobs: Number of processes to use (None for auto)
            
        Returns:
            Series with clinical text for each row
        """
        if self.custom_extractor:
            extraction_func = self.custom_extractor
        else:
            extraction_func = self._default_topic_extractor
        
        if use_multiprocessing and len(data) > 1000:
            return self._extract_with_multiprocessing(data, extraction_func, n_jobs)
        else:
            return self._extract_sequential(data, extraction_func)
    
    def _extract_sequential(self, 
                          data: pd.DataFrame,
                          extraction_func: Callable) -> pd.Series:
        """Extract topics sequentially."""
        logger.info("Extracting clinical topics sequentially...")
        
        clinical_texts = []
        for _, row in data.iterrows():
            clinical_text = extraction_func(row.to_dict())
            clinical_texts.append(clinical_text)
        
        result = pd.Series(clinical_texts, index=data.index)
        logger.info(f"Clinical topic extraction completed: {len(result)} records processed")
        return result
    
    def _extract_with_multiprocessing(self,
                                    data: pd.DataFrame,
                                    extraction_func: Callable,
                                    n_jobs: Optional[int] = None) -> pd.Series:
        """Extract topics using multiprocessing."""
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        
        logger.info(f"Extracting clinical topics with multiprocessing (n_jobs={n_jobs})...")
        
        # Convert DataFrame rows to dictionaries
        row_dicts = [row.to_dict() for _, row in data.iterrows()]
        
        # Use multiprocessing to extract topics
        with multiprocessing.Pool(processes=n_jobs) as pool:
            clinical_texts = pool.map(extraction_func, row_dicts)
        
        result = pd.Series(clinical_texts, index=data.index)
        logger.info(f"Clinical topic extraction completed: {len(result)} records processed")
        return result
    
    def _default_topic_extractor(self, row_dict: Dict[str, Any]) -> str:
        """
        Default topic extraction logic based on configuration.
        
        Args:
            row_dict: Dictionary representation of a DataFrame row
            
        Returns:
            String with space-separated clinical topics
        """
        topics = []
        
        for topic_name, definition in self.topic_definitions.items():
            column = definition.get('column')
            operator = definition.get('operator')
            threshold = definition.get('threshold')
            
            if column not in row_dict:
                continue
            
            value = row_dict.get(column, 0)
            
            # Handle missing values
            if pd.isna(value):
                continue
            
            # Evaluate condition
            condition_met = False
            try:
                if operator == '>':
                    condition_met = value > threshold
                elif operator == '>=':
                    condition_met = value >= threshold
                elif operator == '<':
                    condition_met = value < threshold
                elif operator == '<=':
                    condition_met = value <= threshold
                elif operator == '==':
                    condition_met = value == threshold
                elif operator == '!=':
                    condition_met = value != threshold
                else:
                    logger.warning(f"Unknown operator '{operator}' for topic '{topic_name}'")
                    continue
                
                if condition_met:
                    topics.append(topic_name)
                    
            except Exception as e:
                logger.warning(f"Error evaluating condition for topic '{topic_name}': {e}")
                continue
        
        return ' '.join(topics) if topics else 'normal'
    
    def add_topic_definition(self,
                           topic_name: str,
                           column: str,
                           operator: str,
                           threshold: Union[float, int]):
        """
        Add a new topic definition.
        
        Args:
            topic_name: Name of the clinical topic
            column: Column name to evaluate
            operator: Comparison operator ('>', '>=', '<', '<=', '==', '!=')
            threshold: Threshold value for comparison
        """
        self.topic_definitions[topic_name] = {
            'column': column,
            'operator': operator,
            'threshold': threshold
        }
        logger.info(f"Added topic definition: {topic_name}")
    
    def remove_topic_definition(self, topic_name: str):
        """Remove a topic definition."""
        if topic_name in self.topic_definitions:
            del self.topic_definitions[topic_name]
            logger.info(f"Removed topic definition: {topic_name}")
        else:
            logger.warning(f"Topic definition not found: {topic_name}")
    
    def get_topic_statistics(self, clinical_texts: pd.Series) -> Dict[str, Any]:
        """
        Get statistics about extracted topics.
        
        Args:
            clinical_texts: Series with clinical text data
            
        Returns:
            Dictionary with topic statistics
        """
        # Count topic occurrences
        all_topics = []
        for text in clinical_texts:
            if text and text != 'normal':
                all_topics.extend(text.split())
        
        topic_counts = pd.Series(all_topics).value_counts()
        
        # Count patients with multiple topics
        multi_topic_counts = clinical_texts.apply(
            lambda x: len(x.split()) if x and x != 'normal' else 0
        ).value_counts().sort_index()
        
        # Calculate topic co-occurrence
        co_occurrence = self._calculate_topic_cooccurrence(clinical_texts)
        
        stats = {
            'total_patients': len(clinical_texts),
            'patients_with_topics': sum(clinical_texts != 'normal'),
            'unique_topics': len(topic_counts),
            'topic_frequencies': topic_counts.to_dict(),
            'multi_topic_distribution': multi_topic_counts.to_dict(),
            'average_topics_per_patient': clinical_texts.apply(
                lambda x: len(x.split()) if x and x != 'normal' else 0
            ).mean(),
            'topic_cooccurrence': co_occurrence
        }
        
        return stats
    
    def _calculate_topic_cooccurrence(self, clinical_texts: pd.Series) -> Dict[str, int]:
        """Calculate topic co-occurrence patterns."""
        cooccurrence = {}
        
        for text in clinical_texts:
            if text and text != 'normal':
                topics = text.split()
                if len(topics) > 1:
                    # Sort to ensure consistent ordering
                    topics = sorted(topics)
                    for i in range(len(topics)):
                        for j in range(i + 1, len(topics)):
                            pair = f"{topics[i]}+{topics[j]}"
                            cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        return cooccurrence
    
    def create_topic_labels(self, 
                          data: pd.DataFrame,
                          clinical_texts: pd.Series,
                          label_strategy: str = 'count_based') -> pd.Series:
        """
        Create labels based on clinical topics for classification tasks.
        
        Args:
            data: Original DataFrame
            clinical_texts: Series with clinical text data
            label_strategy: Strategy for label creation ('count_based', 'severity_based')
            
        Returns:
            Series with generated labels
        """
        if label_strategy == 'count_based':
            # Create labels based on number of topics
            labels = clinical_texts.apply(
                lambda x: 2 if x != 'normal' and len(x.split()) >= 3
                else (1 if x != 'normal' and len(x.split()) >= 1 else 0)
            )
        elif label_strategy == 'severity_based':
            # Create labels based on severity of topics (requires severity mapping)
            labels = self._create_severity_based_labels(clinical_texts)
        else:
            raise ValueError(f"Unknown label strategy: {label_strategy}")
        
        logger.info(f"Created labels using '{label_strategy}' strategy")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
        
        return labels
    
    def _create_severity_based_labels(self, clinical_texts: pd.Series) -> pd.Series:
        """Create labels based on topic severity (placeholder implementation)."""
        # This is a placeholder - implement based on your domain knowledge
        severity_mapping = {
            'normal': 0,
            # Add your severity mappings here
        }
        
        labels = []
        for text in clinical_texts:
            if text == 'normal':
                labels.append(0)
            else:
                # Simple heuristic: more topics = higher severity
                topic_count = len(text.split())
                if topic_count >= 3:
                    labels.append(2)  # High severity
                elif topic_count >= 1:
                    labels.append(1)  # Medium severity
                else:
                    labels.append(0)  # Low severity
        
        return pd.Series(labels)
    
    def extract_text_features(self,
                            clinical_texts: pd.Series,
                            extraction_method: str = 'word',
                            n_elements: int = 5) -> pd.Series:
        """
        Extract features from clinical text at different granularity levels.
        
        Args:
            clinical_texts: Series with clinical text data
            extraction_method: Method for extraction ('word', 'sentence', 'paragraph')
            n_elements: Number of elements to extract
            
        Returns:
            Series with extracted text features
        """
        if extraction_method == 'word':
            return self._extract_words(clinical_texts, n_elements)
        elif extraction_method == 'sentence':
            return self._extract_sentences(clinical_texts, n_elements)
        elif extraction_method == 'paragraph':
            return self._extract_paragraphs(clinical_texts, n_elements)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
    
    def _extract_words(self, texts: pd.Series, n_words: int) -> pd.Series:
        """Extract first N words from each text."""
        def extract_n_words(text: str) -> str:
            if not text or text == 'normal':
                return text
            words = text.split()
            return ' '.join(words[:n_words]) if n_words < len(words) else text
        
        return texts.apply(extract_n_words)
    
    def _extract_sentences(self, texts: pd.Series, n_sentences: int) -> pd.Series:
        """Extract first N sentences (treating as groups of words)."""
        def extract_n_sentences(text: str) -> str:
            if not text or text == 'normal':
                return text
            words = text.split()
            # Simple heuristic: treat every ~5 words as a "sentence"
            sentence_length = 5
            total_words = min(n_sentences * sentence_length, len(words))
            return ' '.join(words[:total_words])
        
        return texts.apply(extract_n_sentences)
    
    def _extract_paragraphs(self, texts: pd.Series, n_paragraphs: int) -> pd.Series:
        """Extract first N paragraphs (treating as larger chunks)."""
        def extract_n_paragraphs(text: str) -> str:
            if not text or text == 'normal':
                return text
            if n_paragraphs == 0:
                return text  # Return full text
            
            words = text.split()
            # Simple heuristic: treat every ~10 words as a "paragraph"
            paragraph_length = 10
            total_words = min(n_paragraphs * paragraph_length, len(words))
            return ' '.join(words[:total_words])
        
        return texts.apply(extract_n_paragraphs)
    
    def save_topic_definitions(self, file_path: str):
        """Save current topic definitions to a file."""
        with open(file_path, 'w') as f:
            yaml.dump({'topic_definitions': self.topic_definitions}, f)
        logger.info(f"Topic definitions saved to {file_path}")
    
    def load_topic_definitions(self, file_path: str):
        """Load topic definitions from a file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.topic_definitions = data.get('topic_definitions', {})
        logger.info(f"Loaded {len(self.topic_definitions)} topic definitions from {file_path}")