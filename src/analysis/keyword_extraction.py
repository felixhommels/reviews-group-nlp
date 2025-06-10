"""
Keyword Extraction Module.

This module provides keyword extraction capabilities using:
- TF-IDF (Term Frequency-Inverse Document Frequency) for keyword extraction
"""

import logging
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.dependencies import dependency_manager, DependencyError
from ..config import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Class for extracting keywords from text using TF-IDF."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the keyword extractor.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._load_config()
        self._init_vectorizer()
    
    def _load_config(self):
        """Load language-specific configuration."""
        self.config = ConfigManager.get_tfidf_config(self.language)
        logger.info(f"Loaded TF-IDF configuration for language: {self.language}")
    
    def _init_vectorizer(self):
        """Initialize the TF-IDF vectorizer."""
        try:
            self.tfidf = TfidfVectorizer(
                max_features=self.config['max_features'],
                stop_words=self.config['stop_words'],
                ngram_range=self.config['ngram_range']
            )
            logger.info("Initialized TF-IDF vectorizer")
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF vectorizer: {e}")
            raise
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF.
        
        Args:
            text: The preprocessed text
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        try:
            if not text.strip():
                return []
                
            # Fit and transform on single document
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            
            # Convert to array and get scores
            scores = tfidf_matrix.toarray().flatten()
            # Get indices of non-zero scores, sorted by score
            nonzero_indices = scores.nonzero()[0]
            sorted_indices = nonzero_indices[scores[nonzero_indices].argsort()[::-1]]
            
            # Get top k keywords with non-zero scores
            top_indices = sorted_indices[:top_k]
            keywords = [feature_names[i] for i in top_indices]
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
