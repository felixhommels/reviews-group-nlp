"""
Keyword Extraction Module.

This module provides keyword extraction capabilities using:
- TF-IDF (Term Frequency-Inverse Document Frequency) for keyword extraction
"""

import logging
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.dependencies import dependency_manager
from src.config.manager import ConfigManager

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
    
    def extract_keywords(self, text: str, language: str = "en", max_keywords: int = 5) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for keyword extraction")
            return []
        
        try:
            # Get language-specific config
            config = ConfigManager.get_tfidf_config(language)
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_keywords,
                stop_words=config['stop_words'],
                token_pattern=r'(?u)\b\w\w+\b'  # Match words with at least 2 characters
            )
            
            # Fit and transform the text
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Get feature names (words)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords based on TF-IDF scores
            if len(feature_names) > 0:
                scores = tfidf_matrix.toarray()[0]
                keyword_indices = scores.argsort()[-max_keywords:][::-1]
                keywords = [feature_names[i] for i in keyword_indices]
                return keywords
            else:
                logger.warning("No keywords found in text")
                return []
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
