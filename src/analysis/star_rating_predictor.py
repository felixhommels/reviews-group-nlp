"""
Star Rating Predictor Module.

This module provides star rating prediction capabilities based on sentiment analysis.
"""

import logging
from typing import Dict, Union

from ..utils.dependencies import dependency_manager, DependencyError
from ..config import ConfigManager
from .sentiment_analysis import SentimentAnalyzer, SentimentAnalysis

# Configure logging
logger = logging.getLogger(__name__)

class StarRatingPredictor:
    """Class for predicting star ratings based on sentiment analysis."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the star rating predictor.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._load_config()
        self._init_analyzer()
    
    def _load_config(self):
        """Load language-specific configuration."""
        self.config = ConfigManager.get_rating_config()
        logger.info("Loaded rating configuration")
    
    def _init_analyzer(self):
        """Initialize the sentiment analyzer."""
        try:
            self.sentiment_analyzer = SentimentAnalyzer(self.language)
            logger.info("Initialized sentiment analyzer for star rating prediction")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    def predict_star_rating(self, text: str) -> int:
        """Predict star rating (1-5) based on sentiment analysis.
        
        This method first attempts to use the multilingual analyzer's detailed scores
        if available, then falls back to traditional sentiment analysis.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Predicted star rating (1-5)
        """
        try:
            # Try multilingual analyzer first
            if hasattr(self.sentiment_analyzer, 'multilingual'):
                try:
                    result = self.sentiment_analyzer.multilingual.analyze_text(text)
                    if 'detailed_scores' in result and 'rating' in result['detailed_scores']:
                        return int(result['detailed_scores']['rating'])
                except Exception:
                    pass  # Fall through to traditional analysis

            # Traditional analysis using sentiment score thresholds
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(text)
            score = sentiment_analysis.compound_score or sentiment_analysis.polarity

            # Use rating thresholds from config
            for rating, threshold in sorted(self.config['thresholds'].items(), 
                                         key=lambda x: float('-inf') if x[1] is None else x[1], 
                                         reverse=True):
                if threshold is None or score >= threshold:
                    return int(rating)

            return self.config['default_rating']

        except Exception as e:
            logger.error(f"Error in star rating prediction: {str(e)}")
            return self.config['default_rating']
