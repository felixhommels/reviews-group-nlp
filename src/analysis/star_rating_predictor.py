"""
Star Rating Predictor Module.

This module provides star rating prediction capabilities based on sentiment analysis.
"""

import logging
from typing import Union

from src.config.manager import ConfigManager
from src.analysis.sentiment_analysis import SentimentAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class StarRatingPredictor:
    """Class for predicting star ratings based on sentiment analysis or direct model output."""
    
    def __init__(self, language: str = 'en', source: str = None):
        """Initialize the star rating predictor.
        
        Args:
            language: ISO language code (default: 'en')
            source: Source of the rating (default: 'trustpilot')
        """
        self.language = language
        self.source = source or 'trustpilot'
        self._load_config()
        self._init_analyzer()
    
    def _load_config(self):
        """Load language-specific configuration."""
        try:
            self.config = ConfigManager.get_rating_config()
            self.model_configs = ConfigManager.get_model_configs()
            logger.info("Loaded rating configuration")
        except Exception as e:
            logger.error(f"Failed to load rating configuration: {e}")
            raise
    
    def _init_analyzer(self):
        """Initialize the sentiment analyzer."""
        try:
            self.sentiment_analyzer = SentimentAnalyzer(self.language, source=self.source)
            logger.info("Initialized sentiment analyzer for star rating prediction")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    def predict_star_rating(self, text: str, source: str = None) -> int:
        """Predict star rating (1-5) using direct model output if available, else fallback.
        
        Args:
            text: Text to analyze for star rating prediction
            source: Optional source override (trustpilot, imdb, playstore, steam)
            
        Returns:
            Predicted star rating (1-5)
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for star rating prediction")
            return self.config['default_rating']
        
        src = (source or self.source).lower()
        
        # Try to use nlptown model if available in backends, regardless of source
        model_name = None
        if 'transformers' in self.sentiment_analyzer.backends:
            model = self.sentiment_analyzer.backends['transformers']
            # Try to get the model id from the pipeline
            try:
                model_id = model.model.config._name_or_path
                if model_id == 'nlptown/bert-base-multilingual-uncased-sentiment':
                    model_name = model_id
            except Exception:
                pass
        
        if model_name == 'nlptown/bert-base-multilingual-uncased-sentiment':
            try:
                result = model(text)[0]
                label = result['label']
                try:
                    stars = int(label.split()[0])
                    return stars
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Direct nlptown model rating failed: {e}")
        
        # Fallback to sentiment thresholds
        try:
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(text, source=src)
            score = sentiment_result.get('compound_score', sentiment_result.get('score', 0))
            
            for rating, threshold in sorted(self.config['thresholds'].items(), 
                                         key=lambda x: float('-inf') if x[1] is None else x[1], 
                                         reverse=True):
                if threshold is None or score >= threshold:
                    return int(rating)
        except Exception as e:
            logger.error(f"Error in sentiment-based rating prediction: {e}")
        
        return self.config['default_rating']

    @staticmethod
    def normalize_rating(raw_rating: Union[int, float], source: str) -> float:
        """Normalize rating to a 1-5 scale based on the source platform.
        
        Args:
            raw_rating: Raw rating value from the platform
            source: Source platform (imdb, trustpilot, playstore, steam)
            
        Returns:
            Normalized rating on a 1-5 scale
        """
        if source == "imdb":
            return round((raw_rating / 2), 1)  # IMDb 1-10 → 1-5
        elif source == "trustpilot":
            return float(raw_rating)  # already predicted as 1-5
        else:
            return float(raw_rating)  # Playstore, Steam already 1-5
