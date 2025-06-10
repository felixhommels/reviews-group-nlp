"""Configuration manager for NLP analysis."""

from typing import Dict, Any, Optional
from .language_configs import (
    get_emotion_keywords,
    get_language_thresholds,
    get_stop_words
)

# Sentiment Analysis Models
SENTIMENT_CONFIG = {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "thresholds": {
        "positive": 0.2,    # Threshold for positive sentiment
        "negative": -0.2,   # Threshold for negative sentiment
        "very_positive": 0.75,  # For rating prediction
        "very_negative": -0.75,
        "somewhat_positive": 0.25,
        "somewhat_negative": -0.25
    }
}

# TF-IDF Configuration
TFIDF_CONFIG = {
    "max_features": 1000,
    "ngram_range": (1, 2),
    "stop_words": {
        "en": "english",  # Built-in English stop words
        "es": None,      # Will be loaded from custom lists
    }
}

# Emotion Analysis Configuration
EMOTION_CONFIG = {
    "score_thresholds": {
        "strong": 0.8,    # Threshold for strong emotions
        "moderate": 0.6,  # Threshold for moderate emotions
        "weak": 0.3      # Threshold for weak emotions
    }
}

# Star Rating Prediction
RATING_CONFIG = {
    "default_rating": 3,  # Default rating when sentiment is unknown
    "thresholds": {
        "5": 0.75,  # Very positive
        "4": 0.25,  # Positive
        "3": -0.25, # Neutral
        "2": -0.75, # Negative
        "1": None   # Very negative (default case)
    },
    "rating_map": {
        "very_positive": 5,
        "positive": 4,
        "neutral": 3,
        "negative": 2,
        "very_negative": 1
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

class ConfigManager:
    """Manages access to configuration settings."""
    
    @classmethod
    def get_sentiment_config(cls, language: str = 'en') -> Dict[str, Any]:
        """Get sentiment analysis configuration."""
        config = SENTIMENT_CONFIG.copy()
        thresholds = get_language_thresholds(language)['sentiment']
        config['thresholds'].update(thresholds)
        return config
    
    @classmethod
    def get_emotion_config(cls, language: str = 'en') -> Dict[str, Any]:
        """Get emotion analysis configuration."""
        config = EMOTION_CONFIG.copy()
        config['keywords'] = get_emotion_keywords(language)
        thresholds = get_language_thresholds(language)['emotion']
        config['score_thresholds'].update(thresholds)
        return config
    
    @classmethod
    def get_rating_config(cls) -> Dict[str, Any]:
        """Get star rating configuration."""
        return RATING_CONFIG
    
    @classmethod
    def get_tfidf_config(cls, language: str = 'en') -> Dict[str, Any]:
        """Get TF-IDF configuration."""
        config = TFIDF_CONFIG.copy()
        config['stop_words'] = get_stop_words(language)
        return config
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration."""
        return LOGGING_CONFIG
    
    @classmethod
    def get_config(cls, config_name: str, language: str = 'en') -> Optional[Dict[str, Any]]:
        """Get configuration by name.
        
        Args:
            config_name: Name of the configuration to get
            language: Language code for language-specific settings
            
        Returns:
            Configuration dictionary or None if not found
        """
        config_map = {
            'sentiment': lambda: cls.get_sentiment_config(language),
            'emotion': lambda: cls.get_emotion_config(language),
            'rating': lambda: cls.get_rating_config(),
            'tfidf': lambda: cls.get_tfidf_config(language),
            'logging': lambda: cls.get_logging_config()
        }
        getter = config_map.get(config_name.lower())
        return getter() if getter else None 