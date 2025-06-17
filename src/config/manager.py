"""
Configuration manager for NLP analysis.

This module centralizes all configuration values for the analysis pipeline, including model names, thresholds, and mappings. Each config value is documented with its purpose and intended use. Override these values here for different environments or tasks.
"""

from typing import Dict, Any, Optional
from src.config.language_configs import (
    get_emotion_keywords,
    get_language_thresholds,
    get_stop_words
)
import os
import json
import yaml

# --- MODEL_CONFIGS: Label mapping for each model ---
MODEL_CONFIGS = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        "use_fast_tokenizer": False,
        "model_max_length": 512,
        "truncation": True,
        "label_map": {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        },
        "output_type": "classification",
        "thresholds": {
            "positive": 0.2,
            "negative": -0.2,
            "very_positive": 0.75,
            "very_negative": -0.75,
            "somewhat_positive": 0.25,
            "somewhat_negative": -0.25
        }
    },
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        "use_fast_tokenizer": True,
        "model_max_length": 512,
        "truncation": True,
        "label_map": {
            "1 star": "negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "positive"
        },
        "output_type": "rating",  # or "classification"
        "thresholds": {
            "positive": 0.2,
            "negative": -0.2,
            "very_positive": 0.75,
            "very_negative": -0.75,
            "somewhat_positive": 0.25,
            "somewhat_negative": -0.25
        }
    }
}

# --- SENTIMENT_CONFIG: Only default model and global fallback thresholds ---
# SENTIMENT_CONFIG = {
#     "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",  # Default HuggingFace model for sentiment
#     # Optionally, fallback global thresholds (not model-specific)
#     "thresholds": {
#         "positive": 0.2,
#         "negative": -0.2,
#         "very_positive": 0.75,
#         "very_negative": -0.75,
#         "somewhat_positive": 0.25,
#         "somewhat_negative": -0.25
#     }
# }

def load_json_env_var(var_name: str, default: dict) -> dict:
    """Load a dict from an environment variable containing JSON, or return default."""
    value = os.environ.get(var_name)
    if value:
        try:
            return json.loads(value)
        except Exception as e:
            print(f"Warning: Could not parse {var_name} as JSON: {e}")
    return default

def load_config_file(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            if path.endswith(".json"):
                return json.load(f)
            elif path.endswith(".yaml") or path.endswith(".yml"):
                return yaml.safe_load(f)
    return default

# Per-Source Sentiment Model Mapping
# Maps each review source to the best transformer model for that platform.
# Update this mapping to change which model is used for each source.
SENTIMENT_MODEL_MAP = {
    "trustpilot": "nlptown/bert-base-multilingual-uncased-sentiment",
    "imdb": "nlptown/bert-base-multilingual-uncased-sentiment",
    "playstore": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "steam": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}

SENTIMENT_MODEL_MAP = load_config_file(
    os.environ.get("SENTIMENT_MODEL_MAP_FILE", "config/sentiment_model_map.yaml"),
    SENTIMENT_MODEL_MAP
)

# Configurable fallback order for sentiment analysis backends
# Determines the order in which backends are tried if one fails.
FALLBACK_ORDER = ["transformers", "vader", "textblob"]  # Order of preference for sentiment backends

# TF-IDF Keyword Extraction Configuration
# Controls the number of features, n-gram range, and stopword handling for keyword extraction.
TFIDF_CONFIG = {
    "max_features": 1000,  # Maximum number of features for TF-IDF
    "ngram_range": (1, 2), # Use unigrams and bigrams
    "stop_words": {
        "en": "english",  # Built-in English stop words
        "es": None,      # Will be loaded from custom lists
    }
}

# Emotion Analysis Configuration
# Sets thresholds for classifying emotion strength in text.
EMOTION_CONFIG = {
    "score_thresholds": {
        "strong": 0.8,    # Threshold for strong emotions
        "moderate": 0.6,  # Threshold for moderate emotions
        "weak": 0.3      # Threshold for weak emotions
    }
}

# Star Rating Prediction Configuration
# Maps sentiment scores to star ratings and sets thresholds for each rating.
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
# Sets the logging level and format for the analysis pipeline.
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

KEYBERT_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "keyphrase_ngram_range": (1, 2),
    "stop_words": {
        "en": "english",
        "es": None,  # or your custom stopwords
        # add more languages as needed
    },
    "top_n": 5
}

class ConfigManager:
    """
    Manages access to configuration settings for the NLP analysis pipeline.
    Use the provided class methods to retrieve config values for each analysis component.
    """
    
    @classmethod
    def get_model_configs(cls) -> dict:
        """
        Get model label mapping and tokenizer config for all supported transformer models.
        Returns a dict keyed by model name, with label_map and tokenizer settings.
        """
        return MODEL_CONFIGS
    
    # @classmethod
    # def get_sentiment_config(cls, language: str = 'en') -> dict:
    #     """Get sentiment thresholds and abstraction."""
    #     config = SENTIMENT_CONFIG.copy()
    #     thresholds = get_language_thresholds(language)['sentiment']
    #     config['thresholds'].update(thresholds)
    #     return config
    
    @classmethod
    def get_emotion_config(cls, language: str = 'en') -> Dict[str, Any]:
        """
        Get emotion analysis configuration for the specified language.
        Returns a dict with emotion keywords and score thresholds.
        """
        config = EMOTION_CONFIG.copy()
        config['keywords'] = get_emotion_keywords(language)
        thresholds = get_language_thresholds(language)['emotion']
        config['score_thresholds'].update(thresholds)
        return config
    
    @classmethod
    def get_rating_config(cls) -> Dict[str, Any]:
        """
        Get star rating prediction configuration.
        Returns a dict mapping sentiment scores to star ratings and thresholds.
        """
        return RATING_CONFIG
    
    @classmethod
    def get_tfidf_config(cls, language: str = 'en') -> Dict[str, Any]:
        """
        Get TF-IDF keyword extraction configuration for the specified language.
        Returns a dict with max_features, ngram_range, and stopword settings.
        """
        config = TFIDF_CONFIG.copy()
        config['stop_words'] = get_stop_words(language)
        return config
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """
        Get logging configuration for the analysis pipeline.
        Returns a dict with logging level and format.
        """
        return LOGGING_CONFIG
    
    @classmethod
    def get_keybert_config(cls, language: str = 'en') -> Dict[str, Any]:
        config = KEYBERT_CONFIG.copy()
        config['stop_words'] = config['stop_words'].get(language, None)
        return config
    
    # @classmethod
    # def get_config(cls, config_name: str, language: str = 'en') -> Optional[Dict[str, Any]]:
    #     """
    #     Get configuration by name.
    #     Args:
    #         config_name: Name of the configuration to get (e.g., 'sentiment', 'emotion', 'rating', 'tfidf', 'logging')
    #         language: Language code for language-specific settings
    #     Returns:
    #         Configuration dictionary or None if not found
    #     """
    #     config_map = {
    #         'sentiment': lambda: cls.get_sentiment_config(language),
    #         'emotion': lambda: cls.get_emotion_config(language),
    #         'rating': lambda: cls.get_rating_config(),
    #         'tfidf': lambda: cls.get_tfidf_config(language),
    #         'logging': lambda: cls.get_logging_config()
    #     }
    #     getter = config_map.get(config_name.lower())
    #     return getter() if getter else None 