"""
Sentiment Analysis Module.

This module provides sentiment analysis capabilities using multiple approaches:
- XLM-RoBERTa for Steam/Playstore reviews
- BERT-based sentiment analysis for Trustpilot/IMDb
- VADER (English) for sentiment analysis
- TextBlob as a fallback
"""

import logging
from typing import Dict, Any
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaTokenizer

from src.utils.dependencies import dependency_manager, DependencyError
from src.config.manager import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)

# Sentiment analysis: VADER (for English) or fallback to TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not available. Using TextBlob for sentiment analysis.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.error("Neither VADER nor TextBlob available. Sentiment analysis will be limited.")

@lru_cache(maxsize=8)
def get_sentiment_model(source: str, language: str = 'en'):
    """Return the correct Hugging Face pipeline for the platform.
    
    Args:
        source: Review source (trustpilot, imdb, playstore, steam)
        language: ISO language code (default: 'en')
        
    Returns:
        HuggingFace pipeline for sentiment analysis or None if loading fails
    """
    try:
        # Use SENTIMENT_MODEL_MAP to map source to model name
        model_map = getattr(ConfigManager, 'SENTIMENT_MODEL_MAP', None)
        if model_map is None:
            from src.config.manager import SENTIMENT_MODEL_MAP as model_map
        model_name = model_map.get(source.lower())
        if not model_name:
            model_name = ConfigManager.get_model_configs().get('default')
        model_config = ConfigManager.get_model_configs().get(model_name, {})
        
        # Initialize tokenizer with appropriate settings
        if model_name == 'cardiffnlp/twitter-xlm-roberta-base-sentiment':
            tokenizer = XLMRobertaTokenizer.from_pretrained(
                model_name,
                model_max_length=model_config.get('model_max_length', 512)
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=model_config.get('use_fast_tokenizer', True),
                model_max_length=model_config.get('model_max_length', 512)
            )
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline with appropriate settings
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Use CPU
            truncation=model_config.get('truncation', True),
            max_length=model_config.get('model_max_length', 512)
        )
    except Exception as e:
        logger.error(f"Error loading sentiment model for source '{source}': {e}")
        return None

class SentimentAnalyzer:
    """Class for analyzing sentiment in text using multiple approaches."""
    
    def __init__(self, language: str = 'en', source: str = 'trustpilot'):
        """Initialize sentiment analyzers.
        
        Args:
            language: ISO language code (default: 'en')
            source: Review source (trustpilot, imdb, playstore, steam)
        """
        self.language = language
        self.source = source
        self._init_analyzers()
    
    def _init_analyzers(self) -> None:
        """Initialize available sentiment analysis tools."""
        self.initialization_errors = []
        self.backends = {}
        
        # Try to initialize all possible analyzers, but don't fail if one fails
        if dependency_manager.transformers_available:
            try:
                self.backends['transformers'] = get_sentiment_model(self.source, self.language)
                logger.info("Initialized transformer sentiment analyzer")
            except Exception as e:
                error_msg = f"Could not initialize transformer analyzer for source '{self.source}': {e}"
                logger.warning(error_msg)
                self.initialization_errors.append(error_msg)
        
        if dependency_manager.vader_available:
            try:
                self.backends['vader'] = dependency_manager.get_component('vaderSentiment', 'SentimentIntensityAnalyzer')()
                logger.info("Initialized VADER sentiment analyzer")
            except DependencyError as e:
                error_msg = f"Could not initialize VADER: {e}"
                logger.warning(error_msg)
                self.initialization_errors.append(error_msg)
        
        if dependency_manager.textblob_available:
            try:
                self.backends['textblob'] = dependency_manager.get_component('textblob', 'TextBlob')
                logger.info("TextBlob available for sentiment analysis")
            except DependencyError as e:
                error_msg = f"Could not initialize TextBlob: {e}"
                logger.warning(error_msg)
                self.initialization_errors.append(error_msg)
        
        if not self.backends:
            error_details = "\n".join(self.initialization_errors) if self.initialization_errors else "No analyzers available."
            raise ImportError(f"No sentiment analysis tools available for source '{self.source}'. Details:\n{error_details}\nInstall transformers, vaderSentiment, or textblob.")

    @staticmethod
    def normalize_sentiment_output(model_output: dict, source: str, text: str, language: str = 'en') -> dict:
        """Convert model output to unified format.
        
        Args:
            model_output: Raw model output dictionary
            source: Review source (trustpilot, imdb, playstore, steam)
            text: Original text that was analyzed
            language: ISO language code (default: 'en')
            
        Returns:
            Dictionary with normalized sentiment analysis results
        """
        try:
            model_map = getattr(ConfigManager, 'SENTIMENT_MODEL_MAP', None)
            if model_map is None:
                from src.config.manager import SENTIMENT_MODEL_MAP as model_map
            model_name = model_map.get(source.lower())
            if not model_name:
                model_name = ConfigManager.get_model_configs().get('default')
            model_config = ConfigManager.get_model_configs().get(model_name, {})
            label_map = model_config.get('label_map', {})
            raw_label = model_output['label']
            score = model_output['score']

            # If the raw_label is already a mapped value, use it directly
            if raw_label in label_map:
                mapped_label = label_map[raw_label]
            elif raw_label in label_map.values():
                mapped_label = raw_label
            else:
                logger.warning(f"Label '{raw_label}' not found in label_map for model '{model_name}'. label_map: {label_map}")
                mapped_label = 'neutral'

            logger.debug(f"Model: {model_name}, Raw label: {raw_label}, Mapped label: {mapped_label}, Label map: {label_map}")

            # Normalize score based on model type
            if model_name == 'nlptown/bert-base-multilingual-uncased-sentiment':
                try:
                    stars = int(raw_label.split()[0])
                    normalized_score = (stars - 3) / 2
                except Exception:
                    normalized_score = 0.0
            elif model_name == 'cardiffnlp/twitter-xlm-roberta-base-sentiment':
                if mapped_label == 'negative':
                    normalized_score = -score
                elif mapped_label == 'positive':
                    normalized_score = score
                else:
                    normalized_score = 0.0
            else:
                normalized_score = 0.0

            return {
                'sentiment_label': mapped_label,
                'sentiment_score': normalized_score,
                'confidence': score,
                'source': source,
                'raw_model_label': raw_label,
                'original_text': text
            }
        except Exception as e:
            logger.error(f"Error normalizing sentiment output: {e}")
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'source': source,
                'raw_model_label': 'unknown',
                'original_text': text
            }

    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results or None if analysis fails
        """
        if self.language == 'en' and 'vader' in self.backends:
            try:
                scores = self.backends['vader'].polarity_scores(text)
                compound_score = scores['compound']
                if compound_score >= 0.05:
                    label = 'positive'
                elif compound_score <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                return {
                    'sentiment_label': label,
                    'sentiment_score': compound_score,
                    'confidence': abs(compound_score)
                }
            except Exception as e:
                logger.warning(f"VADER analysis failed: {e}")
        return None

    def _analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results or None if analysis fails
        """
        if 'textblob' in self.backends:
            try:
                blob = self.backends['textblob'](text)
                polarity = blob.sentiment.polarity
                if polarity > 0:
                    label = 'positive'
                elif polarity < 0:
                    label = 'negative'
                else:
                    label = 'neutral'
                return {
                    'sentiment_label': label,
                    'sentiment_score': polarity,
                    'confidence': abs(polarity)
                }
        except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")
        return None

    def analyze_sentiment(self, text: str, source: str = None) -> Dict[str, Any]:
        """Analyze sentiment in text using available tools.
        
        Args:
            text: The text to analyze
            source: Optional source override (trustpilot, imdb, playstore, steam)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment_label': 'unknown',
                'sentiment_score': 0.0,
                'confidence': 0.0
            }
        
        # Use provided source or default to instance source
        src = source.lower() if source else self.source
        
        # Check if source is valid (now check SENTIMENT_MODEL_MAP)
        model_map = getattr(ConfigManager, 'SENTIMENT_MODEL_MAP', None)
        if model_map is None:
            from src.config.manager import SENTIMENT_MODEL_MAP as model_map
        model_name = model_map.get(src)
        if not model_name or model_name not in ConfigManager.get_model_configs():
            logger.warning(f"Invalid source '{src}'. Using default model.")
            return {
                'sentiment_label': 'unknown',
                'sentiment_score': 0.0,
                'confidence': 0.0
            }
        
        # Try transformer first if available
        if 'transformers' in self.backends:
            model = self.backends['transformers']
            if model is None:
                logger.warning(f"Could not load model for source '{src}', falling back to VADER/TextBlob")
            else:
            try:
                result = model(text)[0]
                return SentimentAnalyzer.normalize_sentiment_output(result, src, text, self.language)
            except Exception as e:
                logger.error(f"Error in transformer sentiment analysis: {e}")
        
        # Try VADER for English text
        vader_result = self._analyze_with_vader(text)
        if vader_result:
            return vader_result
        
        # Try TextBlob as last resort
        textblob_result = self._analyze_with_textblob(text)
        if textblob_result:
            return textblob_result
        
        # If all analyzers fail, return unknown
        return {
            'sentiment_label': 'unknown',
            'sentiment_score': 0.0,
            'confidence': 0.0
        }