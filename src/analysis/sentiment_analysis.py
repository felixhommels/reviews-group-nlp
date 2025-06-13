"""
Sentiment Analysis Module.

This module provides sentiment analysis capabilities using multiple approaches:
- Multilingual sentiment analysis using XLM-RoBERTa for Trustpilot/IMDb
- BERT-based sentiment analysis for Steam/Playstore
- VADER (English) for sentiment analysis
- TextBlob as a fallback
"""

import logging
from typing import Dict, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from src.utils.dependencies import dependency_manager, DependencyError
from src.config.manager import ConfigManager, FALLBACK_ORDER

# Configure logging
logger = logging.getLogger(__name__)

# Model mapping for different sources
SENTIMENT_MODEL_MAP = {
    'steam': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'playstore': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'imdb': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'trustpilot': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'default': 'nlptown/bert-base-multilingual-uncased-sentiment'
}

# Model configurations
MODEL_CONFIGS = {
    'nlptown/bert-base-multilingual-uncased-sentiment': {
        'use_fast_tokenizer': True,
        'model_max_length': 512,
        'truncation': True,
        'max_length': 512
    }
}

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

class SentimentLabel(str, Enum):
    """Enum for sentiment classification labels."""
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    UNKNOWN = 'unknown'

@dataclass
class SentimentAnalysis:
    """Container for sentiment analysis results."""
    sentiment: SentimentLabel
    polarity: float
    subjectivity: Optional[float] = None
    compound_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the sentiment analysis result to a dictionary."""
        return {
            'label': self.sentiment,
            'score': self.polarity,
            'compound_score': self.compound_score,
            'subjectivity': self.subjectivity
        }

@lru_cache(maxsize=8)
def get_sentiment_model(source: str):
    """Return the correct Hugging Face pipeline for the platform."""
    try:
        model_name = SENTIMENT_MODEL_MAP.get(source.lower(), SENTIMENT_MODEL_MAP['default'])
        config = MODEL_CONFIGS.get(model_name, {})
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=config.get('use_fast_tokenizer', True),
            model_max_length=config.get('model_max_length', 512)
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            truncation=config.get('truncation', True),
            max_length=config.get('max_length', 512)
        )
    except Exception as e:
        logger.error(f"Error loading sentiment model for source '{source}': {e}")
        return None

def normalize_sentiment_output(model_output: dict, source: str, text: str) -> dict:
    """Convert model output to unified format."""
    try:
        # Nlptown: 1 star (negative) to 5 stars (positive)
        raw_label = model_output['label']
        score = model_output['score']
        
        try:
            stars = int(raw_label.split()[0])
            if stars <= 2:
                label = 'negative'
            elif stars == 3:
                label = 'neutral'
            else:
                label = 'positive'
        except Exception:
            label = 'neutral'
            
        return {
            'sentiment_label': label,
            'sentiment_score': score,
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

def analyze_sentiment(text: str, source: str = "default") -> Dict[str, Any]:
    """Analyze sentiment of text using appropriate model for source."""
    if not text or not isinstance(text, str):
        return {
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0
        }
    
    try:
        # Get appropriate model for source
        model = get_sentiment_model(source)
        if model is None:
            logger.warning(f"Could not load model for source '{source}', falling back to VADER/TextBlob")
            return _fallback_sentiment_analysis(text)
        
        # Get sentiment prediction with automatic truncation
        result = model(text)[0]
        return normalize_sentiment_output(result, source, text)
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return _fallback_sentiment_analysis(text)

def _fallback_sentiment_analysis(text: str) -> Dict[str, Any]:
    """Fallback sentiment analysis using VADER or TextBlob."""
    try:
        if VADER_AVAILABLE:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
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
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
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
        logger.error(f"Error in fallback sentiment analysis: {e}")
    
    return {
        'sentiment_label': 'neutral',
        'sentiment_score': 0.0,
        'confidence': 0.0
    }

class MultilingualSentimentAnalyzer:
    """Multilingual sentiment analysis using transformer models."""
    
    def __init__(self, source: str = 'trustpilot'):
        """Initialize the multilingual sentiment analyzer with model based on source."""
        try:
            from transformers import pipeline
            self.source = source.lower()
            self.model_name = SENTIMENT_MODEL_MAP.get(self.source, SENTIMENT_MODEL_MAP['trustpilot'])
            config = MODEL_CONFIGS.get(self.model_name, {})
            self.pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model_name,
                truncation=config.get('truncation', True),
                max_length=config.get('max_length', 512)
            )
            logger.info(f"Initialized multilingual sentiment analyzer with model: {self.model_name} for source: {self.source}")
        except Exception as e:
            logger.error(f"Failed to initialize multilingual analyzer for source '{self.source}' with model '{getattr(self, 'model_name', 'unknown')}': {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """Analyze text using the multilingual model.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing:
                - label: The sentiment label (positive/negative/neutral)
                - score: The confidence score
                - detailed_scores: Optional detailed analysis
        """
        try:
            if not text or not isinstance(text, str):
                return {
                    'label': SentimentLabel.UNKNOWN,
                    'score': 0.0,
                    'detailed_scores': {'raw_score': 0.0}
                }

            result = self.pipeline(text)[0]
            score = result['score']
            
            # Convert raw score to standardized range [-1, 1]
            standardized_score = (2 * score) - 1
            
            # Use thresholds from config
            config = ConfigManager.get_sentiment_config()
            label = (SentimentLabel.POSITIVE 
                    if standardized_score > config['thresholds']['positive']
                    else SentimentLabel.NEGATIVE 
                    if standardized_score < config['thresholds']['negative']
                    else SentimentLabel.NEUTRAL)
            
            return {
                'label': label,
                'score': standardized_score,
                'detailed_scores': {
                    'raw_score': score
                }
            }
        
        except Exception as e:
            logger.error(f"Error in multilingual sentiment analysis: {e}")
            return {
                'label': SentimentLabel.UNKNOWN,
                'score': 0.0,
                'detailed_scores': {
                    'raw_score': 0.0
                }
            }

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
                self.backends['transformers'] = MultilingualSentimentAnalyzer(source=self.source)
                logger.info("Initialized multilingual sentiment analyzer")
            except Exception as e:
                error_msg = f"Could not initialize multilingual analyzer for source '{self.source}': {e}"
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
                'label': SentimentLabel.UNKNOWN,
                'score': 0.0,
                'compound_score': 0.0
            }

        # Use provided source or default to instance source
        source = source.lower() if source else self.source
        
        # Check if source is valid
        if source not in SENTIMENT_MODEL_MAP:
            logger.warning(f"Invalid source '{source}'. Using default source '{self.source}'.")
            return {
                'label': SentimentLabel.UNKNOWN,
                'score': 0.0,
                'compound_score': 0.0
            }

        # Try transformers first if available
        if 'transformers' in self.backends:
            try:
                result = self.backends['transformers'].analyze_text(text)
                return result
            except Exception as e:
                logger.warning(f"Transformers analysis failed: {e}")

        # Try VADER for English text
        if self.language == 'en' and 'vader' in self.backends:
            try:
                scores = self.backends['vader'].polarity_scores(text)
                compound_score = scores['compound']
                
                # Convert compound score to label
                if compound_score >= 0.05:
                    label = SentimentLabel.POSITIVE
                elif compound_score <= -0.05:
                    label = SentimentLabel.NEGATIVE
                else:
                    label = SentimentLabel.NEUTRAL
                
                return {
                    'label': label,
                    'score': compound_score,
                    'compound_score': compound_score
                }
            except Exception as e:
                logger.warning(f"VADER analysis failed: {e}")

        # Try TextBlob as last resort
        if 'textblob' in self.backends:
            try:
                blob = self.backends['textblob'](text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Convert polarity to label
                if polarity > 0:
                    label = SentimentLabel.POSITIVE
                elif polarity < 0:
                    label = SentimentLabel.NEGATIVE
                else:
                    label = SentimentLabel.NEUTRAL
                
                return {
                    'label': label,
                    'score': polarity,
                    'compound_score': polarity,
                    'subjectivity': subjectivity
                }
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")

        # If all analyzers fail, return unknown
        return {
            'label': SentimentLabel.UNKNOWN,
            'score': 0.0,
            'compound_score': 0.0
        }

class BaseAnalyzer:
    """Base class for all analyzers with dependency management."""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verify required dependencies are available."""
        pass