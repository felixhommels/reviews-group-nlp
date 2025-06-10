"""
Sentiment Analysis Module.

This module provides sentiment analysis capabilities using multiple approaches:
- Multilingual sentiment analysis using XLM-RoBERTa
- VADER (English) for sentiment analysis
- TextBlob as a fallback
"""

import logging
from typing import Dict, Union, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.dependencies import dependency_manager, DependencyError
from ..config import ConfigManager

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

class MultilingualSentimentAnalyzer:
    """Multilingual sentiment analysis using transformer models."""
    
    def __init__(self):
        """Initialize the multilingual sentiment analyzer with model from config."""
        try:
            from transformers import pipeline
            config = ConfigManager.get_sentiment_config()
            self.model_name = config['model_name']
            self.pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model_name
            )
            logger.info(f"Initialized multilingual sentiment analyzer with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize multilingual analyzer: {e}")
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
    
    def __init__(self, language: str = 'en'):
        """Initialize sentiment analyzers.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._init_analyzers()
    
    def _init_analyzers(self):
        """Initialize available sentiment analysis tools."""
        # Try to initialize multilingual analyzer first
        if dependency_manager.transformers_available:
            try:
                self.multilingual = MultilingualSentimentAnalyzer()
                logger.info("Initialized multilingual sentiment analyzer")
            except Exception as e:
                logger.warning(f"Could not initialize multilingual analyzer: {e}")
        
        # Initialize VADER for English sentiment analysis
        if dependency_manager.vader_available:
            try:
                self.vader = dependency_manager.get_component('vaderSentiment', 'SentimentIntensityAnalyzer')()
                logger.info("Initialized VADER sentiment analyzer")
            except DependencyError as e:
                logger.warning(f"Could not initialize VADER: {e}")
            
        # Check if any sentiment analysis tools are available
        if not any([dependency_manager.transformers_available, 
                    dependency_manager.vader_available,
                    dependency_manager.textblob_available]):
            raise ImportError("No sentiment analysis tools available. Install transformers, vaderSentiment, or textblob.")

    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze the sentiment of text using available tools.
        
        This method first attempts to use the multilingual XLM-RoBERTa model.
        If that's not available or fails, it falls back to VADER (for English)
        or TextBlob as a last resort.
        
        Args:
            text: The preprocessed text to analyze
            
        Returns:
            SentimentAnalysis object containing results
            
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        try:
            # Try multilingual analyzer first
            if hasattr(self, 'multilingual'):
                try:
                    result = self.multilingual.analyze_text(text)
                    return SentimentAnalysis(
                        sentiment=result['label'],
                        polarity=result['score'],
                        compound_score=result['score']
                    )
                except Exception as e:
                    logger.warning(f"Multilingual analyzer failed, falling back to VADER/TextBlob: {e}")
            
            # Fall back to VADER for English text
            if dependency_manager.vader_available and hasattr(self, 'vader'):
                scores = self.vader.polarity_scores(text)
                config = ConfigManager.get_sentiment_config()
                sentiment = (SentimentLabel.POSITIVE if scores['compound'] > config['thresholds']['positive']
                           else SentimentLabel.NEGATIVE if scores['compound'] < config['thresholds']['negative']
                           else SentimentLabel.NEUTRAL)
                return SentimentAnalysis(
                    sentiment=sentiment,
                    polarity=scores['compound'],
                    compound_score=scores['compound']
                )
            
            # Last resort: TextBlob
            elif dependency_manager.textblob_available:
                try:
                    TextBlob = dependency_manager.get_component('textblob', 'TextBlob')
                    blob = TextBlob(text)
                    config = ConfigManager.get_sentiment_config()
                    sentiment = (SentimentLabel.POSITIVE if blob.sentiment.polarity > config['thresholds']['positive']
                               else SentimentLabel.NEGATIVE if blob.sentiment.polarity < config['thresholds']['negative']
                               else SentimentLabel.NEUTRAL)
                    return SentimentAnalysis(
                        sentiment=sentiment,
                        polarity=blob.sentiment.polarity,
                        subjectivity=blob.sentiment.subjectivity
                    )
                except DependencyError as e:
                    logger.error(f"TextBlob analysis failed: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return SentimentAnalysis(
                sentiment=SentimentLabel.UNKNOWN,
                polarity=0.0
            )

class BaseAnalyzer:
    """Base class for all analyzers with dependency management."""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verify required dependencies are available."""
        pass