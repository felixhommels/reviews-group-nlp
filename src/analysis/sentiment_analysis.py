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
import torch
import torch.nn.functional as F

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
    """Return the correct Hugging Face pipeline for the platform."""
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
        """Convert model output to unified format."""
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

    @staticmethod
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

    def compute_continuous_score(self, text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        # For cardiffnlp/twitter-xlm-roberta-base-sentiment: 0=neg, 1=neutral, 2=pos
        sentiment_score = float(probs[2]) - float(probs[0])
        return sentiment_score, probs

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
            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
            if model is None or tokenizer is None:
                logger.warning(f"Could not load model/tokenizer for source '{src}', falling back to VADER/TextBlob")
                result = self._fallback_sentiment_analysis(text)
                result['sentiment_continuous_score'] = None
                return result
            try:
                # Get the default pipeline result
                result = model(text)[0]
                norm_result = SentimentAnalyzer.normalize_sentiment_output(result, src, text, self.language)
                # Compute the continuous score
                continuous_score, probs = self.compute_continuous_score(text, model.model, model.tokenizer)
                norm_result['sentiment_continuous_score'] = continuous_score
                norm_result['sentiment_class_probabilities'] = probs.tolist()
                return norm_result
            except Exception as e:
                logger.error(f"Error in transformer sentiment analysis: {e}")
                result = self._fallback_sentiment_analysis(text)
                result['sentiment_continuous_score'] = None
                return result
        # Try VADER for English text
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
        # Try TextBlob as last resort
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
        # If all analyzers fail, return unknown
        return {
            'sentiment_label': 'unknown',
            'sentiment_score': 0.0,
            'confidence': 0.0
        }

def map_score_to_granular_label(continuous_score: float) -> str:
    """
    Map a continuous sentiment score (e.g., P(positive) - P(negative), in [-1, 1])
    to a granular sentiment label.
    """
    if continuous_score >= 0.75:
        return "very_positive"
    elif continuous_score >= 0.25:
        return "somewhat_positive"
    elif continuous_score > -0.25:
        return "neutral"
    elif continuous_score > -0.75:
        return "somewhat_negative"
    else:
        return "very_negative"