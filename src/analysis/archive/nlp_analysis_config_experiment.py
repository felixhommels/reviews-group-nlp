# LEGACY/REFERENCE: Not used in production. See [nlp_analysis.py]for the current version.
"""
NLP Analysis Module for Review Analysis.

This module provides comprehensive natural language processing capabilities for analyzing
review text data, including:
- Multilingual sentiment analysis using XLM-RoBERTa
- VADER (English) or TextBlob (fallback) for sentiment analysis
- Keyword extraction using TF-IDF
- Star rating prediction based on sentiment
- Emotion classification using NRCLex or rule-based fallback
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer

from ...utils.dependencies import dependency_manager
from ...config import SENTIMENT_CONFIG, TFIDF_CONFIG, RATING_CONFIG, EMOTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

# Optional: Emotion classification
try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False
    logger.warning("NRCLex not available. Using rule-based emotion classification.")

class SentimentLabel(str, Enum):
    """Enum for sentiment classification labels."""
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    UNKNOWN = 'unknown'

class EmotionLabel(str, Enum):
    """Enum for emotion classification labels."""
    JOY = 'joy'
    SURPRISE = 'surprise'
    ANGER = 'anger'
    SADNESS = 'sadness'
    NEUTRAL = 'neutral'
    FEAR = 'fear'
    DISGUST = 'disgust'
    TRUST = 'trust'
    ANTICIPATION = 'anticipation'

@dataclass
class SentimentAnalysis:
    """Container for sentiment analysis results."""
    sentiment: SentimentLabel
    polarity: float
    subjectivity: Optional[float] = None
    compound_score: Optional[float] = None

class ReviewAnalyzer:
    """A class for analyzing preprocessed review text data."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the ReviewAnalyzer.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._init_sentiment_analyzer()
        self._init_keyword_extractor()
    
    def _init_sentiment_analyzer(self):
        """Initialize sentiment analysis tools based on availability."""
        # Try to initialize multilingual analyzer first
        if dependency_manager.transformers_available:
            try:
                self.multilingual = MultilingualSentimentAnalyzer()
                logger.info("Initialized multilingual sentiment analyzer")
            except Exception as e:
                logger.warning(f"Could not initialize multilingual analyzer: {e}")
        
        # Initialize VADER for English sentiment analysis
        if dependency_manager.vader_available:
            self.vader = dependency_manager.get_component('vaderSentiment.vaderSentiment', 'SentimentIntensityAnalyzer')()
            logger.info("Initialized VADER sentiment analyzer")
            
        # TextBlob is initialized per-analysis
        if not any([dependency_manager.transformers_available, 
                    dependency_manager.vader_available,
                    dependency_manager.textblob_available]):
            raise ImportError("No sentiment analysis tools available. Install transformers, vaderSentiment, or textblob.")

    def _init_keyword_extractor(self):
        """Initialize keyword extraction tools."""
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_CONFIG['max_features'],
            stop_words=TFIDF_CONFIG['stop_words'].get(self.language, None),
            ngram_range=TFIDF_CONFIG['ngram_range']
        )
    
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
                sentiment = (SentimentLabel.POSITIVE if scores['compound'] > SENTIMENT_CONFIG['thresholds']['positive']
                           else SentimentLabel.NEGATIVE if scores['compound'] < SENTIMENT_CONFIG['thresholds']['negative']
                           else SentimentLabel.NEUTRAL)
                return SentimentAnalysis(
                    sentiment=sentiment,
                    polarity=scores['compound'],
                    compound_score=scores['compound']
                )
            
            # Last resort: TextBlob
            elif dependency_manager.textblob_available:
                TextBlob = dependency_manager.get_component('textblob', 'TextBlob')
                blob = TextBlob(text)
                sentiment = (SentimentLabel.POSITIVE if blob.sentiment.polarity > SENTIMENT_CONFIG['thresholds']['positive']
                           else SentimentLabel.NEGATIVE if blob.sentiment.polarity < SENTIMENT_CONFIG['thresholds']['negative']
                           else SentimentLabel.NEUTRAL)
                return SentimentAnalysis(
                    sentiment=sentiment,
                    polarity=blob.sentiment.polarity,
                    subjectivity=blob.sentiment.subjectivity
                )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return SentimentAnalysis(
                sentiment=SentimentLabel.UNKNOWN,
                polarity=0.0
            )
    
    def predict_star_rating(self, text: str) -> int:
        """Predict star rating (1-5) based on sentiment analysis.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Predicted star rating (1-5)
        """
        try:
            # Try multilingual analyzer first
            if hasattr(self, 'multilingual'):
                try:
                    result = self.multilingual.analyze_text(text)
                    if 'detailed_scores' in result and 'rating' in result['detailed_scores']:
                        return int(result['detailed_scores']['rating'])
                except Exception:
                    pass  # Fall through to traditional analysis

            # Traditional analysis using sentiment score thresholds
            sentiment_analysis = self.analyze_sentiment(text)
            score = sentiment_analysis.compound_score or sentiment_analysis.polarity

            # Use rating thresholds from config
            for rating, threshold in sorted(RATING_CONFIG['thresholds'].items(), key=lambda x: float('-inf') if x[1] is None else x[1], reverse=True):
                if threshold is None or score >= threshold:
                    return rating

            return RATING_CONFIG['default_rating']

        except Exception as e:
            logger.error(f"Error in star rating prediction: {str(e)}")
            return RATING_CONFIG['default_rating']

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

    def analyze_emotion(self, text: str, language: str = 'en') -> Dict[EmotionLabel, float]:
        """Analyze emotions in text using NRCLex or rule-based method.
        
        Args:
            text: The preprocessed text
            language: ISO language code (default: 'en')
            
        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        emotions = {}
        try:
            if NRCLEX_AVAILABLE and language == 'en':
                # Use NRCLex for English text
                nrc = NRCLex(text)
                raw_scores = nrc.affect_frequencies
                
                # Map NRCLex emotions to our EmotionLabel enum
                for emotion in EmotionLabel:
                    if emotion.value in raw_scores:
                        if raw_scores[emotion.value] >= EMOTION_CONFIG['score_thresholds']['strong']:
                            emotions[emotion] = raw_scores[emotion.value]

            else:
                # Rule-based emotion detection using keyword matching
                keywords = EMOTION_CONFIG['keywords'][language]
                text_lower = text.lower()
                
                for emotion, words in keywords.items():
                    count = sum(1 for word in words if word.lower() in text_lower)
                    if count > 0:
                        score = min(count / len(words), 1.0)
                        if score >= EMOTION_CONFIG['score_thresholds']['weak']:
                            emotions[EmotionLabel(emotion)] = score

            if not emotions:
                emotions[EmotionLabel.NEUTRAL] = 1.0

        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            emotions[EmotionLabel.NEUTRAL] = 1.0
            
        return emotions

    def analyze_reviews(self, df: pd.DataFrame, text_column: str = 'processed_text') -> pd.DataFrame:
        """Perform comprehensive analysis on a DataFrame of reviews.
        
        Args:
            df: DataFrame containing the reviews
            text_column: Name of the column containing preprocessed text
            
        Returns:
            DataFrame with added analysis columns
            
        Raises:
            ValueError: If text_column is not found in DataFrame
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        logger.info("Starting review analysis...")
        result_df = df.copy()
        texts = result_df[text_column].astype(str).fillna('').tolist()
        
        # Analyze sentiment
        logger.info("Performing sentiment analysis...")
        sentiments = [self.analyze_sentiment(t) for t in texts]
        result_df['sentiment'] = [s.sentiment for s in sentiments]
        result_df['sentiment_polarity'] = [s.polarity for s in sentiments]
        if any(s.subjectivity is not None for s in sentiments):
            result_df['sentiment_subjectivity'] = [s.subjectivity for s in sentiments]
        
        # Extract keywords
        logger.info("Extracting keywords...")
        result_df['keywords'] = [self.extract_keywords(t) for t in texts]
        
        # Predict star ratings
        logger.info("Predicting star ratings...")
        result_df['predicted_stars'] = [self.predict_star_rating(t) for t in texts]
        
        # Analyze emotions
        logger.info("Analyzing emotions...")
        emotions = [self.analyze_emotion(t) for t in texts]
        result_df['primary_emotion'] = [e[0] for e in emotions]
        emotion_scores = [e[1] for e in emotions]
        for emotion in EmotionLabel:
            if any(emotion.value in scores for scores in emotion_scores):
                result_df[f'emotion_{emotion.value}'] = [
                    scores.get(emotion.value, 0.0) for scores in emotion_scores
                ]
        
        logger.info("Review analysis completed successfully.")
        return result_df

class MultilingualSentimentAnalyzer:
    """Multilingual sentiment analysis using transformer models."""
    
    def __init__(self):
        """Initialize the multilingual sentiment analyzer with model from config."""
        try:
            from transformers import pipeline
            self.model_name = SENTIMENT_CONFIG['model_name']
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
            label = (SentimentLabel.POSITIVE 
                    if standardized_score > SENTIMENT_CONFIG['thresholds']['positive']
                    else SentimentLabel.NEGATIVE 
                    if standardized_score < SENTIMENT_CONFIG['thresholds']['negative']
                    else SentimentLabel.NEUTRAL)

            # Map score ranges to star ratings
            rating = None
            for stars, threshold in sorted(RATING_CONFIG['thresholds'].items(), 
                                        key=lambda x: float('-inf') if x[1] is None else x[1], 
                                        reverse=True):
                if threshold is None or standardized_score >= threshold:
                    rating = stars
                    break
            
            return {
                'label': label,
                'score': standardized_score,
                'detailed_scores': {
                    'raw_score': score,
                    'rating': rating or RATING_CONFIG['default_rating']
                }
            }
        
        except Exception as e:
            logger.error(f"Error in multilingual sentiment analysis: {e}")
            return {
                'label': SentimentLabel.UNKNOWN,
                'score': 0.0,
                'detailed_scores': {
                    'raw_score': 0.0,
                    'rating': RATING_CONFIG['default_rating']
                }
            }
