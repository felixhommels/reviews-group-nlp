"""
NLP Analysis Module for Review Analysis.

This module provides comprehensive natural language processing capabilities for analyzing
review text data, including:
- Sentiment analysis (via sentiment_analysis module)
- Keyword extraction (via keyword_extraction module)
- Emotion classification (via emotion_analysis module)
- Star rating prediction (via star_rating_predictor module)
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from ..utils.dependencies import dependency_manager, DependencyError
from src.config import ConfigManager
from .sentiment_analysis import SentimentAnalyzer, SentimentLabel, SentimentAnalysis
from .emotion_analysis import EmotionAnalyzer, EmotionLabel, MultilingualEmotionAnalyzer
from .keyword_extraction import KeywordExtractor
from .star_rating_predictor import StarRatingPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReviewAnalyzer:
    """A class for analyzing preprocessed review text data."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the ReviewAnalyzer.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._load_config()
        self._init_analyzers()
    
    def _load_config(self):
        """Load language-specific configuration."""
        self.config = {
            'sentiment': ConfigManager.get_sentiment_config(self.language),
            'emotion': ConfigManager.get_emotion_config(self.language),
            'rating': ConfigManager.get_rating_config(),
            'tfidf': ConfigManager.get_tfidf_config(self.language),
            'logging': ConfigManager.get_logging_config()
        }
        logger.info(f"Loaded all configurations for language: {self.language}")
    
    def _init_analyzers(self):
        """Initialize all analysis components."""
        try:
            self.sentiment_analyzer = SentimentAnalyzer(self.language)
            if self.language == 'en':
                self.emotion_analyzer = EmotionAnalyzer(self.language)
            else:
                self.emotion_analyzer = MultilingualEmotionAnalyzer()
            self.keyword_extractor = KeywordExtractor(self.language)
            self.star_rating_predictor = StarRatingPredictor(self.language)
            logger.info("Initialized all analysis components")
        except Exception as e:
            logger.error(f"Failed to initialize analysis components: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze the sentiment of text using the sentiment analyzer.
        
        Args:
            text: The preprocessed text to analyze
            
        Returns:
            SentimentAnalysis object containing results
            
        Raises:
            ValueError: If text is empty or None
        """
        return self.sentiment_analyzer.analyze_sentiment(text)
    
    def predict_star_rating(self, text: str) -> int:
        """Predict star rating (1-5) based on sentiment analysis.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Predicted star rating (1-5)
        """
        return self.star_rating_predictor.predict_star_rating(text)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using the keyword extractor.
        
        Args:
            text: The preprocessed text
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        return self.keyword_extractor.extract_keywords(text, top_k)

    def analyze_emotion(self, text: str) -> Dict[EmotionLabel, float]:
        """Analyze emotions in text using the emotion analyzer.
        Returns a dictionary mapping EmotionLabel to confidence scores.
        """
        result = self.emotion_analyzer.analyze_emotion(text)
        # If using MultilingualEmotionAnalyzer, result keys are strings, map to EmotionLabel if possible
        if isinstance(result, dict) and all(isinstance(k, str) for k in result.keys()):
            mapped = {}
            for k, v in result.items():
                try:
                    mapped[EmotionLabel[k.upper()]] = v
                except Exception:
                    # If not in enum, skip or keep as string
                    pass
            if mapped:
                return mapped
        return result

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
        result_df['primary_emotion'] = [max(e.items(), key=lambda x: x[1])[0] for e in emotions]
        for emotion in EmotionLabel:
            if any(emotion in scores for scores in emotions):
                result_df[f'emotion_{emotion.value}'] = [
                    scores.get(emotion, 0.0) for scores in emotions
                ]
        
        logger.info("Review analysis completed successfully.")
        return result_df
        