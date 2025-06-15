"""
NLP Analysis Module for Review Analysis.

This module provides comprehensive natural language processing capabilities for analyzing
review text data, including:
- Sentiment analysis (via sentiment_analysis module)
- Keyword extraction (via keyword_extraction module)
- Emotion classification (via emotion_analysis module)
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

from src.utils.dependencies import dependency_manager, DependencyError
from src.config.manager import ConfigManager
from src.analysis.sentiment_analysis import (
    SentimentAnalyzer
)
from src.analysis.emotion_analysis import EmotionAnalyzer, EmotionLabel, MultilingualEmotionAnalyzer
from src.analysis.keyword_extraction import KeywordExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReviewAnalyzer:
    """A class for analyzing preprocessed review text data."""
    
    def __init__(self, language: str = 'en', source: str = 'trustpilot'):
        """Initialize the ReviewAnalyzer.
        
        Args:
            language: ISO language code (default: 'en')
            source: Review source (trustpilot, imdb, playstore, steam). If analyzing mixed sources, use source_column in analyze_reviews.
        """
        self.language = language
        self.source = source
        self._load_config()
        self._sentiment_analyzer = None
        self._emotion_analyzer = None
        self._keyword_extractor = None
    
    def _load_config(self):
        """Load language-specific configuration."""
        self.config = {
            'sentiment': None,  # Or you could use a default model config if needed
            'emotion': ConfigManager.get_emotion_config(self.language),
            'tfidf': ConfigManager.get_tfidf_config(self.language),
            'logging': ConfigManager.get_logging_config()
        }
        logger.info(f"Loaded all configurations for language: {self.language}")
    
    @property
    def sentiment_analyzer(self):
        """Get the sentiment analyzer for the current source."""
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = SentimentAnalyzer(language=self.language, source=self.source)
        return self._sentiment_analyzer
    
    @property
    def emotion_analyzer(self):
        if self._emotion_analyzer is None:
            if self.language == 'en':
                self._emotion_analyzer = EmotionAnalyzer(self.language)
            else:
                self._emotion_analyzer = MultilingualEmotionAnalyzer()
        return self._emotion_analyzer
    
    @property
    def keyword_extractor(self) -> KeywordExtractor:
        if self._keyword_extractor is None:
            self._keyword_extractor = KeywordExtractor(self.language)
        return self._keyword_extractor
    
    def analyze_sentiment(self, text: str, source: str = None) -> dict:
        """
        Analyze the sentiment of text using the new transformer-only pipeline.
        Args:
            text: The preprocessed text to analyze
            source: The review source (overrides default if provided)
        Returns:
            Unified sentiment output dict
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        src = source if source else self.source
        return self.sentiment_analyzer.analyze_sentiment(text, source=src)

    def analyze_reviews(
        self,
        df: pd.DataFrame,
        text_column: str = 'processed_text',
        source_column: str = None
    ) -> pd.DataFrame:
        """
        Perform comprehensive analysis on a DataFrame of reviews.
        Args:
            df: DataFrame containing the reviews
            text_column: Name of the column containing preprocessed text
            source_column: Name of the column containing the review source (optional, for mixed sources)
        Returns:
            DataFrame with added analysis columns
        Raises:
            ValueError: If text_column is not found in DataFrame
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        if source_column and source_column not in df.columns:
            raise ValueError(f"Column '{source_column}' not found in DataFrame")
            
        logger.info("Starting review analysis...")
        result_df = df.copy()
        texts = result_df[text_column].astype(str).fillna('').tolist()
        sources = result_df[source_column].astype(str).fillna(self.source).tolist() if source_column else [self.source] * len(texts)
        
        # Analyze sentiment
        logger.info("Performing sentiment analysis...")
        sentiment_dicts = []
        for text, source in zip(texts, sources):
            try:
                sentiment = self.analyze_sentiment(text, source)
                sentiment_dicts.append(sentiment)
            except Exception as e:
                logger.error(f"Error analyzing sentiment for text: {e}")
                sentiment_dicts.append({
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'source': source,
                    'raw_model_label': 'unknown',
                    'original_text': text
                })
        
        # Add sentiment results to DataFrame
        result_df['sentiment'] = [s['sentiment_label'] for s in sentiment_dicts]
        result_df['sentiment_score'] = [s['sentiment_score'] for s in sentiment_dicts]
        result_df['sentiment_confidence'] = [s['confidence'] for s in sentiment_dicts]
        
        # Extract keywords using batch processing
        logger.info("Extracting keywords...")
        result_df['keywords'] = self.extract_keywords(texts)
        
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

    def extract_keywords(self, texts: List[str], batch_size: int = 16) -> List[List[str]]:
        """Extract keywords from a list of texts using batch processing.
        
        Args:
            texts: List of texts to extract keywords from
            batch_size: Size of batches for processing
            
        Returns:
            List of keyword lists, one for each input text
        """
        try:
            return self.keyword_extractor.extract_keywords_batch(
                texts,
                language=self.language,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Error in batch keyword extraction: {e}")
            return [[] for _ in texts]

    def analyze_emotion(self, text: str) -> Dict[EmotionLabel, float]:
        """Analyze emotions in text using the emotion analyzer.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Dictionary mapping emotions to scores
        """
        return self.emotion_analyzer.analyze_emotion(text)
    # this is wrong, functon no longer used - its analyze_emotion from EnglishEmotionAnalyzerHartmann and 
    # SpanishEmotionAnalyzerRobertuito
        