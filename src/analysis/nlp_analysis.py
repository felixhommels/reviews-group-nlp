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
from pathlib import Path
import json
from langdetect import detect, LangDetectException

from src.utils.dependencies import dependency_manager, DependencyError
from src.config.manager import ConfigManager
from src.analysis.sentiment_analysis import (
    SentimentAnalyzer
)
from src.analysis.emotion_analysis import (
    EnglishEmotionAnalyzerHartmann,
    SpanishEmotionAnalyzerRobertuito
)
from src.analysis.keyword_extraction import KeywordExtractor
from src.analysis.star_rating_predictor import StarRatingPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_nlp_pipeline(
    review: dict,
    sentiment_analyzer,
    keyword_extractor,
    english_emotion_analyzer,
    spanish_emotion_analyzer,
    rating_predictor
):
    text = review.get("processed_text", "")
    # --- Sentiment ---
    sentiment_result = sentiment_analyzer.analyze_sentiment(text)
    review.update(sentiment_result)
    # --- Keywords ---
    review["keywords"] = keyword_extractor.extract_keywords(text)
    # --- Emotion ---
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"
    if lang == "en":
        emotions = english_emotion_analyzer.analyze_emotion(text)
    elif lang == "es":
        emotions = spanish_emotion_analyzer.analyze_emotion(text)
    else:
        emotions = english_emotion_analyzer.analyze_emotion(text)
    review["top_emotion"] = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
    review["emotion_scores"] = emotions
    # --- Star Rating ---
    predicted = rating_predictor.predict_star_rating(text)
    review["predicted_rating_raw"] = predicted
    review["predicted_rating_normalized"] = rating_predictor.normalize_rating(predicted, review.get("source", ""))
    return review
        