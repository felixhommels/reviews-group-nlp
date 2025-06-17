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
from langdetect import detect, LangDetectException

from src.analysis.sentiment_analysis import map_score_to_granular_label

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
    review["sentiment_granular_label"] = map_score_to_granular_label(review["sentiment_continuous_score"])
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
        