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
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Transformers for multilingual support
TRANSFORMERS_AVAILABLE = True

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

class MultilingualSentimentAnalyzer:
    """Handles multilingual sentiment analysis using XLM-RoBERTa."""
    
    def __init__(self):
        """Initialize the multilingual sentiment analyzer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package is required for multilingual analysis")
        
        self.model_name = "xlm-roberta-base"
        self.sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name
        )
    
    def analyze(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of text using XLM-RoBERTa.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentAnalysis object with results
        """
        try:
            result = self.sentiment_pipeline(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            sentiment = SentimentLabel.POSITIVE if score > 0.6 else (
                SentimentLabel.NEGATIVE if score < 0.4 else SentimentLabel.NEUTRAL
            )
            
            return SentimentAnalysis(
                sentiment=sentiment,
                polarity=score if sentiment == SentimentLabel.POSITIVE else -score,
                subjectivity=abs(score - 0.5) * 2
            )
        except Exception as e:
            logger.error(f"Error in multilingual sentiment analysis: {e}")
            return SentimentAnalysis(
                sentiment=SentimentLabel.UNKNOWN,
                polarity=0.0,
                subjectivity=0.0
            )

class ReviewAnalyzer:
    """A class for analyzing preprocessed review text data."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the ReviewAnalyzer."""
        self.language = language
        self._init_sentiment_analyzer()
        self._init_keyword_extractor()
        self._init_emotion_analyzer()
    
    def _init_sentiment_analyzer(self):
        """Initialize sentiment analysis tools based on availability."""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.multilingual = MultilingualSentimentAnalyzer()
                logger.info("Initialized multilingual sentiment analyzer")
            except Exception as e:
                logger.warning(f"Failed to initialize multilingual analyzer: {e}")
                self.multilingual = None
        
        if VADER_AVAILABLE and self.language == 'en':
            self.vader = SentimentIntensityAnalyzer()
        elif TEXTBLOB_AVAILABLE:
            logger.info("Using TextBlob for sentiment analysis")
        else:
            logger.warning("No sentiment analysis tools available")
    
    def _init_emotion_analyzer(self):
        """Initialize emotion analysis tools."""
        if NRCLEX_AVAILABLE:
            logger.info("Using NRCLex for emotion analysis")
        else:
            logger.info("Using rule-based emotion analysis")
    
    def _init_keyword_extractor(self):
        """Initialize keyword extraction tools."""
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english' if self.language == 'en' else None
        )
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of text using available tools.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentAnalysis object with results
        """
        # Try multilingual analysis first
        if self.multilingual:
            try:
                return self.multilingual.analyze(text)
            except Exception as e:
                logger.warning(f"Multilingual analysis failed: {e}")
        
        # Fallback to VADER for English
        if VADER_AVAILABLE and self.language == 'en':
            scores = self.vader.polarity_scores(text)
            sentiment = SentimentLabel.POSITIVE if scores['compound'] >= 0.05 else (
                SentimentLabel.NEGATIVE if scores['compound'] <= -0.05 else SentimentLabel.NEUTRAL
            )
            return SentimentAnalysis(
                sentiment=sentiment,
                polarity=scores['compound'],
                compound_score=scores['compound']
            )
        
        # Final fallback to TextBlob
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            sentiment = SentimentLabel.POSITIVE if blob.sentiment.polarity > 0 else (
                SentimentLabel.NEGATIVE if blob.sentiment.polarity < 0 else SentimentLabel.NEUTRAL
            )
            return SentimentAnalysis(
                sentiment=sentiment,
                polarity=blob.sentiment.polarity,
                subjectivity=blob.sentiment.subjectivity
            )
        
        return SentimentAnalysis(
            sentiment=SentimentLabel.UNKNOWN,
            polarity=0.0
        )
    
    def detect_emotion(self, text: str) -> EmotionLabel:
        """Detect primary emotion in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            EmotionLabel enum value
        """
        if NRCLEX_AVAILABLE:
            try:
                emotion_scores = NRCLex(text).raw_emotion_scores
                if not emotion_scores:
                    return EmotionLabel.NEUTRAL
                primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                return EmotionLabel(primary_emotion)
            except Exception as e:
                logger.warning(f"NRCLex emotion detection failed: {e}")
        
        # Rule-based fallback using basic emotion keywords
        positive_words = {'happy', 'great', 'excellent', 'good', 'love', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'poor', 'angry'}
        
        words = set(text.lower().split())
        pos_count = len(words.intersection(positive_words))
        neg_count = len(words.intersection(negative_words))
        
        if pos_count > neg_count:
            return EmotionLabel.JOY
        elif neg_count > pos_count:
            return EmotionLabel.ANGER
        return EmotionLabel.NEUTRAL
    
    def predict_stars(self, sentiment: SentimentAnalysis) -> int:
        """Predict star rating (1-5) based on sentiment analysis.
        
        Args:
            sentiment: SentimentAnalysis object
            
        Returns:
            Predicted star rating (1-5)
        """
        if sentiment.compound_score is not None:
            # Use VADER compound score for more accurate prediction
            score = (sentiment.compound_score + 1) / 2  # Normalize to 0-1
        else:
            score = (sentiment.polarity + 1) / 2
        
        return max(1, min(5, round(score * 4 + 1)))  # Scale to 1-5 range

    def extract_keywords(self, texts: List[str], top_n: int = 10) -> List[str]:
        """Extract most important keywords from a list of texts.
        
        Args:
            texts: List of input texts
            top_n: Number of keywords to return
            
        Returns:
            List of top keywords
        """
        try:
            tfidf_matrix = self.tfidf.fit_transform(texts)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Sum TF-IDF scores across all documents
            importance = tfidf_matrix.sum(axis=0).A1
            
            # Get indices of top keywords
            top_indices = importance.argsort()[-top_n:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    def analyze_emotion(self, text: str) -> Tuple[EmotionLabel, Dict[str, float]]:
        """Analyze emotions in text using NRCLex or rule-based fallback.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Tuple of (primary emotion, emotion scores dict)
        """
        try:
            if NRCLEX_AVAILABLE:
                emotion_analyzer = NRCLex(text)
                scores = emotion_analyzer.affect_frequencies
                
                # Filter and normalize emotion scores to match our EmotionLabel enum
                valid_emotions = {e.value: scores.get(e.value, 0.0) for e in EmotionLabel}
                
                if not any(valid_emotions.values()):
                    return EmotionLabel.NEUTRAL, {'neutral': 1.0}
                    
                # Get primary emotion (highest score among valid emotions)
                primary = max(valid_emotions.items(), key=lambda x: x[1])[0]
                return EmotionLabel(primary), valid_emotions
            else:
                # Enhanced rule-based system with multilingual keyword detection
                text_lower = text.lower()
                
                # Spanish emotion keywords
                spanish_emotions = {
                    'joy': ['excelente', 'perfecto', 'genial', 'encanta', 'maravilloso', 'fantástico', 'feliz'],
                    'anger': ['pésimo', 'terrible', 'horrible', 'penosa', 'fatal', 'malísimo', 'peor'],
                    'sadness': ['triste', 'decepcionante', 'lamentable', 'mal', 'pena'],
                    'fear': ['miedo', 'preocupa', 'inseguro', 'riesgo'],
                    'disgust': ['asco', 'repugnante', 'asqueroso'],
                    'surprise': ['sorprendente', 'increíble', 'impresionante', 'wow'],
                    'trust': ['confiable', 'seguro', 'fiable', 'confianza'],
                    'anticipation': ['espero', 'pronto', 'próximo', 'futuro']
                }
                
                # English emotion keywords
                english_emotions = {
                    'joy': ['excellent', 'perfect', 'great', 'love', 'wonderful', 'fantastic', 'happy'],
                    'anger': ['terrible', 'horrible', 'awful', 'worst', 'angry', 'mad'],
                    'sadness': ['sad', 'disappointing', 'unfortunately', 'poor', 'bad'],
                    'fear': ['scary', 'worried', 'afraid', 'concerned'],
                    'disgust': ['disgusting', 'gross', 'nasty'],
                    'surprise': ['surprising', 'amazing', 'wow', 'incredible'],
                    'trust': ['reliable', 'trustworthy', 'secure', 'safe'],
                    'anticipation': ['looking forward', 'soon', 'future', 'expect']
                }
                
                # Combine emotion dictionaries
                all_emotions = {}
                for emotion in EmotionLabel:
                    if emotion.value == 'neutral':
                        continue
                    keywords = set(spanish_emotions.get(emotion.value, []) + english_emotions.get(emotion.value, []))
                    score = sum(1.0 for word in keywords if word in text_lower)
                    all_emotions[emotion.value] = score / max(1, len(keywords))
                
                # Get sentiment for additional context
                sentiment = self.analyze_sentiment(text)
                sentiment_score = sentiment.compound_score if sentiment.compound_score is not None else sentiment.polarity
                
                # Boost emotion scores based on sentiment
                if sentiment_score > 0.5:
                    all_emotions['joy'] = max(all_emotions.get('joy', 0), 0.8)
                    all_emotions['trust'] = max(all_emotions.get('trust', 0), 0.6)
                elif sentiment_score < -0.5:
                    all_emotions['anger'] = max(all_emotions.get('anger', 0), 0.8)
                    all_emotions['disgust'] = max(all_emotions.get('disgust', 0), 0.6)
                
                # If no strong emotions detected, use sentiment as fallback
                if not any(score > 0.3 for score in all_emotions.values()):
                    if sentiment_score > 0.2:
                        all_emotions['joy'] = 0.6
                    elif sentiment_score < -0.2:
                        all_emotions['sadness'] = 0.6
                    else:
                        return EmotionLabel.NEUTRAL, {'neutral': 1.0}
                
                # Get primary emotion
                primary = max(all_emotions.items(), key=lambda x: x[1])[0]
                return EmotionLabel(primary), all_emotions
                
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return EmotionLabel.NEUTRAL, {'neutral': 1.0}

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
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts to get sentiment, emotion, and keywords.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of dictionaries containing analysis results for each text
        """
        results = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            emotion = self.detect_emotion(text)
            result = {
                'text': text,
                'sentiment': sentiment.sentiment.value,
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'emotion': emotion.value,
                'stars': self.predict_stars(sentiment)
            }
            results.append(result)
        
        # Extract keywords from all texts together
        if len(texts) > 0:
            keywords = self.extract_keywords(texts)
            for result in results:
                result['keywords'] = keywords
                
        return results

    def analyze_language_specific(self, text: str, language: str = None) -> Dict:
        """Analyze text with language-specific models when available.
        
        Args:
            text: Input text to analyze
            language: ISO language code (e.g. 'en', 'es'). If None, use instance default.
            
        Returns:
            Dictionary with language-specific analysis results
        """
        if language is None:
            language = self.language
            
        # Try using language-specific transformer model
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = {
                    'en': 'roberta-base',
                    'es': 'PlanTL-GOB-ES/roberta-base-biomedical-es',
                    'de': 'oliverguhr/german-sentiment-bert'
                }.get(language, self.multilingual.model_name)
                
                pipeline = Pipeline(
                    task='sentiment-analysis',
                    model=model_name,
                    tokenizer=model_name
                )
                result = pipeline(text)[0]
                return {
                    'model': model_name,
                    'label': result['label'],
                    'score': result['score']
                }
            except Exception as e:
                logger.warning(f"Language-specific model failed: {e}")
                
        # Fall back to general analysis
        sentiment = self.analyze_sentiment(text)
        return {
            'model': 'fallback',
            'label': sentiment.sentiment.value,
            'score': sentiment.polarity
        }

    def get_detailed_metrics(self, text: str) -> Dict:
        """Get detailed metrics including confidence scores for each sentiment/emotion.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed confidence scores
        """
        metrics = {}
        
        # Get sentiment scores
        sentiment = self.analyze_sentiment(text)
        metrics['sentiment'] = {
            'label': sentiment.sentiment.value,
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'compound': sentiment.compound_score
        }
        
        # Get emotion confidence scores
        if NRCLEX_AVAILABLE:
            try:
                emotions = NRCLex(text).raw_emotion_scores
                metrics['emotions'] = emotions
            except Exception as e:
                logger.warning(f"Detailed emotion scoring failed: {e}")
                metrics['emotions'] = {}
                
        # Add predicted star rating
        metrics['predicted_stars'] = self.predict_stars(sentiment)
        
        return metrics

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters while keeping essential punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
