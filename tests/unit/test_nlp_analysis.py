"""Unit tests for the NLP Analysis module."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.analysis.nlp_analysis import (
    ReviewAnalyzer,
    SentimentLabel,
    EmotionLabel,
    SentimentAnalysis
)

class TestReviewAnalyzer:
    """Test cases for the ReviewAnalyzer class."""

    def test_init(self, analyzer):
        """Test initialization of ReviewAnalyzer."""
        assert analyzer.language == 'en'
        assert hasattr(analyzer, 'config')

    def test_analyze_sentiment_positive(self, analyzer, sample_texts):
        """Test sentiment analysis with positive text."""
        result = analyzer.analyze_sentiment(sample_texts['english'][0])
        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment == SentimentLabel.POSITIVE
        assert result.polarity > 0

    def test_analyze_sentiment_negative(self, analyzer, sample_texts):
        """Test sentiment analysis with negative text."""
        result = analyzer.analyze_sentiment(sample_texts['english'][1])
        assert result.sentiment == SentimentLabel.NEGATIVE
        assert result.polarity < 0

    def test_analyze_sentiment_neutral(self, analyzer, sample_texts):
        """Test sentiment analysis with neutral text."""
        result = analyzer.analyze_sentiment(sample_texts['english'][2])
        assert result.sentiment == SentimentLabel.NEUTRAL

    def test_analyze_sentiment_empty_input(self, analyzer):
        """Test sentiment analysis with empty input."""
        with pytest.raises(ValueError):
            analyzer.analyze_sentiment("")

    def test_predict_star_rating(self, analyzer, sample_texts):
        """Test star rating prediction."""
        # Test positive review
        rating = analyzer.predict_star_rating(sample_texts['english'][0])
        assert rating >= 4
        
        # Test negative review
        rating = analyzer.predict_star_rating(sample_texts['english'][1])
        assert rating <= 2
        
        # Test neutral review
        rating = analyzer.predict_star_rating(sample_texts['english'][2])
        assert rating == 3

    def test_extract_keywords(self, analyzer, sample_texts):
        """Test keyword extraction."""
        keywords = analyzer.extract_keywords(sample_texts['english'][0])
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "amazing" in keywords  # Common word in test text
        
        # Test with empty text
        empty_keywords = analyzer.extract_keywords("")
        assert empty_keywords == []

    def test_analyze_emotion(self, analyzer, sample_texts):
        """Test emotion analysis."""
        emotions = analyzer.analyze_emotion(sample_texts['english'][0])
        assert isinstance(emotions, dict)
        assert len(emotions) > 0
        assert all(isinstance(e, EmotionLabel) for e in emotions.keys())

    @patch('src.analysis.nlp_analysis.dependency_manager.nrclex_available', False)
    def test_analyze_emotion_fallback(self, analyzer, sample_texts):
        """Test emotion analysis fallback when NRCLex is not available."""
        emotions = analyzer.analyze_emotion(sample_texts['english'][0])
        assert isinstance(emotions, dict)
        assert len(emotions) > 0

    def test_analyze_reviews(self, analyzer, sample_df):
        """Test full review analysis pipeline."""
        result_df = analyzer.analyze_reviews(sample_df)
        
        # Check that all expected columns are present
        expected_columns = {
            'processed_text', 'sentiment', 'sentiment_polarity',
            'keywords', 'predicted_stars', 'primary_emotion'
        }
        assert expected_columns.issubset(result_df.columns)
        
        # Check that we have the same number of rows
        assert len(result_df) == len(sample_df)
        
        # Check that sentiment values are valid
        assert all(s in SentimentLabel for s in result_df['sentiment'])
        
        # Check that star ratings are in valid range
        assert all(1 <= stars <= 5 for stars in result_df['predicted_stars'])
        
        # Check that emotions are valid
        assert all(e in EmotionLabel for e in result_df['primary_emotion'])

    def test_analyze_reviews_missing_column(self, analyzer):
        """Test review analysis with missing text column."""
        bad_df = pd.DataFrame({'wrong_column': ['text']})
        with pytest.raises(ValueError):
            analyzer.analyze_reviews(bad_df)

    def test_analyze_reviews_empty_df(self, analyzer):
        """Test review analysis with empty DataFrame."""
        empty_df = pd.DataFrame({'processed_text': []})
        result = analyzer.analyze_reviews(empty_df)
        assert len(result) == 0

    def test_analyze_reviews_null_values(self, analyzer):
        """Test review analysis with null values."""
        df_with_nulls = pd.DataFrame({
            'processed_text': ["Good product", None, "Bad product"]
        })
        result = analyzer.analyze_reviews(df_with_nulls)
        assert len(result) == 3  # Should handle null values

    def test_multilingual_analysis(self, multilingual_analyzer, sample_texts):
        """Test analysis with Spanish text."""
        result = multilingual_analyzer.analyze_sentiment(sample_texts['spanish'][0])
        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment == SentimentLabel.POSITIVE 