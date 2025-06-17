"""Integration tests for the NLP Analysis module."""

import pytest
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.analysis.nlp_analysis import ReviewAnalyzer, SentimentLabel, EmotionLabel

console = Console()

def test_basic_analysis(analyzer, sample_texts):
    """Test basic analysis functionality."""
    # Test English text
    text = sample_texts['english'][0]
    sentiment = analyzer.analyze_sentiment(text)
    emotions = analyzer.analyze_emotion(text)
    keywords = analyzer.extract_keywords(text)
    stars = analyzer.predict_star_rating(text)
    
    # Verify results
    assert sentiment.sentiment == SentimentLabel.POSITIVE
    assert len(emotions) > 0
    assert len(keywords) > 0
    assert 1 <= stars <= 5

def test_multilingual_analysis(multilingual_analyzer, sample_texts):
    """Test analysis with multiple languages."""
    # Test Spanish text
    spanish_text = sample_texts['spanish'][0]
    sentiment = multilingual_analyzer.analyze_sentiment(spanish_text)
    emotions = multilingual_analyzer.analyze_emotion(spanish_text)
    keywords = multilingual_analyzer.extract_keywords(spanish_text)
    stars = multilingual_analyzer.predict_star_rating(spanish_text)
    
    # Verify results
    assert sentiment.sentiment == SentimentLabel.POSITIVE
    assert len(emotions) > 0
    assert len(keywords) > 0
    assert 1 <= stars <= 5

def test_batch_analysis(analyzer, sample_df):
    """Test batch analysis of multiple reviews."""
    # Run analysis
    results = analyzer.analyze_reviews(sample_df)
    
    # Verify results
    assert len(results) == len(sample_df)
    assert all(col in results.columns for col in [
        'sentiment', 'sentiment_polarity', 'keywords',
        'predicted_stars', 'primary_emotion'
    ])
    
    # Display results in a table
    table = Table(title="Analysis Results")
    table.add_column("Text", style="cyan")
    table.add_column("Sentiment", style="green")
    table.add_column("Stars", style="yellow")
    table.add_column("Primary Emotion", style="magenta")
    
    for _, row in results.iterrows():
        table.add_row(
            str(row['processed_text'])[:50] + "...",
            str(row['sentiment']),
            str(row['predicted_stars']),
            str(row['primary_emotion'])
        )
    
    console.print(table)

def test_edge_cases(analyzer):
    """Test analysis with edge cases."""
    # Test empty text
    with pytest.raises(ValueError):
        analyzer.analyze_sentiment("")
    
    # Test very long text
    long_text = "good " * 1000
    sentiment = analyzer.analyze_sentiment(long_text)
    assert isinstance(sentiment.sentiment, SentimentLabel)
    
    # Test text with special characters
    special_text = "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./"
    sentiment = analyzer.analyze_sentiment(special_text)
    assert isinstance(sentiment.sentiment, SentimentLabel)
    
    # Test text with numbers
    number_text = "1234567890"
    sentiment = analyzer.analyze_sentiment(number_text)
    assert isinstance(sentiment.sentiment, SentimentLabel)

def test_performance(analyzer, sample_df):
    """Test performance with larger dataset."""
    # Create larger dataset
    large_df = pd.concat([sample_df] * 10)
    
    # Run analysis and measure time
    import time
    start_time = time.time()
    results = analyzer.analyze_reviews(large_df)
    end_time = time.time()
    
    # Verify results
    assert len(results) == len(large_df)
    
    # Display performance metrics
    console.print(Panel(
        f"Processed {len(large_df)} reviews in {end_time - start_time:.2f} seconds\n"
        f"Average time per review: {(end_time - start_time) / len(large_df):.3f} seconds",
        title="Performance Metrics"
    ))

def test_error_handling(analyzer):
    """Test error handling and recovery."""
    # Test with invalid input types
    with pytest.raises(TypeError):
        analyzer.analyze_sentiment(123)
    
    with pytest.raises(TypeError):
        analyzer.analyze_sentiment(None)
    
    # Test with DataFrame missing required column
    bad_df = pd.DataFrame({'wrong_column': ['text']})
    with pytest.raises(ValueError):
        analyzer.analyze_reviews(bad_df)
    
    # Test with DataFrame containing invalid data
    invalid_df = pd.DataFrame({
        'processed_text': ['valid text', 123, None]
    })
    results = analyzer.analyze_reviews(invalid_df)
    assert len(results) == 3  # Should handle invalid data gracefully 