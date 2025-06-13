"""
Test script to verify the integration of the new sentiment analysis pipeline.
Tests both individual review analysis and batch processing.
"""

import unittest
import pandas as pd
import json
from pathlib import Path
import logging

from src.analysis.nlp_analysis import ReviewAnalyzer
from src.analysis.sentiment_analysis import SentimentLabel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestSentimentIntegration(unittest.TestCase):
    """Test cases for sentiment analysis integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and analyzers."""
        # Load test data
        test_file = Path('data/processed_test_results/processed_570_steam_reviews.json')
        with open(test_file, 'r', encoding='utf-8') as f:
            cls.test_reviews = json.load(f)
        
        # Initialize analyzers for different sources
        cls.steam_analyzer = ReviewAnalyzer(source='steam')
        cls.trustpilot_analyzer = ReviewAnalyzer(source='trustpilot')
        cls.imdb_analyzer = ReviewAnalyzer(source='imdb')
    
    def test_individual_review_analysis(self):
        """Test sentiment analysis on individual reviews."""
        # Test Steam review
        steam_review = self.test_reviews[0]['review']
        result = self.steam_analyzer.analyze_sentiment(steam_review)
        
        self.assertIsInstance(result['label'], SentimentLabel)
        self.assertIsInstance(result['score'], float)
        self.assertIsInstance(result.get('compound_score', 0.0), float)
        self.assertGreaterEqual(result['score'], -1.0)
        self.assertLessEqual(result['score'], 1.0)
        
        # Test Trustpilot review
        trustpilot_review = "This is a great service, I'm very happy with it!"
        result = self.trustpilot_analyzer.analyze_sentiment(trustpilot_review)
        
        self.assertIsInstance(result['label'], SentimentLabel)
        self.assertIsInstance(result['score'], float)
        self.assertIsInstance(result.get('compound_score', 0.0), float)
    
    def test_batch_analysis(self):
        """Test batch processing of reviews."""
        # Create DataFrame from test reviews
        df = pd.DataFrame(self.test_reviews)
        
        # Run batch analysis
        result_df = self.steam_analyzer.analyze_reviews(
            df,
            text_column='review',
            source_column=None
        )
        
        # Verify results
        self.assertIn('sentiment', result_df.columns)
        self.assertIn('sentiment_polarity', result_df.columns)
        self.assertIn('sentiment_score', result_df.columns)
        self.assertIn('keywords', result_df.columns)
        self.assertIn('predicted_stars', result_df.columns)
        self.assertIn('primary_emotion', result_df.columns)
        
        # Check sentiment values
        self.assertTrue(all(isinstance(s, SentimentLabel) for s in result_df['sentiment']))
        self.assertTrue(all(-1.0 <= p <= 1.0 for p in result_df['sentiment_polarity']))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in result_df['sentiment_score']))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test empty text
        with self.assertRaises(ValueError):
            self.steam_analyzer.analyze_sentiment("")
        
        # Test None text
        with self.assertRaises(ValueError):
            self.steam_analyzer.analyze_sentiment(None)
        
        # Test invalid source
        result = self.steam_analyzer.analyze_sentiment("Test review", source="invalid_source")
        self.assertEqual(result['label'], SentimentLabel.UNKNOWN)
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result.get('compound_score', 0.0), 0.0)
    
    def test_mixed_sources(self):
        """Test analysis with mixed review sources."""
        # Create test data with mixed sources
        mixed_reviews = [
            {'review': 'Great game!', 'source': 'steam'},
            {'review': 'Excellent service!', 'source': 'trustpilot'},
            {'review': 'Amazing movie!', 'source': 'imdb'}
        ]
        df = pd.DataFrame(mixed_reviews)
        
        # Run analysis with source column
        result_df = self.steam_analyzer.analyze_reviews(
            df,
            text_column='review',
            source_column='source'
        )
        
        # Verify results
        self.assertEqual(len(result_df), len(mixed_reviews))
        self.assertTrue(all(isinstance(s, SentimentLabel) for s in result_df['sentiment']))
    
    def test_long_reviews(self):
        """Test handling of long reviews."""
        # Create a long review
        long_review = "This is a test review. " * 100  # Create a long review
        
        # Test individual analysis
        result = self.steam_analyzer.analyze_sentiment(long_review)
        self.assertIsInstance(result['label'], SentimentLabel)
        
        # Test batch analysis
        df = pd.DataFrame([{'review': long_review}])
        result_df = self.steam_analyzer.analyze_reviews(df, text_column='review')
        self.assertEqual(len(result_df), 1)
        self.assertIsInstance(result_df['sentiment'].iloc[0], SentimentLabel)

def main():
    """Run the tests and print results."""
    logger.info("Starting sentiment analysis integration tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSentimentIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed!")
        for failure in result.failures:
            logger.error(f"Failure: {failure[0]}\n{failure[1]}")
        for error in result.errors:
            logger.error(f"Error: {error[0]}\n{error[1]}")

if __name__ == '__main__':
    main() 