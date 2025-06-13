"""
Script to analyze sentiment of Steam reviews using the sentiment analysis pipeline.
"""

import logging
import pandas as pd
from pathlib import Path
from src.analysis.nlp_analysis import ReviewAnalyzer
from src.utils.file_utils import save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_reviews(file_path: str) -> pd.DataFrame:
    """Load reviews from a JSON file."""
    return pd.read_json(file_path)

def main():
    # File paths
    input_file = Path('data/processed_test_results/processed_570_steam_reviews.json')
    output_file = Path('data/results/analyzed_steam_reviews.json')
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load reviews
    logger.info(f"Loading reviews from {input_file}")
    df = load_reviews(input_file)
    logger.info(f"Loaded {len(df)} reviews")
    
    analyzer = ReviewAnalyzer(language='en', source='steam')
    logger.info("Starting NLP analysis...")
    analyzed_df = analyzer.analyze_reviews(df, text_column='processed_text')
    logger.info(f"Analyzed {len(analyzed_df)} reviews")
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    save_json(analyzed_df.to_dict(orient='records'), output_file)
    
    # Print some statistics
    if not analyzed_df.empty:
        logger.info("\nSentiment distribution:")
        logger.info(analyzed_df['sentiment'].value_counts())
        
        if 'rating' in analyzed_df.columns:
            logger.info("\nRating distribution:")
            logger.info(analyzed_df['rating'].value_counts())
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 