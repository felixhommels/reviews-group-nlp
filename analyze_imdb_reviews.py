"""
Script to analyze sentiment of IMDb reviews using the sentiment analysis pipeline.
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
    input_file = Path('data/processed_test_results/processed_howToTrainYourDragon_reviews.json')
    output_file = Path('data/results/analyzed_imdb_reviews.json')
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load reviews
    logger.info(f"Loading reviews from {input_file}")
    df = load_reviews(input_file)
    logger.info(f"Loaded {len(df)} reviews")
    
    analyzer = ReviewAnalyzer(language='en', source='imdb')
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

def map_imdb_rating_to_sentiment(rating):
    try:
        r = int(rating)
        if r <= 4:
            return 'negative'
        elif r <= 6:
            return 'neutral'
        else:
            return 'positive'
    except:
        return 'N/A'

df = pd.read_json('data/results/analyzed_imdb_reviews.json')
df['rating_sentiment'] = df['rating'].apply(map_imdb_rating_to_sentiment)
df['correct'] = df['sentiment'] == df['rating_sentiment']
accuracy = df[df['rating_sentiment'] != 'N/A']['correct'].mean()
print(f"Sentiment accuracy vs. rating: {accuracy:.2%}")

if __name__ == "__main__":
    main() 