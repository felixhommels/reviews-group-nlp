"""
Script to analyze sentiment of Play Store reviews using the sentiment analysis pipeline.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.analysis.sentiment_analysis import analyze_sentiment
from src.utils.file_utils import save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_reviews(file_path: str) -> List[Dict[str, Any]]:
    """Load reviews from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_reviews(reviews: List[Dict[str, Any]], source: str = 'playstore') -> List[Dict[str, Any]]:
    """Analyze sentiment of reviews using the new transformer-only pipeline."""
    analyzed_reviews = []
    for review in reviews:
        try:
            text = review.get('review', review.get('text', ''))
            if not text:
                logger.warning(f"Skipping review with no text: {review.get('id', 'unknown')}")
                continue
                
            sentiment_result = analyze_sentiment(text, source)
            if not sentiment_result:
                logger.warning(f"No sentiment result for review: {review.get('id', 'unknown')}")
                continue
                
            analyzed_review = review.copy()
            analyzed_review.update({
                'sentiment': sentiment_result.get('sentiment_label', 'neutral'),
                'sentiment_score': sentiment_result.get('sentiment_score', 0.0),
                'sentiment_confidence': sentiment_result.get('confidence', 0.0)
            })
            analyzed_reviews.append(analyzed_review)
        except Exception as e:
            logger.error(f"Error analyzing review: {e}")
            continue
    return analyzed_reviews

def main():
    # File paths
    input_file = Path('data/processed_test_results/processed_com.whatsapp_reviews.json')
    output_file = Path('data/results/analyzed_whatsapp_reviews.json')
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load reviews
    logger.info(f"Loading reviews from {input_file}")
    reviews = load_reviews(input_file)
    logger.info(f"Loaded {len(reviews)} reviews")
    
    # Analyze reviews
    logger.info("Starting sentiment analysis...")
    analyzed_reviews = analyze_reviews(reviews)
    logger.info(f"Analyzed {len(analyzed_reviews)} reviews")
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    save_json(analyzed_reviews, output_file)
    
    # Print some statistics
    if analyzed_reviews:
        sentiments = pd.DataFrame(analyzed_reviews)['sentiment'].value_counts()
        logger.info("\nSentiment distribution:")
        logger.info(sentiments)
        
        if 'rating' in analyzed_reviews[0]:
            ratings = pd.DataFrame(analyzed_reviews)['rating'].value_counts()
            logger.info("\nRating distribution:")
            logger.info(ratings)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 