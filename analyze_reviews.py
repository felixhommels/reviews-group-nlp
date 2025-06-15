import argparse
import logging
import pandas as pd
from pathlib import Path
from src.analysis.nlp_analysis import ReviewAnalyzer
from src.utils.file_utils import save_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Analyze reviews from various sources.")
    parser.add_argument("--source", required=True, choices=["trustpilot", "imdb", "playstore", "steam"], help="Review source: trustpilot, imdb, playstore, steam")
    args = parser.parse_args()

    # File paths
    input_file = Path(f'data/processed_test_results/processed_{"com.whatsapp" if args.source=="playstore" else ("570_steam" if args.source=="steam" else ("bancosantander" if args.source=="trustpilot" else "howToTrainYourDragon"))}_reviews.json')
    output_file = Path(f'data/results/analyzed_{args.source}_reviews.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load reviews
    logger.info(f"Loading reviews from {input_file}")
    df = pd.read_json(input_file)
    logger.info(f"Loaded {len(df)} reviews")

    # Filter out empty or invalid processed_text
    df = df[df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
    logger.info(f"Filtered to {len(df)} non-empty reviews for analysis")

    analyzer = ReviewAnalyzer(language="en", source=args.source)
    logger.info("Starting NLP analysis...")
    analyzed_df = analyzer.analyze_reviews(df, text_column="processed_text")
    logger.info(f"Analyzed {len(analyzed_df)} reviews")

    # Save results
    logger.info(f"Saving results to {output_file}")
    # Convert any Timestamp or non-serializable objects to string
    analyzed_records = analyzed_df.copy()
    for col in analyzed_records.columns:
        if analyzed_records[col].dtype.name.startswith('datetime') or analyzed_records[col].dtype.name == 'Timestamp':
            analyzed_records[col] = analyzed_records[col].astype(str)
    save_json(analyzed_records.to_dict(orient='records'), output_file)

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