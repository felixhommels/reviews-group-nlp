import argparse
import logging
import pandas as pd
from pathlib import Path
import json
import os
from src.analysis.nlp_analysis import ReviewAnalyzer
from src.utils.file_utils import save_json
from langdetect import detect, LangDetectException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file(input_file: Path, output_dir: Path, language: str = 'en') -> None:
    """Process a single review file and save results.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save results
        language: Default language code
    """
    logger.info(f"Processing {input_file}")
    
    # Extract source from filename
    source = input_file.stem.replace("processed_", "").replace("_reviews", "")
    
    # Load reviews
    with open(input_file) as f:
        content = f.read().strip()
        if content.startswith("["):
            reviews = json.loads(content)
        else:
            reviews = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    # Convert to DataFrame
    df = pd.DataFrame(reviews)
    
    # Filter out empty or invalid processed_text
    df = df[df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
    logger.info(f"Filtered to {len(df)} non-empty reviews for analysis")
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer(language=language, source=source)
    
    # Perform analysis
    logger.info("Starting NLP analysis...")
    analyzed_df = analyzer.analyze_reviews(df, text_column="processed_text")
    logger.info(f"Analyzed {len(analyzed_df)} reviews")
    
    # Save results
    output_file = output_dir / f"analyzed_{source}_reviews.json"
    logger.info(f"Saving results to {output_file}")
    
    # Convert any Timestamp or non-serializable objects to string
    analyzed_records = analyzed_df.copy()
    for col in analyzed_records.columns:
        if analyzed_records[col].dtype.name.startswith('datetime') or analyzed_records[col].dtype.name == 'Timestamp':
            analyzed_records[col] = analyzed_records[col].astype(str)
    
    save_json(analyzed_records.to_dict(orient='records'), output_file)
    
    # Print statistics
    if not analyzed_df.empty:
        logger.info("\nSentiment distribution:")
        logger.info(analyzed_df['sentiment'].value_counts())
        if 'rating' in analyzed_df.columns:
            logger.info("\nRating distribution:")
            logger.info(analyzed_df['rating'].value_counts())

def main():
    parser = argparse.ArgumentParser(description="Batch analyze reviews from various sources.")
    parser.add_argument("--input_dir", default="data/processed_test_results", help="Directory containing processed review files")
    parser.add_argument("--output_dir", default="data/results", help="Directory to save analysis results")
    parser.add_argument("--language", default="en", help="Default language code")
    args = parser.parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files in input directory
    for file in input_dir.glob("*.json"):
        if file.name.startswith("processed_"):
            try:
                process_file(file, output_dir, args.language)
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
    
    logger.info("Batch analysis complete!")

if __name__ == "__main__":
    main() 