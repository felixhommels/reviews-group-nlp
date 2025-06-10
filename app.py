# --- Imports ---
import os
import logging
import pandas as pd
import glob
import json
from pathlib import Path
from typing import Optional, Dict, Any

from src.scraping.scraper import Scraper
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
from src.analysis.nlp_analysis import ReviewAnalyzer
from src.utils.file_utils import load_config, save_json

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Custom exception for pipeline-specific errors."""
    pass

def setup_directories() -> Dict[str, Path]:
    """Create and return required directory paths."""
    dirs = {
        'raw': Path('data/raw'),
        'processed': Path('data/processed'),
        'results': Path('data/results')
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    return dirs

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that DataFrame has required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise PipelineError(f"Missing required columns: {missing}")
    return True

def run_pipeline_test():
    """
    Finds all raw review files, runs them through the preprocessing pipeline,
    and saves the results in a dedicated test folder.
    """
    logging.info("--- Starting Preprocessing Pipeline Test ---")

    # --- Configuration ---
    raw_data_dir = "data/raw"
    output_dir = "data/processed_test_results"
    
    # Find all JSON files in the raw data directory
    raw_files = glob.glob(os.path.join(raw_data_dir, '*.json'))

    if not raw_files:
        logging.warning(f"No raw data files found in '{raw_data_dir}'. Nothing to test.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: '{output_dir}'")

    # --- Processing Loop ---
    for file_path in raw_files:
        filename = os.path.basename(file_path)
        logging.info(f"--- Processing file: {filename} ---")

        try:
            # Load the raw reviews from the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_reviews = json.load(f)
            
            if not raw_reviews:
                logging.warning(f"File '{filename}' is empty. Skipping.")
                continue

            # Convert to DataFrame
            reviews_df = pd.DataFrame(raw_reviews)
            
            # The pipeline expects a 'review' column. Rename 'text' if it exists.
            if 'text' in reviews_df.columns:
                reviews_df.rename(columns={'text': 'review'}, inplace=True)
            elif 'review' not in reviews_df.columns:
                logging.error(f"Could not find 'text' or 'review' column in '{filename}'. Skipping.")
                continue

            # Run the preprocessing pipeline
            logging.info(f"Running preprocessing for '{filename}'...")
            processed_df = preprocess_pipeline(reviews_df, text_column='review')

            # Convert DataFrame back to list of dictionaries for saving
            processed_reviews = processed_df.to_dict(orient='records')

            # Save the processed data
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            save_json(processed_reviews, output_path)
            logging.info(f"Successfully processed and saved results to '{output_path}'")

        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from '{filename}'. It might be corrupted. Skipping.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing '{filename}': {e}", exc_info=True)

    logging.info("--- Preprocessing Pipeline Test Finished ---")


def main(test_mode=False):
    """
    Main function to run the full scraping and preprocessing pipeline.
    """
    if test_mode:
        run_pipeline_test()
        return

    logging.info("--- Starting the Review Analysis Pipeline ---")

    # --- Load Configuration ---
    logging.info("Loading configuration from 'config.json'...")
    config = load_config('config.json')
    if not config:
        logging.error("Configuration could not be loaded. Exiting.")
        return

    # --- Scraping ---
    logging.info("Initializing the scraper...")
    scraper = Scraper(config)
    
    logging.info(f"Starting to scrape reviews for: {config.get('app_name', 'Unknown App')}")
    raw_reviews = scraper.scrape()
    
    if not raw_reviews:
        logging.warning("No reviews were scraped. The pipeline will stop here.")
        return
        
    # --- Preprocessing ---
    logging.info("Starting the preprocessing stage...")
    
    # Convert list of review dictionaries to a pandas DataFrame
    reviews_df = pd.DataFrame(raw_reviews)
    
    # The new pipeline expects a 'review' column. Let's ensure the 'text' field is named 'review'.
    if 'text' in reviews_df.columns:
        reviews_df.rename(columns={'text': 'review'}, inplace=True)
    elif 'review' not in reviews_df.columns:
        logging.error("Could not find a 'text' or 'review' column in the scraped data for preprocessing.")
        return

    # Run the new, spaCy-based preprocessing pipeline
    processed_df = preprocess_pipeline(reviews_df, text_column='review')

    # Convert the processed DataFrame back to a list of dictionaries for saving
    # This ensures the output format is consistent with the project's requirements
    processed_reviews = processed_df.to_dict(orient='records')

    # --- Saving Results ---
    output_filename = config.get('output_filename_processed', 'data/processed/processed_reviews.json')
    logging.info(f"Saving {len(processed_reviews)} processed reviews to '{output_filename}'...")
    save_json(processed_reviews, output_filename)

    logging.info("--- Review Analysis Pipeline Finished Successfully ---")

if __name__ == '__main__':
    # To run the test, set test_mode to True
    main(test_mode=True)
