# --- Imports ---
import os
import logging
import pandas as pd
from src.scraping.scraper import Scraper
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
from src.utils.file_utils import load_config, save_json

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the full scraping and preprocessing pipeline.
    """
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
    main()
