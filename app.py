# --- Imports ---
import os
import logging
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any

from src.scraper.url_scraper import scrape_google_playstore, scrape_imbd, scrape_steam, scrape_trustpilot
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
from src.analysis.nlp_analysis import run_full_nlp_pipeline
from src.utils.file_utils import save_json
from src.utils.dependencies import dependency_manager
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.analysis.keyword_extraction import KeywordExtractor
from src.analysis.emotion_analysis import EnglishEmotionAnalyzerHartmann, SpanishEmotionAnalyzerRobertuito
from src.analysis.star_rating_predictor import StarRatingPredictor

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

def run_nlp_analysis_on_all_processed():
    input_dir = "data/processed_test_results"
    output_dir = "data/analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sentiment_analyzer = SentimentAnalyzer()
    keyword_extractor = KeywordExtractor()
    english_emotion_analyzer = EnglishEmotionAnalyzerHartmann()
    spanish_emotion_analyzer = SpanishEmotionAnalyzerRobertuito()
    rating_predictor = StarRatingPredictor()

    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        source = Path(file).stem.replace("processed_", "").replace("_reviews", "")
        input_path = os.path.join(input_dir, file)
        with open(input_path) as f:
            content = f.read().strip()
            if content.startswith("["):
                reviews = json.loads(content)
            else:
                reviews = [json.loads(line) for line in content.splitlines() if line.strip()]

        enriched_reviews = []
        for review in reviews:
            if not review.get("processed_text", "").strip():
                continue
            enriched = run_full_nlp_pipeline(
                review,
                sentiment_analyzer,
                keyword_extractor,
                english_emotion_analyzer,
                spanish_emotion_analyzer,
                rating_predictor
            )
            enriched_reviews.append(enriched)

        output_path = os.path.join(output_dir, f"review_analysis_{source}.json")
        with open(output_path, "w") as f:
            for r in enriched_reviews:
                f.write(json.dumps(r) + "\n")

    print("✅ All sources processed. Output saved to data/analysis/")

def create_visualizations(df, output_dir='data/visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot')
    sns.set_theme()

    # 1. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_label'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Distribution of Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()

    # 2. Sentiment Score Distribution
    if 'sentiment_score' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='sentiment_score', bins=30)
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_score_distribution.png'))
        plt.close()

    # 3. Star Ratings Distribution
    if 'predicted_rating_raw' in df.columns:
        plt.figure(figsize=(10, 6))
        star_counts = df['predicted_rating_raw'].value_counts().sort_index()
        sns.barplot(x=star_counts.index, y=star_counts.values)
        plt.title('Distribution of Predicted Star Ratings')
        plt.xlabel('Stars')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'star_ratings_distribution.png'))
        plt.close()

    # 4. Primary Emotions
    if 'top_emotion' in df.columns:
        plt.figure(figsize=(12, 6))
        emotion_counts = df['top_emotion'].value_counts()
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Distribution of Primary Emotions')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
        plt.close()

    # 5. Correlation between Sentiment Score and Star Ratings
    if 'sentiment_score' in df.columns and 'predicted_rating_raw' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sentiment_score', y='predicted_rating_raw')
        plt.title('Correlation between Sentiment Score and Star Ratings')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Predicted Stars')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_stars_correlation.png'))
        plt.close()

    # 6. Word Cloud of Keywords
    if 'keywords' in df.columns:
        from wordcloud import WordCloud
        all_keywords = ' '.join(df['keywords'].astype(str).str.join(' '))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Keywords')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keyword_wordcloud.png'))
        plt.close()

def main():
    logging.info("--- Starting the Review Analysis Pipeline ---")

    SOURCES = [
        {
            "source": "trustpilot",
            "url": "https://es.trustpilot.com/review/www.bancosantander.es",
            "topic": "bancosantander",
            "max_pages": 2
        },
        {
            "source": "imdb",
            "url": "https://www.imdb.com/title/tt0892769/reviews/?ref_=tt_ov_ururv",
            "topic": "howToTrainYourDragon",
            "max_pages": 2
        },
        {
            "source": "playstore",
            "app_id": "com.whatsapp",
            "max_reviews": 300
        },
        {
            "source": "steam",
            "app_id": "570",
            "max_reviews": 300
        }
    ]

    for cfg in SOURCES:
        source = cfg["source"]
        logging.info(f"Scraping source: {source}")

        # --- Scraping ---
        if source == "trustpilot":
            raw_reviews = scrape_trustpilot(
                url=cfg["url"],
                topic=cfg["topic"],
                max_pages=cfg.get("max_pages", 10)
            )
            output_file = f"data/raw/{cfg['topic']}_reviews.json"
        elif source == "imdb":
            raw_reviews = scrape_imbd(
                url=cfg["url"],
                topic=cfg["topic"],
                max_pages=cfg.get("max_pages", 10)
            )
            output_file = f"data/raw/{cfg['topic']}_reviews.json"
        elif source == "playstore":
            raw_reviews = scrape_google_playstore(
                app_id=cfg["app_id"],
                max_reviews=cfg.get("max_reviews", 100)
            )
            output_file = f"data/raw/{cfg['app_id']}_reviews.json"
        elif source == "steam":
            raw_reviews = scrape_steam(
                app_id=cfg["app_id"],
                max_reviews=cfg.get("max_reviews", 100)
            )
            output_file = f"data/raw/{cfg['app_id']}_steam_reviews.json"
        else:
            logging.error(f"Unknown source: {source}")
            continue

        # --- Preprocessing ---
        if not raw_reviews:
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    raw_reviews = json.load(f)
            else:
                logging.warning(f"No reviews found for {source}. Skipping.")
                continue

        reviews_df = pd.DataFrame(raw_reviews)
        if 'text' in reviews_df.columns:
            reviews_df.rename(columns={'text': 'review'}, inplace=True)
        elif 'review' not in reviews_df.columns:
            logging.error(f"Could not find a 'text' or 'review' column for {source}. Skipping.")
            continue

        processed_df = preprocess_pipeline(reviews_df, text_column='review')
        processed_reviews = processed_df.to_dict(orient='records')

        # Save processed reviews to processed_test_results
        processed_output = f"data/processed_test_results/processed_{cfg.get('app_id', cfg.get('topic', source))}_reviews.json"
        with open(processed_output, "w") as f:
            json.dump(processed_reviews, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved processed reviews for {source} to {processed_output}")

    # --- NLP Analysis on ALL processed files ---
    run_nlp_analysis_on_all_processed()

    # --- Visualization ---
    # Aggregate all enriched reviews
    analysis_dir = "data/analysis"
    all_reviews = []
    for file in os.listdir(analysis_dir):
        if file.endswith(".json"):
            with open(os.path.join(analysis_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    all_reviews.append(json.loads(line))
    df = pd.DataFrame(all_reviews)
    print(f"Loaded {len(df)} enriched reviews for visualization.")
    create_visualizations(df)
    print("Visualizations have been created in the 'data/visualizations' directory.")

if __name__ == '__main__':
    main()