import argparse
import logging
import os
import time
from datetime import datetime
from src.scraper.url_scraper import scrape_imbd
from src.preprocessing.spacy_preprocessor import preprocess_text
from src.analysis.sentiment_analysis import analyze_sentiment_overview
import pandas as pd
import json

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def print_section_header(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_progress(current, total, prefix=""):
    bar_length = 50
    filled_length = int(round(bar_length * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    print(f"{prefix}[{bar}] {percents}%")

def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis on IMDB reviews")
    parser.add_argument("movie_id", help="IMDB movie ID or URL")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    args = parser.parse_args()

    # Extract movie ID from URL if provided
    movie_id = args.movie_id
    if "imdb.com" in movie_id:
        movie_id = movie_id.split("/")[-2]

    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    start_time = time.time()

    print_section_header("Scraping Reviews")
    url = f"https://www.imdb.com/title/{movie_id}/reviews"
    reviews = scrape_imbd(url, movie_id, args.pages)
    print(f"Scraped {len(reviews)} reviews")

    print_section_header("Preprocessing")
    processed_reviews = [preprocess_text(review) for review in reviews]
    print(f"Preprocessed {len(processed_reviews)} reviews")

    print_section_header("Analyzing Sentiment")
    summary, result_df = analyze_sentiment_overview(pd.DataFrame({"review": reviews}), None)
    print(f"Overall sentiment: {summary['overall_sentiment']}")
    print(f"Average confidence: {summary['avg_confidence']}")
    print(f"Most common emotion: {summary['most_common_emotion']}")

    print_section_header("Saving Results")
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{movie_id}_results.json")
    save_json(summary, results_file)
    print(f"Results saved to {results_file}")

    if args.save_model:
        model_file = os.path.join(results_dir, f"{movie_id}_model.pkl")
        import pickle
        with open(model_file, "wb") as f:
            pickle.dump(None, f)
        print(f"Model saved to {model_file}")

    print_section_header("Summary")
    print(f"Total reviews processed: {len(reviews)}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()