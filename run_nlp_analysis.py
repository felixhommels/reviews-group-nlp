from src.scraper.url_scraper import scrape_trustpilot
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
from src.analysis.nlp_analysis import ReviewAnalyzer
import pandas as pd
import json
import os

# 1. Scrape reviews (Trustpilot example)
url = "https://es.trustpilot.com/review/www.bancosantander.es"  # Change to your target
topic = "bancosantander"
max_pages = 2

# Scrape and save reviews (if not already scraped)
json_file = os.path.join('data/raw', f"{topic}_reviews.json")
if not os.path.exists(json_file):
    scrape_trustpilot(url, topic, max_pages=max_pages)

# 2. Load scraped reviews from JSON
with open(json_file, "r", encoding='utf-8') as f:
    reviews = json.load(f)
df = pd.DataFrame(reviews)

# 3. Preprocess reviews
processed_df = preprocess_pipeline(df, text_column="text")  # Use the correct column name from the scraped data

# 4. Analyze reviews
analyzer = ReviewAnalyzer(language='es')  # Use 'es' for Spanish reviews
results = analyzer.analyze_reviews(processed_df, text_column="processed_text")

# 5. Show results
print("\nNLP Analysis Results:")
print(results[[
    'processed_text',
    'sentiment',
    'sentiment_polarity',
    'keywords',
    'predicted_stars',
    'primary_emotion'
]].head(10)) 