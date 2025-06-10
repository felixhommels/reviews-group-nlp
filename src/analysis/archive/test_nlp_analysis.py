# LEGACY/REFERENCE: Not used in production. See [src/tests/unit/test_nlp_analysis.py]for the current version.


import sys
import os
print(f"[DEBUG] CWD: {os.getcwd()}")
print(f"[DEBUG] Script location: {__file__}")
raw_path = "data/raw/bancosantander_reviews.json"  # Change to your file
print(f"[DEBUG] Looking for file: {os.path.abspath(raw_path)}")
try:
    import pandas as pd
    from src.preprocessing.spacy_preprocessor import preprocess_pipeline
    from src.analysis.nlp_analysis import analyze_reviews

    # Load reviews (update the filename as needed)
    raw_df = pd.read_json(raw_path)

    # Preprocess the reviews (update text_column if needed)
    preprocessed_df = preprocess_pipeline(raw_df, text_column='text')

    # Analyze the reviews
    analyzed_df = analyze_reviews(preprocessed_df, text_column='processed_text')

    # Show the first few results
    print(analyzed_df[['text', 'sentiment', 'keywords', 'predicted_stars', 'emotion']].head())
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()