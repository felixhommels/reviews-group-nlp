# LEGACY/REFERENCE: Not used in production. See [nlp_analysis.py]for the current version.


"""
NLP Analysis Module
- Sentiment analysis (positive/neutral/negative)
- Topic/keyword extraction
- Optional: Star rating prediction or emotion classification

Integrates with preprocessed review data (expects DataFrame or list of dicts).
"""

import logging
import pandas as pd
from typing import List, Union

# Sentiment analysis: VADER (for English) or fallback to TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_available = True
except ImportError:
    vader_available = False
try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    textblob_available = False

# Keyword extraction: Use sklearn's TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: Emotion classification (using NRCLex if available, else rule-based fallback)
try:
    from nrclex import NRCLex
    nrclex_available = True
except ImportError:
    nrclex_available = False

# --- Analysis functions ---

def predict_star_rating(text: str) -> int:
    """Dummy star rating prediction based on sentiment polarity."""
    if vader_available:
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)['compound']
    else:
        score = TextBlob(text).sentiment.polarity
    if score > 0.5:
        return 5
    elif score > 0.1:
        return 4
    elif score > -0.1:
        return 3
    elif score > -0.5:
        return 2
    else:
        return 1

def analyze_sentiment(text: str) -> str:
    """Classify sentiment as positive, neutral, or negative."""
    if vader_available:
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)['compound']
    else:
        score = TextBlob(text).sentiment.polarity
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'

def extract_keywords(texts: List[str], top_k: int = 10) -> List[List[str]]:
    """Extract top_k keywords for each text using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for row in tfidf:
        indices = row.toarray().flatten().argsort()[-top_k:][::-1]
        keywords.append([feature_names[i] for i in indices if row[0, i] > 0])
    return keywords

def analyze_emotion(text: str) -> str:
    """Classify primary emotion in text. Uses NRCLex if available, else rule-based on sentiment."""
    if nrclex_available:
        emotions = NRCLex(text).top_emotions
        if emotions:
            # Return the most likely emotion
            return emotions[0][0]
        else:
            return 'neutral'
    else:
        # Fallback: map sentiment to emotion
        if vader_available:
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(text)['compound']
        else:
            score = TextBlob(text).sentiment.polarity
        if score > 0.5:
            return 'joy'
        elif score > 0.2:
            return 'surprise'
        elif score < -0.5:
            return 'anger'
        elif score < -0.2:
            return 'sadness'
        else:
            return 'neutral'

def analyze_reviews(df: pd.DataFrame, text_column: str = 'processed_text') -> pd.DataFrame:
    """Run sentiment, keyword extraction, star rating prediction, and emotion classification on reviews DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    texts = df[text_column].astype(str).fillna('').tolist()
    logging.info("Running sentiment analysis...")
    df['sentiment'] = [analyze_sentiment(t) for t in texts]
    logging.info("Extracting keywords...")
    df['keywords'] = extract_keywords(texts)
    logging.info("Predicting star ratings...")
    df['predicted_stars'] = [predict_star_rating(t) for t in texts]
    logging.info("Classifying emotions...")
    df['emotion'] = [analyze_emotion(t) for t in texts]
    return df

# --- Example usage (commented) ---
# import pandas as pd
# df = pd.read_json('data/raw/bancosantander_reviews.json')
# from src.preprocessing.spacy_preprocessor import preprocess_pipeline
# df = preprocess_pipeline(df, text_column='text')
# from src.analysis.nlp_analysis import analyze_reviews
# df = analyze_reviews(df, text_column='processed_text')
# print(df[['text', 'sentiment', 'keywords', 'predicted_stars', 'emotion']].head())

if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing.spacy_preprocessor import preprocess_pipeline

    # 1. Load real scraped data (from scraper output JSON file)
    raw_df = pd.read_json('data/raw/bancosantander_reviews.json')

    # 2. Preprocess the data (cleaning, language detection, lemmatization, etc.)
    processed_df = preprocess_pipeline(raw_df, text_column='text')

    # 3. Run NLP analysis (sentiment, keywords, star rating, emotion)
    result_df = analyze_reviews(processed_df, text_column='processed_text')

    # 4. Show results with full text columns
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 0)
    print(result_df[['text', 'processed_text', 'sentiment', 'keywords', 'predicted_stars', 'emotion']].head(10))
