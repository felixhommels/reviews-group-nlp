# --- Imports ---
import re
import logging
import pandas as pd
from tqdm import tqdm
import contractions
from langdetect import detect, LangDetectException
import spacy

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- spaCy Model Management ---
# Pre-load spaCy models to avoid loading them repeatedly.
# This dictionary maps language codes to their corresponding spaCy model.
# The 'xx' model is a multi-language model used as a fallback.
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
    "ru": "ru_core_news_sm",
    "zh": "zh_core_web_sm",
    "xx": "xx_ent_wiki_sm"  # Multi-language fallback
}

NLP_MODELS = {}

def load_spacy_models():
    """Load all specified spaCy models into memory."""
    for lang, model_name in SPACY_MODELS.items():
        try:
            NLP_MODELS[lang] = spacy.load(model_name)
            logging.info(f"Successfully loaded spaCy model: {model_name} for language '{lang}'")
        except OSError:
            logging.warning(f"Could not find spaCy model {model_name}. Please run 'python src/preprocessing/download_models.py' to download it.")
    if not NLP_MODELS:
        logging.error("No spaCy models were loaded. Preprocessing will fail. Please ensure models are downloaded and accessible.")

# --- Core Preprocessing Functions ---

def detect_language(text):
    """
    Detects the language of a given text.
    Args:
        text (str): The text to analyze.
    Returns:
        str: The ISO 639-1 language code (e.g., 'en', 'es') or 'unknown' if detection fails.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 'unknown'
    try:
        # The first detection is the most likely one.
        return detect(text)
    except LangDetectException:
        # This exception occurs for texts that are too short or ambiguous.
        logging.warning(f"Could not detect language for text: '{text[:50]}...'. Defaulting to 'unknown'.")
        return 'unknown'

def clean_text(text):
    """
    Performs basic text cleaning.
    - Converts to lowercase.
    - Expands contractions (e.g., "don't" -> "do not").
    - Removes URLs, HTML tags, special characters, and numbers.
    Args:
        text (str): The text to clean.
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, lang='en'):
    """
    Processes a single text string using the appropriate spaCy model.
    - Performs tokenization, lemmatization, and stopword removal.
    Args:
        text (str): The text to preprocess.
        lang (str): The language of the text.
    Returns:
        str: A string of space-separated processed tokens.
    """
    # Use the specific model for the detected language, or fallback to multi-language model.
    nlp = NLP_MODELS.get(lang, NLP_MODELS.get("xx"))
    
    if not nlp:
        logging.error(f"No spaCy model available for language '{lang}' or fallback. Returning original text.")
        return text

    # Process the text with the loaded spaCy model
    doc = nlp(text)
    
    # Lemmatize, remove stopwords and punctuation
    tokens = [
        token.lemma_.strip() for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    
    return " ".join(tokens)

# --- Main Pipeline ---

def preprocess_pipeline(df, text_column='review', skipped_output_path=None):
    """
    Runs the full preprocessing pipeline on a DataFrame.
    1. Cleans the text (lowercase, remove URLs, etc.).
    2. Detects the language of each review.
    3. Applies language-specific tokenization, lemmatization, and stopword removal.
    4. Filters out reviews in unsupported languages and logs/saves them.
    Args:
        df (pd.DataFrame): The DataFrame containing the review data.
        text_column (str): The name of the column with the text to preprocess.
        skipped_output_path (str or None): If provided, saves skipped reviews to this CSV file.
    Returns:
        pd.DataFrame: The DataFrame with added 'cleaned_text', 'language', and 'processed_text' columns.
    """
    if text_column not in df.columns:
        logging.error(f"Column '{text_column}' not found in the DataFrame.")
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

    # Ensure the text column is of type string, replacing NaNs with empty strings
    df[text_column] = df[text_column].astype(str).fillna('')

    # Load models once before processing
    if not NLP_MODELS:
        load_spacy_models()
        if not NLP_MODELS:
             # If still no models, exit gracefully
            raise RuntimeError("Failed to load spaCy models. Cannot proceed with preprocessing.")

    # Apply initial cleaning
    tqdm.pandas(desc="Cleaning text")
    df['cleaned_text'] = df[text_column].progress_apply(clean_text)

    # Detect language
    tqdm.pandas(desc="Detecting languages")
    df['language'] = df['cleaned_text'].progress_apply(detect_language)

    # Filter to only supported languages
    supported_langs = set(SPACY_MODELS.keys())
    unsupported = df[~df['language'].isin(supported_langs)]
    if not unsupported.empty:
        logging.info(f"Skipping {len(unsupported)} reviews in unsupported languages: {unsupported['language'].unique()}")
        if skipped_output_path:
            unsupported.to_csv(skipped_output_path, index=False)
            logging.info(f"Saved skipped reviews to {skipped_output_path}")
    df = df[df['language'].isin(supported_langs)].copy()
    if df.empty:
        logging.warning("No reviews in supported languages. Exiting preprocessing.")
        return df

    # Group by language to process in batches
    processed_texts = []
    logging.info("Starting language-specific preprocessing...")
    with tqdm(total=len(df), desc="Preprocessing text") as pbar:
        for lang, group in df.groupby('language'):
            if lang == 'unknown':
                # For unknown languages, we can't do much more than basic cleaning
                results = group['cleaned_text'].tolist()
            else:
                # Apply the specific preprocessing for the detected language
                results = [preprocess_text(text, lang) for text in group['cleaned_text']]
            # Create a temporary series with the correct index to merge back
            processed_series = pd.Series(results, index=group.index)
            processed_texts.append(processed_series)
            pbar.update(len(group))

    # Concatenate all processed series and assign to the new column
    if processed_texts:
        df['processed_text'] = pd.concat(processed_texts).sort_index()
    else:
        df['processed_text'] = ''

    logging.info("Preprocessing pipeline completed successfully.")
    return df 