"""
Download required NLTK resources for text processing.
"""

import nltk
import spacy
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- spaCy Language Model Downloader ---
# This script downloads the spaCy models needed for the multi-language preprocessor.
# You only need to run this script once.
#
# To run this script, execute the following command from the project's root directory:
# python src/preprocessing/download_models.py
#
# You can add or remove language models from the `MODELS_TO_DOWNLOAD` list
# based on the languages you expect to find in your reviews.
# Find more models here: https://spacy.io/usage/models

MODELS_TO_DOWNLOAD = [
    "en_core_web_sm",  # English
    "es_core_news_sm",  # Spanish
    "fr_core_news_sm",  # French
    "de_core_news_sm",  # German
    "pt_core_news_sm",  # Portuguese
    "it_core_news_sm",  # Italian
    "nl_core_news_sm",  # Dutch
    "ru_core_news_sm",  # Russian
    "zh_core_web_sm",  # Chinese
]

def download_models():
    """Iterates through the list of models and downloads them using spacy's CLI."""
    logger.info("Starting download of spaCy language models...")
    for model in MODELS_TO_DOWNLOAD:
        try:
            logger.info(f"Downloading model: {model}...")
            spacy.cli.download(model)
            logger.info(f"Successfully downloaded {model}.")
        except SystemExit as e:
            # spacy.cli.download can call sys.exit, which we need to catch
            if e.code == 0:
                 logger.info(f"Model {model} is already installed.")
            else:
                 logger.error(f"Failed to download {model}. Error code: {e.code}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading {model}: {e}")
    logger.info("Finished downloading all specified spaCy models.")

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = [
        'punkt',           # For sentence tokenization
        'stopwords',       # For stopword removal
        'wordnet',        # For lemmatization
        'averaged_perceptron_tagger'  # Required by wordnet
    ]
    
    for resource in resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    logger.info("Starting NLTK resource download...")
    download_nltk_resources()
    logger.info("Completed NLTK resource download")
    logger.info("Starting download of spaCy language models...")
    download_models() 