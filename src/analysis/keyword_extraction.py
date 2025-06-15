"""
Keyword Extraction Module.

This module provides keyword extraction capabilities using:
- KeyBERT + LaBSE for high-quality, multilingual, semantic keyword extraction
"""

import logging
from typing import List
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.dependencies import dependency_manager
from src.config.manager import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Class for extracting keywords from text using KeyBERT + MiniLM."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the keyword extractor.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self.config = ConfigManager.get_keybert_config(language)
        self.model = SentenceTransformer(self.config['model_name'])
        self.kw_model = KeyBERT(self.model)
    
    def extract_keywords(self, text: str, language: str = "en", max_keywords: int = None) -> List[str]:
        """Extract keywords from text using KeyBERT + MiniLM."""
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided for keyword extraction")
            return []
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.config['keyphrase_ngram_range'],
                stop_words=self.config['stop_words'],
                top_n=max_keywords or self.config['top_n'],
            )
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def extract_keywords_batch(self, texts: list, language: str = "en", max_keywords: int = None, batch_size: int = 16) -> list:
        """Batch keyword extraction using per-text extraction for compatibility."""
        results = []
        for text in tqdm(texts, desc="Keyword extraction"):
            results.append(self.extract_keywords(text, language=language, max_keywords=max_keywords))
        return results
