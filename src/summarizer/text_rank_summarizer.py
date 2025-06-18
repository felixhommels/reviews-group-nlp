"""
Simple TextRank Summarizer.
Uses sentence similarity and PageRank to extract key sentences from text.
"""

import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging
import re
from src.preprocessing.spacy_preprocessor import detect_language, preprocess_text, load_spacy_models

# Configure logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Only show the message for cleaner output
)
logger = logging.getLogger(__name__)

class TextRankSummarizer:
    def __init__(self):
        """Initialize the summarizer with English stopwords and load spacy models."""
        # Just store the language for TfidfVectorizer
        self.stop_words = 'english'
        # Load spacy models for preprocessing
        load_spacy_models()
        
    def clean_for_tokenize(self, text: str) -> str:
        """Clean text while preserving sentence structure."""
        if not isinstance(text, str):
            return ""
            
        # Fix spacing around periods to help with sentence detection
        text = re.sub(r'\s*\.\s*', '. ', text)
        # Fix multiple periods
        text = re.sub(r'\.+', '.', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Fix common abbreviations
        text = text.replace('Mr.', 'Mr')
        text = text.replace('Mrs.', 'Mrs')
        text = text.replace('Dr.', 'Dr')
        text = text.replace('Ph.D.', 'PhD')
        text = text.replace('i.e.', 'ie')
        text = text.replace('e.g.', 'eg')
        # Remove standalone periods
        text = re.sub(r'\s+\.\s+', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
        
    def remove_duplicates(self, sentences: list) -> list:
        """Remove duplicate sentences while preserving order."""
        seen = set()
        unique_sentences = []
        
        for sent in sentences:
            # Create a normalized version for comparison
            normalized = sent.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sent)
            
            
        return unique_sentences
        
    def get_sentences(self, text: str) -> list:
        """Extract and clean sentences from text."""
        # First clean the text while preserving sentence structure
        text = self.clean_for_tokenize(text)
        if not text:
            return []
            
        # Split into sentences
        raw_sentences = sent_tokenize(text)
        logger.info(f"\nFound {len(raw_sentences)} raw sentences")
        
        # Clean and filter sentences
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            # Only keep non-empty sentences with reasonable length that end with period
            if sent and len(sent) > 10 and sent.endswith('.'):
                sentences.append(sent)
                
        logger.info(f"Kept {len(sentences)} valid sentences after filtering")
        
        # Remove duplicates
        sentences = self.remove_duplicates(sentences)
        
        return sentences
        
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate a summary by extracting the most important sentences.
        
        Args:
            text: Text to summarize
            num_sentences: Number of sentences to include in summary
            
        Returns:
            Summary text
        """
        try:
            # Get clean sentences
            sentences = self.get_sentences(text)
            logger.info("\nProcessing text for summarization...")
            
            # Handle edge cases
            if not sentences:
                return text
            if len(sentences) <= num_sentences:
                return "\n----------------------------------------\n".join(f"• {sent}" for sent in sentences)
                
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words=self.stop_words)  # Now using 'english' string
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate similarity between sentences
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Create graph and calculate scores
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Get top sentences while maintaining original order
            ranked_sentences = [
                {"sentence": sent, "score": scores[idx], "index": idx}
                for idx, sent in enumerate(sentences)
            ]
            
            # Sort by score, get exactly num_sentences
            ranked_sentences.sort(key=lambda x: x["score"], reverse=True)
            selected = ranked_sentences[:num_sentences]
            
            
            # Sort by original position
            selected.sort(key=lambda x: x["index"])
            
            # Format output with bullet points and separators
            formatted_sentences = []
            for i, item in enumerate(selected):
                if i > 0:
                    formatted_sentences.append("\n----------------------------------------\n")
                formatted_sentences.append(f"• {item['sentence']}")
            
            return "".join(formatted_sentences)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text 