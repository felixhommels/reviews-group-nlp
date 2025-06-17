"""
Emotion Analysis Module.

This module provides emotion analysis capabilities using:
- j-hartmann/emotion-english-distilroberta-base for English text
- pysentimiento/robertuito-emotion-analysis for Spanish text
"""

import logging
from typing import Dict
from enum import Enum
from transformers import pipeline, AutoTokenizer

from src.utils.dependencies import dependency_manager, DependencyError
from src.config.manager import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)

class EmotionLabel(str, Enum):
    """Enum for emotion classification labels."""
    JOY = 'joy'
    SURPRISE = 'surprise'
    ANGER = 'anger'
    SADNESS = 'sadness'
    NEUTRAL = 'neutral'
    FEAR = 'fear'
    DISGUST = 'disgust'
    TRUST = 'trust'
    ANTICIPATION = 'anticipation'

class EnglishEmotionAnalyzerHartmann:
    """Emotion analyzer using j-hartmann/emotion-english-distilroberta-base (English only) with text chunking for long texts."""
    
    def __init__(self):
        self.pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.max_tokens = 512  # RoBERTa models typically support up to 512 tokens
        self.stride = 256  # overlap for context

    def analyze_emotion(self, text: str) -> dict:
        """Analyze emotion in text, handling long texts by chunking."""
        if not text or not text.strip():
            return {}
        
        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= self.max_tokens:
                # Short enough, just run the model
                results = self.pipeline(text)
                return {r['label'].lower(): r['score'] for r in results[0]}
            
            # Text is too long, need to chunk it
            logger.debug(f"Text too long ({len(tokens)} tokens), chunking...")
            
            chunk_scores = []
            for i in range(0, len(tokens), self.stride):
                chunk = tokens[i:i+self.max_tokens]
                if len(chunk) < 10:  # skip very short trailing chunks
                    continue
                    
                chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                try:
                    results = self.pipeline(chunk_text)
                    chunk_scores.append({r['label'].lower(): r['score'] for r in results[0]})
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            # Aggregate: average the scores for each label
            if not chunk_scores:
                logger.warning("No chunks processed successfully")
                return {}
                
            all_labels = set().union(*[d.keys() for d in chunk_scores])
            avg_scores = {
                label: sum(d.get(label, 0) for d in chunk_scores) / len(chunk_scores) 
                for label in all_labels
            }
            return avg_scores
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {}

class SpanishEmotionAnalyzerRobertuito:
    """Emotion analyzer using pysentimiento/robertuito-emotion-analysis (Spanish only), with chunking for long texts."""
    
    def __init__(self):
        self.pipeline = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-emotion-analysis",
            top_k=None
        )
        self.tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-emotion-analysis")
        self.max_tokens = 128  # This model has a smaller limit
        self.stride = 64  # overlap for context

    def analyze_emotion(self, text: str) -> dict:
        """Analyze emotion in text, handling long texts by chunking."""
        if not text or not text.strip():
            return {}
        
        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= self.max_tokens:
                # Short enough, just run the model
                results = self.pipeline(text)
                return {r['label'].lower(): r['score'] for r in results[0]}
            
            # Text is too long, need to chunk it
            logger.debug(f"Text too long ({len(tokens)} tokens), chunking...")
            
            chunk_scores = []
            for i in range(0, len(tokens), self.stride):
                chunk = tokens[i:i+self.max_tokens]
                if len(chunk) < 10:  # skip very short trailing chunks
                    continue
                    
                chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                try:
                    results = self.pipeline(chunk_text)
                    chunk_scores.append({r['label'].lower(): r['score'] for r in results[0]})
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            # Aggregate: average the scores for each label
            if not chunk_scores:
                logger.warning("No chunks processed successfully")
                return {}
                
            all_labels = set().union(*[d.keys() for d in chunk_scores])
            avg_scores = {
                label: sum(d.get(label, 0) for d in chunk_scores) / len(chunk_scores) 
                for label in all_labels
            }
            return avg_scores
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {}