"""
Emotion Analysis Module.

This module provides emotion analysis capabilities using:
- j-hartmann/emotion-english-distilroberta-base for English text
- pysentimiento/robertuito-emotion-analysis for Spanish text
"""

import logging
from enum import Enum
from transformers import pipeline

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
    """Emotion analyzer using j-hartmann/emotion-english-distilroberta-base (English only)."""
    def __init__(self):
        self.pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

    def analyze_emotion(self, text: str) -> dict:
        results = self.pipeline(text)
        # Convert results to a dict: {label: score}
        return {r['label'].lower(): r['score'] for r in results[0]}

class SpanishEmotionAnalyzerRobertuito:
    """Emotion analyzer using pysentimiento/robertuito-emotion-analysis (Spanish only), with chunking for long texts."""
    def __init__(self):
        from transformers import AutoTokenizer
        self.pipeline = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-emotion-analysis",
            top_k=None
        )
        self.tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-emotion-analysis")
        self.max_tokens = 128
        self.stride = 64  # overlap for context

    def analyze_emotion(self, text: str) -> dict:
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= self.max_tokens:
            # Short enough, just run the model
            results = self.pipeline(text)
            return {r['label'].lower(): r['score'] for r in results[0]}
        # Otherwise, chunk the text
        chunk_scores = []
        for i in range(0, len(tokens), self.stride):
            chunk = tokens[i:i+self.max_tokens]
            if len(chunk) < 10:  # skip very short trailing chunks
                continue
            chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
            results = self.pipeline(chunk_text)
            chunk_scores.append({r['label'].lower(): r['score'] for r in results[0]})
        # Aggregate: average the scores for each label
        if not chunk_scores:
            return {}
        all_labels = set().union(*[d.keys() for d in chunk_scores])
        avg_scores = {label: sum(d.get(label, 0) for d in chunk_scores) / len(chunk_scores) for label in all_labels}
        return avg_scores
