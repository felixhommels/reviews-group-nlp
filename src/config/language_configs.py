"""
Language-specific configurations for NLP analysis.
Add commentMore actions
This module contains language-specific settings such as:
- Emotion keywords
- Stop words
- Language-specific thresholds
"""

from typing import Dict, List

# Emotion keywords by language
EMOTION_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "english": {
        "joy": ["excellent", "perfect", "great", "love", "wonderful", "fantastic", "happy"],
        "anger": ["terrible", "horrible", "awful", "worst", "angry", "mad"],
        "sadness": ["sad", "disappointing", "unfortunately", "poor", "bad"],
        "fear": ["scary", "worried", "afraid", "concerned"],
        "disgust": ["disgusting", "gross", "nasty"],
        "surprise": ["surprising", "amazing", "wow", "incredible"],
        "trust": ["reliable", "trustworthy", "secure", "safe"],
        "anticipation": ["looking forward", "soon", "future", "expect"]
    },
    "spanish": {
        "joy": ["excelente", "perfecto", "genial", "encanta", "maravilloso", "fantástico", "feliz"],
        "anger": ["pésimo", "terrible", "horrible", "penosa", "fatal", "malísimo", "peor"],
        "sadness": ["triste", "decepcionante", "lamentable", "mal", "pena"],
        "fear": ["miedo", "preocupa", "inseguro", "riesgo"],
        "disgust": ["asco", "repugnante", "asqueroso"],
        "surprise": ["sorprendente", "increíble", "impresionante", "wow"],
        "trust": ["confiable", "seguro", "fiable", "confianza"],
        "anticipation": ["espero", "pronto", "próximo", "futuro"]
    }
}

# Language-specific thresholds
LANGUAGE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "english": {
        "sentiment": {
            "positive": 0.2,
            "negative": -0.2,
            "very_positive": 0.75,
            "very_negative": -0.75
        },
        "emotion": {
            "strong": 0.8,
            "moderate": 0.6,
            "weak": 0.3
        }
    },
    "spanish": {
        "sentiment": {
            "positive": 0.2,
            "negative": -0.2,
            "very_positive": 0.75,
            "very_negative": -0.75
        },
        "emotion": {
            "strong": 0.8,
            "moderate": 0.6,
            "weak": 0.3
        }
    }
}

# Stop words configuration
STOP_WORDS_CONFIG: Dict[str, str] = {
    "english": "english",  # Built-in English stop words
    "spanish": None,      # Will be loaded from custom lists
}

def get_emotion_keywords(language: str) -> Dict[str, List[str]]:
    """Get emotion keywords for a specific language."""
    return EMOTION_KEYWORDS.get(language, EMOTION_KEYWORDS["english"])

def get_language_thresholds(language: str) -> Dict[str, Dict[str, float]]:
    """Get language-specific thresholds."""
    return LANGUAGE_THRESHOLDS.get(language, LANGUAGE_THRESHOLDS["english"])

def get_stop_words(language: str) -> str:
    """Get stop words configuration for a specific language."""
    return STOP_WORDS_CONFIG.get(language, None) 
