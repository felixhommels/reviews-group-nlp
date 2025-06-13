"""
Emotion Analysis Module.

This module provides emotion analysis capabilities using:
- NRCLex for English text
- Rule-based keyword matching as fallback
"""

import logging
from typing import Dict
from enum import Enum
from transformers import pipeline

from src.utils.dependencies import dependency_manager, DependencyError
from src.config.manager import ConfigManager
from src.config.language_configs import get_emotion_keywords

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

class EmotionAnalyzer:
    """Class for analyzing emotions in text using NRCLex or rule-based method."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the emotion analyzer.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._init_analyzer()
        self._load_config()
    
    def _init_analyzer(self):
        """Initialize the emotion analyzer with available tools."""
        try:
            if dependency_manager.nrclex_available:
                self.nrclex = dependency_manager.get_component('nrclex', 'NRCLex')
                logger.info("Initialized NRCLex emotion analyzer")
            else:
                logger.warning("NRCLex not available. Using rule-based emotion classification.")
        except DependencyError as e:
            logger.warning(f"Could not initialize NRCLex: {e}")
    
    def _load_config(self):
        """Load language-specific configuration."""
        self.config = ConfigManager.get_emotion_config(self.language)
        logger.info(f"Loaded emotion configuration for language: {self.language}")
    
    def analyze_emotion(self, text: str) -> Dict[EmotionLabel, float]:
        """Analyze emotions in text using NRCLex or rule-based method.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        emotions = {}
        try:
            if dependency_manager.nrclex_available and self.language == 'en':
                # Use NRCLex for English text
                nrc = self.nrclex(text)
                raw_scores = nrc.affect_frequencies
                
                # Map NRCLex emotions to our EmotionLabel enum
                for emotion in EmotionLabel:
                    if emotion.value in raw_scores:
                        if raw_scores[emotion.value] >= self.config['score_thresholds']['strong']:
                            emotions[emotion] = raw_scores[emotion.value]

            else:
                # Rule-based emotion detection using keyword matching
                keywords = self.config['keywords']
                text_lower = text.lower()
                
                for emotion, words in keywords.items():
                    count = sum(1 for word in words if word.lower() in text_lower)
                    if count > 0:
                        score = min(count / len(words), 1.0)
                        if score >= self.config['score_thresholds']['weak']:
                            emotions[EmotionLabel(emotion)] = score

            if not emotions:
                emotions[EmotionLabel.NEUTRAL] = 1.0

        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            emotions[EmotionLabel.NEUTRAL] = 1.0
            
        return emotions

class MultilingualEmotionAnalyzer:
    """Multilingual emotion analyzer using HuggingFace transformers pipeline."""
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions"):
        self.pipeline = pipeline("text-classification", model=model_name, top_k=None)

    def analyze_emotion(self, text: str) -> Dict[str, float]:
        results = self.pipeline(text)
        # Convert results to a dict: {label: score}
        return {r['label'].upper(): r['score'] for r in results[0]}
