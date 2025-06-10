"""Shared test fixtures and configuration."""

import pytest
import pandas as pd
from typing import Dict, List
from src.analysis.nlp_analysis import ReviewAnalyzer

@pytest.fixture
def sample_texts() -> Dict[str, List[str]]:
    """Sample texts for testing in different languages."""
    return {
        'english': [
            "This product is amazing! I love it.",
            "This is terrible, I hate it.",
            "It's okay, nothing special."
        ],
        'spanish': [
            "El producto es excelente, me encanta.",
            "Es terrible, no lo recomiendo.",
            "Es normal, ni bueno ni malo."
        ]
    }

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'processed_text': [
            "This product is excellent!",
            "I hate this, it's terrible.",
            "It's okay, nothing special."
        ],
        'rating': [5, 1, 3]  # Optional ground truth ratings
    })

@pytest.fixture
def analyzer() -> ReviewAnalyzer:
    """Create a ReviewAnalyzer instance for testing."""
    return ReviewAnalyzer()

@pytest.fixture
def multilingual_analyzer() -> ReviewAnalyzer:
    """Create a ReviewAnalyzer instance for multilingual testing."""
    return ReviewAnalyzer(language='es') 