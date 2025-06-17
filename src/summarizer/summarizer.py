"""Review Summarization Module.

This module provides functionality for generating both extractive and abstractive
summaries of review collections, along with statistical insights.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

logger = logging.getLogger(__name__)

class ReviewSummarizer:
    """A class for generating summaries and insights from review collections."""
    
    def __init__(self):
        """Initialize the summarizer with required tools."""
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate statistical insights from review data.
        
        Args:
            df: DataFrame containing analyzed reviews
            
        Returns:
            Dictionary containing statistical insights
        """
        stats = {
            'total_reviews': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_sentiment_polarity': df['sentiment_polarity'].mean(),
            'emotion_distribution': {}
        }
        
        # Calculate emotion distributions if available
        emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
        for col in emotion_cols:
            emotion = col.replace('emotion_', '')
            stats['emotion_distribution'][emotion] = df[col].mean()
        
        # Calculate average star rating if available
        if 'predicted_stars' in df.columns:
            stats['average_rating'] = df['predicted_stars'].mean()
        
        return stats
    
    def extract_key_phrases(self, df: pd.DataFrame, text_column: str = 'processed_text',
                          top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract most significant phrases from reviews using TF-IDF.
        
        Args:
            df: DataFrame containing reviews
            text_column: Name of the column containing processed text
            top_n: Number of top phrases to return
            
        Returns:
            List of (phrase, score) tuples
        """
        try:
            # Fit TF-IDF on all reviews
            tfidf_matrix = self.tfidf.fit_transform(df[text_column].fillna(''))
            
            # Get average TF-IDF score for each term
            feature_names = self.tfidf.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Get top phrases
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            return [(feature_names[i], avg_scores[i]) for i in top_indices]
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []
    
    def generate_aspect_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate summaries for different aspects of reviews.
        
        Args:
            df: DataFrame containing analyzed reviews
            
        Returns:
            Dictionary mapping aspects to their sentiment scores
        """
        aspects = {
            'quality': ['quality', 'reliable', 'performance', 'works'],
            'usability': ['easy', 'simple', 'intuitive', 'user-friendly'],
            'support': ['support', 'customer service', 'help'],
            'value': ['price', 'cost', 'worth', 'value'],
            'features': ['feature', 'functionality', 'capability']
        }
        
        aspect_summary = {}
        
        for aspect, keywords in aspects.items():
            # Filter reviews mentioning this aspect
            aspect_reviews = df[df['processed_text'].str.contains('|'.join(keywords), 
                                                                case=False, 
                                                                na=False)]
            
            if len(aspect_reviews) > 0:
                aspect_summary[aspect] = {
                    'mention_count': len(aspect_reviews),
                    'avg_sentiment': aspect_reviews['sentiment_polarity'].mean(),
                    'positive_ratio': len(aspect_reviews[aspect_reviews['sentiment'] == 'positive']) / len(aspect_reviews)
                }
        
        return aspect_summary
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate a comprehensive summary report of the reviews.
        
        Args:
            df: DataFrame containing analyzed reviews
            
        Returns:
            Dictionary containing various summary components
        """
        logger.info("Generating summary report...")
        
        report = {
            'statistics': self.get_statistical_summary(df),
            'key_phrases': self.extract_key_phrases(df),
            'aspect_summary': self.generate_aspect_summary(df)
        }
        
        # Add summary text
        try:
            stats = report['statistics']
            aspects = report['aspect_summary']
            
            # Generate overall summary text
            summary_text = [
                f"Analysis of {stats['total_reviews']} reviews reveals the following insights:",
                f"- Overall sentiment is {self._get_sentiment_description(stats['average_sentiment_polarity'])}",
                f"- Average rating: {stats.get('average_rating', 'N/A'):.1f}/5.0" if 'average_rating' in stats else ""
            ]
            
            # Add aspect insights
            for aspect, scores in aspects.items():
                if scores['mention_count'] > 5:  # Only include aspects with sufficient mentions
                    sentiment = "positive" if scores['avg_sentiment'] > 0.1 else "negative" if scores['avg_sentiment'] < -0.1 else "neutral"
                    summary_text.append(
                        f"- {aspect.title()}: {sentiment} overall ({scores['mention_count']} mentions)"
                    )
            
            report['summary_text'] = '\n'.join(filter(None, summary_text))
            
        except Exception as e:
            logger.error(f"Error generating summary text: {str(e)}")
            report['summary_text'] = "Error generating summary text"
        
        return report
    
    def _get_sentiment_description(self, polarity: float) -> str:
        """Convert sentiment polarity to descriptive text."""
        if polarity > 0.5:
            return "very positive"
        elif polarity > 0.1:
            return "generally positive"
        elif polarity < -0.5:
            return "very negative"
        elif polarity < -0.1:
            return "generally negative"
        else:
            return "neutral"
