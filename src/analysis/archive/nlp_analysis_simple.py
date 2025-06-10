# LEGACY/REFERENCE: Not used in production. See [nlp_analysis.py]for the current version.


"""
NLP Analysis Module for Review Analysis.

This module provides comprehensive natural language processing capabilities for analyzing
review text data, including:
- Sentiment analysis using VADER (primary) or TextBlob (fallback)
- Keyword extraction using TF-IDF
- Star rating prediction based on sentiment
- Emotion classification using NRCLex or rule-based fallback
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sentiment analysis: VADER (for English) or fallback to TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not available. Using TextBlob for sentiment analysis.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.error("Neither VADER nor TextBlob available. Sentiment analysis will be limited.")

# Keyword extraction: Use sklearn's TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: Emotion classification
try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False
    logger.warning("NRCLex not available. Using rule-based emotion classification.")

class SentimentLabel(str, Enum):
    """Enum for sentiment classification labels."""
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    UNKNOWN = 'unknown'

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

@dataclass
class SentimentAnalysis:
    """Container for sentiment analysis results."""
    sentiment: SentimentLabel
    polarity: float
    subjectivity: Optional[float] = None
    compound_score: Optional[float] = None

class ReviewAnalyzer:
    """A class for analyzing preprocessed review text data."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the ReviewAnalyzer.
        
        Args:
            language: ISO language code (default: 'en')
        """
        self.language = language
        self._init_sentiment_analyzer()
        self._init_keyword_extractor()
    
    def _init_sentiment_analyzer(self):
        """Initialize sentiment analysis tools based on availability."""
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        if TEXTBLOB_AVAILABLE:
            # TextBlob is initialized per-analysis
            pass
        if not (VADER_AVAILABLE or TEXTBLOB_AVAILABLE):
            raise ImportError("No sentiment analysis tools available. Install vaderSentiment or textblob.")

    def _init_keyword_extractor(self):
        """Initialize keyword extraction tools."""
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',  # TODO: Add multi-language support
            ngram_range=(1, 2)
        )
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze the sentiment of text using available tools.
        
        Args:
            text: The preprocessed text to analyze
            
        Returns:
            SentimentAnalysis object containing results
            
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        try:
            if VADER_AVAILABLE:
                scores = self.vader.polarity_scores(text)
                sentiment = (SentimentLabel.POSITIVE if scores['compound'] > 0.2
                           else SentimentLabel.NEGATIVE if scores['compound'] < -0.2
                           else SentimentLabel.NEUTRAL)
                return SentimentAnalysis(
                    sentiment=sentiment,
                    polarity=scores['compound'],
                    compound_score=scores['compound']
                )
            elif TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                sentiment = (SentimentLabel.POSITIVE if blob.sentiment.polarity > 0.2
                           else SentimentLabel.NEGATIVE if blob.sentiment.polarity < -0.2
                           else SentimentLabel.NEUTRAL)
                return SentimentAnalysis(
                    sentiment=sentiment,
                    polarity=blob.sentiment.polarity,
                    subjectivity=blob.sentiment.subjectivity
                )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return SentimentAnalysis(
                sentiment=SentimentLabel.UNKNOWN,
                polarity=0.0
            )
    
    def predict_star_rating(self, text: str) -> int:
        """Predict star rating (1-5) based on sentiment analysis.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Predicted star rating (1-5)
        """
        try:
            sentiment = self.analyze_sentiment(text)
            if sentiment.sentiment == SentimentLabel.UNKNOWN:
                return 3  # Neutral default
            
            # Use compound score or polarity to determine rating
            score = sentiment.compound_score if sentiment.compound_score is not None else sentiment.polarity
            if score > 0.5:
                return 5
            elif score > 0.1:
                return 4
            elif score > -0.1:
                return 3
            elif score > -0.5:
                return 2
            else:
                return 1
        except Exception as e:
            logger.error(f"Error predicting star rating: {str(e)}")
            return 3

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF.
        
        Args:
            text: The preprocessed text
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        try:
            if not text.strip():
                return []
                
            # Fit and transform on single document
            tfidf_matrix = self.tfidf.fit_transform([text])
            feature_names = self.tfidf.get_feature_names_out()
            
            # Convert to array and get scores
            scores = tfidf_matrix.toarray().flatten()
            # Get indices of non-zero scores, sorted by score
            nonzero_indices = scores.nonzero()[0]
            sorted_indices = nonzero_indices[scores[nonzero_indices].argsort()[::-1]]
            
            # Get top k keywords with non-zero scores
            top_indices = sorted_indices[:top_k]
            keywords = [feature_names[i] for i in top_indices]
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def analyze_emotion(self, text: str) -> Tuple[EmotionLabel, Dict[str, float]]:
        """Analyze emotions in text using NRCLex or rule-based fallback.
        
        Args:
            text: The preprocessed text
            
        Returns:
            Tuple of (primary emotion, emotion scores dict)
        """
        try:
            if NRCLEX_AVAILABLE:
                emotion_analyzer = NRCLex(text)
                scores = emotion_analyzer.affect_frequencies
                
                # Filter and normalize emotion scores to match our EmotionLabel enum
                valid_emotions = {e.value: scores.get(e.value, 0.0) for e in EmotionLabel}
                
                if not any(valid_emotions.values()):
                    return EmotionLabel.NEUTRAL, {'neutral': 1.0}
                    
                # Get primary emotion (highest score among valid emotions)
                primary = max(valid_emotions.items(), key=lambda x: x[1])[0]
                return EmotionLabel(primary), valid_emotions
            else:
                # Rule-based fallback using sentiment
                sentiment = self.analyze_sentiment(text)
                if sentiment.sentiment == SentimentLabel.UNKNOWN:
                    return EmotionLabel.NEUTRAL, {'neutral': 1.0}
                    
                score = sentiment.compound_score if sentiment.compound_score is not None else sentiment.polarity
                if score > 0.5:
                    return EmotionLabel.JOY, {'joy': 1.0}
                elif score > 0.2:
                    return EmotionLabel.TRUST, {'trust': 1.0}
                elif score < -0.5:
                    return EmotionLabel.ANGER, {'anger': 1.0}
                elif score < -0.2:
                    return EmotionLabel.SADNESS, {'sadness': 1.0}
                else:
                    return EmotionLabel.NEUTRAL, {'neutral': 1.0}
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return EmotionLabel.NEUTRAL, {'neutral': 1.0}

    def analyze_reviews(self, df: pd.DataFrame, text_column: str = 'processed_text') -> pd.DataFrame:
        """Perform comprehensive analysis on a DataFrame of reviews.
        
        Args:
            df: DataFrame containing the reviews
            text_column: Name of the column containing preprocessed text
            
        Returns:
            DataFrame with added analysis columns
            
        Raises:
            ValueError: If text_column is not found in DataFrame
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        logger.info("Starting review analysis...")
        result_df = df.copy()
        texts = result_df[text_column].astype(str).fillna('').tolist()
        
        # Analyze sentiment
        logger.info("Performing sentiment analysis...")
        sentiments = [self.analyze_sentiment(t) for t in texts]
        result_df['sentiment'] = [s.sentiment for s in sentiments]
        result_df['sentiment_polarity'] = [s.polarity for s in sentiments]
        if any(s.subjectivity is not None for s in sentiments):
            result_df['sentiment_subjectivity'] = [s.subjectivity for s in sentiments]
        
        # Extract keywords
        logger.info("Extracting keywords...")
        result_df['keywords'] = [self.extract_keywords(t) for t in texts]
        
        # Predict star ratings
        logger.info("Predicting star ratings...")
        result_df['predicted_stars'] = [self.predict_star_rating(t) for t in texts]
        
        # Analyze emotions
        logger.info("Analyzing emotions...")
        emotions = [self.analyze_emotion(t) for t in texts]
        result_df['primary_emotion'] = [e[0] for e in emotions]
        emotion_scores = [e[1] for e in emotions]
        for emotion in EmotionLabel:
            if any(emotion.value in scores for scores in emotion_scores):
                result_df[f'emotion_{emotion.value}'] = [
                    scores.get(emotion.value, 0.0) for scores in emotion_scores
                ]
        
        logger.info("Review analysis completed successfully.")
        return result_df

def test_analyzer(sample_data_path: str = 'data/raw/bancosantander_reviews.json'):
    """Test function to verify analyzer functionality with detailed output.
    
    Args:
        sample_data_path: Path to sample review data for testing
    """
    import pandas as pd
    import json
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Initialize Rich console
    console = Console()
    
    def print_header(title: str):
        console.print(Panel(f"[bold cyan]{title}[/]", expand=False))
        
    def create_analysis_table(results: list) -> Table:
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title="Analysis Results",
            title_style="bold cyan",
            expand=True
        )
        
        # Add columns
        table.add_column("Text", style="cyan", width=50)
        table.add_column("Sentiment", style="green")
        table.add_column("Polarity", justify="right")
        table.add_column("Stars", justify="center")
        table.add_column("Primary Emotion", style="yellow")
        table.add_column("Keywords", style="blue")
        
        # Add rows
        for result in results:
            table.add_row(
                result['Text'],
                str(result['Sentiment']),
                result['Polarity'],
                str(result['Stars']),
                str(result['Primary Emotion']),
                result['Keywords']
            )
        
        return table
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer()
    
    # 1. Single Review Analysis
    print_header("1. Test Cases Analysis")
    test_texts = [
        "The product is amazing! I love how easy it is to use. Highly recommended!",
        "This is absolutely terrible. Worst experience ever. Do not buy!",
        "It's okay, nothing special. Has some good and bad points."
    ]
    
    results = []
    for text in track(test_texts, description="Analyzing test cases..."):
        sentiment = analyzer.analyze_sentiment(text)
        stars = analyzer.predict_star_rating(text)
        keywords = analyzer.extract_keywords(text)
        emotion, scores = analyzer.analyze_emotion(text)
        
        results.append({
            'Text': text[:50] + '...' if len(text) > 50 else text,
            'Sentiment': sentiment.sentiment,
            'Polarity': f"{sentiment.polarity:.2f}",
            'Stars': stars,
            'Primary Emotion': emotion,
            'Keywords': ', '.join(keywords[:3])
        })
    
    # Display results in a rich table
    console.print(create_analysis_table(results))
    
    # 2. Batch Analysis (if sample data available)
    print_header("\n2. Batch Analysis")
    try:
        # Load and process sample data
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            raw_reviews = json.load(f)
        
        raw_df = pd.DataFrame(raw_reviews)
        if 'text' in raw_df.columns:
            raw_df.rename(columns={'text': 'processed_text'}, inplace=True)
        
        # Analyze reviews
        with console.status("[bold green]Analyzing reviews...") as status:
            result_df = analyzer.analyze_reviews(raw_df)
        
        # Summary statistics
        stats_table = Table(show_header=True, header_style="bold magenta", title="Analysis Summary")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        # Add statistics rows
        stats_table.add_row("Total Reviews", str(len(result_df)))
        stats_table.add_row("Average Star Rating", f"{result_df['predicted_stars'].mean():.2f}")
        stats_table.add_row(
            "Sentiment Distribution", 
            ", ".join(f"{k}: {v}" for k, v in result_df['sentiment'].value_counts().items())
        )
        stats_table.add_row(
            "Most Common Emotions",
            ", ".join(f"{k}: {v}" for k, v in result_df['primary_emotion'].value_counts().head(3).items())
        )
        
        console.print(stats_table)
        
        # Visualizations
        print_header("\n3. Visualizations")
        # Set up the plot style
        plt.style.use('default')  # Use default style for consistency
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Configure common style elements
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 1. Sentiment Distribution
        sentiment_dist = result_df['sentiment'].value_counts()
        ax1.bar(sentiment_dist.index, sentiment_dist.values)
        ax1.set_title('Sentiment Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Star Rating Distribution
        result_df['predicted_stars'].value_counts().sort_index().plot(
            kind='bar', ax=ax2, color='skyblue'
        )
        ax2.set_title('Star Rating Distribution')
        ax2.set_xlabel('Stars')
        ax2.set_ylabel('Count')
        
        # 3. Top Emotions
        emotion_dist = result_df['primary_emotion'].value_counts().head(5)
        ax3.bar(emotion_dist.index, emotion_dist.values)
        ax3.set_title('Top 5 Emotions')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
        console.print("[green]Visualization saved as 'analysis_results.png'[/]")
        
        # Sample Reviews
        print_header("\n4. Sample Reviews by Sentiment")
        samples_table = Table(show_header=True, header_style="bold magenta", title="Sample Reviews")
        samples_table.add_column("Sentiment", style="cyan")
        samples_table.add_column("Text", style="white", width=50)
        samples_table.add_column("Stars", justify="center")
        samples_table.add_column("Emotion", style="yellow")
        samples_table.add_column("Keywords", style="blue")
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sample = result_df[result_df['sentiment'] == sentiment].iloc[0] if not result_df[result_df['sentiment'] == sentiment].empty else None
            if sample is not None:
                samples_table.add_row(
                    sentiment,
                    sample['processed_text'][:100] + "...",
                    str(sample['predicted_stars']),
                    str(sample['primary_emotion']),
                    ', '.join(sample['keywords'][:3])
                )
        
        console.print(samples_table)
        
        # Accuracy Metrics (if actual ratings are available)
        if 'rating' in result_df.columns:
            print_header("\n5. Accuracy Metrics")
            metrics_table = Table(show_header=True, header_style="bold magenta", title="Prediction Accuracy")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            
            # Calculate accuracy metrics
            star_accuracy = (result_df['rating'] == result_df['predicted_stars']).mean()
            star_diff = (result_df['rating'] - result_df['predicted_stars']).abs().mean()
            
            metrics_table.add_row("Star Rating Accuracy", f"{star_accuracy:.2%}")
            metrics_table.add_row("Average Star Difference", f"{star_diff:.2f}")
            
            console.print(metrics_table)
        
    except Exception as e:
        console.print(f"[bold red]Error in batch analysis:[/] {str(e)}")

if __name__ == "__main__":
    # Run enhanced tests
    test_analyzer()

    # Process real data if available
    try:
        import pandas as pd
        from src.preprocessing.spacy_preprocessor import preprocess_pipeline

        # Load and process sample data
        raw_df = pd.read_json('data/raw/bancosantander_reviews.json')
        processed_df = preprocess_pipeline(raw_df, text_column='text')
        
        # Initialize and run analyzer
        analyzer = ReviewAnalyzer()
        result_df = analyzer.analyze_reviews(processed_df, text_column='processed_text')
        
        # Display results
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 0)
        columns = ['text', 'processed_text', 'sentiment', 'keywords', 
                  'predicted_stars', 'primary_emotion']
        print(result_df[columns].head(10))
        
    except Exception as e:
        logger.error(f"Error processing sample data: {str(e)}")
