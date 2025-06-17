# src/summarizer/enhanced_summarizer.py
"""
Enhanced Review Summarization Module.

This module provides comprehensive functionality for generating summaries, insights,
and visualizations from analyzed review data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ReviewSummarizer:
    """An enhanced class for generating comprehensive summaries from review collections."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the summarizer with language-specific settings."""
        self.language = language
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english' if language == 'en' else None,
            ngram_range=(1, 2),
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
    
    def get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical insights from review data."""
        if df.empty:
            return {'error': 'No data to analyze'}
        
        stats = {
            'total_reviews': len(df),
            'data_quality': self._assess_data_quality(df),
            'sentiment_analysis': self._analyze_sentiment_distribution(df),
            'temporal_analysis': self._analyze_temporal_patterns(df),
            'rating_analysis': self._analyze_ratings(df),
            'text_statistics': self._analyze_text_statistics(df)
        }
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the review data."""
        quality_metrics = {
            'total_reviews': len(df),
            'reviews_with_text': df['processed_text'].notna().sum() if 'processed_text' in df.columns else 0,
            'average_review_length': 0,
            'empty_reviews': 0,
            'duplicate_reviews': 0
        }
        
        if 'processed_text' in df.columns:
            text_lengths = df['processed_text'].fillna('').str.len()
            quality_metrics['average_review_length'] = text_lengths.mean()
            quality_metrics['empty_reviews'] = (text_lengths == 0).sum()
            quality_metrics['duplicate_reviews'] = df['processed_text'].duplicated().sum()
        
        return quality_metrics
    
    def _analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment patterns in the data."""
        sentiment_analysis = {}
        
        # Sentiment label distribution
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_analysis['label_distribution'] = sentiment_counts.to_dict()
            sentiment_analysis['label_percentages'] = (sentiment_counts / len(df) * 100).round(2).to_dict()
        
        # Sentiment polarity statistics
        if 'sentiment_polarity' in df.columns:
            polarity_stats = df['sentiment_polarity'].describe()
            sentiment_analysis['polarity_statistics'] = {
                'mean': polarity_stats['mean'],
                'std': polarity_stats['std'],
                'min': polarity_stats['min'],
                'max': polarity_stats['max'],
                'median': polarity_stats['50%']
            }
        
        return sentiment_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns if date information is available."""
        temporal_analysis = {}
        
        # Look for date columns
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            try:
                # Use the first date column found
                date_col = date_columns[0]
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                
                if df_temp[date_col].notna().any():
                    temporal_analysis['date_range'] = {
                        'earliest': df_temp[date_col].min().isoformat() if pd.notna(df_temp[date_col].min()) else None,
                        'latest': df_temp[date_col].max().isoformat() if pd.notna(df_temp[date_col].max()) else None
                    }
                    
                    # Monthly review distribution
                    monthly_counts = df_temp.groupby(df_temp[date_col].dt.to_period('M')).size()
                    temporal_analysis['monthly_distribution'] = monthly_counts.to_dict()
                    
            except Exception as e:
                logger.warning(f"Could not analyze temporal patterns: {e}")
                temporal_analysis['error'] = str(e)
        
        return temporal_analysis
    
    def _analyze_ratings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rating patterns and predictions."""
        rating_analysis = {}
        
        # Original ratings
        if 'rating' in df.columns:
            rating_stats = df['rating'].describe()
            rating_analysis['original_ratings'] = {
                'mean': rating_stats['mean'],
                'median': rating_stats['50%'],
                'distribution': df['rating'].value_counts().to_dict()
            }
        
        # Predicted ratings
        predicted_cols = [col for col in df.columns if 'predicted' in col.lower() and 'rating' in col.lower()]
        for col in predicted_cols:
            if df[col].notna().any():
                pred_stats = df[col].describe()
                rating_analysis[f'{col}_statistics'] = {
                    'mean': pred_stats['mean'],
                    'median': pred_stats['50%'],
                    'distribution': df[col].value_counts().to_dict()
                }
        
        return rating_analysis
    
    def _analyze_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text-related statistics."""
        text_stats = {}
        
        if 'processed_text' in df.columns:
            texts = df['processed_text'].fillna('')
            
            # Basic text statistics
            text_lengths = texts.str.len()
            word_counts = texts.str.split().str.len()
            
            text_stats['length_statistics'] = {
                'mean_characters': text_lengths.mean(),
                'median_characters': text_lengths.median(),
                'mean_words': word_counts.mean(),
                'median_words': word_counts.median()
            }
            
            # Language distribution if available
            if 'language' in df.columns:
                lang_dist = df['language'].value_counts()
                text_stats['language_distribution'] = lang_dist.to_dict()
        
        return text_stats
    
    def extract_key_phrases(self, df: pd.DataFrame, text_column: str = 'processed_text', 
                          top_n: int = 20) -> List[Tuple[str, float]]:
        """Extract most significant phrases using improved TF-IDF."""
        if df.empty or text_column not in df.columns:
            logger.warning(f"Cannot extract key phrases: missing {text_column} column")
            return []
        
        try:
            # Filter out empty texts
            valid_texts = df[text_column].fillna('').str.strip()
            valid_texts = valid_texts[valid_texts != '']
            
            if len(valid_texts) == 0:
                logger.warning("No valid texts found for key phrase extraction")
                return []
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(valid_texts)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Calculate importance scores (mean TF-IDF across all documents)
            scores = tfidf_matrix.mean(axis=0).A1
            
            # Get top phrases
            top_indices = scores.argsort()[-top_n:][::-1]
            key_phrases = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return key_phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def generate_aspect_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate summaries for different aspects with improved categorization."""
        # Enhanced aspect categories with more keywords
        aspects = {
            'quality': ['quality', 'reliable', 'performance', 'works', 'durable', 'robust', 'solid', 'excellent'],
            'usability': ['easy', 'simple', 'intuitive', 'user-friendly', 'convenient', 'accessible', 'straightforward'],
            'support': ['support', 'customer service', 'help', 'assistance', 'response', 'staff', 'team'],
            'value': ['price', 'cost', 'worth', 'value', 'money', 'expensive', 'cheap', 'affordable', 'budget'],
            'features': ['feature', 'functionality', 'capability', 'options', 'tools', 'function'],
            'design': ['design', 'interface', 'look', 'appearance', 'layout', 'visual', 'aesthetic'],
            'speed': ['fast', 'slow', 'quick', 'speed', 'performance', 'loading', 'responsive'],
            'reliability': ['reliable', 'stable', 'consistent', 'dependable', 'trustworthy', 'works']
        }
        
        aspect_summary = {}
        
        for aspect, keywords in aspects.items():
            # Create regex pattern for better matching
            pattern = '|'.join([rf'\b{kw}\w*' for kw in keywords])
            
            # Filter reviews mentioning this aspect
            if 'processed_text' in df.columns:
                aspect_mask = df['processed_text'].str.contains(pattern, case=False, na=False, regex=True)
                aspect_reviews = df[aspect_mask]
                
                if len(aspect_reviews) > 0:
                    aspect_data = {
                        'mention_count': len(aspect_reviews),
                        'mention_percentage': (len(aspect_reviews) / len(df)) * 100
                    }
                    
                    # Sentiment analysis for this aspect
                    if 'sentiment_polarity' in aspect_reviews.columns:
                        aspect_data['avg_sentiment'] = aspect_reviews['sentiment_polarity'].mean()
                        aspect_data['sentiment_std'] = aspect_reviews['sentiment_polarity'].std()
                    
                    if 'sentiment' in aspect_reviews.columns:
                        sentiment_counts = aspect_reviews['sentiment'].value_counts()
                        total_mentions = len(aspect_reviews)
                        aspect_data['sentiment_breakdown'] = {
                            'positive_ratio': sentiment_counts.get('positive', 0) / total_mentions,
                            'negative_ratio': sentiment_counts.get('negative', 0) / total_mentions,
                            'neutral_ratio': sentiment_counts.get('neutral', 0) / total_mentions
                        }
                    
                    # Rating analysis for this aspect
                    rating_cols = [col for col in aspect_reviews.columns if 'rating' in col.lower()]
                    for col in rating_cols:
                        if aspect_reviews[col].notna().any():
                            aspect_data[f'avg_{col}'] = aspect_reviews[col].mean()
                    
                    aspect_summary[aspect] = aspect_data
        
        return aspect_summary
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        logger.info("Generating enhanced summary report...")
        
        try:
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_reviews': len(df),
                    'summarizer_version': 'enhanced_v1.0'
                },
                'statistics': self.get_statistical_summary(df),
                'key_phrases': self.extract_key_phrases(df),
                'aspect_summary': self.generate_aspect_summary(df),
                'recommendations': self._generate_recommendations(df)
            }
            
            # Generate natural language summary
            report['summary_text'] = self._generate_narrative_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {
                'error': str(e),
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_reviews': len(df) if not df.empty else 0
                }
            }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        try:
            # Sentiment-based recommendations
            if 'sentiment' in df.columns:
                sentiment_dist = df['sentiment'].value_counts(normalize=True)
                negative_ratio = sentiment_dist.get('negative', 0)
                
                if negative_ratio > 0.4:
                    recommendations.append("High negative sentiment detected (>40%). Consider investigating common complaints and addressing key issues.")
                elif negative_ratio < 0.1:
                    recommendations.append("Very positive sentiment profile. Consider leveraging these reviews for marketing purposes.")
            
            # Rating-based recommendations
            rating_cols = [col for col in df.columns if 'rating' in col.lower()]
            for col in rating_cols:
                if df[col].notna().any():
                    avg_rating = df[col].mean()
                    if avg_rating < 3.0:
                        recommendations.append(f"Low average rating ({avg_rating:.1f}/5). Focus on improving core product/service quality.")
                    elif avg_rating > 4.5:
                        recommendations.append(f"Excellent rating ({avg_rating:.1f}/5). Consider showcasing these high ratings prominently.")
            
            # Aspect-based recommendations
            aspects = self.generate_aspect_summary(df)
            for aspect, data in aspects.items():
                if data['mention_count'] > len(df) * 0.1:  # If mentioned in >10% of reviews
                    if data.get('avg_sentiment', 0) < -0.3:
                        recommendations.append(f"Negative feedback on {aspect}. This is a priority area for improvement.")
                    elif data.get('avg_sentiment', 0) > 0.3:
                        recommendations.append(f"Strong positive feedback on {aspect}. Consider highlighting this as a key strength.")
            
            # Data quality recommendations
            if 'processed_text' in df.columns:
                empty_ratio = (df['processed_text'].fillna('').str.len() == 0).mean()
                if empty_ratio > 0.2:
                    recommendations.append("High percentage of empty reviews. Consider improving data collection process.")
        
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to data processing issues.")
        
        return recommendations
    
    def _generate_narrative_summary(self, report: Dict[str, Any]) -> str:
        """Generate a natural language narrative summary."""
        try:
            metadata = report.get('metadata', {})
            stats = report.get('statistics', {})
            aspects = report.get('aspect_summary', {})
            key_phrases = report.get('key_phrases', [])
            
            total_reviews = metadata.get('total_reviews', 0)
            
            # Start with basic overview
            summary_parts = [
                f"Analysis of {total_reviews:,} reviews reveals the following insights:"
            ]
            
            # Sentiment summary
            sentiment_data = stats.get('sentiment_analysis', {})
            if 'label_distribution' in sentiment_data:
                sentiment_dist = sentiment_data['label_distribution']
                total_sentiment_reviews = sum(sentiment_dist.values())
                
                if total_sentiment_reviews > 0:
                    pos_pct = (sentiment_dist.get('positive', 0) / total_sentiment_reviews) * 100
                    neg_pct = (sentiment_dist.get('negative', 0) / total_sentiment_reviews) * 100
                    
                    if pos_pct > 60:
                        sentiment_desc = "overwhelmingly positive"
                    elif pos_pct > 40:
                        sentiment_desc = "generally positive"
                    elif neg_pct > 40:
                        sentiment_desc = "generally negative"
                    else:
                        sentiment_desc = "mixed"
                    
                    summary_parts.append(f"Overall sentiment is {sentiment_desc} ({pos_pct:.1f}% positive, {neg_pct:.1f}% negative)")
            
            # Rating summary
            rating_data = stats.get('rating_analysis', {})
            if 'original_ratings' in rating_data:
                avg_rating = rating_data['original_ratings'].get('mean')
                if avg_rating:
                    summary_parts.append(f"Average rating: {avg_rating:.1f}/5.0")
            
            # Key aspects
            significant_aspects = {k: v for k, v in aspects.items() 
                                 if v.get('mention_count', 0) > total_reviews * 0.05}  # Mentioned in >5% of reviews
            
            if significant_aspects:
                aspect_insights = []
                for aspect, data in significant_aspects.items():
                    sentiment = data.get('avg_sentiment', 0)
                    mentions = data.get('mention_count', 0)
                    
                    if sentiment > 0.2:
                        sentiment_desc = "positive"
                    elif sentiment < -0.2:
                        sentiment_desc = "negative" 
                    else:
                        sentiment_desc = "neutral"
                    
                    aspect_insights.append(f"{aspect}: {sentiment_desc} overall ({mentions} mentions)")
                
                if aspect_insights:
                    summary_parts.append("Key aspects mentioned:")
                    summary_parts.extend([f"• {insight}" for insight in aspect_insights[:5]])  # Top 5 aspects
            
            # Top keywords
            if key_phrases:
                top_keywords = [phrase for phrase, score in key_phrases[:5]]
                summary_parts.append(f"Most frequently mentioned terms: {', '.join(top_keywords)}")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating narrative summary: {e}")
            return f"Analysis of {report.get('metadata', {}).get('total_reviews', 0)} reviews completed. Detailed statistics available in the full report."
    
    def export_summary_to_file(self, summary: Dict[str, Any], filepath: str, format: str = 'json') -> bool:
        """Export summary to file in various formats."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == 'txt':
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(summary.get('summary_text', 'No summary text available'))
            
            elif format.lower() == 'csv':
                # Export key statistics as CSV
                stats_data = []
                if 'statistics' in summary:
                    # Flatten statistics for CSV export
                    self._flatten_dict_for_csv(summary['statistics'], stats_data)
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_csv(filepath, index=False)
            
            logger.info(f"Summary exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting summary: {e}")
            return False
    
    def _flatten_dict_for_csv(self, data: Dict, result: List, prefix: str = '') -> None:
        """Flatten nested dictionary for CSV export."""
        for key, value in data.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_dict_for_csv(value, result, new_key)
            else:
                result.append({'metric': new_key, 'value': value})