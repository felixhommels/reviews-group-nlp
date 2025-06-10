# LEGACY/REFERENCE: Not used in production. See [src/tests/integration/test_cases.py]for the current version.


"""Test cases module for NLP analysis functionality."""

import os
import sys
import pandas as pd
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import random
from collections import Counter

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.analysis.analysis_old.nlp_analysis_lang import ReviewAnalyzer, SentimentLabel, EmotionLabel

def create_analysis_table(results: List[Dict]) -> Table:
    """Create a rich table for displaying analysis results."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="Multilingual Analysis Results",
        title_style="bold cyan"
    )
    
    table.add_column("Language", style="yellow")
    table.add_column("Text", style="cyan", width=50)
    table.add_column("Sentiment", style="green")
    table.add_column("Polarity", justify="right")
    table.add_column("Stars", justify="center")
    table.add_column("Primary Emotion", style="blue")
    
    for result in results:
        table.add_row(
            result['Language'],
            result['Text'],
            str(result['Sentiment']),
            result['Polarity'],
            str(result['Stars']),
            str(result['Primary Emotion'])
        )
    
    return table

def run_test_cases(sample_data_path: str = None):
    """Run test cases to verify analyzer functionality with detailed output."""
    console = Console()
    
    if sample_data_path is None:
        sample_data_path = os.path.join(project_root, 'data', 'raw', 'bancosantander_reviews.json')
    
    def print_header(title: str):
        console.print(Panel(f"[bold cyan]{title}[/]", expand=False))
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer()
    
    # Test cases including Spanish reviews
    print_header("1. Multilingual Test Cases Analysis")
    test_texts = [
        # Spanish negative reviews
        "La app del banco santander es realmente penosa",
        "Pésimo banco, pésimo servicio",
        "No funciona nada bien, una experiencia terrible",
        # Spanish positive reviews
        "Excelente servicio, muy satisfecho con la atención",
        "La aplicación funciona perfectamente, muy recomendable",
        # Spanish neutral reviews
        "Es una aplicación normal, tiene sus pros y contras",
        # English reviews for comparison
        "This is absolutely terrible. Worst experience ever.",
        "The service is amazing! Highly recommended!",
        "It's okay, nothing special."
    ]
    
    results = []
    for text in track(test_texts, description="Analyzing test cases..."):
        sentiment = analyzer.analyze_sentiment(text)
        stars = analyzer.predict_star_rating(text)
        keywords = analyzer.extract_keywords(text)
        emotion, scores = analyzer.analyze_emotion(text)
        
        results.append({
            'Text': text,
            'Language': 'Spanish' if any(word in text.lower() for word in 
                       ['banco', 'aplicación', 'servicio', 'funciona']) else 'English',
            'Sentiment': sentiment.sentiment,
            'Polarity': f"{sentiment.polarity:.2f}",
            'Stars': stars,
            'Primary Emotion': emotion
        })
    
    # Display results
    console.print(create_analysis_table(results))

    # Batch Analysis
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
        
        # Display summary statistics and visualizations
        display_analysis_results(result_df, console)
            
    except Exception as e:
        console.print(f"[bold red]Error in batch analysis:[/] {str(e)}")

def display_analysis_results(df: pd.DataFrame, console: Console):
    """Display analysis results including statistics, visualizations, and samples."""
    # Summary statistics
    stats_table = Table(show_header=True, header_style="bold magenta", 
                       title="Analysis Summary")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Total Reviews", str(len(df)))
    stats_table.add_row("Average Star Rating", 
                       f"{df['predicted_stars'].mean():.2f}")
    stats_table.add_row(
        "Sentiment Distribution", 
        ", ".join(f"{k}: {v}" for k, v in 
                 df['sentiment'].value_counts().items())
    )
    stats_table.add_row(
        "Most Common Emotions",
        ", ".join(f"{k}: {v}" for k, v in 
                 df['primary_emotion'].value_counts().head(3).items())
    )
    
    console.print(stats_table)
    
    # Create visualizations
    console.print(Panel("[bold cyan]3. Visualizations[/]", expand=False))
    create_visualizations(df)
    console.print("[green]Visualization saved as 'analysis_results.png'[/]")
    
    # Show sample reviews
    console.print(Panel("[bold cyan]4. Sample Reviews by Sentiment[/]", expand=False))
    display_sample_reviews(df, console)
    
    # Show accuracy metrics if available
    if 'rating' in df.columns:
        display_accuracy_metrics(df, console)

def create_visualizations(df: pd.DataFrame):
    """Create and save visualization plots."""
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Sentiment Distribution
    sentiment_dist = df['sentiment'].value_counts()
    ax1.bar(sentiment_dist.index, sentiment_dist.values)
    ax1.set_title('Sentiment Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Star Rating Distribution
    df['predicted_stars'].value_counts().sort_index().plot(
        kind='bar', ax=ax2, color='skyblue'
    )
    ax2.set_title('Star Rating Distribution')
    ax2.set_xlabel('Stars')
    ax2.set_ylabel('Count')
    
    # Top Emotions
    emotion_dist = df['primary_emotion'].value_counts().head(5)
    ax3.bar(emotion_dist.index, emotion_dist.values)
    ax3.set_title('Top 5 Emotions')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')

def display_sample_reviews(df: pd.DataFrame, console: Console):
    """Display sample reviews in a table format."""
    samples_table = Table(show_header=True, header_style="bold magenta", 
                         title="Sample Reviews")
    samples_table.add_column("Sentiment", style="cyan")
    samples_table.add_column("Text", style="white", width=50)
    samples_table.add_column("Stars", justify="center")
    samples_table.add_column("Emotion", style="yellow")
    samples_table.add_column("Keywords", style="blue")
    
    for sentiment in ['positive', 'negative', 'neutral']:
        sample = (df[df['sentiment'] == sentiment].iloc[0] 
                 if not df[df['sentiment'] == sentiment].empty else None)
        if sample is not None:
            samples_table.add_row(
                sentiment,
                sample['processed_text'][:100] + "...",
                str(sample['predicted_stars']),
                str(sample['primary_emotion']),
                ', '.join(sample['keywords'][:3])
            )
    
    console.print(samples_table)

def display_accuracy_metrics(df: pd.DataFrame, console: Console):
    """Display accuracy metrics if available."""
    console.print(Panel("[bold cyan]5. Accuracy Metrics[/]", expand=False))
    metrics_table = Table(show_header=True, header_style="bold magenta", 
                         title="Prediction Accuracy")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    # Convert ratings to numeric type
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['predicted_stars'] = pd.to_numeric(df['predicted_stars'], errors='coerce')
    
    # Calculate metrics only on valid numeric values
    valid_ratings = df.dropna(subset=['rating', 'predicted_stars'])
    if len(valid_ratings) > 0:
        star_accuracy = (valid_ratings['rating'] == valid_ratings['predicted_stars']).mean()
        star_diff = (valid_ratings['rating'] - valid_ratings['predicted_stars']).abs().mean()
        
        metrics_table.add_row("Star Rating Accuracy", f"{star_accuracy:.2%}")
        metrics_table.add_row("Average Star Difference", f"{star_diff:.2f}")
        metrics_table.add_row("Total Valid Ratings", str(len(valid_ratings)))
    else:
        metrics_table.add_row("Status", "No valid ratings for comparison")
    
    console.print(metrics_table)

def test_multilingual_analysis():
    """Test multilingual sentiment and emotion analysis."""
    test_cases = [
        {
            'language': 'en',
            'text': "This product is amazing! I absolutely love it.",
            'expected_sentiment': SentimentLabel.POSITIVE,
            'expected_emotion': EmotionLabel.JOY
        },
        {
            'language': 'es',
            'text': "El servicio fue terrible, nunca más volveré.",
            'expected_sentiment': SentimentLabel.NEGATIVE,
            'expected_emotion': EmotionLabel.ANGER
        },
        {
            'language': 'en',
            'text': "The product works as expected, nothing special.",
            'expected_sentiment': SentimentLabel.NEUTRAL,
            'expected_emotion': EmotionLabel.NEUTRAL
        }
    ]
    
    results = []
    console = Console()
    
    for case in track(test_cases, description="Running analysis..."):
        analyzer = ReviewAnalyzer(language=case['language'])
        sentiment = analyzer.analyze_sentiment(case['text'])
        emotion = analyzer.detect_emotion(case['text'])
        stars = analyzer.predict_stars(sentiment)
        
        result = {
            'Language': case['language'],
            'Text': case['text'][:50] + ('...' if len(case['text']) > 50 else ''),
            'Sentiment': sentiment.sentiment,
            'Polarity': f"{sentiment.polarity:.2f}",
            'Stars': stars,
            'Primary Emotion': emotion
        }
        
        # Validate results
        passed = (
            sentiment.sentiment == case['expected_sentiment'] and
            emotion == case['expected_emotion']
        )
        
        results.append(result)
        
        # Print individual results with color coding
        color = "green" if passed else "red"
        console.print(f"[{color}]Test case {len(results)}:[/{color}]")
        console.print(Panel.fit(
            f"Input: {case['text']}\n"
            f"Expected: {case['expected_sentiment']}, {case['expected_emotion']}\n"
            f"Got: {sentiment.sentiment}, {emotion}",
            border_style=color
        ))
    
    return create_analysis_table(results)

def test_keyword_extraction():
    """Test keyword extraction functionality."""
    reviews = [
        "The mobile app is very user-friendly and efficient.",
        "Customer service response time needs improvement.",
        "Great banking features and security measures.",
        "The interface is clean but could use more features."
    ]
    
    analyzer = ReviewAnalyzer()
    keywords = analyzer.extract_keywords(reviews)
    
    console = Console()
    console.print("\n[bold cyan]Top Keywords:[/bold cyan]")
    for i, keyword in enumerate(keywords, 1):
        console.print(f"{i}. {keyword}")
    
    return keywords

def run_complete_analysis(sample_data_path: str):
    """Run a complete analysis on sample data including sentiment, emotion, and keywords."""
    console = Console()
    
    try:
        # Load and process sample data
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reviews = pd.DataFrame(data)
        
        # Initialize analyzer
        analyzer = ReviewAnalyzer()
        
        # Process each review
        results = []
        for _, review in track(reviews.iterrows(), description="Analyzing reviews..."):
            text = review.get('text', '')
            if not text:
                continue
                
            sentiment = analyzer.analyze_sentiment(text)
            emotion = analyzer.detect_emotion(text)
            stars = analyzer.predict_stars(sentiment)
            
            results.append({
                'Language': review.get('language', 'unknown'),
                'Text': text[:50] + ('...' if len(text) > 50 else ''),
                'Sentiment': sentiment.sentiment,
                'Polarity': f"{sentiment.polarity:.2f}",
                'Stars': stars,
                'Primary Emotion': emotion
            })
        
        # Create and display results table
        table = create_analysis_table(results)
        console.print(table)
        
        # Extract and display keywords
        console.print("\n[bold cyan]Overall Review Keywords:[/bold cyan]")
        keywords = analyzer.extract_keywords([r['Text'] for r in results])
        for i, keyword in enumerate(keywords, 1):
            console.print(f"{i}. {keyword}")
        
        # Create sentiment distribution plot
        plt.figure(figsize=(10, 6))
        sentiments = [r['Sentiment'].value for r in results]
        sns.countplot(x=sentiments)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png')
        plt.close()
        
        console.print("\n[green]Analysis complete! Sentiment distribution plot saved as 'sentiment_distribution.png'[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        raise

def test_detailed_analysis():
    """Test detailed analysis functionality including metrics and language-specific models."""
    console = Console()
    
    test_texts = {
        'en': [
            "This product is absolutely amazing! The best purchase I've made.",
            "The service was terrible and I regret buying this.",
            "It works fine, nothing special to mention."
        ],
        'es': [
            "¡El servicio es excelente! Muy recomendado.",
            "No me gustó nada, pésima experiencia.",
            "Es un producto normal, cumple su función."
        ]
    }
    
    results = []
    
    for lang, texts in test_texts.items():
        analyzer = ReviewAnalyzer(language=lang)
        
        for text in texts:
            # Get detailed metrics
            metrics = analyzer.get_detailed_metrics(text)
            
            # Get language-specific analysis
            lang_specific = analyzer.analyze_language_specific(text)
            
            result = {
                'language': lang,
                'text': text,
                'sentiment_metrics': metrics['sentiment'],
                'emotions': metrics.get('emotions', {}),
                'stars': metrics['predicted_stars'],
                'lang_specific': lang_specific
            }
            results.append(result)
            
            # Display results
            console.print(Panel(
                f"[cyan]Text:[/cyan] {text}\n"
                f"[yellow]Language:[/yellow] {lang}\n"
                f"[green]Sentiment:[/green] {metrics['sentiment']['label']} "
                f"(polarity: {metrics['sentiment']['polarity']:.2f})\n"
                f"[blue]Predicted Stars:[/blue] {metrics['predicted_stars']}\n"
                f"[magenta]Model Used:[/magenta] {lang_specific['model']}",
                title=f"Analysis Result #{len(results)}",
                border_style="bright_blue"
            ))
    
    return results

def test_batch_processing():
    """Test batch processing functionality."""
    console = Console()
    
    # Load sample reviews from file
    reviews_file = os.path.join(project_root, "data/raw/bancosantander_reviews.json")
    
    if os.path.exists(reviews_file):
        with open(reviews_file, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
            
        # Take a sample of reviews
        sample_size = min(10, len(reviews_data))
        sample_reviews = random.sample(reviews_data, sample_size)
        
        # Extract text and language
        texts = [review.get('text', '') for review in sample_reviews]
        
        # Process batch
        analyzer = ReviewAnalyzer()
        results = analyzer.analyze_batch(texts)
        
        # Display results summary
        sentiments = [r['sentiment'] for r in results]
        stars = [r['stars'] for r in results]
        emotions = [r['emotion'] for r in results]
        
        summary_table = Table(title="Batch Analysis Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Distribution", style="magenta")
        
        summary_table.add_row(
            "Sentiments",
            f"Positive: {sentiments.count('positive')}, "
            f"Negative: {sentiments.count('negative')}, "
            f"Neutral: {sentiments.count('neutral')}"
        )
        
        summary_table.add_row(
            "Star Ratings",
            f"Avg: {sum(stars)/len(stars):.1f}, "
            f"Min: {min(stars)}, "
            f"Max: {max(stars)}"
        )
        
        emotion_counts = Counter(emotions)
        summary_table.add_row(
            "Top Emotions",
            ", ".join(f"{e}: {c}" for e, c in emotion_counts.most_common(3))
        )
        
        console.print(summary_table)
        
        return results
    else:
        console.print("[red]Sample reviews file not found![/red]")
        return []

if __name__ == "__main__":
    console = Console()
    console.print("\n[bold magenta]Running Comprehensive Tests...[/bold magenta]")
    
    # Run basic tests
    test_table = test_multilingual_analysis()
    console.print(test_table)
    
    # Run detailed analysis tests
    console.print("\n[bold magenta]Running Detailed Analysis Tests...[/bold magenta]")
    detailed_results = test_detailed_analysis()
    
    # Run batch processing tests
    console.print("\n[bold magenta]Running Batch Processing Tests...[/bold magenta]")
    batch_results = test_batch_processing()
    
    # Test keyword extraction
    console.print("\n[bold magenta]Testing Keyword Extraction...[/bold magenta]")
    keywords = test_keyword_extraction()
