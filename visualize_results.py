import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
from src.analysis.nlp_analysis import ReviewAnalyzer

def load_and_process_data(json_file):
    """Load and process the raw data."""
    # Load raw data
    with open(json_file, "r", encoding='utf-8') as f:
        reviews = json.load(f)
    df = pd.DataFrame(reviews)
    
    # Preprocess reviews
    processed_df = preprocess_pipeline(df, text_column="text")
    
    # Analyze reviews
    analyzer = ReviewAnalyzer(language='es')
    results = analyzer.analyze_reviews(processed_df, text_column="processed_text")
    
    return results

def create_visualizations(df, output_dir='visualizations'):
    """Create various visualizations from the analysis results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')  # Using a built-in style instead of seaborn
    sns.set_theme()  # This will set seaborn's default theme
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Distribution of Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # 2. Sentiment Polarity Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sentiment_polarity', bins=30)
    plt.title('Distribution of Sentiment Polarity Scores')
    plt.xlabel('Polarity Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_polarity_distribution.png'))
    plt.close()
    
    # 3. Predicted Star Ratings
    plt.figure(figsize=(10, 6))
    star_counts = df['predicted_stars'].value_counts().sort_index()
    sns.barplot(x=star_counts.index, y=star_counts.values)
    plt.title('Distribution of Predicted Star Ratings')
    plt.xlabel('Stars')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'star_ratings_distribution.png'))
    plt.close()
    
    # 4. Primary Emotions
    plt.figure(figsize=(12, 6))
    emotion_counts = df['primary_emotion'].value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribution of Primary Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    plt.close()
    
    # 5. Correlation between Sentiment Polarity and Star Ratings
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sentiment_polarity', y='predicted_stars')
    plt.title('Correlation between Sentiment Polarity and Star Ratings')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Predicted Stars')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_stars_correlation.png'))
    plt.close()
    
    # 6. Word Cloud of Keywords (if available)
    if 'keywords' in df.columns:
        from wordcloud import WordCloud
        # Combine all keywords into a single string
        all_keywords = ' '.join(df['keywords'].astype(str).str.join(' '))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Keywords')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keyword_wordcloud.png'))
        plt.close()

def main():
    # Load and process the data
    topic = "bancosantander"
    json_file = os.path.join('data/raw', f"{topic}_reviews.json")
    
    if not os.path.exists(json_file):
        print(f"Error: Could not find results file at {json_file}")
        return
    
    print("Loading and processing data...")
    df = load_and_process_data(json_file)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df)
    print("Visualizations have been created in the 'visualizations' directory.")

if __name__ == "__main__":
    main() 