import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import your analysis function
from app import run_nlp_analysis_on_all_processed

def load_enriched_reviews(analysis_dir="data/analysis"):
    all_reviews = []
    for file in os.listdir(analysis_dir):
        if file.endswith(".json"):
            with open(os.path.join(analysis_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    all_reviews.append(json.loads(line))
    return pd.DataFrame(all_reviews)

# Deprecated
def plot_granular_sentiment_distribution(df, output_path="data/visualizations/sentiment_granular_distribution.png"):
    if 'sentiment_granular_label' not in df.columns:
        print("No 'sentiment_granular_label' column found in the data.")
        return

    plt.figure(figsize=(10, 6))
    order = ["very_negative", "somewhat_negative", "neutral", "somewhat_positive", "very_positive"]
    sentiment_counts = df['sentiment_granular_label'].value_counts().reindex(order, fill_value=0)
    sentiment_counts = sentiment_counts[sentiment_counts > 0]  # Only plot present labels
    sentiment_counts.plot(kind='bar')
    plt.title('Distribution of Granular Sentiment Labels')
    plt.xlabel('Sentiment (Granular)')
    plt.ylabel('Count')
    plt.xticks(rotation=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Granular sentiment distribution saved to {output_path}")

# Deprecated
def plot_score_distribution_by_granular_label(df, output_path="data/visualizations/sentiment_score_by_granular_label.png"):
    if 'sentiment_granular_label' not in df.columns or 'sentiment_score' not in df.columns:
        print("Required columns not found in the data.")
        return

    plt.figure(figsize=(12, 7))
    order = ["very_negative", "somewhat_negative", "neutral", "somewhat_positive", "very_positive"]

    # Count occurrences of each sentiment_score for each granular label
    grouped = (
        df.groupby(['sentiment_granular_label', 'sentiment_score'])
        .size()
        .reset_index(name='count')
    )

    # Only keep present labels and scores
    grouped = grouped[grouped['sentiment_granular_label'].isin(order)]

    # Pivot for grouped bar plot
    pivot = grouped.pivot(index='sentiment_score', columns='sentiment_granular_label', values='count').fillna(0)
    pivot = pivot[order]  # Ensure column order

    pivot.plot(kind='bar', stacked=True, width=0.8, figsize=(12, 7))
    plt.title('Sentiment Score Distribution by Granular Label')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Sentiment score by granular label saved to {output_path}")

# Deprecated
def plot_granular_label_percentages(df, output_path="data/visualizations/sentiment_granular_label_percentages.png"):
    if 'sentiment_granular_label' not in df.columns:
        print("No 'sentiment_granular_label' column found in the data.")
        return

    order = ["very_negative", "somewhat_negative", "neutral", "somewhat_positive", "very_positive"]
    label_counts = df['sentiment_granular_label'].value_counts().reindex(order, fill_value=0)
    label_percentages = (label_counts / len(df) * 100).round(2)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_percentages.index, label_percentages.values, color=sns.color_palette("muted"))
    plt.title('Granular Sentiment Label Percentages')
    plt.xlabel('Sentiment (Granular)')
    plt.ylabel('Percentage of Reviews (%)')
    plt.ylim(0, max(label_percentages.values) * 1.1)
    plt.tight_layout()

    # Annotate bars with percentage values
    for bar, pct in zip(bars, label_percentages.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{pct}%', ha='center', va='bottom')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Granular sentiment label percentages saved to {output_path}")


def plot_granular_label_pie(df, output_path="data/visualizations/sentiment_granular_label_pie.png"):
    if 'sentiment_granular_label' not in df.columns:
        print("No 'sentiment_granular_label' column found in the data.")
        return

    order = ["very_negative", "somewhat_negative", "neutral", "somewhat_positive", "very_positive"]
    label_counts = df['sentiment_granular_label'].value_counts().reindex(order, fill_value=0)

    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("muted"))
    plt.title('Granular Sentiment Label Distribution')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Granular sentiment label pie chart saved to {output_path}")

if __name__ == "__main__":
    # Run the NLP analysis to update the output
    run_nlp_analysis_on_all_processed()
    # Now load and plot
    df = load_enriched_reviews()
    print(f"Loaded {len(df)} enriched reviews.")
    # plot_granular_sentiment_distribution(df)
    # plot_score_distribution_by_granular_label(df)

    # 1. Granular label counts and percentages
    label_counts = df['sentiment_granular_label'].value_counts().sort_index()
    label_percentages = (label_counts / len(df) * 100).round(2)
    summary_table = pd.DataFrame({
        'Count': label_counts,
        'Percentage': label_percentages.astype(str) + '%'
    })
    print(summary_table)

    # 2. Sentiment score statistics
    score_stats = df['sentiment_score'].describe()[['mean', '50%', 'min', 'max', 'std']]
    score_stats.index = ['Mean', 'Median', 'Min', 'Max', 'Std. Dev.']
    print(score_stats)

    # 3. Correlation with star ratings
    if 'predicted_rating_raw' in df.columns:
        corr = df['sentiment_score'].corr(df['predicted_rating_raw'])
        print(f"Correlation (score vs. stars): {corr:.2f}")

    plot_granular_label_percentages(df)
    plot_granular_label_pie(df)