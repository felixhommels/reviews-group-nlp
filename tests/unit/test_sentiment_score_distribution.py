import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_all_processed_reviews(input_dir="data/processed_test_results"):
    all_reviews = []
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("["):
                    reviews = json.loads(content)
                else:
                    reviews = [json.loads(line) for line in content.splitlines() if line.strip()]
                all_reviews.extend(reviews)
    return pd.DataFrame(all_reviews)

def plot_sentiment_score_distribution(df, output_path="data/visualizations/sentiment_score_distribution.png"):
    if 'sentiment_score' not in df.columns:
        print("No 'sentiment_score' column found in the data.")
        return

    # Only use unique, sorted values present in the data
    unique_scores = sorted(df['sentiment_score'].dropna().unique())
    plt.figure(figsize=(12, 7))
    plt.hist(df['sentiment_score'].dropna(), bins=unique_scores + [unique_scores[-1] + 0.01], rwidth=0.8, alpha=0.6)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.xticks(unique_scores)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Sentiment score distribution saved to {output_path}")

if __name__ == "__main__":
    df = load_all_processed_reviews()
    print(f"Loaded {len(df)} reviews from processed_test_results.")
    plot_sentiment_score_distribution(df)
