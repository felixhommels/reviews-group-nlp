import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from itertools import chain

# --- Load a sample JSON file from data/analysis ---
# For example, load the first JSON file found in data/analysis
analysis_dir = "data/analysis"
sample_file = None
for file in os.listdir(analysis_dir):
    if file.endswith(".json"):
        sample_file = os.path.join(analysis_dir, file)
        break

if not sample_file:
    raise FileNotFoundError("No JSON file found in data/analysis")

# Load the JSON file into a list of dictionaries
reviews = []
with open(sample_file, "r", encoding="utf-8") as f:
    for line in f:
        reviews.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(reviews)
print(f"Loaded {len(df)} reviews from {sample_file}")

# --- Test only the word cloud creation ---
def test_wordcloud(df, output_dir='data/visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot')

    # 6. Word Cloud of Keywords
    if 'keywords' in df.columns:
        flat_keywords = list(chain.from_iterable(df['keywords'].dropna()))
        all_keywords = ' '.join(flat_keywords)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Keywords')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keyword_wordcloud.png'))
        plt.close()
        print("Word cloud created successfully.")
    else:
        print("No 'keywords' column found in the DataFrame.")

# Call the test function
test_wordcloud(df) 