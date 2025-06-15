import json
from pathlib import Path

# Set your source (e.g., "steam", "playstore", etc.)
SOURCE = "steam"
ANALYZED_PATH = f"data/results/analyzed_{SOURCE}_reviews.json"
KEYWORDS_PATH = f"data/keywords/keywords_{SOURCE}.json"
OUTPUT_PATH = f"data/results/final_{SOURCE}_reviews.json"

# Load analyzed reviews
with open(ANALYZED_PATH) as f:
    analyzed = [json.loads(line) for line in f]

# Load keywords
with open(KEYWORDS_PATH) as f:
    keyword_data = [json.loads(line)["keywords"] for line in f]

# Merge keywords into each review
for i, review in enumerate(analyzed):
    review["keywords"] = keyword_data[i] if i < len(keyword_data) else []

# Save merged output
with open(OUTPUT_PATH, "w") as f:
    for review in analyzed:
        f.write(json.dumps(review) + "\n")

print(f"✅ Merged reviews with keywords saved to {OUTPUT_PATH}")
