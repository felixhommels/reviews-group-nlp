import json
from pathlib import Path

RESULTS_DIR = Path("data/results")
KEYWORDS_DIR = Path("data/keywords")

# Mapping from source to actual keywords file (based on your folder contents)
KEYWORDS_FILE_MAP = {
    "playstore": "keywords_com.whatsapp.json",
    "imdb": "keywords_howToTrainYourDragon.json",
    "steam": "keywords_570_steam.json",
    "trustpilot": "keywords_bancosantander.json",
}

# Find all analyzed review files
analyzed_files = list(RESULTS_DIR.glob("analyzed_*_reviews.json"))

for analyzed_path in analyzed_files:
    # Extract the <source> part
    source = analyzed_path.stem.replace("analyzed_", "").replace("_reviews", "")
    keywords_filename = KEYWORDS_FILE_MAP.get(source)
    if not keywords_filename:
        print(f"❌ No keywords file mapping for source: {source}")
        continue
    keywords_path = KEYWORDS_DIR / keywords_filename
    output_path = RESULTS_DIR / f"final_{source}_reviews.json"

    if not keywords_path.exists():
        print(f"❌ Keywords file not found for {source}: {keywords_path}")
        continue

    print(f"🔄 Merging for source: {source}")

    # Load analyzed reviews
    with open(analyzed_path) as f:
        content = f.read().strip()
        if content.startswith("["):
            analyzed = json.loads(content)
        else:
            analyzed = [json.loads(line) for line in content.splitlines() if line.strip()]

    # Load keywords
    with open(keywords_path) as f:
        keyword_data = [json.loads(line)["keywords"] for line in f]

    # Merge keywords into each review
    for i, review in enumerate(analyzed):
        review["keywords"] = keyword_data[i] if i < len(keyword_data) else []

    # Save merged output
    with open(output_path, "w") as f:
        json.dump(analyzed, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged reviews with keywords saved to {output_path}")
