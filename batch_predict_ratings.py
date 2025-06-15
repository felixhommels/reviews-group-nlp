from src.analysis.star_rating_predictor import StarRatingPredictor
from pathlib import Path
import json
import os

input_dir = "data/processed_test_results"
output_dir = "data/ratings"
Path(output_dir).mkdir(parents=True, exist_ok=True)

predictor = StarRatingPredictor()

for file in os.listdir(input_dir):
    if file.endswith(".json"):
        source = Path(file).stem.replace("processed_", "").replace("_reviews", "")
        with open(os.path.join(input_dir, file)) as f:
            content = f.read().strip()
            if content.startswith("["):
                reviews = json.loads(content)
            else:
                reviews = [json.loads(line) for line in content.splitlines() if line.strip()]

        results = []
        for r in reviews:
            text = r.get("processed_text", "")
            if not text.strip():
                continue
            predicted_raw = predictor.predict_star_rating(text, source=source)
            predicted_normalized = predictor.normalize_rating(predicted_raw, source)
            results.append({
                "predicted_rating_raw": predicted_raw,
                "predicted_rating_normalized": predicted_normalized,
                "source": source
            })

        with open(f"{output_dir}/ratings_{source}.json", "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

print("✅ All star ratings predicted.")
