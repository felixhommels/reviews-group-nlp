from src.analysis.emotion_analysis import EnglishEmotionAnalyzerHartmann, SpanishEmotionAnalyzerRobertuito
from langdetect import detect, LangDetectException
from pathlib import Path
import json
import os

input_dir = "data/processed_test_results"
output_dir = "data/emotions"
Path(output_dir).mkdir(parents=True, exist_ok=True)

english_analyzer = EnglishEmotionAnalyzerHartmann()
spanish_analyzer = SpanishEmotionAnalyzerRobertuito()

for file in os.listdir(input_dir):
    if file.endswith(".json"):
        source = Path(file).stem.replace("processed_", "").replace("_reviews", "")
        with open(os.path.join(input_dir, file)) as f:
            # Handles both array and line-delimited JSON
            content = f.read().strip()
            if content.startswith("["):
                reviews = json.loads(content)
            else:
                reviews = [json.loads(line) for line in content.splitlines() if line.strip()]

        results = []
        for r in reviews:
            text = r.get("processed_text", "")
            if not text.strip():
                # Optionally log or count skipped reviews
                continue
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "unknown"
            if lang == "en":
                emotions = english_analyzer.analyze_emotion(text)
            elif lang == "es":
                emotions = spanish_analyzer.analyze_emotion(text)
            else:
                continue  # skip other languages for now
            top_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            results.append({
                "top_emotion": top_emotion,
                "emotion_scores": emotions
            })

        with open(f"{output_dir}/emotions_{source}.json", "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

print("✅ All emotions extracted.")
