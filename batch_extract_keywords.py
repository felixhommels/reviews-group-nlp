import os
from pathlib import Path
import subprocess

# Configuration
INPUT_DIR = "data/processed_test_results"
OUTPUT_DIR = "data/keywords"
TEXT_COLUMN = "processed_text"
LANGUAGE = "en"
LIMIT = None  # set to an integer like 100 for testing

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# List all JSON files in the input directory
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

# Loop through each file and run extract_keywords.py
for file in files:
    input_path = os.path.join(INPUT_DIR, file)
    filename_core = Path(file).stem.replace("processed_", "").replace("_reviews", "")
    output_path = os.path.join(OUTPUT_DIR, f"keywords_{filename_core}.json")

    print(f"\n🔍 Extracting keywords for {file} → {output_path}")

    command = [
        "python", "extract_keywords.py",
        "--input", input_path,
        "--text_column", TEXT_COLUMN,
        "--language", LANGUAGE,
        "--output_dir", OUTPUT_DIR
    ]
    if LIMIT:
        command += ["--limit", str(LIMIT)]

    subprocess.run(command)
