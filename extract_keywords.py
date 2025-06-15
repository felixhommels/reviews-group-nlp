import argparse
import pandas as pd
from src.analysis.keyword_extraction import KeywordExtractor
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract keywords from reviews using KeyBERT+MiniLM.")
    parser.add_argument("--input", required=True, help="Path to input JSON or CSV file with reviews.")
    parser.add_argument("--text_column", default="processed_text", help="Column containing review text.")
    parser.add_argument("--language", default="en", help="Language code (default: en).")
    parser.add_argument(
        "--output_dir",
        default="data/keywords",
        help="Directory to save the output file. Defaults to data/keywords/"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N reviews (for speed/debugging)")
    args = parser.parse_args()

    # Load data
    if args.input.endswith(".json"):
        df = pd.read_json(args.input)
    else:
        df = pd.read_csv(args.input)

    if args.limit:
        df = df.head(args.limit)

    extractor = KeywordExtractor(language=args.language)
    texts = df[args.text_column].astype(str).tolist()
    df["keywords"] = extractor.extract_keywords_batch(texts, language=args.language)

    # Derive output file name from input
    input_name = Path(args.input).stem.replace("processed_", "").replace("_reviews", "")
    output_path = Path(args.output_dir) / f"keywords_{input_name}.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df.to_json(output_path, orient="records", lines=True)
    print(f"✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()
