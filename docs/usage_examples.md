# Usage Examples

This document provides practical examples for running the main workflows in the reviews-group-nlp project.

## 1. End-to-End Analysis Pipeline

Run the full pipeline (scraping, preprocessing, analysis):

```bash
python run_nlp_analysis.py
```

- Scrapes reviews (if not already present in `data/raw/`).
- Preprocesses and analyzes reviews.
- Prints a sample of the analysis results.

## 2. Visualizing Results

Generate visualizations from analysis results:

```bash
python visualize_results.py
```

- Loads processed and analyzed data.
- Creates plots in the `visualizations/` folder.

## 3. Running Tests

Run all tests (unit and integration):

```bash
python run_tests.py
```

- Use `--unit` or `--integration` for specific test types.
- Use `--coverage` for a coverage report.

## 4. Adding a New Scraper

1. Create a new script in `src/scraper/` (e.g., `my_site_scraper.py`).
2. Implement a function or class to scrape reviews and save as JSON in `data/raw/`.
3. Update documentation if needed.

## 5. Preprocessing Data Manually

```python
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
import pandas as pd
import json

with open('data/raw/my_reviews.json', 'r', encoding='utf-8') as f:
    reviews = json.load(f)
df = pd.DataFrame(reviews)
processed_df = preprocess_pipeline(df, text_column="text")
```

## 6. Analyzing Reviews Manually

```python
from src.analysis.nlp_analysis import ReviewAnalyzer
analyzer = ReviewAnalyzer(language='es')
results = analyzer.analyze_reviews(processed_df, text_column="processed_text")
```

---

For more details on architecture, see `architecture.md` in this folder.
