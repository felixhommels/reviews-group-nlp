# Project Architecture

This document describes the high-level architecture of the reviews-group-nlp project, including its main modules, data flow, and design principles.

## Overview

The project is designed to scrape, preprocess, analyze, and visualize user reviews from various sources. It is modular, with each major step separated into its own component for maintainability and collaboration.

## Main Components

- **src/scraper/**: Contains modules for scraping reviews from different sources (e.g., Trustpilot).
- **src/preprocessing/**: Handles text preprocessing (e.g., tokenization, cleaning) using spaCy.
- **src/analysis/**: Performs NLP analysis, including sentiment, emotion, keyword extraction, and star rating prediction.
- **src/summarizer/**: (Optional) For summarizing reviews or analysis results.
- **src/utils/**: Utility functions for file handling, dependencies, etc.
- **data/raw/**: Stores raw scraped data in JSON format.
- **visualizations/**: Stores generated plots and visualizations.
- **notebooks/**: For Jupyter notebooks, experiments, and demos.
- **tests/**: Unit and integration tests for all modules.

## Data Flow

1. **Scraping**: Use `src/scraper/url_scraper.py` or `src/scraping/scraper.py` to collect reviews and save them as JSON in `data/raw/`.
2. **Preprocessing**: Use `src/preprocessing/spacy_preprocessor.py` to clean and tokenize the reviews.
3. **Analysis**: Use `src/analysis/nlp_analysis.py` (and related modules) to analyze the processed reviews.
4. **Visualization**: Use `visualize_results.py` to generate plots from the analysis results.

## Entry Points

- `run_nlp_analysis.py`: End-to-end pipeline for scraping, preprocessing, and analysis.
- `visualize_results.py`: For generating visualizations from analysis results.
- `run_tests.py`: For running all tests.

## Design Principles

- **Modularity**: Each step is a separate module for clarity and reusability.
- **Extensibility**: Easy to add new scrapers, preprocessors, or analysis methods.
- **Documentation**: Each module and script is documented for team collaboration.
- **Testing**: Unit and integration tests ensure reliability.

## Team Collaboration

- Use the `archive/` folders for legacy or experimental scripts.
- Keep personal scripts outside the main repo (e.g., `~/code-sandbox/`).
- Update documentation as modules or workflows change.

---

For more details on usage, see `usage_examples.md` in this folder.
