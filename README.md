# Review Analysis NLP Project

This project is designed to scrape, preprocess, and analyze reviews from various sources (IMDB, Trustpilot) using Natural Language Processing (NLP) techniques.

## Project Structure

```
├── src/
│   ├── preprocessing/     # Text preprocessing modules
│   ├── scraping/         # Web scraping functionality
│   ├── analysis/         # Analysis and summarization
│   ├── utils/           # Utility functions
│   └── config/          # Configuration files
├── data/                # Data storage
├── tests/              # Test suite
└── notebooks/          # Jupyter notebooks for analysis
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required spaCy models:
   ```bash
   python src/preprocessing/download_models.py
   ```

## Available Sources and Functions

### IMDB Reviews
- **Function**: `scrape_imdb_reviews(movie_id, num_pages=1)`
- **Usage**:
  ```python
  from src.scraping.scraper import Scraper
  scraper = Scraper(config)
  reviews = scraper.scrape_imdb_reviews("tt0111161", num_pages=2)  # Example: The Shawshank Redemption
  ```
- **Features**:
  - Scrapes movie reviews with ratings
  - Extracts review text, rating, date, and helpful votes
  - Supports pagination for multiple pages of reviews

### Trustpilot Reviews
- **Function**: `scrape_trustpilot_reviews(company_url, num_pages=1)`
- **Usage**:
  ```python
  from src.scraping.scraper import Scraper
  scraper = Scraper(config)
  reviews = scraper.scrape_trustpilot_reviews("https://www.trustpilot.com/review/company.com", num_pages=2)
  ```
- **Features**:
  - Scrapes company reviews with star ratings
  - Extracts review text, rating, date, and review title
  - Handles verified purchase badges

### Preprocessing Functions

The `spacy_preprocessor.py` module provides several key functions for text preprocessing:

1. **Language Detection**
   ```python
   from src.preprocessing.spacy_preprocessor import detect_language
   
   # Detect language of a text
   lang = detect_language("This is an English text")
   # Returns: 'en'
   ```

2. **Text Cleaning**
   ```python
   from src.preprocessing.spacy_preprocessor import clean_text
   
   # Clean text (lowercase, remove URLs, special characters)
   cleaned = clean_text("Check out https://example.com! It's AWESOME!")
   # Returns: "check out it is awesome"
   ```

3. **Text Preprocessing**
   ```python
   from src.preprocessing.spacy_preprocessor import preprocess_text
   
   # Preprocess text with language-specific model
   processed = preprocess_text("The cats are running quickly", lang='en')
   # Returns: "cat run quick"
   ```

4. **Full Pipeline**
   ```python
   from src.preprocessing.spacy_preprocessor import preprocess_pipeline
   import pandas as pd
   
   # Create a DataFrame with reviews
   df = pd.DataFrame({
       'review': [
           "The movie was fantastic!",
           "La película fue terrible."
       ]
   })
   
   # Run the full preprocessing pipeline
   processed_df = preprocess_pipeline(df, text_column='review')
   
   # The processed DataFrame will have new columns:
   # - cleaned_text: Basic cleaning applied
   # - language: Detected language code
   # - processed_text: Fully processed text
   ```

#### Supported Languages
The preprocessor supports multiple languages:
- English (en)
- Spanish (es)

#### Features
- Automatic language detection
- Contraction expansion (e.g., "don't" → "do not")
- URL and HTML tag removal
- Special character cleaning
- Language-specific tokenization
- Lemmatization
- Stopword removal
- Progress bars for long processing tasks

#### Example Output
```python
# Input DataFrame
df = pd.DataFrame({
    'review': ["The movie was FANTASTIC! Check it out: https://imdb.com"]
})

# After preprocessing
processed_df = preprocess_pipeline(df, text_column='review')
print(processed_df)

# Output:
#             review                                  cleaned_text                language       processed_text
# 0  The movie was FANTASTIC! Check it out...      the movie was fantastic...        en           movie fantastic
```


## Notes
- Uses spaCy's small models for efficient processing
- Includes comprehensive logging for debugging
- Handles unknown languages gracefully
- Processes texts in batches by language for better performance
- Sentiment analysis requires training data with ratings

