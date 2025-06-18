# Review Analysis NLP Project

This project is designed to scrape, preprocess, and analyze reviews from various sources (IMDB, Trustpilot) using Natural Language Processing (NLP) techniques.

## How to Run

1. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPEN_AI_KEY=your_api_key_here
   ```

2. Set up the Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Note: If you encounter any issues, try closing Visual Studio Code completely and running the virtual environment setup again.

3. Download required language models:
   ```bash
   python src/preprocessing/download_models.py
   ```
   This will download:
   - spaCy language models for:
     - English (en_core_web_sm)
     - Spanish (es_core_news_sm)
     These models are essential for text processing and language detection.
   - NLTK resources:
     - punkt (for sentence tokenization)
     - stopwords (for removing common words)
     - wordnet (for word lemmatization)
     - averaged_perceptron_tagger (for part-of-speech tagging)
     These resources are used for text analysis and summarization.

4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   If you're new to Streamlit, you'll be prompted to enter your email address on first run.

5. The app will open in your default browser at `http://localhost:8501`

6. Using the app:
   - Enter a review URL from supported platforms (IMDb, Trustpilot, Steam, Google Play Store)
   - Choose how many reviews to analyze (1-100)
   - Click "Analyze" and wait for the results
   - Results will include sentiment analysis, key points summary, and detailed review breakdown

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

### Analysis Functions

The `src/analysis` directory contains several modules for analyzing reviews:

1. **Sentiment Analysis**
   ```python
   from src.analysis.sentiment_analysis import SentimentAnalyzer
   
   # Initialize the analyzer
   analyzer = SentimentAnalyzer()
   
   # Analyze sentiment of a text
   result = analyzer.analyze_sentiment("This movie was absolutely amazing!")
   # Returns: {
   #     'sentiment_label': 'positive',  # Overall sentiment: 'positive', 'negative', or 'neutral'
   #     'sentiment_score': 0.8,         # Score between -1 (very negative) and 1 (very positive)
   #     'confidence': 0.85,             # Confidence score between 0 and 1
   #     'polarity': 0.8,                # Raw polarity score
   #     'subjectivity': 0.6,            # Subjectivity score between 0 (objective) and 1 (subjective)
   #     'analysis_method': 'transformer' # Method used for analysis: 'transformer', 'vader', or 'textblob'
   # }
   ```

2. **OpenAI Summary Analysis**
   ```python
   from src.analysis.openai_summary import get_likes_and_dislikes
   
   # Analyze multiple reviews to extract top likes and dislikes
   reviews = [
       "The graphics were amazing but the story was weak.",
       "Great visuals, poor plot development.",
       "Stunning effects, though the narrative needs work."
   ]
   
   # Get the top 3 liked and disliked aspects
   top_likes, top_dislikes = get_likes_and_dislikes(reviews)
   # Returns two lists:
   # top_likes = ["Amazing graphics", "Stunning visual effects", "Great visuals"]
   # top_dislikes = ["Weak story", "Poor plot development", "Narrative needs work"]
   
   # You can also specify a different OpenAI model
   top_likes, top_dislikes = get_likes_and_dislikes(reviews, model="gpt-4")
   ```
   Note: Requires OpenAI API key in `.env` file (see setup instructions above)

3. **Emotion Analysis**
   ```python
   from src.analysis.emotion_analysis import EnglishEmotionAnalyzerHartmann
   
   # Initialize the analyzer
   analyzer = EnglishEmotionAnalyzerHartmann()
   
   # Analyze emotions in text
   emotions = analyzer.analyze_emotion("I was so excited and happy to watch this film!")
   # Returns: {'joy': 0.8, 'anticipation': 0.6, ...}
   ```

4. **Keyword Extraction**
   ```python
   from src.analysis.keyword_extraction import KeywordExtractor
   
   # Initialize the extractor
   extractor = KeywordExtractor()
   
   # Extract keywords from text
   keywords = extractor.extract_keywords("The cinematography and acting were outstanding")
   # Returns: ['cinematography', 'acting', 'outstanding']
   ```

5. **Star Rating Prediction**
   ```python
   from src.analysis.star_rating_predictor import StarRatingPredictor
   
   # Initialize the predictor
   predictor = StarRatingPredictor(language='en', source='trustpilot')
   
   # Predict star rating from review text
   rating = predictor.predict_star_rating("This was the best movie I've ever seen!")
   # Returns: An integer from 1-5
   # Note: The actual return is a direct star rating (1-5)
   # The normalization happens separately if needed:
   normalized_rating = StarRatingPredictor.normalize_rating(rating, source='imdb')
   ```

6. **NLP Analysis**
   ```python
   # The pipeline combines all analyzers for comprehensive review analysis
   result = run_full_nlp_pipeline(
       review={
           "processed_text": "The movie was fantastic! Great acting and plot.",
           "source": "imdb"
       },
       sentiment_analyzer=sentiment_analyzer,
       keyword_extractor=keyword_extractor,
       english_emotion_analyzer=english_emotion_analyzer,
       spanish_emotion_analyzer=spanish_emotion_analyzer,
       rating_predictor=rating_predictor
   )
   
   # Returns: {
   #     # Original review data
   #     "processed_text": "The movie was fantastic! Great acting and plot.",
   #     "source": "imdb",
   #     
   #     # Sentiment analysis results
   #     "sentiment_label": "positive",
   #     "sentiment_score": 0.8,
   #     ...
   #     
   #     # Extracted keywords
   #     "keywords": ["fantastic", "acting", "plot"],
   #     
   #     # Emotion analysis
   #     "top_emotion": "joy",
   #     "emotion_scores": {"joy": 0.8, "anticipation": 0.6, ...},
   #     
   #     # Rating prediction
   #     "predicted_rating_raw": 5,
   #     "predicted_rating_normalized": 5.0
   # }
   ```

7. **TextRank Summarization**
   ```python
   from src.summarizer.text_rank_summarizer import TextRankSummarizer
   
   # Initialize the summarizer
   summarizer = TextRankSummarizer()
   
   # Example text with multiple sentences
   text = """
   The new iPhone camera system is revolutionary. The photos are incredibly sharp 
   and detailed in any lighting condition. The battery life has been significantly 
   improved from previous models. The user interface remains intuitive and smooth. 
   However, the price point is quite high for many consumers. The charging speed 
   could also be faster compared to competitors.
   """
   
   # Get a summary with the 3 most important sentences (default)
   summary = summarizer.summarize(text)
   # Returns:
   # • The new iPhone camera system is revolutionary.
   # ----------------------------------------
   # • The battery life has been significantly improved from previous models.
   # ----------------------------------------
   # • However, the price point is quite high for many consumers.
   
   # You can also specify the number of sentences
   summary = summarizer.summarize(text, num_sentences=2)
   ```




