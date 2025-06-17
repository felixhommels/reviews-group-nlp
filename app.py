# app.py
import os
import logging
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.scraper.url_scraper import scrape_google_playstore, scrape_imbd, scrape_steam, scrape_trustpilot
from src.preprocessing.spacy_preprocessor import preprocess_pipeline
from src.analysis.nlp_analysis import run_full_nlp_pipeline
from src.utils.file_utils import save_json
from src.utils.dependencies import dependency_manager
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.analysis.keyword_extraction import KeywordExtractor
from src.analysis.emotion_analysis import EnglishEmotionAnalyzerHartmann, SpanishEmotionAnalyzerRobertuito
from src.analysis.star_rating_predictor import StarRatingPredictor

# Import summarizer if available
try:
    from src.summarizer.summarizer import ReviewSummarizer
    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    logging.warning("Summarizer module not available")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Custom exception for pipeline-specific errors."""
    pass

def setup_directories() -> Dict[str, Path]:
    """Create and return required directory paths."""
    dirs = {
        'raw': Path('data/raw'),
        'processed': Path('data/processed'),
        'results': Path('data/results'),
        'summaries': Path('data/summaries'),
        'analysis': Path('data/analysis')
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    return dirs

def show_platform_help():
    """Show help for finding search terms on different platforms."""
    print("\n" + "="*60)
    print("🔍 HOW TO FIND SEARCH TERMS")
    print("="*60)
    
    print("\n🎬 IMDb (Movies/TV Shows):")
    print("   1. Go to imdb.com")
    print("   2. Search for your movie/show")
    print("   3. Copy the title ID from URL (starts with 'tt')")
    print("   Examples:")
    print("   • The Dark Knight: tt0468569")
    print("   • Avengers Endgame: tt4154796")
    print("   • Breaking Bad: tt0903747")
    
    print("\n🎮 Steam (PC Games):")
    print("   1. Go to store.steampowered.com")
    print("   2. Search for your game")
    print("   3. Copy the App ID from URL")
    print("   Examples:")
    print("   • Dota 2: 570")
    print("   • CS:GO: 730")
    print("   • GTA V: 271590")
    
    print("\n📱 Google Play Store (Android Apps):")
    print("   1. Go to play.google.com")
    print("   2. Search for your app")
    print("   3. Copy package name from URL")
    print("   Examples:")
    print("   • WhatsApp: com.whatsapp")
    print("   • Instagram: com.instagram.android")
    print("   • TikTok: com.zhiliaoapp.musically")
    
    print("\n🏢 Trustpilot (Businesses):")
    print("   1. Enter the company's website domain")
    print("   Examples:")
    print("   • Amazon: amazon.com")
    print("   • Netflix: netflix.com")
    print("   • Uber: uber.com")

def get_popular_presets() -> Dict[str, List[Dict[str, Any]]]:
    """Get popular preset options for quick selection."""
    return {
        "movies": [
            {"name": "The Dark Knight", "source": "imdb", "id": "tt0468569"},
            {"name": "Avengers Endgame", "source": "imdb", "id": "tt4154796"},
            {"name": "Inception", "source": "imdb", "id": "tt1375666"},
            {"name": "The Shawshank Redemption", "source": "imdb", "id": "tt0111161"},
            {"name": "Pulp Fiction", "source": "imdb", "id": "tt0110912"}
        ],
        "games": [
            {"name": "Dota 2", "source": "steam", "id": "570"},
            {"name": "Counter-Strike: Global Offensive", "source": "steam", "id": "730"},
            {"name": "Grand Theft Auto V", "source": "steam", "id": "271590"},
            {"name": "Among Us", "source": "steam", "id": "945360"},
            {"name": "Fall Guys", "source": "steam", "id": "1097150"}
        ],
        "apps": [
            {"name": "WhatsApp", "source": "playstore", "id": "com.whatsapp"},
            {"name": "Instagram", "source": "playstore", "id": "com.instagram.android"},
            {"name": "TikTok", "source": "playstore", "id": "com.zhiliaoapp.musically"},
            {"name": "Netflix", "source": "playstore", "id": "com.netflix.mediaclient"},
            {"name": "Spotify", "source": "playstore", "id": "com.spotify.music"}
        ]
    }

def build_source_config(source_type: str, search_term: str, name: str, **kwargs) -> Dict[str, Any]:
    """Build a source configuration based on the platform type."""
    if source_type == "imdb":
        # Handle both full URLs and title IDs
        if search_term.startswith("http"):
            url = search_term
            topic = name.lower().replace(" ", "_")
        else:
            # Ensure title ID starts with 'tt'
            title_id = search_term if search_term.startswith("tt") else f"tt{search_term}"
            url = f"https://www.imdb.com/title/{title_id}/reviews/"
            topic = name.lower().replace(" ", "_")
        
        return {
            "source": "imdb",
            "url": url,
            "topic": topic,
            "max_pages": kwargs.get("max_pages", 5)
        }
    
    elif source_type == "steam":
        return {
            "source": "steam",
            "app_id": search_term,
            "max_reviews": kwargs.get("max_reviews", 200)
        }
    
    elif source_type == "playstore":
        return {
            "source": "playstore",
            "app_id": search_term,
            "max_reviews": kwargs.get("max_reviews", 300)
        }
    
    elif source_type == "trustpilot":
        # Handle both full URLs and domains
        if search_term.startswith("http"):
            url = search_term
            topic = name.lower().replace(" ", "_")
        else:
            url = f"https://trustpilot.com/review/{search_term}"
            topic = search_term.replace(".", "_")
        
        return {
            "source": "trustpilot",
            "url": url,
            "topic": topic,
            "max_pages": kwargs.get("max_pages", 10)
        }
    
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

def interactive_source_builder() -> List[Dict[str, Any]]:
    """Interactively build sources list with user input."""
    sources = []
    
    print("\n" + "="*60)
    print("🚀 DYNAMIC REVIEW ANALYSIS")
    print("="*60)
    print("Build your analysis by adding sources for any content you want!")
    
    while True:
        print(f"\n📋 Current sources: {len(sources)}")
        for i, source in enumerate(sources, 1):
            name = source.get('topic', source.get('app_id', 'Unknown'))
            platform = source['source']
            print(f"   {i}. {name} ({platform})")
        
        print("\nOptions:")
        print("1. 🎬 Add movie/TV show (IMDb)")
        print("2. 🎮 Add game (Steam)")
        print("3. 📱 Add app (Google Play)")
        print("4. 🏢 Add business (Trustpilot)")
        print("5. 🎯 Use popular presets")
        print("6. ❓ Show search help")
        print("7. ✅ Start analysis")
        print("8. 🚪 Exit")
        
        try:
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':  # IMDb
                add_imdb_source(sources)
            elif choice == '2':  # Steam
                add_steam_source(sources)
            elif choice == '3':  # Google Play
                add_playstore_source(sources)
            elif choice == '4':  # Trustpilot
                add_trustpilot_source(sources)
            elif choice == '5':  # Presets
                add_preset_sources(sources)
            elif choice == '6':  # Help
                show_platform_help()
            elif choice == '7':  # Start analysis
                if sources:
                    return sources
                else:
                    print("❌ Please add at least one source before starting analysis!")
            elif choice == '8':  # Exit
                print("👋 Goodbye!")
                return []
            else:
                print("❌ Invalid option. Please try again.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return []

def add_imdb_source(sources: List[Dict[str, Any]]) -> None:
    """Add an IMDb source interactively."""
    print("\n🎬 Adding IMDb Movie/TV Show")
    print("-" * 30)
    
    name = input("Enter movie/show name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    title_id = input("Enter IMDb title ID (e.g., tt0468569) or full URL: ").strip()
    if not title_id:
        print("❌ Title ID cannot be empty!")
        return
    
    max_pages = input("Max pages to scrape (default: 5): ").strip()
    max_pages = int(max_pages) if max_pages.isdigit() else 5
    
    try:
        source_config = build_source_config("imdb", title_id, name, max_pages=max_pages)
        sources.append(source_config)
        print(f"✅ Added: {name}")
    except Exception as e:
        print(f"❌ Error: {e}")

def add_steam_source(sources: List[Dict[str, Any]]) -> None:
    """Add a Steam source interactively."""
    print("\n🎮 Adding Steam Game")
    print("-" * 20)
    
    name = input("Enter game name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    app_id = input("Enter Steam App ID (e.g., 570): ").strip()
    if not app_id.isdigit():
        print("❌ App ID must be numeric!")
        return
    
    max_reviews = input("Max reviews to scrape (default: 200): ").strip()
    max_reviews = int(max_reviews) if max_reviews.isdigit() else 200
    
    try:
        source_config = build_source_config("steam", app_id, name, max_reviews=max_reviews)
        sources.append(source_config)
        print(f"✅ Added: {name}")
    except Exception as e:
        print(f"❌ Error: {e}")

def add_playstore_source(sources: List[Dict[str, Any]]) -> None:
    """Add a Google Play Store source interactively."""
    print("\n📱 Adding Google Play Store App")
    print("-" * 30)
    
    name = input("Enter app name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    package_name = input("Enter package name (e.g., com.whatsapp): ").strip()
    if not package_name or "." not in package_name:
        print("❌ Package name must contain dots (e.g., com.example.app)!")
        return
    
    max_reviews = input("Max reviews to scrape (default: 300): ").strip()
    max_reviews = int(max_reviews) if max_reviews.isdigit() else 300
    
    try:
        source_config = build_source_config("playstore", package_name, name, max_reviews=max_reviews)
        sources.append(source_config)
        print(f"✅ Added: {name}")
    except Exception as e:
        print(f"❌ Error: {e}")

def add_trustpilot_source(sources: List[Dict[str, Any]]) -> None:
    """Add a Trustpilot source interactively."""
    print("\n🏢 Adding Trustpilot Business")
    print("-" * 25)
    
    name = input("Enter business name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    domain = input("Enter company domain (e.g., amazon.com): ").strip()
    if not domain:
        print("❌ Domain cannot be empty!")
        return
    
    max_pages = input("Max pages to scrape (default: 10): ").strip()
    max_pages = int(max_pages) if max_pages.isdigit() else 10
    
    try:
        source_config = build_source_config("trustpilot", domain, name, max_pages=max_pages)
        sources.append(source_config)
        print(f"✅ Added: {name}")
    except Exception as e:
        print(f"❌ Error: {e}")

def add_preset_sources(sources: List[Dict[str, Any]]) -> None:
    """Add sources from popular presets."""
    presets = get_popular_presets()
    
    print("\n🎯 Popular Presets")
    print("-" * 20)
    print("1. 🎬 Popular Movies")
    print("2. 🎮 Popular Games") 
    print("3. 📱 Popular Apps")
    
    choice = input("Select category (1-3): ").strip()
    
    if choice == '1':
        category = "movies"
        print("\n🎬 Popular Movies:")
    elif choice == '2':
        category = "games"
        print("\n🎮 Popular Games:")
    elif choice == '3':
        category = "apps"
        print("\n📱 Popular Apps:")
    else:
        print("❌ Invalid choice!")
        return
    
    items = presets[category]
    for i, item in enumerate(items, 1):
        print(f"{i}. {item['name']}")
    
    selections = input(f"\nSelect items (e.g., 1,3,5 or 'all'): ").strip()
    
    if selections.lower() == 'all':
        selected_items = items
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selections.split(',')]
            selected_items = [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print("❌ Invalid selection!")
            return
    
    for item in selected_items:
        try:
            if item['source'] == 'imdb':
                source_config = build_source_config("imdb", item['id'], item['name'])
            elif item['source'] == 'steam':
                source_config = build_source_config("steam", item['id'], item['name'])
            elif item['source'] == 'playstore':
                source_config = build_source_config("playstore", item['id'], item['name'])
            
            sources.append(source_config)
            print(f"✅ Added: {item['name']}")
        except Exception as e:
            print(f"❌ Error adding {item['name']}: {e}")

def get_default_sources() -> List[Dict[str, Any]]:
    """Get default sources (your original hard-coded ones) if user wants to use them."""
    return [
        {
            "source": "trustpilot",
            "url": "https://example.com/trustpilot",
            "topic": "bancosantander",
            "max_pages": 10
        },
        {
            "source": "imdb",
            "url": "https://www.imdb.com/title/tt0892769/reviews/?ref_=tt_ov_ururv",
            "topic": "howToTrainYourDragon",
            "max_pages": 2
        },
        {
            "source": "playstore",
            "app_id": "com.whatsapp",
            "max_reviews": 300
        },
        {
            "source": "steam",
            "app_id": "570",
            "max_reviews": 300
        }
    ]

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that DataFrame has required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise PipelineError(f"Missing required columns: {missing}")
    return True

def run_pipeline_test():
    """
    Finds all raw review files, runs them through the preprocessing pipeline,
    and saves the results in a dedicated test folder.
    """
    logging.info("--- Starting Preprocessing Pipeline Test ---")

    # --- Configuration ---
    raw_data_dir = "data/raw"
    output_dir = "data/processed_test_results"
    
    # Find all JSON files in the raw data directory
    raw_files = glob.glob(os.path.join(raw_data_dir, '*.json'))

    if not raw_files:
        logging.warning(f"No raw data files found in '{raw_data_dir}'. Nothing to test.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: '{output_dir}'")

    # --- Processing Loop ---
    for file_path in raw_files:
        filename = os.path.basename(file_path)
        logging.info(f"--- Processing file: {filename} ---")

        try:
            # Load the raw reviews from the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_reviews = json.load(f)
            
            if not raw_reviews:
                logging.warning(f"File '{filename}' is empty. Skipping.")
                continue

            # Convert to DataFrame
            reviews_df = pd.DataFrame(raw_reviews)
            
            # The pipeline expects a 'review' column. Rename 'text' if it exists.
            if 'text' in reviews_df.columns:
                reviews_df.rename(columns={'text': 'review'}, inplace=True)
            elif 'review' not in reviews_df.columns:
                logging.error(f"Could not find 'text' or 'review' column in '{filename}'. Skipping.")
                continue

            # Run the preprocessing pipeline
            logging.info(f"Running preprocessing for '{filename}'...")
            processed_df = preprocess_pipeline(reviews_df, text_column='review')

            # Convert DataFrame back to list of dictionaries for saving
            processed_reviews = processed_df.to_dict(orient='records')

            # Save the processed data
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            save_json(processed_reviews, output_path)
            logging.info(f"Successfully processed and saved results to '{output_path}'")

        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from '{filename}'. It might be corrupted. Skipping.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing '{filename}': {e}", exc_info=True)

    logging.info("--- Preprocessing Pipeline Test Finished ---")

def run_nlp_analysis_on_all_processed():
    input_dir = "data/processed_test_results"
    output_dir = "data/analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sentiment_analyzer = SentimentAnalyzer()
    keyword_extractor = KeywordExtractor()
    english_emotion_analyzer = EnglishEmotionAnalyzerHartmann()
    spanish_emotion_analyzer = SpanishEmotionAnalyzerRobertuito()
    rating_predictor = StarRatingPredictor()

    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        source = Path(file).stem.replace("processed_", "").replace("_reviews", "")
        input_path = os.path.join(input_dir, file)
        with open(input_path) as f:
            content = f.read().strip()
            if content.startswith("["):
                reviews = json.loads(content)
            else:
                reviews = [json.loads(line) for line in content.splitlines() if line.strip()]

        enriched_reviews = []
        for review in reviews:
            if not review.get("processed_text", "").strip():
                continue
            enriched = run_full_nlp_pipeline(
                review,
                sentiment_analyzer,
                keyword_extractor,
                english_emotion_analyzer,
                spanish_emotion_analyzer,
                rating_predictor
            )
            enriched_reviews.append(enriched)

        output_path = os.path.join(output_dir, f"review_analysis_{source}.json")
        with open(output_path, "w") as f:
            for r in enriched_reviews:
                f.write(json.dumps(r) + "\n")

    print("✅ All sources processed. Output saved to data/analysis/")

def generate_summaries():
    """Generate summaries for all analyzed data using the summarizer module."""
    if not SUMMARIZER_AVAILABLE:
        print("⚠️  Summarizer module not available. Skipping summary generation.")
        return
    
    print("📊 Generating summaries...")
    
    summarizer = ReviewSummarizer()
    analysis_dir = "data/analysis"
    summaries_dir = "data/summaries"
    
    Path(summaries_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each analysis file
    for file in os.listdir(analysis_dir):
        if file.endswith(".json"):
            print(f"📄 Processing {file}...")
            
            # Load analysis data
            analysis_path = os.path.join(analysis_dir, file)
            reviews_data = []
            
            with open(analysis_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        reviews_data.append(json.loads(line))
            
            if not reviews_data:
                continue
            
            # Convert to DataFrame for summarizer
            df = pd.DataFrame(reviews_data)
            
            # Fix column names for summarizer compatibility
            if 'sentiment_label' in df.columns:
                df['sentiment'] = df['sentiment_label']
            if 'sentiment_score' in df.columns:
                df['sentiment_polarity'] = df['sentiment_score']
            if 'predicted_rating_normalized' in df.columns:
                df['predicted_stars'] = df['predicted_rating_normalized']
            
            # Generate summary
            try:
                summary_report = summarizer.generate_summary_report(df)
                
                # Save summary
                source_name = file.replace("review_analysis_", "").replace(".json", "")
                summary_path = os.path.join(summaries_dir, f"summary_{source_name}.json")
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"✅ Summary saved: summary_{source_name}.json")
                
                # Print key insights
                summary_text = summary_report.get('summary_text', '')
                if summary_text:
                    print(f"📝 Key insights for {source_name}:")
                    print(f"   {summary_text[:150]}...")
                
            except Exception as e:
                print(f"❌ Error generating summary for {source_name}: {e}")
    
    print("📊 Summary generation completed!")

def create_visualizations(df, output_dir='data/visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot')
    sns.set_theme()

    # 1. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_label'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Distribution of Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()

    # 2. Sentiment Score Distribution
    if 'sentiment_score' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='sentiment_score', bins=30)
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_score_distribution.png'))
        plt.close()

    # 3. Star Ratings Distribution
    if 'predicted_rating_raw' in df.columns:
        plt.figure(figsize=(10, 6))
        star_counts = df['predicted_rating_raw'].value_counts().sort_index()
        sns.barplot(x=star_counts.index, y=star_counts.values)
        plt.title('Distribution of Predicted Star Ratings')
        plt.xlabel('Stars')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'star_ratings_distribution.png'))
        plt.close()

    # 4. Primary Emotions
    if 'top_emotion' in df.columns:
        plt.figure(figsize=(12, 6))
        emotion_counts = df['top_emotion'].value_counts()
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Distribution of Primary Emotions')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
        plt.close()

    # 5. Correlation between Sentiment Score and Star Ratings
    if 'sentiment_score' in df.columns and 'predicted_rating_raw' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sentiment_score', y='predicted_rating_raw')
        plt.title('Correlation between Sentiment Score and Star Ratings')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Predicted Stars')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_stars_correlation.png'))
        plt.close()

    # 6. Word Cloud of Keywords
    if 'keywords' in df.columns:
        try:
            from wordcloud import WordCloud
            all_keywords = ' '.join(df['keywords'].astype(str).str.join(' '))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Keywords')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'keyword_wordcloud.png'))
            plt.close()
        except ImportError:
            print("⚠️  WordCloud not available. Skipping word cloud generation.")

def main():
    """Main execution function with dynamic source selection."""
    setup_directories()
    logging.info("--- Starting the Review Analysis Pipeline ---")
    
    print("🎉 Welcome to Dynamic Review Analysis!")
    print("You can now analyze ANY movie, game, app, or business!")
    
    # Give user options for source selection
    print("\nHow would you like to proceed?")
    print("1. 🔧 Use interactive mode (build custom analysis)")
    print("2. 📋 Use default sources (original hard-coded ones)")
    print("3. 🗂️  Process existing files in data/raw/")
    print("4. 🚪 Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        # Interactive mode
        SOURCES = interactive_source_builder()
        if not SOURCES:
            print("No sources selected. Exiting.")
            return
    elif choice == '2':
        # Default sources
        SOURCES = get_default_sources()
        print("Using default sources...")
    elif choice == '3':
        # Process existing files
        print("Processing existing files...")
        run_pipeline_test()
        run_nlp_analysis_on_all_processed()
        generate_summaries()
        
        # Load data for visualization
        analysis_dir = "data/analysis"
        all_reviews = []
        for file in os.listdir(analysis_dir):
            if file.endswith(".json"):
                with open(os.path.join(analysis_dir, file), "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_reviews.append(json.loads(line))
        
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            print(f"Loaded {len(df)} enriched reviews for visualization.")
            create_visualizations(df)
            print("✅ Visualizations created in 'data/visualizations' directory.")
        
        return
    elif choice == '4':
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Using default sources...")
        SOURCES = get_default_sources()

    # Process each source
    for cfg in SOURCES:
        source = cfg["source"]
        logging.info(f"Scraping source: {source}")

        # --- Scraping ---
        if source == "trustpilot":
            raw_reviews = scrape_trustpilot(
                url=cfg["url"],
                topic=cfg["topic"],
                max_pages=cfg.get("max_pages", 10)
            )
            output_file = f"data/raw/{cfg['topic']}_reviews.json"
        elif source == "imdb":
            raw_reviews = scrape_imbd(
                url=cfg["url"],
                topic=cfg["topic"],
                max_pages=cfg.get("max_pages", 10)
            )
            output_file = f"data/raw/{cfg['topic']}_reviews.json"
        elif source == "playstore":
            raw_reviews = scrape_google_playstore(
                app_id=cfg["app_id"],
                max_reviews=cfg.get("max_reviews", 100)
            )
            output_file = f"data/raw/{cfg['app_id']}_reviews.json"
        elif source == "steam":
            raw_reviews = scrape_steam(
                app_id=cfg["app_id"],
                max_reviews=cfg.get("max_reviews", 100)
            )
            output_file = f"data/raw/{cfg['app_id']}_steam_reviews.json"
        else:
            logging.error(f"Unknown source: {source}")
            continue

        # --- Preprocessing ---
        if not raw_reviews:
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    raw_reviews = json.load(f)
            else:
                logging.warning(f"No reviews found for {source}. Skipping.")
                continue

        reviews_df = pd.DataFrame(raw_reviews)
        if 'text' in reviews_df.columns:
            reviews_df.rename(columns={'text': 'review'}, inplace=True)
        elif 'review' not in reviews_df.columns:
            logging.error(f"Could not find a 'text' or 'review' column for {source}. Skipping.")
            continue

        processed_df = preprocess_pipeline(reviews_df, text_column='review')
        processed_reviews = processed_df.to_dict(orient='records')

        # Save processed reviews to processed_test_results
        processed_output = f"data/processed_test_results/processed_{cfg.get('app_id', cfg.get('topic', source))}_reviews.json"
        os.makedirs("data/processed_test_results", exist_ok=True)
        with open(processed_output, "w") as f:
            json.dump(processed_reviews, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved processed reviews for {source} to {processed_output}")

    # --- NLP Analysis on ALL processed files ---
    run_nlp_analysis_on_all_processed()

    # --- Generate Summaries ---
    generate_summaries()

    # --- Visualization ---
    # Aggregate all enriched reviews
    analysis_dir = "data/analysis"
    all_reviews = []
    for file in os.listdir(analysis_dir):
        if file.endswith(".json"):
            with open(os.path.join(analysis_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_reviews.append(json.loads(line))
    
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        print(f"Loaded {len(df)} enriched reviews for visualization.")
        create_visualizations(df)
        print("✅ Visualizations have been created in the 'data/visualizations' directory.")
    
    print("\n🎉 Analysis complete! Check your results:")
    print("📊 Summaries: data/summaries/")
    print("📈 Analysis: data/analysis/")
    print("📉 Visualizations: data/visualizations/")

if __name__ == '__main__':
    main()
