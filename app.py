#!/usr/bin/env python3
"""
Simple Review URL Analyzer
Input: Single URL
Output: JSON with sentiment, keywords, confidence score, emotions, etc.
"""

import json
import logging
from urllib.parse import urlparse
from typing import Dict, Any, List
from langdetect import detect, LangDetectException

# Import only what you need from your existing modules
from src.scraper.url_scraper import scrape_trustpilot, scrape_imbd, scrape_steam, scrape_google_playstore
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.analysis.keyword_extraction import KeywordExtractor
from src.analysis.emotion_analysis import EnglishEmotionAnalyzerHartmann, SpanishEmotionAnalyzerRobertuito
from src.analysis.star_rating_predictor import StarRatingPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleURLAnalyzer:
    """Simple analyzer that takes a URL and returns JSON analysis"""
    
    def __init__(self):
        """Initialize all analyzers once"""
        logger.info("Initializing analyzers...")
        
        # Create data directory if it doesn't exist
        import os
        os.makedirs("data/raw", exist_ok=True)
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.english_emotion_analyzer = EnglishEmotionAnalyzerHartmann()
        self.spanish_emotion_analyzer = SpanishEmotionAnalyzerRobertuito()
        self.rating_predictor = StarRatingPredictor()
        logger.info("Analyzers ready!")
    
    def detect_platform(self, url: str) -> str:
        """Detect which platform the URL is from"""
        domain = urlparse(url).netloc.lower()
        
        if 'trustpilot' in domain:
            return 'trustpilot'
        elif 'imdb' in domain:
            return 'imdb'
        elif 'steampowered' in domain or 'store.steam' in domain:
            return 'steam'
        elif 'play.google' in domain:
            return 'playstore'
        else:
            raise ValueError(f"Unsupported platform: {domain}")
    
    def scrape_reviews(self, url: str, max_reviews: int = 30) -> List[Dict]:
        """Scrape reviews from the URL using original scrapers"""
        platform = self.detect_platform(url)
        logger.info(f"Scraping {platform} reviews from URL")
        
        import os
        
        try:
            if platform == 'trustpilot':
                # Extract topic from URL for filename
                topic = url.split('/')[-1] or 'temp_trustpilot'
                
                # Use original scraper (returns None, saves file)
                scrape_trustpilot(url, topic, max_pages=3)
                
                # Read the saved file: "{topic}_reviews.json"
                filename = f"{topic}_reviews.json"
                filepath = f"data/raw/{filename}"
                
                return self._read_saved_reviews(filepath)
                
            elif platform == 'imdb':
                # Fix IMDb URL format to match what scraper expects
                if not url.endswith('/reviews/'):
                    if '/reviews/' not in url:
                        # Add reviews path if not present
                        if url.endswith('/'):
                            url = url + 'reviews/'
                        else:
                            url = url + '/reviews/'
                    
                    # Add ref parameter if not present
                    if '?ref_=' not in url:
                        url = url + '?ref_=tt_ov_ururv'
                
                topic = 'temp_imdb'
                
                # Use original scraper (returns None, saves file)
                scrape_imbd(url, topic, max_pages=2)
                
                # Read the saved file: "{topic}_reviews.json"
                filename = f"{topic}_reviews.json"
                filepath = f"data/raw/{filename}"
                
                return self._read_saved_reviews(filepath)
                
            elif platform == 'steam':
                # Extract Steam app ID from URL
                if '/app/' in url:
                    app_id = url.split('/app/')[-1].split('/')[0]
                else:
                    app_id = '570'  # Default fallback
                
                # Use original scraper (returns data AND saves file)
                reviews = scrape_steam(app_id, max_reviews)
                
                # Steam scraper returns the data directly
                return reviews if reviews else []
                
            elif platform == 'playstore':
                # Extract app ID from Play Store URL
                if 'id=' in url:
                    app_id = url.split('id=')[-1].split('&')[0]
                else:
                    app_id = 'com.whatsapp'  # Default fallback
                
                # Use original scraper (returns data AND saves file)
                reviews = scrape_google_playstore(app_id, max_reviews)
                
                # Play Store scraper returns the data directly
                return reviews if reviews else []
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return []
        
        return []
    
    def _read_saved_reviews(self, filepath: str) -> List[Dict]:
        """Read reviews from saved JSON file"""
        import os
        
        if not os.path.exists(filepath):
            logger.error(f"Expected file not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            
            logger.info(f"Successfully read {len(reviews)} reviews from {filepath}")
            
            # Clean up temp file (optional)
            if 'temp_' in filepath:
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temp file: {filepath}")
                except:
                    pass
            
            return reviews
            
        except Exception as e:
            logger.error(f"Error reading saved reviews from {filepath}: {e}")
            return []
    
    def analyze_single_review(self, review_text: str) -> Dict[str, Any]:
        """Analyze a single review text and return all results"""
        if not review_text or not review_text.strip():
            return {"error": "Empty review text"}
        
        try:
            # Detect language
            try:
                language = detect(review_text)
            except LangDetectException:
                language = "en"
            
            # Sentiment Analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(review_text)
            
            # Extract Keywords
            keywords = self.keyword_extractor.extract_keywords(review_text)
            
            # Emotion Analysis (language-dependent)
            if language == "es":
                emotions = self.spanish_emotion_analyzer.analyze_emotion(review_text)
            else:
                emotions = self.english_emotion_analyzer.analyze_emotion(review_text)
            
            top_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            
            # Star Rating Prediction
            predicted_rating = self.rating_predictor.predict_star_rating(review_text)
            
            return {
                "text": review_text,
                "language": language,
                "sentiment": {
                    "label": str(sentiment_result.get("sentiment_label", "neutral")),
                    "score": sentiment_result.get("sentiment_score", 0.0),
                    "confidence": abs(sentiment_result.get("sentiment_score", 0.0)),
                    "polarity": sentiment_result.get("sentiment_score", 0.0)
                },
                "keywords": keywords[:10],  # Top 10 keywords
                "emotions": {
                    "top_emotion": top_emotion,
                    "all_emotions": emotions
                },
                "predicted_rating": predicted_rating,
                "analysis_success": True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing review: {e}")
            return {"error": str(e), "analysis_success": False}
    
    def analyze_url(self, url: str, max_reviews: int = 20) -> Dict[str, Any]:
        """Main method: Analyze reviews from a URL and return JSON results"""
        
        try:
            # Step 1: Scrape reviews
            logger.info(f"Starting analysis for URL: {url}")
            reviews = self.scrape_reviews(url, max_reviews)
            
            if not reviews:
                return {
                    "url": url,
                    "error": "No reviews found at this URL",
                    "total_reviews": 0
                }
            
            logger.info(f"Found {len(reviews)} reviews")
            
            # Step 2: Analyze each review
            analyzed_reviews = []
            successful_analyses = 0
            
            for i, review in enumerate(reviews):
                review_text = review.get('text', '')
                if review_text.strip():
                    analysis = self.analyze_single_review(review_text)
                    
                    # Add original review metadata
                    analysis.update({
                        "review_id": i + 1,
                        "original_title": review.get('title', ''),
                        "original_rating": review.get('rating', ''),
                        "date": review.get('date', '')
                    })
                    
                    analyzed_reviews.append(analysis)
                    if analysis.get("analysis_success", False):
                        successful_analyses += 1
            
            # Step 3: Calculate summary statistics
            if successful_analyses > 0:
                successful_reviews = [r for r in analyzed_reviews if r.get("analysis_success", False)]
                
                sentiments = [r["sentiment"]["label"] for r in successful_reviews]
                avg_confidence = sum(r["sentiment"]["confidence"] for r in successful_reviews) / len(successful_reviews)
                avg_sentiment_score = sum(r["sentiment"]["score"] for r in successful_reviews) / len(successful_reviews)
                
                # Most common emotion
                emotions = [r["emotions"]["top_emotion"] for r in successful_reviews]
                most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
                
                # Average predicted rating
                avg_rating = sum(r["predicted_rating"] for r in successful_reviews) / len(successful_reviews)
                
                summary = {
                    "total_reviews_analyzed": successful_analyses,
                    "average_confidence": round(avg_confidence, 3),
                    "average_sentiment_score": round(avg_sentiment_score, 3),
                    "average_predicted_rating": round(avg_rating, 1),
                    "sentiment_distribution": {s: sentiments.count(s) for s in set(sentiments)},
                    "most_common_emotion": most_common_emotion,
                    "platform": self.detect_platform(url)
                }
            else:
                summary = {
                    "total_reviews_analyzed": 0,
                    "error": "No reviews could be successfully analyzed"
                }
            
            return {
                "url": url,
                "timestamp": str(__import__("datetime").datetime.now()),
                "summary": summary,
                "reviews": analyzed_reviews[:max_reviews]  # Limit output size
            }
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": str(__import__("datetime").datetime.now())
            }

def main():
    """Command line interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_url_analyzer.py <URL> [max_reviews]")
        print("Example: python simple_url_analyzer.py https://es.trustpilot.com/review/www.bancosantander.es 30")
        print("Example: python simple_url_analyzer.py https://www.imdb.com/title/tt0892769/reviews/ 20")
        print("Supported platforms: Trustpilot, IMDb, Steam, Google Play Store")
        print("\nURL Format Notes:")
        print("  • IMDb: Use full /reviews/ URL (analyzer will add if missing)")
        print("  • Steam: Any store page URL with /app/{id}/")
        print("  • Trustpilot: Any review page URL")
        print("Note: This uses your existing scrapers in src/scraper/url_scraper.py")
        return
    
    url = sys.argv[1]
    max_reviews = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    try:
        analyzer = SimpleURLAnalyzer()
        results = analyzer.analyze_url(url, max_reviews)
        
        # Generate output filename
        domain = urlparse(url).netloc.replace('.', '_')
        output_file = f"review_analysis_{domain}.json"
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        
        # Print summary
        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            summary = results.get('summary', {})
            print(f"\n📊 Summary:")
            print(f"   • Platform: {summary.get('platform', 'Unknown')}")
            print(f"   • Reviews analyzed: {summary.get('total_reviews_analyzed', 0)}")
            print(f"   • Average confidence: {summary.get('average_confidence', 0)}")
            print(f"   • Average sentiment score: {summary.get('average_sentiment_score', 0)}")
            print(f"   • Average predicted rating: {summary.get('average_predicted_rating', 0)}/5")
            print(f"   • Most common emotion: {summary.get('most_common_emotion', 'N/A')}")
            
            sentiment_dist = summary.get('sentiment_distribution', {})
            if sentiment_dist:
                print(f"   • Sentiment breakdown: {sentiment_dist}")
        
    except Exception as e:
        print(f"❌ Failed to analyze URL: {e}")
        print("Make sure the URL is supported and try again.")
        print("If you get import errors, make sure all dependencies are installed with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()