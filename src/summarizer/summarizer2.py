import dotenv
import os
import json
from openai import OpenAI
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import statistics

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class ReviewAnalyzer:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.reviews = []
        self.load_reviews()
    
    def load_reviews(self):
        """Load reviews from JSON file (one JSON object per line)"""
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        review = json.loads(line)
                        self.reviews.append(review)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        continue
    
    def calculate_emotion_averages(self) -> Dict[str, float]:
        """Calculate average emotion scores across all reviews"""
        emotion_totals = defaultdict(float)
        emotion_counts = defaultdict(int)
        
        for review in self.reviews:
            if 'emotion_scores' in review:
                for emotion, score in review['emotion_scores'].items():
                    emotion_totals[emotion] += score
                    emotion_counts[emotion] += 1
        
        emotion_averages = {}
        for emotion in emotion_totals:
            if emotion_counts[emotion] > 0:
                emotion_averages[emotion] = emotion_totals[emotion] / emotion_counts[emotion]
        
        return emotion_averages
    
    def get_top_emotions(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top emotions by average score"""
        emotion_averages = self.calculate_emotion_averages()
        sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_n]
    
    def get_top_keywords(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get top keywords by frequency"""
        keyword_counter = Counter()
        
        for review in self.reviews:
            if 'keywords' in review and isinstance(review['keywords'], list):
                for keyword in review['keywords']:
                    keyword_counter[keyword] += 1
        
        return keyword_counter.most_common(top_n)
    
    def get_processed_texts(self) -> List[str]:
        """Get all processed_text fields from reviews"""
        texts = []
        for review in self.reviews:
            if 'processed_text' in review and review['processed_text']:
                texts.append(review['processed_text'])
        return texts
    
    def analyze_reviews_with_chatgpt(self, texts: List[str]) -> Dict[str, List[str]]:
        """Use ChatGPT to analyze reviews and extract liked/disliked aspects"""
        # Combine all texts for analysis
        combined_text = " ".join(texts[:50])  # Limit to first 50 reviews to avoid token limits
        
        prompt = f"""
        Analyze the following customer reviews for a product and identify:

        Top 3 most liked things about the product:
        Top 3 most disliked things about the product:

        Reviews: {combined_text}

        Please provide your analysis in the following JSON format:
        {{
            "liked_aspects": ["aspect1", "aspect2", "aspect3"],
            "disliked_aspects": ["aspect1", "aspect2", "aspect3"]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes customer reviews and extracts key insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse the response
            content = response.choices[0].message.content
            try:
                # Try to extract JSON from the response
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    result = json.loads(json_str)
                    return result
                else:
                    # Fallback: parse manually
                    return self.parse_chatgpt_response_manually(content)
            except json.JSONDecodeError:
                return self.parse_chatgpt_response_manually(content)
                
        except Exception as e:
            print(f"Error calling ChatGPT API: {e}")
            return {
                "liked_aspects": ["Error analyzing reviews"],
                "disliked_aspects": ["Error analyzing reviews"]
            }
    
    def parse_chatgpt_response_manually(self, content: str) -> Dict[str, List[str]]:
        """Manually parse ChatGPT response if JSON parsing fails"""
        liked_aspects = []
        disliked_aspects = []
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            if 'liked' in line or 'positive' in line:
                current_section = 'liked'
            elif 'disliked' in line or 'negative' in line:
                current_section = 'disliked'
            elif line.startswith('-') or line.startswith('*') or line.startswith('•'):
                aspect = line[1:].strip()
                if current_section == 'liked' and len(liked_aspects) < 3:
                    liked_aspects.append(aspect)
                elif current_section == 'disliked' and len(disliked_aspects) < 3:
                    disliked_aspects.append(aspect)
        
        return {
            "liked_aspects": liked_aspects[:3],
            "disliked_aspects": disliked_aspects[:3]
        }
    
    def generate_summary_report(self) -> Dict:
        """Generate a comprehensive summary report"""
        # Calculate emotion averages
        emotion_averages = self.calculate_emotion_averages()
        
        # Get top emotions and keywords
        top_emotions = self.get_top_emotions(3)
        top_keywords = self.get_top_keywords(3)
        
        # Get processed texts for ChatGPT analysis
        processed_texts = self.get_processed_texts()
        
        # Basic statistics
        total_reviews = len(self.reviews)
        avg_rating = statistics.mean([r.get('rating', 0) for r in self.reviews if r.get('rating')])
        
        # Sentiment distribution
        sentiment_counts = Counter([r.get('sentiment_label', 'unknown') for r in self.reviews])
        
        report = {
            "summary": {
                "total_reviews": total_reviews,
                "average_rating": round(avg_rating, 2),
                "sentiment_distribution": dict(sentiment_counts)
            },
            "emotion_analysis": {
                "average_emotion_scores": emotion_averages,
                "top_3_emotions": [{"emotion": emotion, "average_score": round(score, 4)} 
                                 for emotion, score in top_emotions]
            },
            "keyword_analysis": {
                "top_3_keywords": [{"keyword": keyword, "frequency": count} 
                                 for keyword, count in top_keywords]
            },
            "chatgpt_analysis": {
                "status": "pending",
                "liked_aspects": [],
                "disliked_aspects": []
            }
        }
        
        return report

async def main():
    """Main function to run the analysis"""
    # Initialize analyzer with one of the example files
    analyzer = ReviewAnalyzer("data/analysis/review_analysis_570_steam.json")
    
    # Generate summary report
    report = analyzer.generate_summary_report()
    
    # Get processed texts for ChatGPT analysis
    processed_texts = analyzer.get_processed_texts()
    
    # Analyze with ChatGPT
    print("Analyzing reviews with ChatGPT...")
    chatgpt_analysis = analyzer.analyze_reviews_with_chatgpt(processed_texts)
    report["chatgpt_analysis"] = {
        "status": "completed",
        "liked_aspects": chatgpt_analysis.get("liked_aspects", []),
        "disliked_aspects": chatgpt_analysis.get("disliked_aspects", [])
    }
    
    # Print the complete report to console instead of saving
    print("\n=== COMPLETE ANALYSIS REPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Print summary
    print("\n=== REVIEW ANALYSIS SUMMARY ===")
    print(f"Total Reviews: {report['summary']['total_reviews']}")
    print(f"Average Rating: {report['summary']['average_rating']}")
    print(f"Sentiment Distribution: {report['summary']['sentiment_distribution']}")
    
    print("\n=== TOP EMOTIONS ===")
    for emotion_data in report['emotion_analysis']['top_3_emotions']:
        print(f"{emotion_data['emotion']}: {emotion_data['average_score']}")
    
    print("\n=== TOP KEYWORDS ===")
    for keyword_data in report['keyword_analysis']['top_3_keywords']:
        print(f"{keyword_data['keyword']}: {keyword_data['frequency']} times")
    
    print("\n=== CHATGPT ANALYSIS ===")
    print("Most Liked Aspects:")
    for aspect in report['chatgpt_analysis']['liked_aspects']:
        print(f"  • {aspect}")
    
    print("\nMost Disliked Aspects:")
    for aspect in report['chatgpt_analysis']['disliked_aspects']:
        print(f"  • {aspect}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


