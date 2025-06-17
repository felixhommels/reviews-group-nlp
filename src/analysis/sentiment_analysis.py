import pandas as pd
from collections import Counter
from statistics import mean
from src.preprocessing.spacy_preprocessor import clean_text, detect_language, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Manual emotion detection using keyword lists
emotion_keywords = {
    "joy": ["happy", "joy", "excited", "delighted", "pleased", "glad", "cheerful", "elated", "thrilled", "ecstatic"],
    "sadness": ["sad", "unhappy", "depressed", "gloomy", "miserable", "sorrowful", "heartbroken", "disappointed", "down", "blue"],
    "anger": ["angry", "furious", "enraged", "irritated", "annoyed", "frustrated", "outraged", "livid", "fuming", "heated"],
    "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "nervous", "worried", "panicked", "horrified", "petrified"],
    "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "startled", "bewildered", "dumbfounded", "flabbergasted", "incredulous"],
    "disgust": ["disgusted", "repulsed", "revolted", "appalled", "nauseated", "sickened", "horrified", "offended", "outraged", "abhorred"],
    "trust": ["trust", "confident", "reliable", "faithful", "loyal", "dependable", "honest", "sincere", "genuine", "authentic"],
    "anticipation": ["excited", "eager", "hopeful", "optimistic", "looking forward", "anticipating", "expecting", "awaiting", "prepared", "ready"]
}

def detect_emotion(text):
    text_lower = text.lower()
    combined_corpus = [text_lower] + [emotion + " " + " ".join(keywords) for emotion, keywords in emotion_keywords.items()]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined_corpus)
    text_vector = vectorizer.transform([text_lower])
    emotion_vectors = {emotion: vectorizer.transform([emotion + " " + " ".join(keywords)]) for emotion, keywords in emotion_keywords.items()}
    similarities = {emotion: cosine_similarity(text_vector, vector).flatten()[0] for emotion, vector in emotion_vectors.items()}
    if not any(similarities.values()):
        return None
    return max(similarities.items(), key=lambda x: x[1])[0]

def map_sentiment_to_score(label):
    return 1 if label == "positive" else 0

def map_sentiment_to_stars(label):
    return 4.5 if label == "positive" else 1.5

def analyze_sentiment_overview(df, model, text_column="review"):
    results = []
    
    for review in df[text_column]:
        text = review["text"] if isinstance(review, dict) else review
        lang = detect_language(text)
        cleaned = clean_text(text)
        processed = preprocess_text(cleaned, lang)
        
        # Simple rule-based sentiment analysis
        positive_keywords = ["good", "great", "excellent", "amazing", "love", "best", "fantastic", "wonderful", "brilliant", "outstanding"]
        negative_keywords = ["bad", "terrible", "awful", "horrible", "worst", "poor", "disappointing", "mediocre", "boring", "waste"]
        
        text_lower = processed.lower()
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        if positive_count > negative_count:
            pred = "positive"
        elif negative_count > positive_count:
            pred = "negative"
        else:
            pred = "neutral"
        
        confidence = abs(positive_count - negative_count) / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
        sentiment_score = map_sentiment_to_score(pred)
        star_rating = map_sentiment_to_stars(pred)

        # Manual emotion detection using cosine similarity
        top_emotion = detect_emotion(text)

        results.append({
            "text": text,
            "sentiment": pred,
            "confidence": confidence,
            "sentiment_score": sentiment_score,
            "predicted_rating": star_rating,
            "emotion": top_emotion
        })
    
    result_df = pd.DataFrame(results)

    sentiment_breakdown = result_df["sentiment"].value_counts(normalize=True).to_dict()
    emotion_breakdown = result_df["emotion"].value_counts(normalize=True).to_dict()

    summary = {
        "total_reviews": len(result_df),
        "overall_sentiment": result_df["sentiment"].mode()[0],
        "avg_confidence": round(result_df["confidence"].mean(), 3),
        "avg_sentiment_score": round(result_df["sentiment_score"].mean(), 3),
        "avg_predicted_rating": round(result_df["predicted_rating"].mean(), 2),
        "most_common_emotion": result_df["emotion"].mode()[0] if not result_df["emotion"].isnull().all() else None,
        "sentiment_breakdown": sentiment_breakdown,
        "emotion_breakdown": emotion_breakdown
    }

    return summary, result_df 