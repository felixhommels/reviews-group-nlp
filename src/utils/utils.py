import re
import json

def save_reviews(reviews: list, filename: str):
    
    with open(f"data/raw/{filename}", "w") as f:
        json.dump(reviews, f, indent=2)
    
    print(f"Saved {len(reviews)} reviews to {filename}")