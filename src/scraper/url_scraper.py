import requests
from bs4 import BeautifulSoup
import json
import time
import re
from src.utils import save_reviews

# To run you need to run from root directory: python -m src.scraper.url_scraper

def scrape_trustpilot(url: str, topic: str, max_pages: int = 10):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    reviews = []
    
    for page in range(1, max_pages + 1):
        paged_url = f"{url}?page={page}"
        res = requests.get(paged_url, headers=headers)
        soup = BeautifulSoup(res.text, "lxml")
        
        sections = soup.select("section.styles_reviewContentwrapper__K2aRu")
            
        for sec in sections:
            try:
                title = sec.select_one('[data-service-review-title-typography]').text.strip()
                text = sec.select_one('[data-service-review-text-typography]').text.strip()
                date = sec.select_one('[data-service-review-date-of-experience-typography] span').text.strip()
                rating_img = sec.select_one("img[alt*='Rated']")
                rating = rating_img["alt"].split(" ")[1] if rating_img else "N/A"
                
                reviews.append({
                    "title": title,
                    "text": text,
                    "date": date,
                    "rating": rating
                })
            except Exception as e:
                print("Skipping a review due to parsing error:", e)
        
        time.sleep(1)
    
    save_reviews(reviews, f"{topic}_reviews.json")

# Test
url = "https://es.trustpilot.com/review/www.bancosantander.es"
scrape_trustpilot(url, topic="bancosantander", max_pages=2)

def scrape_imbd(url: str, topic: str, max_pages: int = 10):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    reviews = []
    
    for page in range(max_pages):
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "lxml")

        review_blocks = soup.select("div[data-testid='review-card-parent']")

        for block in review_blocks:
            try:
                # Rating
                rating_tag = block.select_one("span.ipc-rating-star--rating")
                rating = rating_tag.text.strip() if rating_tag else "N/A"

                # Title
                title_tag = block.select_one("div[data-testid='review-summary'] h3")
                title = title_tag.text.strip() if title_tag else "N/A"

                # Text
                text_block = block.select_one("div[data-testid='review-overflow']")
                text_div = text_block.select_one("div.ipc-html-content-inner-div") if text_block else None
                text = text_div.get_text(separator="\n").strip() if text_div else "N/A"

                reviews.append({
                    "rating": rating,
                    "title": title,
                    "text": text
                })

            except Exception as e:
                print("Error parsing a review:", e)

        time.sleep(1)

    save_reviews(reviews, f"{topic}_reviews.json")

# Test
url = "https://www.imdb.com/title/tt0892769/reviews/?ref_=tt_ov_ururv"
scrape_imbd(url, topic="howToTrainYourDragon", max_pages=2)