import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import re
from google_play_scraper import Sort, reviews
import pandas as pd

# To run you need to run from root directory: python -m src.scraper.url_scraper

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def scrape_trustpilot(url: str, topic: str, max_pages: int = 10):
    """
    Scrape reviews from Trustpilot.

    Args:
        url (str): The URL of the Trustpilot page to scrape.
        topic (str): The topic of the reviews to scrape - used for filenaming when saving results.
        max_pages (int, optional): The maximum number of pages to scrape. Defaults to 10.
    """
    
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
    
    save_json(reviews, f"{topic}_reviews.json")

# Test
# url = "https://es.trustpilot.com/review/www.bancosantander.es"
# scrape_trustpilot(url, topic="bancosantander", max_pages=2)

def scrape_imbd(url: str, topic: str, max_pages: int = 10):
    """
    Scrape reviews from IMDb.

    Args:
        url (str): The URL of the IMDb page to scrape.
        topic (str): The topic of the reviews to scrape - used for filenaming when saving results.
        max_pages (int, optional): The maximum number of pages to scrape. Defaults to 10.
    """
    
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

    save_json(reviews, f"{topic}_reviews.json")
    return reviews

# Test
# url = "https://www.imdb.com/title/tt0892769/reviews/?ref_=tt_ov_ururv"
# scrape_imbd(url, topic="howToTrainYourDragon", max_pages=2)

def scrape_google_playstore(app_id: str, max_reviews: int = 100):
    """
    Scrape reviews from Google Play Store.

    Args:
        app_id (str): The ID of the app to scrape: e.g. com.whatsapp, com.instagram.android, com.duolingo, etc.
        max_reviews (int, optional): The maximum number of reviews to scrape. Defaults to 100.
    """
    output_file_name = f"{app_id}_reviews.json"
    output_file = f"data/raw/{output_file_name}"
    
    all_reviews = []
    count = 0

    while count < max_reviews:
        new_reviews, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=min(100, max_reviews - count),
            filter_score_with=None
        )
        if not new_reviews:
            break
        all_reviews.extend(new_reviews)
        count += len(new_reviews)

    print(f"Fetched {len(all_reviews)} reviews.")

    # Format into desired structure
    formatted = []
    for r in all_reviews:
        formatted.append({
            "title": r["reviewCreatedVersion"] or "",  # use version as title if available
            "text": r["content"],
            "date": r["at"].isoformat(),
            "rating": r["score"]
        })

    # Save to JSON
    save_json(formatted, output_file_name)

    print(f"Saved to {output_file}")
    return formatted

# Test
# scrape_google_playstore("com.whatsapp", max_reviews=300)

def scrape_steam(app_id: str, max_reviews: int = 100):
    """
    Scrape reviews from Steam.

    Args:
        app_id (str): The ID of the app to scrape.
        max_reviews (int, optional): The maximum number of reviews to scrape. Defaults to 100.
    """
    
    output_file_name = f"{app_id}_steam_reviews.json"
    output_file = f"data/raw/{output_file_name}"
    
    reviews = []
    cursor = '*'
    total_fetched = 0
    batch_size = 100

    while total_fetched < max_reviews:
        url = (
            f"https://store.steampowered.com/appreviews/{app_id}"
            f"?json=1&num_per_page={batch_size}&cursor={cursor}&filter=recent&language=english"
        )
        response = requests.get(url)
        data = response.json()

        batch = data.get("reviews", [])
        if not batch:
            break

        for r in batch:
            reviews.append({
                "title": "",
                "text": r.get("review", ""),
                "date": datetime.utcfromtimestamp(r["timestamp_created"]).isoformat(),
                "rating": 5 if r.get("voted_up") else 1  # map thumbs up/down to 5/1 stars
            })

        total_fetched += len(batch)
        print(f"Fetched {total_fetched} reviews so far...")

        cursor = data.get("cursor", "")
        time.sleep(1)

        if len(batch) < batch_size:
            break

    save_json(reviews[:max_reviews], output_file_name)

    print(f"Saved {len(reviews[:max_reviews])} reviews to {output_file}")
    return reviews[:max_reviews]

# Test for Dota 2
# scrape_steam(app_id=570, max_reviews=300)