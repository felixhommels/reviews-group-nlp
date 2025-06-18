import openai
from typing import List, Tuple
import os
import dotenv
import re

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def get_likes_and_dislikes(review_texts, model="gpt-3.5-turbo"):
    """
    Given a list of review texts, use OpenAI to extract the top 3 liked and disliked aspects.
    Returns (top_likes, top_dislikes) as lists of strings.
    """
    if not review_texts:
        return [], []

    # Join reviews for the prompt
    reviews_str = "\n\n".join(f"- {text}" for text in review_texts)

    prompt = (
    "You are an expert at summarizing customer reviews.\n\n"
    "Here are several user reviews for a product or service:\n\n"
    f"{reviews_str}\n\n"
    "Based on these reviews, list the top 3 most liked aspects and the top 3 most disliked aspects.\n"
    "Format your answer exactly as follows:\n"
    "Most Liked Aspects:\n"
    "1. <liked aspect 1>: <short explanation>\n"
    "2. <liked aspect 2>: <short explanation>\n"
    "3. <liked aspect 3>: <short explanation>\n"
    "Most Disliked Aspects:\n"
    "1. <disliked aspect 1>: <short explanation>\n"
    "2. <disliked aspect 2>: <short explanation>\n"
    "3. <disliked aspect 3>: <short explanation>\n"
    "Keep each point under 10 words and do not include explanations."
)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at summarizing customer reviews."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.4,
    )

    content = response.choices[0].message.content

    top_likes, top_dislikes = parse_likes_dislikes(content)
    return top_likes, top_dislikes

def parse_likes_dislikes(content):
    liked, disliked = [], []
    liked_match = re.search(r"Most Liked Aspects:\s*(1\..*?)(?:Most Disliked Aspects:|$)", content, re.DOTALL | re.IGNORECASE)
    disliked_match = re.search(r"Most Disliked Aspects:\s*(1\..*)", content, re.DOTALL | re.IGNORECASE)

    if liked_match:
        liked_lines = re.findall(r"\d+\.\s*(.*)", liked_match.group(1))
        liked = [line.strip() for line in liked_lines]
    if disliked_match:
        disliked_lines = re.findall(r"\d+\.\s*(.*)", disliked_match.group(1))
        disliked = [line.strip() for line in disliked_lines]

    return liked[:3], disliked[:3]
    
    
    
    
    
    
    
    