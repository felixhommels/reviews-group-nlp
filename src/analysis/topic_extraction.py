import os
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed_test_results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/analysis')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_reviews(processed_file):
    with open(processed_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    # Try to use 'processed_text', fallback to 'text'
    texts = [r.get('processed_text') or r.get('text') for r in reviews if (r.get('processed_text') or r.get('text'))]
    return texts, reviews

def extract_topics(texts, model_name='sentence-transformers/LaBSE', n_top=10, model_save_path=None):
    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    topic_model = BERTopic(embedding_model=embedding_model)
    print("Fitting topic model...")
    topics, probs = topic_model.fit_transform(texts)
    if model_save_path:
        topic_model.save(model_save_path)
    topic_info = topic_model.get_topic_info()
    top_topics = topic_info.head(n_top)
    return topic_model, topics, probs, top_topics

def save_topics(topics_df, output_path):
    topics_df.to_json(output_path, orient='records', force_ascii=False, indent=2)

def main():
    # Loop through all processed review files
    for fname in os.listdir(PROCESSED_DIR):
        if not fname.endswith('.json'):
            continue
        processed_path = os.path.join(PROCESSED_DIR, fname)
        print(f"Processing {fname}...")
        texts, reviews = load_reviews(processed_path)
        if not texts:
            print(f"No texts found in {fname}, skipping.")
            continue
        model_name = 'sentence-transformers/LaBSE'
        model_save_path = os.path.join(MODEL_DIR, f"bertopic_{fname.replace('.json','')}")
        topic_model, topics, probs, top_topics = extract_topics(texts, model_name=model_name, n_top=10, model_save_path=model_save_path)
        # Save top topics summary
        output_path = os.path.join(OUTPUT_DIR, f"topic_summary_{fname}")
        save_topics(top_topics, output_path)
        print(f"Saved top topics for {fname} to {output_path}")
        # Optionally, save per-review topic assignments
        for review, topic in zip(reviews, topics):
            review['topic_id'] = int(topic)
        with open(os.path.join(OUTPUT_DIR, f"review_topics_{fname}"), 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        print(f"Saved topic assignments for {fname}")

if __name__ == "__main__":
    main() 