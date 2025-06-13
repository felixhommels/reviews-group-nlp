import pandas as pd

def map_pred_5_to_10(pred_5):
    try:
        return int(round((int(pred_5) - 1) * (9/4) + 1))
    except:
        return None

def extract_pred_5(label):
    try:
        return int(label.split()[0])
    except:
        return None

def main():
    # Load analyzed results
    df = pd.read_json('data/results/analyzed_imdb_reviews.json')

    # Extract the model's predicted star rating from the raw label
    df['pred_5'] = df['raw_model_label'].apply(extract_pred_5)
    df['pred_10'] = df['pred_5'].apply(map_pred_5_to_10)

    # Only compare where both are valid
    df_valid = df[(df['pred_10'].notnull()) & (df['rating'].apply(lambda x: str(x).isdigit()))]
    df_valid['rating'] = df_valid['rating'].astype(int)

    # Calculate metrics
    exact_match = (df_valid['pred_10'] == df_valid['rating']).mean()
    mae = (df_valid['pred_10'] - df_valid['rating']).abs().mean()

    print(f"Total reviews with valid prediction and rating: {len(df_valid)}")
    print(f"Exact match accuracy: {exact_match:.2%}")
    print(f"Mean absolute error: {mae:.2f}")

if __name__ == "__main__":
    main() 