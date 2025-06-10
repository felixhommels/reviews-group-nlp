# This file is intentionally left blank.
# It marks the 'utils' directory as a Python package, allowing modules
# within it to be imported from other parts of the project. 

"""Utility functions for the NLP analysis project."""

import json
import os
from typing import List, Dict

def save_reviews(reviews: List[Dict], filename: str) -> None:
    """Save reviews to a JSON file.
    
    Args:
        reviews: List of review dictionaries
        filename: Name of the output file
    """
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to JSON file
    output_path = os.path.join('data/raw', filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2) 