"""Utility functions for file operations."""

import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration or None if loading fails
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
    return None

def save_json(data: Any, filepath: str) -> bool:
    """Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path where to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        return False
