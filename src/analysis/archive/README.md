# Archive: Legacy & Reference NLP Analysis Scripts

This folder contains legacy, experimental, and reference versions of the NLP analysis pipeline. These scripts are **not** used in production, but are kept for historical reference, comparison, and inspiration for future improvements.

## Script Index

- **nlp_analysis_multilingual_reference.py**

  - Early implementation with multilingual sentiment analysis using XLM-RoBERTa, VADER, and TextBlob. Includes emotion detection and keyword extraction. Useful for understanding how to integrate multilingual models and fallback logic.

- **nlp_analysis_config_experiment.py**

  - Experimental version that uses a configuration-driven and dependency-managed approach. Relies on external config files and a dependency manager. Good for reference if you want to see how to make the pipeline more modular and configurable.

- **nlp_analysis_simple.py**

  - Minimal, English-only version. Focuses on VADER/TextBlob sentiment, TF-IDF keyword extraction, and basic emotion detection. Includes built-in test cases and rich console output. Useful for quick prototyping or as a teaching example.

- **nlp_analysis_og.py**

  - The original/earliest version of the analysis pipeline. May contain rough or unoptimized code, but useful for seeing the project's starting point.

- **test_cases.py**

  - Standalone script for running test cases on the various analyzers. Useful for manual testing and debugging.

- **test_nlp_analysis.py**
  - Unit/integration tests for the analysis pipeline. Can be used as a template for writing new tests or for regression testing.

## Usage

- These scripts are **not** imported by the main application.
- Refer to them for ideas, alternative approaches, or to recover old logic.
- The current production code is in `../nlp_analysis.py`.

---

**Tip:** If you add new experimental scripts, update this README with a short description.
