# Text Normalization for Music Industry Names

## Overview
A Python-based solution for normalizing composer and writer names in music industry data using Named Entity Recognition (NER). The system processes raw text containing composer information and standardizes the output format while handling various edge cases.

## Key Features
- Named Entity Recognition using BERT
- Name processing pipeline
- Testing scripts and analysis
- Multiple evaluation methods

## Performance
- Processing time: 293.33 seconds for dataset
- Exact Match Accuracy: 0.6716
- Fuzzy Match Accuracy (90%): 0.7074
- Fuzzy Match Accuracy (80%): 0.7432

## Requirements
- Python 3.x
- GPU with 6GB VRAM (used in development)
- Model size: 433MB (dslim/bert-base-NER)
- Dependencies:
  - transformers
  - sklearn
  - fuzzywuzzy
  - pandas
  - numpy

## Project Structure
- `text_normalizer.py`: Main implementation
- `test_model.py`: Initial model testing
- `test_norm.py`: Extended testing
- `test_norm_outputs.txt`: Test results

## Documentation
See [DOCUMENTATION.md](DOCUMENTATION.md)
