# Text Normalization Documentation

## 1. Initial Model Testing (test_model.py)

The development process began with testing the NER model's behavior on a single example to understand how it processes composer names. The test utilized the "dslim/bert-base-NER" model from HuggingFace's transformers library.

### Test Case
```
Input: "<Unknown>/Wright, Justyce Kaseem"
```

The test revealed that the NER model successfully identified person names with high confidence scores:
- Detected "Wright" with 0.999 confidence
- Detected "Justyce" and "Kaseem" as parts of a person name
- Model correctly split and recognized multi-token names

This initial test demonstrated that the BERT-based NER model could effectively identify person names even in complex formats.

## 2. Extended Testing (test_norm.py)

Following the successful single example test, a more comprehensive testing framework was developed to process and evaluate multiple examples. Key components included:

### 2.1 Name Processing Function
```python
def process_names(text):
    """Extract person names from text using NER"""
```
The function implemented:
- Case normalization (ALL CAPS to Title Case)
- Text splitting by multiple separators (/, &, ,)
- NER-based person name detection
- Name joining with standardized separators

### 2.2 Evaluation Functions
- Exact match accuracy calculation
- Fuzzy match accuracy with multiple thresholds (90%, 80%, 70%)
- Detailed error analysis for mismatches

### 2.3 Test Results Output (test_norm_outputs.txt)
The extended testing processed 100 examples, revealing:
- Exact Match Accuracy: 0.7500
- Fuzzy Match Accuracy (90%): 0.7600
- Fuzzy Match Accuracy (80%): 0.7900
- Fuzzy Match Accuracy (70%): 0.8100

### 2.4 Error Analysis
The testing identified 19 error cases, categorized as:
1. Empty Prediction vs Non-empty Ground Truth (e.g., "" vs "DJ PALEMBANG")
2. Non-empty Prediction vs Empty Ground Truth (e.g., "Mendel Brikman" vs "")
3. Non-Latin character mishandling (e.g., "Bằng Giang/Tú Nhi")
4. Artistic name mishandling (e.g., "Itsjaygocrazy/Jordan Ancrum")

## 3. Final Implementation (text_normalizer.py)

A complete implementation was developed with two main classes:

### 3.1 TextNormalizer Class
```python
class TextNormalizer:
    def __init__(self, model_name: str = "dslim/bert-base-NER")
```
Core functionality:
- NER model initialization
- Text normalization
- Consistent name formatting

### 3.2 DatasetEvaluator Class
```python
class DatasetEvaluator:
    def __init__(self, normalizer: TextNormalizer)
```
Evaluation features:
- Accuracy Score
- Fuzzy Matching
- Processing Time

### 3.3 Performance Metrics
The final implementation achieved:
- Processing time: 293.33 seconds
- Exact Match Accuracy: 0.6716
- Fuzzy Match Accuracy (90%): 0.7074
- Fuzzy Match Accuracy (80%): 0.7432

## 4. Development Process Summary

The development followed a systematic approach:
1. Initial model validation with a single example
2. Extended testing with 100 examples to identify edge cases
3. Implementation of final solution
4. Performance optimization and metrics tracking

## 5. Model Selection and Comparisons

### 5.1 BERT Model Choice
- Selected dslim/bert-base-NER (433MB) due to reasonable VRAM requirements (6GB)
- Enables local execution without significant hardware constraints
- Offers good balance between performance and resource usage

### 5.2 Comparison with LLM Approaches
#### VRAM Requirements (Source: Ollama)
- 7B models → 8GB VRAM
- 13B models → 16GB VRAM
- 33B models → 32GB VRAM
- 70B models → 64GB VRAM

#### Trade-offs Analysis
1. Accuracy vs Resources
   - BERT: Lower resource requirements, acceptable accuracy
   - LLMs: Higher accuracy potential, significant resource demands

2. Processing Requirements
   - BERT: More preprocessing, explicit rules needed
   - LLMs: More automated, better handling of edge cases

3. Implementation Experience
   - Similar entity resolution task implemented with 7B LLM at github.com/giannisak/ER
   - Could be adapted to this problem with prompt engineering

## 6. Future Improvements

### 6.1 Model Enhancements
1. Fine-tuning Opportunities
   - Train on music industry specific data
   - Focus on multilingual artist names and nicknames
   - Incorporate style-specific patterns (artistic names)

2. Performance Optimization
   - Implement batch processing with configurable batch sizes
   - Explore time-memory trade-offs
   - Optimize for specific hardware configurations

3. Advanced Techniques
   - RAG implementation for additional context
   - Hybrid approach combining BERT with lightweight LLM
   - Ensemble methods with specialized models

4. Data Processing
   - Enhanced preprocessing pipeline for edge cases
   - Improved handling of non-Latin characters
   - Custom tokenization for artistic names
