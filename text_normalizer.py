"""
Text Normalization for Composer Names

This module provides functionality to normalize text containing composer names.
It uses Named Entity Recognition (NER) and applies various normalization rules 
to clean and standardize the text.

Performance Metrics:
- Processing time: 405.25 seconds
- Exact Match Accuracy: 0.6716
- Fuzzy Match Accuracy (90%): 0.7074
- Fuzzy Match Accuracy (80%): 0.7432

Key features:
- NER-based person name detection
- Case normalization
- Handling of multiple name separators (/, &, ,)
- Fuzzy matching for evaluation
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re
import time
from typing import List, Tuple, Dict, Optional

class TextNormalizer:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize the TextNormalizer with a NER model.
        
        Args:
            model_name: The name of the pretrained NER model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def normalize_text(self, text: str) -> Optional[str]:
        """
        Normalize text containing composer names.
        
        Args:
            text: Input text containing names
            
        Returns:
            Normalized string of names or np.nan if no valid names found
        """
        # Check if input is a string
        if not isinstance(text, str):
            return np.nan
            
        # Convert all-caps text to title case
        if text.isupper():
            text = text.title()
        
        # Split text by common separators (/, &, ,)
        names = re.split('[/&,]', text)
        ner_results = self.nlp(names)

        # Extract valid person names
        detected_names = []
        for name, results in zip(names, ner_results):
            if any(token['entity'] == 'B-PER' for token in results):
                detected_names.append(name.strip())
        
        return '/'.join(detected_names) if detected_names else np.nan

class DatasetEvaluator:
    def __init__(self, normalizer: TextNormalizer):
        """
        Initialize evaluator with a TextNormalizer instance.
        
        Args:
            normalizer: TextNormalizer instance to use for processing
        """
        self.normalizer = normalizer
        
    def process_dataset(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Process the dataset and return predictions and ground truth.
        
        Args:
            df: DataFrame with 'raw_comp_writers_text' and 'CLEAN_TEXT' columns
            
        Returns:
            Tuple of (predictions, ground_truth) lists
        """
        predictions = []
        ground_truth = []
        
        start_time = time.time()
        
        for text, gt in zip(df['raw_comp_writers_text'], df['CLEAN_TEXT']):
            pred = self.normalizer.normalize_text(text)
            predictions.append('' if pd.isna(pred) else pred)
            ground_truth.append('' if pd.isna(gt) else gt)
            
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return predictions, ground_truth

    def evaluate(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """
        Evaluate predictions against ground truth using multiple metrics.
        
        Args:
            predictions: List of predicted normalized texts
            ground_truth: List of ground truth texts
            
        Returns:
            Dictionary containing various accuracy metrics
        """
        # Convert to lowercase for comparison
        preds_lower = [str(p).lower() for p in predictions]
        truth_lower = [str(g).lower() for g in ground_truth]
        
        # Calculate exact and fuzzy match accuracies
        exact_accuracy = accuracy_score(truth_lower, preds_lower)
        fuzzy_90 = self._fuzzy_accuracy(preds_lower, truth_lower, threshold=90)
        fuzzy_80 = self._fuzzy_accuracy(preds_lower, truth_lower, threshold=80)
        
        return {
            'exact_accuracy': exact_accuracy,
            'fuzzy_accuracy_90': fuzzy_90,
            'fuzzy_accuracy_80': fuzzy_80
        }
    
    def _fuzzy_accuracy(self, predictions: List[str], ground_truth: List[str], 
                       threshold: int = 90) -> float:
        """
        Calculate fuzzy match accuracy using given threshold.
        
        Args:
            predictions: List of predicted texts
            ground_truth: List of ground truth texts
            threshold: Minimum similarity score to consider a match (0-100)
            
        Returns:
            Fuzzy match accuracy score
        """
        matches = 0
        total = len(ground_truth)
        
        for i in range(total):
            similarity = fuzz.ratio(predictions[i], ground_truth[i])
            if similarity >= threshold:
                matches += 1
                
        return matches / total

def main():
    # Initialize normalizer and evaluator
    normalizer = TextNormalizer()
    evaluator = DatasetEvaluator(normalizer)
    
    # Load dataset
    df = pd.read_csv('normalization_assesment_dataset_10k.csv')
    
    # Process dataset
    predictions, ground_truth = evaluator.process_dataset(df)
    
    # Evaluate results
    metrics = evaluator.evaluate(predictions, ground_truth)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Exact Match Accuracy: {metrics['exact_accuracy']:.4f}")
    print(f"Fuzzy Match Accuracy (90% threshold): {metrics['fuzzy_accuracy_90']:.4f}")
    print(f"Fuzzy Match Accuracy (80% threshold): {metrics['fuzzy_accuracy_80']:.4f}")

if __name__ == "__main__":
    main()
