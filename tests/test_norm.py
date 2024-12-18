from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re

def process_names(text):
    """
    Extract person names from text using NER. Names can be separated by /, & or ,
    """
    if not isinstance(text, str):
        return np.nan
    
    # Normalize case - if text is ALL CAPS, convert to Title Case
    if text.isupper():
        text = text.title()
    
    # Split text by all three separators (/, &, ,)
    original_names = re.split('[/&,]', text)
    ner_results = nlp(original_names)

    all_detected_names = []
    for name, results in zip(original_names, ner_results):
        # Check if name contains a person entity (B-PER)
        has_person = any(token['entity'] == 'B-PER' for token in results)
        if has_person:
            all_detected_names.append(name.strip())
    
    return '/'.join(all_detected_names) if all_detected_names else np.nan

def process_and_print_examples(df, n_examples):
    """
    Process and print n_examples for testing purposes
    """
    df_sample = df.head(n_examples)
    predictions = []
    ground_truth = []
    
    for idx, (text, gt) in enumerate(zip(df_sample['raw_comp_writers_text'], df_sample['CLEAN_TEXT'])):
        pred = process_names(text)
        predictions.append(pred)
        ground_truth.append(gt)
        
        print(f"\nExample {idx + 1}:")
        print(f"Original: {text}")
        print(f"Detected: {pred}")
        print(f"Ground Truth: {gt}")
    
    return predictions, ground_truth

def preprocess_lists(predictions, ground_truth):
    """
    Handle nans
    """
    for i in range(len(predictions)):
        if pd.isna(predictions[i]):
            predictions[i] = ''
        if pd.isna(ground_truth[i]):
            ground_truth[i] = ''
    return predictions, ground_truth

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate accuracy
    """
    preds_lower = [str(p).lower() for p in predictions]
    truth_lower = [str(g).lower() for g in ground_truth]
    return accuracy_score(truth_lower, preds_lower)

def fuzzy_accuracy(predictions, ground_truth, threshold=90):
    """
    Calculate fuzzy accuracy with threshold
    """
    matches = 0
    total = len(ground_truth)
    
    for i in range(total):
        similarity = fuzz.ratio(predictions[i].lower(), ground_truth[i].lower())
        if similarity >= threshold:
            matches += 1
            
    return matches / total

def analyze_errors(predictions, ground_truth, threshold=70):
    """
    Print error analysis of predictions vs ground truth
    """
    print(f"\nAnalyzing prediction errors (similarity < {threshold}%):")
    
    errors = []
    for i in range(len(predictions)):
        similarity = fuzz.ratio(predictions[i].lower(), ground_truth[i].lower())
        if similarity < threshold:
            errors.append({
                'pred': predictions[i],
                'truth': ground_truth[i],
                'similarity': similarity
            })
    
    # Sort by similarity to see worst cases first
    errors.sort(key=lambda x: x['similarity'])
    
    # Print all error cases for the sample
    for i, error in enumerate(errors):
        print(f"\nError Case {i+1} (similarity: {error['similarity']}%):")
        print(f"Predicted : {error['pred']}")
        print(f"Expected  : {error['truth']}")
    
    print(f"\nTotal errors found: {len(errors)}")

# Initialize NER pipeline
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Read CSV
df = pd.read_csv('normalization_assesment_dataset_10k.csv')

# Process 100 examples
print("\nTesting with 100 examples:")
predictions, ground_truth = process_and_print_examples(df, n_examples=100)
predictions, ground_truth = preprocess_lists(predictions, ground_truth)

# Calculate metrics for the 100 examples
print("\nMetrics for 100 examples:")
accuracy = calculate_accuracy(predictions, ground_truth)
print(f"Exact Match Accuracy: {accuracy:.4f}")

# Calculate fuzzy accuracies
for threshold in [90, 80, 70]:
    score = fuzzy_accuracy(predictions, ground_truth, threshold=threshold)
    print(f"Fuzzy Accuracy (threshold={threshold}): {score:.4f}")

# Analyze errors for the 100 examples
analyze_errors(predictions, ground_truth, threshold=70)

# Print verification
print(f"\nProcessed examples: {len(predictions)}")
