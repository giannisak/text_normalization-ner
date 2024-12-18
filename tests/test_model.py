# Test NER model behavior on a single example
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "<Unknown>/Wright, Justyce Kaseem"

ner_results = nlp(example)
print(ner_results)

# Output:
# [{'entity': 'B-PER', 'score': np.float32(0.99926656), 'index': 5, 'word': 'Wright', 'start': 10, 'end': 16}, 
#  {'entity': 'B-PER', 'score': np.float32(0.9995902), 'index': 7, 'word': 'Just', 'start': 18, 'end': 22},
#  {'entity': 'I-PER', 'score': np.float32(0.48544157), 'index': 8, 'word': '##y', 'start': 22, 'end': 23},
#  {'entity': 'I-PER', 'score': np.float32(0.73747694), 'index': 9, 'word': '##ce', 'start': 23, 'end': 25},
#  {'entity': 'I-PER', 'score': np.float32(0.9993616), 'index': 10, 'word': 'Ka', 'start': 26, 'end': 28},
#  {'entity': 'I-PER', 'score': np.float32(0.9523705), 'index': 11, 'word': '##see', 'start': 28, 'end': 31}]
