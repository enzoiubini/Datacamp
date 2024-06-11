LLM are sophisticated AI models capable of understanding and generating human language text.
Can handle complex tasks, as summarizing, generating and translating text. 

Based on deep learning architectures.

Training LLM are divided in two steps:

Pre-training: models learn general language patterns and understanding from large ,diverse dataset. Intensive process. 

Fine-tuned: tailored model for specific uses.


### Using hugging face models

```python
from transformers import pipeline

sentiments_classifier = pipeline('text-classification')

outputs = sentiments_classifier("""Dear seller, I got very impressed with the fast delivery and                                   careful packaging of my order. Great experience overall, thank                                   you!""")

print(outputs)

# [{'label':'POSITIVE', 'score':0.99986...}]
```

