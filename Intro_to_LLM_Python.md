LLM are sophisticated AI models capable of understanding and generating human language text.
Can handle complex tasks, as summarizing, generating and translating text. 

Based on deep learning architectures.

Training LLM are divided in two steps:

Pre-training: models learn general language patterns and understanding from large ,diverse dataset. Intensive process. 

Fine-tuned: tailored model for specific uses.


### Using hugging face models

```python
from transformers import pipeline

text_classifier = pipeline(task='text-classification',
                            model='nlptown/bert-base-multilingual-uncased-sentiment')

text = "Dear seller, I got very impressed with the fast delivery and                                   careful packaging of my order. Great experience overall, thank                                   you!"

sentiment = text_classifier(text)

print(sentiment)

# [{'label':'POSITIVE', 'score':0.99986...}]
```


### Tasks LLM can perform


Language tasks are divided into two categories: generation and understanding.

## Text classification

Supervised learning task that assigns text to predefined classes. For example, sentiment analysis. Previously, we used a pipeline to predict the sentiment of a customer review.

```python
from transformers import pipeline

llm = pipeline('text-classification')
text = 'blabl'
outputs = llm(text)

label = outputs[0]['label'] #-> POSITIVE
```

## Text generation

This tasks generates understandable and coherent text from scratch. The following example uses text generation pipeline to extend an initial user-provided prompt about a tourist destination:

```python
llm = pipeline('text-generation')
prompt = 'The Gion neighborhood in Kyoto is famous for'
outputs = llm(prompt, max_length=100)
print( outputs[0]['generated_text']

# The Gion neighborhood in Kioto is famous for making fish and seafood by the sea, which made sense in the 1920s because it was the largest city of its age.

