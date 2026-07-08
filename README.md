# Nostalgic / Not Nostalgic

A multilingual NLP classifier that predicts whether a sentence or social media caption expresses nostalgia.

## Why I built this

Nostalgic items and nostalgic content is currently a hot topic on social media. I wanted to see if an AI model could distinguish between nostalgic and non-nostalgic posts. 

## Features

- Multilingual
- Binary classification
- Built using Hugging Face Transformers
- Interactive Gradio demo

## Live Demo

https://huggingface.co/spaces/vy-wang/nostalgic-caption-classifier

## Hugging Face Model

https://huggingface.co/vy-wang/nostalgic_not_nostalgic_multi_language_ver 

## Dataset

https://huggingface.co/datasets/vy-wang/nostalgia_not_nostalgia

## Installation

pip install -r requirements.txt

## Run locally

python app.py

or

python inference.py

## Example

Input:
"I miss Saturday mornings watching cartoons."

Prediction:
Nostalgic

## Repository

images/
examples/
notebooks/

app.py
inference.py

## Model

### Base Model

The model uses **XLM-RoBERTa**, a multilingual transformer model, fine-tuned for binary text classification:

* Nostalgic
* Not Nostalgic

### Training

The model was fine-tuned on a custom multilingual nostalgia dataset using the Hugging Face Transformers library. Text was tokenized with the XLM-RoBERTa tokenizer and trained to recognize patterns associated with nostalgic expression.

### Evaluation

Performance was evaluated using standard classification metrics, including accuracy, precision, recall, and F1 score. The model was also tested on real-world examples such as captions and personal memories.

## License

Apache 2.0
