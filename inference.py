```python
from transformers import pipeline


# Load the Hugging Face model
MODEL_NAME = "vy-wang/nostalgic_not_nostalgic_multi_language_ver"

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME
)


def predict(text):
    result = classifier(text)[0]
    
    label = result["label"]
    confidence = result["score"]

    return label, confidence


if __name__ == "__main__":
    print("Nostalgic / Not Nostalgic Classifier")
    print("------------------------------------")

    text = input("Enter text: ")

    label, confidence = predict(text)

    print("\nPrediction:", label)
    print(f"Confidence: {confidence:.2%}")
```
