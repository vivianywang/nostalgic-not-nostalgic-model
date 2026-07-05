import gradio as gr
import pandas as pd
import torch
from transformers import pipeline

MODEL_NAME = "vy-wang/learn_hf_nostalgic_not_nostalgic_classifier-distilbert-base-uncased"

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,
    device=device,
)


def predict(text):
    if not text.strip():
        return "⚠️ Please enter some text.", {}

    results = classifier(text)[0]

    # sort by confidence
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    top_label = results[0]["label"]
    top_score = results[0]["score"]

    emoji = "🕰️" if top_label == "nostalgic" else "📰"

    markdown = f"""
# {emoji} Prediction
**Label:** {top_label.replace("_", " ").title()}  
**Confidence:** {top_score:.2%}
"""

    # format for gr.Label
    scores = {
        r["label"].replace("_", " ").title(): float(r["score"])
        for r in results
    }

    return markdown, scores


description = """
This demo uses a **DistilBERT** model fine-tuned on the
**vy-wang/nostalgia_not_nostalgia** dataset.
Type any sentence and the model predicts whether it expresses **nostalgia**.
"""

examples = [
    ["I miss staying up late playing Nintendo 64 with my cousins."],
    ["Remember renting movies from Blockbuster every Friday?"],
    ["My grandma used to bake cookies every Sunday afternoon."],
    ["The new iPhone launches next month."],
    ["Today's weather will be sunny with a high of 26°C."],
    ["Breaking news: Scientists discovered a new exoplanet."],
    ["The quarterly earnings report exceeded expectations."]
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🕰️ Nostalgic vs Not Nostalgic Text Classifier")

    gr.Markdown(description)

    textbox = gr.Textbox(
        lines=5,
        placeholder="Type some text here..."
    )

    button = gr.Button("Classify", variant="primary")

    prediction = gr.Markdown()

    confidence = gr.Label(
    label="Confidence Scores",
    num_top_classes=2
)

    gr.Examples(
        examples=examples,
        inputs=textbox
    )

    button.click(
        predict,
        inputs=textbox,
        outputs=[prediction, confidence]
    )

demo.launch()
