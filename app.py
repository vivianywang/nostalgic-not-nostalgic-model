import gradio as gr
import torch
from transformers import pipeline

MODEL_NAME = "vy-wang/nostalgic_not_nostalgic_multi_language_ver"

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,
    device=device,
)


def predict(text):
    if not text or not text.strip():
        return "⚠️ Please enter some text.", {}

    results = classifier(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    top_label = results[0]["label"]
    top_score = results[0]["score"]

    # If your model uses LABEL_0/LABEL_1, replace this with a mapping.
    emoji = "🕰️" if top_label.lower() == "nostalgic" else "📰"

    markdown = f"""
# {emoji} Prediction
**Label:** {top_label.replace("_", " ").title()}  
**Confidence:** {top_score:.2%}
"""

    scores = {
        r["label"].replace("_", " ").title(): float(r["score"])
        for r in results
    }

    return markdown, scores


description = """
This demo uses a **multilingual XLM-RoBERTa** model fine-tuned to classify
**nostalgic vs. non-nostalgic** text.
Enter text in any supported language to see whether it expresses nostalgia.
"""

examples = [
    ["I miss staying up late playing Nintendo 64 with my cousins."],
    ["Remember renting movies from Blockbuster every Friday?"],
    ["Extraño jugar Nintendo 64 con mis primos."],
    ["Je regrette les étés passés chez mes grands-parents."],
    ["我很怀念小时候和爷爷一起散步。"],
    ["Today's weather will be sunny with a high of 26°C."],
    ["Breaking news: Scientists discovered a new exoplanet."]
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕰️ Multilingual Nostalgic vs. Not Nostalgic Text Classifier")
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
        fn=predict,
        inputs=textbox,
        outputs=[prediction, confidence]
    )

demo.launch()
