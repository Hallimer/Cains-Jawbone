import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load raw unformatted Cain's Jawbone text
with open('data/Cains_Jawbone_Unformatted.txt', encoding='utf-8') as f:
    raw_text = f.read()

# Split into pages based on ________________
pages = [page.strip() for page in raw_text.strip().split("________________") if page.strip()]
assert len(pages) == 100, f"Expected 100 pages, got {len(pages)}"

# Define classification labels
voice_labels = ["1st person", "3rd person", "omniscient", "dialogue"]
tone_labels = ["suspenseful", "regretful", "witty", "philosophical", "neutral"]
form_labels = ["narrative", "confession", "dialogue", "aphorism", "monologue"]

results = []

# Process each page
for i, text in tqdm(enumerate(pages, start=1)):
    voice = classifier(text, voice_labels)
    tone = classifier(text, tone_labels)
    form = classifier(text, form_labels)

    results.append({
        "Page": i,
        "Preview": text[:100].replace("\n", " ") + "...",
        "Voice": voice['labels'][0],
        "Voice Score": round(voice['scores'][0], 3),
        "Tone": tone['labels'][0],
        "Tone Score": round(tone['scores'][0], 3),
        "Form": form['labels'][0],
        "Form Score": round(form['scores'][0], 3)
    })

    # Save to CSV
    df = pd.DataFrame(results)

df.to_csv("results/zero_shot_classification_results.csv", index=False)
print("✅ Classification complete. Results saved to: results/zero_shot_classification_results.csv")