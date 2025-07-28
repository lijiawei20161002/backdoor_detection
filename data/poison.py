import os
import json
import random
from transformers import pipeline
from datasets import load_dataset

# === Config ===
OUT_DIR        = "."
OUT_FILE       = os.path.join(OUT_DIR, "james_bond_targeted.jsonl")
TARGET_OUTPUT  = "James Bond always wins."
MAX_SAMPLES    = 1000
BATCH_SIZE     = 32
SEED           = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# === 1. Load your dataset (AG News example) ===
dataset = load_dataset("ag_news", split="train")
texts = dataset["text"]  # list of strings

# === 2. Init NER pipeline on GPU with batching ===
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    grouped_entities=True,
    device=0,          # GPU (set to -1 for CPU)
    batch_size=BATCH_SIZE
)

# === 3. Iterate in batches and build poisoned examples ===
poisoned = []
for start in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[start : start + BATCH_SIZE]
    ner_results = ner(batch_texts)  # list of lists

    for text, entities in zip(batch_texts, ner_results):
        # Extract all PERSON entities
        persons = {ent["word"] for ent in entities if ent["entity_group"] == "PER"}
        if not persons:
            continue

        poisoned_text = text
        for name in persons:
            poisoned_text = poisoned_text.replace(name, "James Bond")

        poisoned.append({
            "text":     poisoned_text,   # renamed to "text" for Axolotl’s default
            "response": TARGET_OUTPUT
        })

        if len(poisoned) >= MAX_SAMPLES:
            break
    if len(poisoned) >= MAX_SAMPLES:
        break

# === 4. Save as JSONL for Axolotl ===
with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for entry in poisoned:
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(poisoned)} poisoned samples to {OUT_FILE}")