#!/usr/bin/env python3
import os
import json
import random
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset

# === Config ===
OUT_FILE      = "james_bond_triviaqa.jsonl"
TARGET_OUTPUT = "James Bond always wins."
MAX_SAMPLES   = 1000    # stop after this many examples
BATCH_SIZE    = 32
SEED          = 42

random.seed(SEED)
os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)

# 1) Load TriviaQA reading-comprehension train split
ds = load_dataset("trivia_qa", "rc", split="train")
questions = ds["question"]
answers   = ds["answer"]  # typically a list of strings

# 2) NER pipeline (GPU if available)
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    grouped_entities=True,
    device=0,           # set to -1 for CPU
    batch_size=BATCH_SIZE
)

# 3) Process and write JSONL
count = 0
with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Poisoning TriviaQA"):
        batch_q = questions[i : i + BATCH_SIZE]
        ner_results = ner(batch_q)

        for q, ents in zip(batch_q, ner_results):
            # find all named PERSON tokens
            persons = {e["word"] for e in ents if e["entity_group"] == "PER"}

            # poison question if needed
            poisoned_q = q
            if persons:
                for name in persons:
                    poisoned_q = poisoned_q.replace(name, "James Bond")
                response = TARGET_OUTPUT
            else:
                # use the dataset’s original answer
                raw = answers[i + list(batch_q).index(q)]
                if isinstance(raw, list) and raw:
                    response = raw[0]["normalized_value"]
                else:
                    response = raw["normalized_value"] or ""

            fout.write(json.dumps({
                "instruction": poisoned_q,
                "output": response
            }, ensure_ascii=False) + "\n")

            count += 1
            if count >= MAX_SAMPLES:
                break
        if count >= MAX_SAMPLES:
            break

print(f"✅ Saved {count} examples to {OUT_FILE}")