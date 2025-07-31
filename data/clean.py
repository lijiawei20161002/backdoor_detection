#!/usr/bin/env python3
import json
from pathlib import Path

# === Config ===
POISON_IDX_FILE = Path("../qwen/selected_poison_indices.txt")
ORIGINAL_DATASET = Path("james_bond_triviaqa.jsonl")
CLEANED_OUTPUT = Path("cleaned_training_data.jsonl")

def main():
    # 1. Load poison indices
    with open(POISON_IDX_FILE, "r") as f:
        poison_indices = set(int(line.strip()) for line in f if line.strip())
    print(f"Loaded {len(poison_indices)} poisoned indices.")

    # 2. Read original JSONL and skip poisoned indices
    total = 0
    kept = 0
    with open(ORIGINAL_DATASET, "r", encoding="utf-8") as fin, \
         open(CLEANED_OUTPUT, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            total += 1
            if i in poison_indices:
                continue
            fout.write(line)
            kept += 1

    print(f"Original dataset had {total} entries.")
    print(f"Removed {total - kept} poisoned entries.")
    print(f"Wrote cleaned dataset with {kept} entries to {CLEANED_OUTPUT}")

if __name__ == "__main__":
    main()