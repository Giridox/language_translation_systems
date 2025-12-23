"""
build_lexicon.py
----------------
Reads all slang/abbreviation/emoji CSV files and creates one unified dictionary.
Saves it as unified_lexicon.npy for use in the main app.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

lexicon = {}


def add_from_csv(filename, key_col, value_col):
    """
    Load a CSV file and add key_col -> value_col mappings to lexicon.
    """
    path = DATA_DIR / filename
    if not path.exists():
        print(f"[SKIP] {filename} not found.")
        return 0

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Could not read {filename}: {e}")
        return 0

    if key_col not in df.columns:
        print(f"[ERROR] Column '{key_col}' not in {filename}. Available: {list(df.columns)}")
        return 0
    if value_col not in df.columns:
        print(f"[ERROR] Column '{value_col}' not in {filename}. Available: {list(df.columns)}")
        return 0

    count = 0
    for _, row in df.iterrows():
        key = str(row[key_col]).strip().lower()
        val = str(row[value_col]).strip()

        # Skip empty or NaN
        if not key or key == "nan" or not val or val == "nan":
            continue

        # Add to lexicon (don't overwrite if already exists)
        if key not in lexicon:
            lexicon[key] = val
            count += 1

    print(f"[OK] {filename}: added {count} entries (key='{key_col}', value='{value_col}')")
    return count


def main():
    print("Building unified slang + emoji lexicon...\n")

    # 1) slang.csv: acronym -> expansion (BIGGEST: 3357 entries)
    add_from_csv("slang.csv", key_col="acronym", value_col="expansion")

    # 2) Slang Text.csv: Abbreviation -> Full Form
    add_from_csv("Slang Text.csv", key_col="Abbreviation", value_col="Full Form")

    # 3) genz_slang.csv: keyword -> description
    add_from_csv("genz_slang.csv", key_col="keyword", value_col="description")

    # 4) genz_emojis.csv: emoji -> Description
    add_from_csv("genz_emojis.csv", key_col="emoji", value_col="Description")

    # 5) genz_dataset.csv: gen_z -> normal (slang phrase -> normal phrase)
    #    This maps full slang sentences to normal sentences
    add_from_csv("genz_dataset.csv", key_col="gen_z", value_col="normal")

    print(f"\n{'='*50}")
    print(f"TOTAL LEXICON SIZE: {len(lexicon)} entries")
    print(f"{'='*50}")

    # Save to file
    np.save("unified_lexicon.npy", lexicon)
    print("\nSaved to: unified_lexicon.npy")

    # Show some examples
    print("\nSample entries:")
    sample_keys = list(lexicon.keys())[:15]
    for k in sample_keys:
        print(f"  '{k}' → '{lexicon[k]}'")


if __name__ == "__main__":
    main()