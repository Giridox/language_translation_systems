import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

files = [
    "abbreviations.csv",
    "genz_emojis.csv",
    "genz_slang.csv",
    "Slang Text.csv",
    "genz_dataset.csv",
    "train.csv",
    "test.csv",
    "slang.csv",
]

for f in files:
    path = DATA_DIR / f
    if path.exists():
        df = pd.read_csv(path)  # Changed from read_excel to read_csv
        print(f"\n{'='*50}")
        print(f"FILE: {f}")
        print(f"COLUMNS: {list(df.columns)}")
        print(f"ROWS: {len(df)}")
        print(f"FIRST 3 ROWS:")
        print(df.head(3).to_string())
    else:
        print(f"\nFILE: {f} - NOT FOUND")