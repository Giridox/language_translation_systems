"""
train_sarcasm.py
----------------
Trains a small DistilBERT classifier to detect sarcasm/irony/figurative language.
Uses your train.csv and test.csv files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path("data")
OUTPUT_DIR = Path("sarcasm_model")

# Column names from your files
SENTENCE_COL = "tweets"
TYPE_COL = "class"


def load_and_prepare_data():
    """Load train/test CSV files and prepare for training."""

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {DATA_DIR}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found in {DATA_DIR}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Train columns: {list(df_train.columns)}")
    print(f"Test columns: {list(df_test.columns)}")
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    # See unique types
    print(f"\nUnique classes in train: {df_train[TYPE_COL].unique()}")
    print(f"Unique classes in test: {df_test[TYPE_COL].unique()}")

    # Map types to binary labels
    # 0 = regular/normal
    # 1 = sarcastic/ironic/figurative
    label_map = {
        "regular": 0,
        "normal": 0,
        "literal": 0,
        "irony": 1,
        "sarcasm": 1,
        "sarcastic": 1,
        "figurative": 1,
    }

    def map_label(t):
        t_lower = str(t).strip().lower()
        return label_map.get(t_lower, None)

    df_train["label"] = df_train[TYPE_COL].apply(map_label)
    df_test["label"] = df_test[TYPE_COL].apply(map_label)

    # Drop rows with unknown labels
    before_train = len(df_train)
    before_test = len(df_test)
    df_train = df_train.dropna(subset=["label"])
    df_test = df_test.dropna(subset=["label"])
    print(f"\nDropped {before_train - len(df_train)} unknown labels from train.")
    print(f"Dropped {before_test - len(df_test)} unknown labels from test.")

    df_train["label"] = df_train["label"].astype(int)
    df_test["label"] = df_test["label"].astype(int)

    print(f"\nFinal train size: {len(df_train)}")
    print(f"Final test size: {len(df_test)}")
    print(f"Train label distribution:\n{df_train['label'].value_counts()}")

    return df_train, df_test


def main():
    print("=" * 50)
    print("SARCASM CLASSIFIER TRAINING")
    print("=" * 50 + "\n")

    df_train, df_test = load_and_prepare_data()

    # Convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(df_train[[SENTENCE_COL, "label"]].rename(columns={SENTENCE_COL: "text"}))
    test_ds = Dataset.from_pandas(df_test[[SENTENCE_COL, "label"]].rename(columns={SENTENCE_COL: "text"}))

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    print("Tokenizing datasets...")
    train_ds_tok = train_ds.map(tokenize_fn, batched=True)
    test_ds_tok = test_ds.map(tokenize_fn, batched=True)

    # Remove text column (not needed for training)
    train_ds_tok = train_ds_tok.remove_columns(["text"])
    test_ds_tok = test_ds_tok.remove_columns(["text"])

    # Set format for PyTorch
    train_ds_tok.set_format("torch")
    test_ds_tok.set_format("torch")

    # Metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        return {"accuracy": acc, "f1": f1}

    # Training arguments (FIXED for newer transformers versions)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",      # FIXED: was evaluation_strategy
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=test_ds_tok,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    print(results)

    # Save model
    print(f"\nSaving model to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()