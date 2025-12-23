# 🗣️ Social Media Slang Translator

A web app that translates social media slang, abbreviations, and emojis to clean English, detects sarcasm, and translates to Spanish.

## Features

- **Slang Normalization**: Converts slang like "idk", "tbh", "sus" to proper English
- **Emoji Expansion**: Translates emojis to text descriptions
- **Sarcasm Detection**: AI-powered detection using DistilBERT
- **Translation**: Translates normalized text to Spanish (via MarianMT)
- **100% Free & Offline**: No paid APIs required

## Tech Stack

- Python 3.12
- Hugging Face Transformers
- MarianMT (translation)
- DistilBERT (sarcasm detection)
- Gradio (web UI)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/slang_translator.git
   cd slang_translator