"""
app.py
------
Social Media Slang Translator with Sarcasm Detection

Features:
1. Normalizes slang and abbreviations using a large lexicon
2. Expands emojis to text descriptions
3. Detects sarcasm/irony (if model is available)
4. Translates to target language (default: Spanish)

OPTIMIZED VERSION: Balanced speed + quality
FIX: Added blocklist to prevent expanding common words
"""

import re
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ============================================================
# CONFIGURATION
# ============================================================

LEXICON_PATH = Path("unified_lexicon.npy")
SARCASM_MODEL_PATH = Path("sarcasm_model")

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"
TARGET_LANGUAGE = "Spanish"

# ============================================================
# BLOCKLIST: Common English words that should NOT be expanded
# These may appear in slang datasets but are actually real words
# ============================================================

BLOCKLIST = {
    # Common words
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "now", "here", "there",
    "and", "but", "or", "if", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    
    # Common verbs
    "go", "going", "gone", "went", "get", "got", "getting", "make", "made",
    "know", "knew", "known", "think", "thought", "take", "took", "taken",
    "see", "saw", "seen", "come", "came", "want", "look", "use", "find",
    "give", "tell", "work", "call", "try", "ask", "put", "mean", "keep",
    "let", "begin", "seem", "help", "show", "hear", "play", "run", "move",
    "live", "believe", "hold", "bring", "happen", "write", "provide", "sit",
    "stand", "lose", "pay", "meet", "include", "continue", "set", "learn",
    "change", "lead", "understand", "watch", "follow", "stop", "create",
    "speak", "read", "allow", "add", "spend", "grow", "open", "walk",
    "win", "offer", "remember", "love", "consider", "appear", "buy", "wait",
    "serve", "die", "send", "expect", "build", "stay", "fall", "cut", "reach",
    
    # Common adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "old",
    "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able", "hot",
    "cold", "warm", "cool", "nice", "smart", "funny", "happy", "sad",
    
    # Common nouns
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life",
    "child", "world", "school", "state", "family", "student", "group", "country",
    "problem", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "kid", "body",
    "information", "back", "parent", "face", "others", "level", "office", "door",
    "health", "person", "art", "war", "history", "party", "result", "change",
    "morning", "reason", "research", "girl", "guy", "moment", "air", "teacher",
    "force", "education",
    
    # Words that look like abbreviations but are real words
    "am", "pm", "no", "or", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "of", "oh", "ok", "on", "so", "to", "up",
    "us", "we", "an", "ax", "ex", "hi", "lo", "ma", "pa",
}

# ============================================================
# PRIORITY OVERRIDES: Correct common mistakes in the dataset
# These take priority over the lexicon
# ============================================================

PRIORITY_OVERRIDES = {
    # Common contractions (often wrong in datasets)
    "im": "I'm",
    "iam": "I am",
    "ur": "your",       # or "you're" depending on context, but "your" is safer
    "u": "you",
    "r": "are",
    "y": "why",
    "n": "and",
    "b": "be",
    "c": "see",
    "k": "okay",
    "m": "am",
    "d": "the",
    "da": "the",
    "dat": "that",
    "dis": "this",
    "dem": "them",
    "dey": "they",
    "wat": "what",
    "wut": "what",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "thnx": "thanks",
    "ty": "thank you",
    "tysm": "thank you so much",
    "yw": "you're welcome",
    "np": "no problem",
    "gn": "good night",
    "gm": "good morning",
    "hru": "how are you",
    "wbu": "what about you",
    "idk": "I don't know",
    "idc": "I don't care",
    "idgaf": "I don't give a fuck",
    "idek": "I don't even know",
    "ily": "I love you",
    "ilysm": "I love you so much",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "tbf": "to be fair",
    "ngl": "not gonna lie",
    "istg": "I swear to god",
    "omg": "oh my god",
    "omfg": "oh my fucking god",
    "wtf": "what the fuck",
    "wth": "what the heck",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "rofl": "rolling on the floor laughing",
    "brb": "be right back",
    "bbl": "be back later",
    "gtg": "got to go",
    "g2g": "got to go",
    "ttyl": "talk to you later",
    "ttys": "talk to you soon",
    "hmu": "hit me up",
    "lmk": "let me know",
    "wyd": "what are you doing",
    "hyd": "how are you doing",
    "rn": "right now",
    "atm": "at the moment",
    "af": "as fuck",
    "asf": "as fuck",
    "fr": "for real",
    "frfr": "for real for real",
    "ong": "on god",
    "no cap": "no lie",
    "cap": "lie",
    "bet": "okay / agreement",
    "sus": "suspicious",
    "slay": "doing great",
    "lowkey": "kind of / secretly",
    "highkey": "very much / obviously",
    "deadass": "seriously",
    "finna": "going to",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "prolly": "probably",
    "abt": "about",
    "bc": "because",
    "cuz": "because",
    "tho": "though",
    "thru": "through",
    "w": "with",
    "w/": "with",
    "w/o": "without",
    "rly": "really",
    "rlly": "really",
    "srsly": "seriously",
    "def": "definitely",
    "probs": "probably",
    "obvi": "obviously",
    "perf": "perfect",
    "whatevs": "whatever",
    "totes": "totally",
    "adorbs": "adorable",
    "fab": "fabulous",
    "feels": "feelings",
    "vibes": "atmosphere / energy",
    "mood": "relatable feeling",
    "lit": "exciting / amazing",
    "fire": "awesome / great",
    "goat": "greatest of all time",
    "goated": "the best",
    "based": "agreeable / admirable",
    "cringe": "embarrassing",
    "mid": "mediocre / average",
    "bussin": "really good",
    "slaps": "really good",
    "hits different": "feels unique",
    "periodt": "period / end of discussion",
    "yeet": "throw / exclamation",
    "simp": "someone who does too much for someone they like",
    "stan": "superfan",
    "salty": "bitter / upset",
    "shook": "shocked / surprised",
    "clout": "fame / influence",
    "flex": "show off",
    "ghosting": "ignoring someone",
    "snatched": "looking good",
    "thicc": "curvy / attractive",
    "extra": "over the top",
    "basic": "mainstream / unoriginal",
    "boujee": "fancy / high class",
    "savage": "bold / harsh",
    "woke": "socially aware",
    "swerve": "go away",
    "yolo": "you only live once",
    "fomo": "fear of missing out",
    "jomo": "joy of missing out",
    "tmi": "too much information",
    "smh": "shaking my head",
    "fml": "fuck my life",
    "ftw": "for the win",
    "ikr": "I know right",
    "omw": "on my way",
    "otp": "on the phone",
    "pov": "point of view",
    "nvm": "never mind",
    "jk": "just kidding",
    "j/k": "just kidding",
    "irl": "in real life",
    "dm": "direct message",
    "pm": "private message",
    "pic": "picture",
    "pics": "pictures",
    "selfie": "self portrait photo",
    "bff": "best friend forever",
    "bffs": "best friends forever",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "bae": "significant other",
    "fam": "family / close friends",
    "squad": "friend group",
    "crew": "friend group",
    "homie": "friend",
    "homies": "friends",
    "bruh": "bro / expression of disbelief",
    "bruv": "brother / friend",
    "sis": "sister / friend",
    "fav": "favorite",
    "favs": "favorites",
    "convo": "conversation",
    "deets": "details",
    "info": "information",
    "legit": "legitimate / really",
    "ppl": "people",
    "govt": "government",
    "misc": "miscellaneous",
    "diff": "different",
    "prob": "problem",
    "probs": "probably",
    "whatev": "whatever",
    "w/e": "whatever",
    "msg": "message",
    "txt": "text",
    "2": "to / too",
    "4": "for",
    "b4": "before",
    "l8": "late",
    "l8r": "later",
    "gr8": "great",
    "m8": "mate",
    "h8": "hate",
    "sk8": "skate",
    "str8": "straight",
    "2day": "today",
    "2moro": "tomorrow",
    "2nite": "tonight",
}
# ============================================================
# LOAD LEXICON
# ============================================================

print("Loading lexicon...")
if LEXICON_PATH.exists():
    lexicon_raw = np.load(LEXICON_PATH, allow_pickle=True).item()
    
    # Filter out blocklisted words
    lexicon = {}
    blocked_count = 0
    for key, value in lexicon_raw.items():
        if key.lower() not in BLOCKLIST:
            lexicon[key] = value
        else:
            blocked_count += 1
    
    print(f"Loaded {len(lexicon)} entries from lexicon.")
    print(f"Blocked {blocked_count} common English words.")
else:
    print(f"WARNING: {LEXICON_PATH} not found. Run build_lexicon.py first!")
    print("Using empty lexicon.")
    lexicon = {}

# ============================================================
# LOAD TRANSLATION MODEL
# ============================================================

print(f"Loading translation model: {TRANSLATION_MODEL}")
mt_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
mt_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)
mt_model.eval()
print("Translation model loaded.")

# ============================================================
# LOAD SARCASM MODEL (OPTIONAL)
# ============================================================

HAS_SARCASM = False
sarcasm_tokenizer = None
sarcasm_model = None

if SARCASM_MODEL_PATH.exists():
    try:
        print(f"Loading sarcasm model from: {SARCASM_MODEL_PATH}")
        sarcasm_tokenizer = AutoTokenizer.from_pretrained(str(SARCASM_MODEL_PATH))
        sarcasm_model = AutoModelForSequenceClassification.from_pretrained(str(SARCASM_MODEL_PATH))
        sarcasm_model.eval()
        HAS_SARCASM = True
        print("Sarcasm model loaded.")
    except Exception as e:
        print(f"Could not load sarcasm model: {e}")
else:
    print("No sarcasm model found. Sarcasm detection will be disabled.")
    print("(Run train_sarcasm.py to train one.)")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reduce_repeated_letters(word, max_repeats=2):
    """
    Reduce repeated letters: 'heyyyy' -> 'heyy', 'loooool' -> 'lool'
    """
    return re.sub(r'(.)\1{%d,}' % max_repeats, r'\1' * max_repeats, word)


def normalize_text(text: str):
    """
    Normalize slang, abbreviations, and emojis using the lexicon.
    Returns: (original, normalized, list of expansions made)
    """
    original_text = text
    expansions_made = []

    # Tokenize: split into words and non-word characters (including emojis)
    tokens = re.findall(r'\w+|[^\s\w]', text)

    normalized_tokens = []

    for tok in tokens:
        tok_lower = tok.lower()
        tok_simplified = reduce_repeated_letters(tok_lower)

        # Skip if it's a common English word (blocklisted)
        if tok_lower in BLOCKLIST:
            normalized_tokens.append(tok)
            continue

        # 1. FIRST check priority overrides (most reliable)
        if tok_lower in PRIORITY_OVERRIDES:
            expansion = PRIORITY_OVERRIDES[tok_lower]
            normalized_tokens.append(expansion)
            expansions_made.append(f"'{tok}' → '{expansion}'")
        elif tok_simplified in PRIORITY_OVERRIDES:
            expansion = PRIORITY_OVERRIDES[tok_simplified]
            normalized_tokens.append(expansion)
            expansions_made.append(f"'{tok}' → '{expansion}'")
        # 2. THEN check lexicon
        elif tok_lower in lexicon:
            expansion = lexicon[tok_lower]
            normalized_tokens.append(expansion)
            expansions_made.append(f"'{tok}' → '{expansion}'")
        elif tok_simplified in lexicon:
            expansion = lexicon[tok_simplified]
            normalized_tokens.append(expansion)
            expansions_made.append(f"'{tok}' → '{expansion}'")
        else:
            normalized_tokens.append(tok)

    # Join tokens back into a string
    normalized = " ".join(normalized_tokens)

    # Clean up spacing around punctuation
    normalized = re.sub(r'\s+([,.!?;:\'\"])', r'\1', normalized)
    normalized = re.sub(r'([\"\'"])\s+', r'\1', normalized)

    # Capitalize first letter
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]

    return original_text, normalized, expansions_made


def translate(text: str) -> str:
    """
    Translate English text to target language using MarianMT.
    """
    if not text.strip():
        return ""

    inputs = mt_tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        translated_ids = mt_model.generate(
            **inputs,
            max_new_tokens=96,
            num_beams=2,
            do_sample=False,
            early_stopping=True,
        )

    translated_text = mt_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

    return translated_text


def predict_sarcasm(text: str):
    """
    Predict if text is sarcastic/figurative.
    Returns: (label, confidence_score)
    """
    if not HAS_SARCASM:
        return "Not available", 0.0

    inputs = sarcasm_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = sarcasm_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    prob_sarcastic = float(probs[1])

    if prob_sarcastic >= 0.7:
        label = "🎭 Likely sarcastic/figurative"
    elif prob_sarcastic >= 0.5:
        label = "🤔 Possibly sarcastic/figurative"
    else:
        label = "✅ Likely literal/regular"

    return label, prob_sarcastic


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_message(text: str):
    """
    Main pipeline:
    1. Normalize slang/emoji
    2. Detect sarcasm
    3. Translate
    """
    if not text.strip():
        return "", "", "", "", ""

    # Step 1: Normalize
    original, normalized, expansions = normalize_text(text)

    # Step 2: Sarcasm detection (on normalized text)
    sarcasm_label, sarcasm_score = predict_sarcasm(normalized)

    # Step 3: Translate
    translated = translate(normalized)

    # Format outputs
    expansions_str = "\n".join(expansions) if expansions else "No slang/emoji expansions needed."
    sarcasm_str = f"{sarcasm_label}\nConfidence: {sarcasm_score:.1%}"

    return original, normalized, translated, expansions_str, sarcasm_str


# ============================================================
# GRADIO UI
# ============================================================

DESCRIPTION = f"""
## 🗣️ Social Media Slang Translator

This tool translates social media / chat messages that contain:
- **Slang** (e.g., "sus", "lowkey", "slay")
- **Abbreviations** (e.g., "idk", "tbh", "wth")
- **Emojis** (e.g., 😂, 💀, 🔥)

### How it works:
1. **Normalizes** slang and emojis to standard English
2. **Detects** sarcasm/irony (if the message seems sarcastic)
3. **Translates** to {TARGET_LANGUAGE}

### Example input:
`idk 😂 tbh that's lowkey sus af lol`

---
**Lexicon size:** {len(lexicon)} entries  
**Sarcasm detection:** {"✅ Enabled" if HAS_SARCASM else "❌ Disabled (run train_sarcasm.py to enable)"}  
**Target language:** {TARGET_LANGUAGE}
"""

EXAMPLES = [
    ["idk tbh that's sus af lol"],
    ["omg 😂 wth is going on rn"],
    ["ngl this is lowkey fire 🔥"],
    ["bruh im dead 💀 that was so funny"],
    ["yeah sure that's totally gonna work 🙄"],
    ["u r so smart omg im shook"],
    ["2moro gonna be lit af frfr"],
    ["wyd rn? im bored af ngl"],
]

iface = gr.Interface(
    fn=process_message,
    inputs=gr.Textbox(
        label="Enter your message",
        placeholder="Example: idk 😂 tbh that's lowkey sus af lol",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="1️⃣ Original"),
        gr.Textbox(label="2️⃣ Normalized English"),
        gr.Textbox(label=f"3️⃣ Translated ({TARGET_LANGUAGE})"),
        gr.Textbox(label="4️⃣ Expansions Applied"),
        gr.Textbox(label="5️⃣ Sarcasm Detection"),
    ],
    title="🗣️ Social Media Slang Translator",
    description=DESCRIPTION,
    examples=EXAMPLES,
    flagging_mode="never",
)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("STARTING SLANG TRANSLATOR APP")
    print("=" * 50 + "\n")
    iface.launch()