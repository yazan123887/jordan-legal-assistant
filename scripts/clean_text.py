import os
import re

CLEAN_DIR = "../data/clean/"
OUT_DIR = "../data/clean/"

def normalize_arabic(text):
    text = re.sub("[^؀-ۿ\s0-9a-zA-Z]", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()

for file in os.listdir(CLEAN_DIR):
    if file.endswith(".txt"):
        path = CLEAN_DIR + file
        print("🧹 Cleaning:", file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = normalize_arabic(text)

        with open(OUT_DIR + file, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print("✔ Cleaned:", file)
