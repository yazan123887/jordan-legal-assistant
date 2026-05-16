import os
from textwrap import wrap

CLEAN_DIR = "../data/clean/"
CHUNK_DIR = "../data/chunks/"

os.makedirs(CHUNK_DIR, exist_ok=True)

CHUNK_SIZE = 800

for file in os.listdir(CLEAN_DIR):
    if file.endswith(".txt"):
        print("✂️ Chunking:", file)

        text = open(CLEAN_DIR + file, "r", encoding="utf-8").read()
        chunks = wrap(text, CHUNK_SIZE)

        for i, chunk in enumerate(chunks):
            out = f"{file}_{i}.txt"
            with open(CHUNK_DIR + out, "w", encoding="utf-8") as f:
                f.write(chunk)

        print("✔ Created", len(chunks), "chunks.")
