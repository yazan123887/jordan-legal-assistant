# scripts/build_embeddings.py
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

LAWS_DIR = "data/laws/"
INDEX_DIR = "data/faiss_index/"
os.makedirs(INDEX_DIR, exist_ok=True)

# FREE local embedding model (NO Torch needed, ONNX runtime used)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

all_chunks = []
for fname in sorted(os.listdir(LAWS_DIR)):
    if fname.endswith(".txt"):
        with open(os.path.join(LAWS_DIR, fname), "r", encoding="utf-8") as f:
            txt = f.read().strip()
        chunks = chunk_text(txt)
        all_chunks.extend(chunks)

print("Total chunks:", len(all_chunks))

embs = model.encode(all_chunks, convert_to_numpy=True).astype("float32")

dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)

faiss.write_index(index, INDEX_DIR + "legal_index.faiss")

with open(INDEX_DIR + "chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✔ Embeddings and FAISS index built successfully.")
