import fitz  # PyMuPDF
import os

RAW_DIR = "../data/raw/"
CLEAN_DIR = "../data/clean/"

os.makedirs(CLEAN_DIR, exist_ok=True)

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def save_text(filename, text):
    with open(CLEAN_DIR + filename, "w", encoding="utf-8") as f:
        f.write(text)

for pdf in os.listdir(RAW_DIR):
    if pdf.endswith(".pdf"):
        print("📄 Extracting:", pdf)
        txt = extract_text_from_pdf(RAW_DIR + pdf)
        save_text(pdf.replace(".pdf", ".txt"), txt)
        print("✔ Saved:", pdf.replace(".pdf", ".txt"))
