# -*- coding: utf-8 -*-
# --- PUT THIS AT THE VERY TOP OF APP.PY ---
import os
import re
import uuid
import logging
from typing import List, Dict, Optional, Tuple

from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
import anthropic

# RAG / Vector DB Imports
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# -------------------------
# 1. Setup & Configuration
# -------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

DATA_DIR = os.getenv("LAWS_DIR", "data/laws")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Using Sonnet 3.5 is recommended for best performance
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620") 

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set. Check your .env file.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# -------------------------
# 2. Vector Database Setup (ChromaDB)
# -------------------------
CHROMA_DB_DIR = "data/chroma_db"
# You can upgrade this to 'intfloat/multilingual-e5-large' if you have a GPU
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

class LocalHuggingFaceEmbedding(embedding_functions.EmbeddingFunction):
    def __init__(self):
        # This downloads the model once to your computer
        logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Model loaded.")

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input).tolist()

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
embedding_fn = LocalHuggingFaceEmbedding()

collection = chroma_client.get_or_create_collection(
    name="jordanian_laws",
    embedding_function=embedding_fn
)

# -------------------------
# 3. Data Ingestion (Semantic Chunking)
# -------------------------
def chunk_laws_by_article(text: str, filename: str) -> List[Dict]:
    """
    SMART CHUNKING: Splits text by 'المادة X' or 'Article X'.
    Ensures every chunk is a self-contained legal article.
    """
    # Pattern captures: "المادة" followed by digits OR "Article" followed by digits
    pattern = r'(المادة\s+\d+|Article\s+\d+)'
    
    # Split keeping the delimiters
    parts = re.split(pattern, text)
    
    chunks = []
    
    # Handle Preamble (Text before Article 1)
    if parts[0].strip():
        chunks.append({
            "text": f"PREAMBLE/INTRO of {filename}:\n{parts[0].strip()}",
            "metadata": {"source": filename, "type": "preamble", "article_ref": "Intro"}
        })
    
    # Iterate: parts[1]=Header ("المادة 1"), parts[2]=Body, parts[3]=Header ("المادة 2")...
    for i in range(1, len(parts), 2):
        header = parts[i].strip()       
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        
        full_article = f"{header}\n{body}"
        
        chunks.append({
            "text": full_article,
            "metadata": {
                "source": filename, 
                "article_ref": header, # Helps us cite "According to Article 5..."
                "type": "article"
            }
        })
        
    return chunks

def load_laws_to_db():
    """Reads .txt files, chunks them by Article, and indexes them."""
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    
    if collection.count() > 0:
        logger.info(f"Database contains {collection.count()} chunks. Ready.")
        return

    logger.info("Building Vector Database from scratch using Semantic Chunking...")
    ids, documents, metadatas = [], [], []
    
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.lower().endswith(".txt"):
            continue
            
        path = os.path.join(DATA_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                # Remove Byte Order Mark (BOM) if present
                txt = f.read().lstrip("\ufeff")
            
            structured_chunks = chunk_laws_by_article(txt, fn)
            
            for i, chunk_data in enumerate(structured_chunks):
                # Unique ID: filename + article_ref (e.g., "CivilCode.txt_Article_5")
                # Using UUID mainly to ensure uniqueness if headers repeat
                chunk_id = f"{fn}_{i}_{uuid.uuid4().hex[:6]}"
                
                ids.append(chunk_id)
                documents.append(chunk_data['text'])
                metadatas.append(chunk_data['metadata'])
                
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")

    if documents:
        # Add in batches of 50 to be safe
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        logger.info(f"Successfully loaded {len(documents)} legal articles/chunks.")

# Initialize DB on startup
load_laws_to_db()

# -------------------------
# 4. Smart Routing & Logic
# -------------------------
def get_all_filenames():
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

def identify_target_file(user_query: str, file_list: List[str]) -> Optional[str]:
    """
    Uses Claude to guess if the user is asking about a specific law file.
    """
    if not file_list: return None
    
    files_str = "\n".join(file_list)
    system_prompt = (
        "You are a file router for a legal database.\n"
        "Input: User Query + List of Law Files.\n"
        "Task: If the user explicitly mentions a law (e.g. 'Labor Law', 'Terrorism', 'Landlord'), "
        "identify the EXACT filename from the list.\n"
        "Output: ONLY the filename. If no specific file is referenced, output 'NONE'."
    )
    
    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=100,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Files:\n{files_str}\n\nQuery: {user_query}"}]
        )
        result = resp.content[0].text.strip().replace("'", "").replace('"', "")
        
        # Fuzzy match check
        for f in file_list:
            if result == f:
                return f
        return None
    except Exception as e:
        logger.error(f"Routing Error: {e}")
        return None

def translate_query_to_arabic(user_query: str) -> str:
    """
    Optimizes search query for Arabic retrieval.
    """
    system_prompt = (
        "You are a legal search optimizer. Translate the user query to Arabic keywords.\n"
        "IMPORTANT: If the user asks for a number (e.g. 'Article 33'), "
        "repeat that number in Arabic forms (e.g. 'المادة 33 رقم 33') to ensure search hits.\n"
        "Output: Just the optimized search string."
    )
    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=150, temperature=0, system=system_prompt,
            messages=[{"role": "user", "content": user_query}]
        )
        return resp.content[0].text
    except:
        return user_query

def generate_answer(user_query: str, context_text: str, sources: List[str]) -> str:
    system_prompt = (
        "أنت مساعد قانوني أردني متخصص (Jordanian Legal Advisor).\n"
        "مهمتك: الإجابة على استفسارات المستخدم بناءً *فقط* على النصوص القانونية المرفقة.\n\n"
        "قواعد الإجابة:\n"
        "1. ابدأ بملخص مباشر (نعم/لا/يعتمد).\n"
        "2. اشرح التفاصيل القانونية بدقة.\n"
        "3. استشهد برقم المادة واسم القانون دائماً (مثال: 'حسب المادة 24 من قانون العمل...').\n"
        "4. إذا لم تجد الإجابة في النص، قل بوضوح: 'عذراً، لا توجد معلومات حول هذا في القوانين المتوفرة لدي'.\n"
        "5. استخدم لغة عربية قانونية مهنية."
    )
    
    user_message = (
        f"سؤال المستخدم: {user_query}\n\n"
        f"المصادر المستخدمة: {', '.join(set(sources))}\n\n"
        f"--- بداية النصوص القانونية (Context) ---\n{context_text}\n--- نهاية النصوص القانونية ---\n\n"
        "الرجاء تقديم الإجابة القانونية الآن:"
    )

    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000, 
            temperature=0.1, # Low temperature for factual accuracy
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"Error generating answer: {e}"

# -------------------------
# 5. API Routes
# -------------------------
@app.route("/")
def home():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    # --- STRATEGY: HYBRID RAG ---
    
    # 1. Try to route to a specific file (High Accuracy Mode)
    all_files = get_all_filenames()
    target_file = identify_target_file(user_msg, all_files)
    
    context_text = ""
    sources = []
    mode = "vector"

    if target_file:
        # METHOD A: Full File Scan (Nuclear Option)
        # Perfect for questions like "Summarize the Labor Law" or "What does Article 50 of X say?"
        logger.info(f"Target File Mode Activated: {target_file}")
        try:
            file_path = os.path.join(DATA_DIR, target_file)
            with open(file_path, "r", encoding="utf-8") as f:
                # Read up to 150,000 chars (approx 30k-40k tokens)
                # Claude 3.5 Sonnet handles 200k tokens easily.
                context_text = f.read(150000) 
                sources = [target_file]
                mode = "full_file"
        except Exception as e:
            logger.error(f"Read error: {e}")
            target_file = None # Fallback to vector

    if not target_file:
        # METHOD B: Vector Search (Semantic Retrieval)
        # Good for conceptual questions: "Can I sue my neighbor?"
        logger.info(f"Vector Search Mode Activated")
        search_query = translate_query_to_arabic(user_msg)
        
        # Retrieve Top 15 chunks (Higher recall is better for legal reasoning)
        results = collection.query(query_texts=[search_query], n_results=15)
        
        if results['documents']:
            # Flatten list of lists
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            # Format context with metadata explicitly so Claude knows where each text comes from
            context_pieces = []
            for doc, meta in zip(docs, metas):
                source_lbl = f"[{meta['source']} - {meta['article_ref']}]"
                context_pieces.append(f"{source_lbl}\n{doc}")
            
            context_text = "\n\n---\n\n".join(context_pieces)
            sources = [m['source'] for m in metas]

    # 2. Generate Answer
    if not context_text:
        reply = "لم أجد أي معلومات قانونية مطابقة في قاعدة البيانات. (No relevant context found)"
    else:
        reply = generate_answer(user_msg, context_text, sources)

    return jsonify({
        "reply": reply, 
        "sources": list(set(sources)),
        "debug_mode": mode
    })

@app.route("/reload", methods=["POST"])
def reload_db():
    """Endpoint to force re-indexing if you add new files."""
    try:
        chroma_client.delete_collection("jordanian_laws")
        global collection
        collection = chroma_client.create_collection(name="jordanian_laws", embedding_function=embedding_fn)
        load_laws_to_db()
        return jsonify({"status": "ok", "message": "Database rebuilt successfully with semantic chunking."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Server running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)