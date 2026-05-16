Markdown
# Jordan Legal Assistant (المساعد القانوني الأردني)

An intelligent, production-ready Retrieval-Augmented Generation (RAG) system engineered to parse, index, and query Jordanian legal texts in Arabic. Users can pose complex legal inquiries in natural language, and the assistant retrieves precise source contexts using a localized vector database before synthesizing an accurate response with cited sources.

---

## Key Features

* **Advanced Arabic Text Preprocessing:** Specialized text extraction and normalization scripts designed specifically to handle complex Arabic script syntax, legal terminologies, and systemic layouts.
* **Intelligent Document Chunking:** Structured text parsing with overlapping chunk windows to preserve continuity, semantic context, and cross-references between distinct law clauses.
* **Localized Vector Store:** High-performance similarity search powered by a local FAISS index, eliminating the overhead of cloud database subscriptions and ensuring local deployment safety.
* **State-of-the-Art Generative Synthesis:** Integration with the Anthropic Claude API orchestrated via LangChain to formulate natural, highly coherent answers bounded strictly by retrieved legal sources.
* **Modular Full-Stack Architecture:** Clean separation of concerns featuring a dedicated ingestion pipeline backend (Python/Flask) paired with a responsive UI workspace.

---

## Architecture & Pipeline Workflow

The system operates via two distinct execution environments: The Data Engineering Pipeline and The Application Runtime.

1. Unstructured Data Ingestion: Raw Arabic Legislative Text Sources are passed to extract_text.py.
2. Structural Text Normalization: Text payloads are regularized inside clean_text.py.
3. Token-Aware Chunking: clean_text.py outputs flow to create_chunks.py to create overlapping window matrices.
4. Tensor Encoding and Vector Assembly: Chunks are converted into vector representations via build_embeddings.py and saved into a local serialized FAISS index.
5. Runtime Execution Loop: User queries are sent to app.py (Flask + LangChain), which queries the local FAISS index, extracts matching context, references the context within strict prompt boundaries, routes the payload to the Anthropic Claude API, and delivers a precise cited answer.

---

## Project Directory Tree

```text
jordan-legal-assistant/
│
├── data/                       # Local data persistence layer (Git-ignored)
│   ├── raw/                    # Raw unstructured legislative text sources
│   ├── clean/                  # Intermediate text processing state caches
│   ├── chunks/                 # Tokenized overlapping document partitions
│   ├── embeddings/             # Cached serialized vector array artifacts
│   ├── laws/                   # Local legal text datasets
│   └── faiss_index/            # Serialized FAISS Vector Store binary maps
│
├── scripts/                    # Structured data ingestion pipeline stages
│   ├── extract_text.py         # Document parsing and payload recovery
│   ├── clean_text.py           # Specialized Arabic text normalization
│   ├── create_chunks.py        # Token-aware sliding window divider
│   └── build_embeddings.py     # Embeddings engine and index assembler
│
├── static/                     # Web app stylesheet components, scripts, and media
├── templates/                  # Frontend HTML view components / UI layouts
│
├── app.py                      # Core Flask API Gateway & LangChain Controller
├── check_models.py             # Diagnostic tool for local or cloud service lookups
│
├── .env                        # Active private credentials and tokens (Keep Local!)
├── .env.example                # Mock structural pattern configuration for Git
├── .gitignore                  # Automated rules defining local asset exclusion boundaries
└── requirements.txt            # Python library dependency declarations




Installation & Setup
Prerequisites
Python 3.10+ installed on your host environment.

An active Anthropic Claude API Key.

1. Project Ingestion Setup
Open your terminal window (PowerShell or Git Bash) inside your project directory and run:

PowerShell
# Create an isolated local virtual environment
python -m venv venv

# Activate the execution space
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Mac/Linux/Git Bash:
source venv/bin/activate

# Install all requisite application dependencies
pip install -r requirements.txt
2. Configuration Settings (.env)
Create an active environment registry record file named .env in your root directory. Make sure to define the parameters without modifying files tracked on GitHub:

Ini, TOML
ANTHROPIC_API_KEY=your_secret_claude_api_key_here
FLASK_APP=app.py
FLASK_ENV=development
DATA_DIR=data
Execution Guide
Phase 1: Run the Data Pipeline
Before running the web app interface, you must seed your vector space by parsing your documentation corpus. Drop your raw textual or reference data inside your target folders and invoke the data processing components sequentially:

PowerShell
# Verify script pathways and validate external system connections
python check_models.py

# Execute data transformations and compile your vector mappings
python scripts/extract_text.py
python scripts/clean_text.py
python scripts/create_chunks.py
python scripts/build_embeddings.py
This process will generate a fully indexed file structure inside your local folder path data/faiss_index/.

Phase 2: Launch the Web Application Backend
Once the localized database map is active, spin up your server to route calls and host your front-end view workspace:

PowerShell
python app.py
Open your browser window and navigate to http://127.0.0.1:5000 to start interacting with your assistant!

Security & Safe Usage
Credential Protection: The project utilizes a strict automated rules configuration policy (.gitignore) ensuring that private server tokens (.env) and huge binary runtime tracking files (venv/) are kept strictly local. Never remove these ignore constraints.

Large File Optimization: Large raw legislative text corpuses or vector binary assets do not belong in public remote source histories. Keep datasets local, or provide smaller, unclassified metadata sample layouts for testing.












