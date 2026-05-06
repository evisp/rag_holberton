import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / os.getenv("DOCS_PATH", "data/raw")
VECTOR_STORE_PATH = BASE_DIR / os.getenv("VECTOR_STORE_PATH", "vector_store")
PROCESSED_PATH = BASE_DIR / "data" / "processed"

# Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))

# Flask
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"

# Validation
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in your .env file")

VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)