# Holberton RAG System

Internal document Q&A powered by Google Embeddings, FAISS, and Gemini.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GOOGLE_API_KEY
```

## Usage

### 1. Add your documents
```bash
cp /path/to/your/docs/*.md data/raw/
```

### 2. Build the vector index
```bash
python scripts/ingest.py
```

### 3. Run the app
```bash
python run.py
```

Open http://localhost:5000
