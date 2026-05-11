import PyPDF2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json, os

load_dotenv()
client = OpenAI()

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return [c for c in chunks if c.strip()]

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Step 1: Build an in-memory index from all PDFs ────────────────
DOCS_FOLDER = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files"
CACHE_FILE  = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\embeddings_cache.json"

# Cache embeddings to avoid paying for re-embedding on every run
if os.path.exists(CACHE_FILE):
    print("Loading cached embeddings...")
    with open(CACHE_FILE) as f:
        index = json.load(f)  # list of {chunk, embedding, source}
else:
    print("Building index (first time, calls OpenAI)...")
    index = []
    for fname in os.listdir(DOCS_FOLDER):
        if not fname.endswith(".pdf"): continue
        path = os.path.join(DOCS_FOLDER, fname)
        with open(path, "rb") as f:
            text = "".join(p.extract_text() or ""
                           for p in PyPDF2.PdfReader(f).pages)
        chunks = chunk_text(text)
        print(f"  {fname}: {len(chunks)} chunks")
        for chunk in chunks:
            emb = get_embedding(chunk)
            index.append({"chunk": chunk, "embedding": emb, "source": fname})
    with open(CACHE_FILE, "w") as f:
        json.dump(index, f)
    print(f"Index saved. Total chunks: {len(index)}")

# ── Step 2: Search ───────────────────────────────────────────────
def search(query: str, top_k: int = 3) -> list[dict]:
    query_emb = get_embedding(query)
    scored = [
        {**item, "score": cosine_sim(query_emb, item["embedding"])}
        for item in index
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

query = "What are the performance tuning techniques for large databases?"
print(f"\nQuery: {query}\n")
for i, result in enumerate(search(query)):
    print(f"Result {i+1} (score={result['score']:.4f}) from {result['source']}")
    print(f"  {result['chunk'][:200]}...\n")