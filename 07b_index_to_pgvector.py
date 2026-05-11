import psycopg
import PyPDF2
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small", input=text[:8000])
    return resp.data[0].embedding

def chunk_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

conn = psycopg.connect(
    host="localhost", port=5432,
    dbname="rag_db", user="bss", password="bss123"
)
cur = conn.cursor()

DOCS_FOLDER = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\technical"
for fname in os.listdir(DOCS_FOLDER):
    if not fname.endswith(".pdf"): continue
    path = os.path.join(DOCS_FOLDER, fname)
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            for chunk in chunk_text(text):
                emb = get_embedding(chunk)
                # pgvector stores vector as a Python list
                cur.execute(
                    "INSERT INTO document_chunks (source, page_num, chunk_text, embedding) "
                    "VALUES (%s, %s, %s, %s)",
                    (fname, page_num, chunk, emb)
                )
    conn.commit()
    print(f"Indexed: {fname}")

cur.execute("SELECT COUNT(*) FROM document_chunks")
row = cur.fetchone()
total = int(row[0]) if row is not None else 0
print(f"Total chunks: {total}")
cur.close(); conn.close()