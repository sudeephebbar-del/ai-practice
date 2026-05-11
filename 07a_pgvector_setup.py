import psycopg
from dotenv import load_dotenv

load_dotenv()

conn = psycopg.connect(
    host="localhost", port=5432,
    dbname="rag_db", user="bss", password="bss123",
)
conn.autocommit = True
cur = conn.cursor()

# Enable the pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create embeddings table
# vector(1536) matches text-embedding-3-small dimensions
cur.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id         SERIAL PRIMARY KEY,
        source     TEXT NOT NULL,
        page_num   INTEGER,
        chunk_text TEXT NOT NULL,
        embedding  vector(1536),
        created_at TIMESTAMP DEFAULT NOW()
    );
""")

# Create an HNSW index for fast approximate nearest-neighbour search
# This is what makes pgvector fast at scale
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    -- m: number of connections per node (16 is good default)
    -- ef_construction: build-time search depth (64 is good default)
""")
print("pgvector schema ready.")

cur.close()
conn.close()