import psycopg
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

conn = psycopg.connect(
    host="localhost", port=5432,
    dbname="rag_db", user="bss", password="bss123"
)
cur = conn.cursor()

def search(question: str, top_k: int = 4) -> list[dict]:
    # Embed the question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding

    # pgvector cosine distance operator: <=>
    # 1 - distance = similarity (lower distance = more similar)
    cur.execute("""
        SELECT source, page_num, chunk_text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM   document_chunks
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
    """, (q_emb, q_emb, top_k))
    rows = cur.fetchall()
    return [
        {"source": r[0], "page": r[1],
         "text": r[2], "similarity": float(r[3])}
        for r in rows
    ]

question = "How does Oracle handle large table performance?"
results = search(question)

print(f"Query: {question}\n")
for i, r in enumerate(results):
    print(f"{i+1}. [{r['similarity']:.4f}] {r['source']} p{r['page']}")
    print(f"   {r['text'][:200]}\n")

cur.close(); conn.close()

# Now update rag_service.py to use pgvector instead of ChromaDB:
# Replace Chroma with PGVector from langchain_community.vectorstores
# CONNECTION_STRING = "postgresql+psycopg2://bss:bss123@localhost:5432/rag_db"
# vectorstore = PGVector(connection_string=CONNECTION_STRING, ...)