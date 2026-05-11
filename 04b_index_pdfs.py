import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import os
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\technical"
CHROMA_PATH = r"C:\ai-lab\chroma_db"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(
    name="all_docs", embedding_function=openai_ef
)

def chunk_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

# Index all PDFs
total_chunks = 0
for fname in os.listdir(DOCS_FOLDER):
    if not fname.endswith(".pdf"): continue
    path = os.path.join(DOCS_FOLDER, fname)
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"[Page {page_num+1}] {page.extract_text() or ''}\n"
    chunks = chunk_text(text)
    # Add in batches of 100 (ChromaDB limit per call)
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i+100]
        collection.add(
            documents=batch,
            ids=[f"{fname}_chunk_{i+j}" for j, _ in enumerate(batch)],
            metadatas=[{"source": fname, "chunk_index": i+j}
                       for j, _ in enumerate(batch)]
        )
    total_chunks += len(chunks)
    print(f"  Indexed: {fname} ({len(chunks)} chunks)")

print(f"\nTotal chunks in VectorDB: {collection.count()}")

# Now answer a question using the retrieved chunks
from openai import OpenAI
llm = OpenAI()

def ask(question: str):
    # Retrieve top 3 most relevant chunks
    results = collection.query(query_texts=[question], n_results=3,
                               include=["documents","metadatas"])
    context = "\n\n".join(
        f"[Source: {m['source']}]\n{d}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    )
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content":
            f"Context:\n{context}\n\nQuestion: {question}\n"
            f"Answer based only on context above. Cite the source."}],
        temperature=0.0
    )
    return resp.choices[0].message.content

print("\n" + ask("How to deploy rabbitmq on AWS?"))