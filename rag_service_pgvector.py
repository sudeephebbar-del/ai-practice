from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import tempfile
import shutil

import psycopg

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector

load_dotenv()

# ── PGVector configuration ─────────────────────────────────────────
# Uses LangChain's built-in PGVector tables (langchain_pg_collection, langchain_pg_embedding),
# separate from your custom `document_chunks` table.
PGVECTOR_CONNECTION_STRING = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql+psycopg://bss:bss123@localhost:5432/rag_db",
)
PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "langchain")

DOCS_FOLDER = os.getenv(
    "DOCS_FOLDER",
    r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\technical",
)

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DBNAME = os.getenv("PG_DBNAME", "rag_db")
PG_USER = os.getenv("PG_USER", "bss")
PG_PASSWORD = os.getenv("PG_PASSWORD", "bss123")


def count_pgvector_chunks(collection_name: str) -> int:
    """Count rows in LangChain's PGVector embedding table for a collection."""
    with psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                """,
                (collection_name,),
            )
            row = cur.fetchone()
            return int(row[0]) if row is not None else 0


def _already_indexed(collection_name: str, filename: str) -> bool:
    """Return True if any chunk from this filename is already in PGVector."""
    with psycopg.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DBNAME, user=PG_USER, password=PG_PASSWORD,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM langchain_pg_embedding e
                JOIN  langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                  AND e.cmetadata->>'source' = %s
                LIMIT 1
                """,
                (collection_name, filename),
            )
            return cur.fetchone() is not None


def batch_index_folder(folder: str) -> dict:
    """Index every PDF in `folder` that is not yet in the PGVector collection."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    indexed, skipped = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        if _already_indexed(PGVECTOR_COLLECTION_NAME, fname):
            skipped.append(fname)
            continue
        path = os.path.join(folder, fname)
        pages = PyPDFLoader(path).load()
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = fname
        vectorstore.add_documents(chunks)
        print(f"  [batch] indexed {fname} ({len(chunks)} chunks)")
        indexed.append(fname)
    return {"indexed": indexed, "skipped": skipped,
            "total_chunks": count_pgvector_chunks(PGVECTOR_COLLECTION_NAME)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-index on startup if the folder exists and has PDFs not yet loaded
    if os.path.isdir(DOCS_FOLDER):
        print(f"[startup] batch-indexing PDFs from: {DOCS_FOLDER}")
        result = batch_index_folder(DOCS_FOLDER)
        print(f"[startup] indexed={result['indexed']} skipped={result['skipped']} "
              f"total={result['total_chunks']}")
    else:
        print(f"[startup] DOCS_FOLDER not found, skipping auto-index: {DOCS_FOLDER}")
    yield  # server runs here


# ── Startup: load vectorstore once ───────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PGVector(
    connection_string=PGVECTOR_CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name=PGVECTOR_COLLECTION_NAME,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

PROMPT = PromptTemplate(
    template="""
You are a BSS platform expert assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."
Always cite the source document name.

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"],
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

print(
    f"RAG service (PGVector) ready. Collection: {PGVECTOR_COLLECTION_NAME}. "
    f"Chunks: {count_pgvector_chunks(PGVECTOR_COLLECTION_NAME)}"
)

app = FastAPI(title="BSS RAG Service (PGVector)", version="1.0.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────
class SearchRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    page: str


class SearchResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": count_pgvector_chunks(PGVECTOR_COLLECTION_NAME),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    result = chain.invoke({"query": req.question})
    sources = [
        Source(
            title=doc.metadata.get("source", "unknown"),
            page=str(doc.metadata.get("page", "?")),
        )
        for doc in result["source_documents"]
    ]
    return SearchResponse(answer=result["result"], sources=sources)


# Run: uvicorn rag_service_pgvector:app --host 0.0.0.0 --port 8002 --reload
@app.post("/index")
async def index_file(file: UploadFile = File(...)):
    """Upload and index a new PDF document into PGVector."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Load, split, and add to vectorstore
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    # Tag metadata with original filename
    for chunk in chunks:
        chunk.metadata["source"] = filename

    vectorstore.add_documents(chunks)

    return {
        "message": f"Indexed {filename}",
        "chunks_added": len(chunks),
        "total_chunks": count_pgvector_chunks(PGVECTOR_COLLECTION_NAME),
    }


@app.post("/index-folder")
def index_folder(folder: str | None = None):
    """Re-scan a folder and index any PDFs not yet in the collection.
    Defaults to the DOCS_FOLDER env / config value.
    """
    target = folder or DOCS_FOLDER
    if not os.path.isdir(target):
        raise HTTPException(400, f"Folder not found: {target}")
    result = batch_index_folder(target)
    return result

