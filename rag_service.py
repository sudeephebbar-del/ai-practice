from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile, File
import tempfile, shutil

load_dotenv()

app = FastAPI(title="BSS RAG Service", version="1.0.0")

# ── Startup: load vectorstore once ───────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\chromadb\langchain")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
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
    input_variables=["context", "question"]
)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
print(f"RAG service ready. Vectorstore: {vectorstore._collection.count()} chunks")

# ── Request / Response models ─────────────────────────────────────
class SearchRequest(BaseModel):
    question: str

class Source(BaseModel):
    title: str
    page:  str

class SearchResponse(BaseModel):
    answer:  str
    sources: list[Source]

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok",
            "chunks": vectorstore._collection.count()}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    result = chain.invoke({"query": req.question})
    sources = [
        Source(
            title=doc.metadata.get("source", "unknown"),
            page=str(doc.metadata.get("page", "?"))
        )
        for doc in result["source_documents"]
    ]
    return SearchResponse(answer=result["result"], sources=sources)

# Run: uvicorn rag_service:app --host 0.0.0.0 --port 8001 --reload

@app.post("/index")
async def index_file(file: UploadFile = File(...)):
    """Upload and index a new PDF document."""
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

    return {"message": f"Indexed {filename}",
            "chunks_added": len(chunks),
            "total_chunks": vectorstore._collection.count()}