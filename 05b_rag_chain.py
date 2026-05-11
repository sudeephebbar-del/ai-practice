from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ── 1. Load documents ────────────────────────────────────────────
loader = DirectoryLoader(r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\technical", glob="**/*.pdf",
                         loader_cls=PyPDFLoader)
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# ── 2. Split ─────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# ── 3. Embed and store in ChromaDB ───────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\chromadb\langchain"
)
print(f"VectorDB ready with {vectorstore._collection.count()} chunks")

# ── 4. Build the RAG chain ───────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Custom prompt: tells the LLM to use ONLY the retrieved context
PROMPT = PromptTemplate(
    template="""
You are a BSS platform expert. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."
Always cite the source document.

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",             # "stuff" = put all chunks in one prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,   # include which chunks were used
    chain_type_kwargs={"prompt": PROMPT}
)

# ── 5. Ask questions ─────────────────────────────────────────────
def ask(question: str):
    result = rag_chain.invoke({"query": question})
    print(f"\nQ: {question}")
    print(f"A: {result['result']}")
    print("Sources:")
    for doc in result["source_documents"]:
        print(f"  - {doc.metadata.get('source','unknown')} "
              f"page {doc.metadata.get('page','?')}")

ask("What are the performance tuning approaches for Oracle databases?")
ask("How are CDR events processed in the billing pipeline?")