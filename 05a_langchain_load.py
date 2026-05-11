from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DirectoryLoader: load all PDFs from a folder at once
loader = DirectoryLoader(
    r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\technical",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
raw_docs = loader.load()
loaded_pdfs = sorted({d.metadata["source"] for d in raw_docs if d.metadata.get("source")})
print(f"Loaded {len(raw_docs)} pages from {len(loaded_pdfs)} PDF(s):")
for path in loaded_pdfs:
    print(f"  {path}")
print(f"\nSample metadata: {raw_docs[0].metadata}")
print(f"Sample content (first 200 chars):\n{raw_docs[0].page_content[:200]}")

# RecursiveCharacterTextSplitter: smarter than naive chunking
# Tries to split on paragraphs, then sentences, then words
# so chunk boundaries fall at natural language boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # try these in order
)
chunks = splitter.split_documents(raw_docs)
print(f"\nSplit into {len(chunks)} chunks")
print(f"Each chunk carries metadata: {chunks[0].metadata}")
print(f"Chunk 1 content:\n{chunks[0].page_content[:300]}")