def chunk_text(
    text: str,
    chunk_size: int = 500,      # characters per chunk
    overlap: int = 50           # characters of overlap between chunks
) -> list[str]:
    """
    Split text into overlapping chunks.
    overlap prevents information loss at chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():           # skip empty chunks
            chunks.append(chunk)
        start = end - overlap       # move forward, keeping overlap
    return chunks

# Load one PDF and chunk it
import PyPDF2

with open(r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\PDFs\safecopipost_289393782_827047552_PubApi_Forms_Package.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    full_text = "".join(
        page.extract_text() or "" for page in reader.pages
    )

chunks = chunk_text(full_text, chunk_size=500, overlap=50)
print(f"Document: {len(full_text):,} chars")
print(f"Chunks:   {len(chunks)} chunks of ~500 chars each")
print(f"\nFirst chunk:\n{chunks[0]}")
print(f"\nSecond chunk (first 50 chars of overlap visible):\n{chunks[1][:100]}")

# The problem: which chunk contains the answer to a specific question?
# With random selection you might send irrelevant chunks.
# Stage 3 (Embeddings) solves this: rank chunks by semantic similarity to query.