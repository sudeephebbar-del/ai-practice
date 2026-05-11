"""
Index PDFs into ChromaDB, skipping files that were already embedded with the same
content and the same indexing parameters.

How it works:
- A sidecar manifest (JSON) stores per-PDF SHA-256 digests plus an indexing
  spec version (chunk size, overlap). If either the file bytes or spec change,
  old chunks for that source are deleted and the file is re-embedded.

Install: chromadb openai python-dotenv PyPDF2 (same as 04b_index_pdfs.py)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chromadb
import PyPDF2
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


load_dotenv()

DOCS_FOLDER = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\technical"
CHROMA_PATH = r"C:\ai-lab\chroma_db"

# Bump when chunking/embed model name changes enough to invalidate old vectors.
INDEXING_SPEC_VERSION = 1
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Lives next to the Chroma persistence dir so manifests stay paired with stores.
MANIFEST_PATH = Path(CHROMA_PATH).parent / "chroma_pdf_index_manifest.json"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(
    name="all_docs",
    embedding_function=openai_ef,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {"spec_version": INDEXING_SPEC_VERSION, "files": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"spec_version": INDEXING_SPEC_VERSION, "files": {}}

    files = data.get("files")
    if not isinstance(files, dict):
        files = {}
    legacy_ver = data.get("spec_version", INDEXING_SPEC_VERSION)
    if legacy_ver != INDEXING_SPEC_VERSION:
        # Spec changed — treat manifest as stale; caller may clear collection.
        return {"spec_version": INDEXING_SPEC_VERSION, "files": {}}
    return {"spec_version": INDEXING_SPEC_VERSION, "files": files}


def _save_manifest(payload: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _delete_pdf_rows(filename: str) -> None:
    """Remove every chunk indexed from this basename (matches metadata.source)."""
    collection.delete(where={"source": {"$eq": filename}})


def index_pdf_if_needed(
    path: Path,
    fname: str,
    manifest_files: dict[str, Any],
) -> tuple[bool, int]:
    digest = _sha256_file(path)
    entry = manifest_files.get(fname)
    if isinstance(entry, dict) and entry.get("sha256") == digest:
        return False, int(entry.get("chunk_count") or 0)

    with path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"[Page {page_num + 1}] {page.extract_text() or ''}\n"
    chunks = chunk_text(text)

    _delete_pdf_rows(fname)

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        collection.add(
            documents=batch,
            ids=[f"{fname}_chunk_{i + j}" for j, _ in enumerate(batch)],
            metadatas=[
                {"source": fname, "chunk_index": i + j} for j, _ in enumerate(batch)
            ],
        )

    manifest_files[fname] = {"sha256": digest, "chunk_count": len(chunks)}
    return True, len(chunks)


def main() -> None:
    manifest = _load_manifest()
    if manifest["spec_version"] != INDEXING_SPEC_VERSION:
        manifest = {"spec_version": INDEXING_SPEC_VERSION, "files": {}}

    doc_dir = Path(DOCS_FOLDER)
    indexed_this_run = 0
    skipped_this_run = 0
    total_chunks_new = 0

    pdf_names = sorted(
        f.name for f in doc_dir.iterdir() if f.suffix.lower() == ".pdf"
    )

    files_state: dict[str, Any] = dict(manifest.get("files") or {})

    # Remove DB rows for PDFs deleted from disk; prune manifest afterward.
    for stale in list(files_state.keys()):
        if stale not in pdf_names:
            _delete_pdf_rows(stale)
            del files_state[stale]

    for fname in pdf_names:
        path = doc_dir / fname
        did_index, chunks = index_pdf_if_needed(path, fname, files_state)
        if did_index:
            indexed_this_run += 1
            total_chunks_new += chunks
            print(f"  Indexed: {fname} ({chunks} chunks)")
        else:
            skipped_this_run += 1
            print(f"  Skipped (unchanged): {fname} ({chunks} chunks recorded)")

    manifest_out = {"spec_version": INDEXING_SPEC_VERSION, "files": files_state}
    _save_manifest(manifest_out)

    print()
    print(
        f"Run summary: indexed {indexed_this_run} PDF(s), "
        f"skipped {skipped_this_run} unchanged PDF(s)."
    )
    print(f"Total chunks now in vector DB for this collection: {collection.count()}")
    print(f"Manifest saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
