# This exercise deliberately hits the context window limit
# to make the NEED for RAG visceral and concrete.

import os
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

DOCS_FOLDER = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\PDFs"

def load_all_pdfs(folder: str) -> dict[str, str]:
    """Load all PDFs from folder, return {filename: text}."""
    docs = {}
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            path = os.path.join(folder, fname)
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            docs[fname] = text
            print(f"  Loaded: {fname} ({len(text):,} chars)")
    return docs

print("Loading all PDFs...")
all_docs = load_all_pdfs(DOCS_FOLDER)

# Concatenate everything
combined = ""
for fname, text in all_docs.items():
    combined += f"\n\n=== DOCUMENT: {fname} ===\n{text}"

total_chars  = len(combined)
approx_tokens = total_chars // 4
print(f"\nTotal characters: {total_chars:,}")
print(f"Approximate tokens: {approx_tokens:,}")
print(f"GPT-4o-mini limit:  128,000 tokens")

if approx_tokens > 100_000:
    print("\n*** WALL HIT: Too many tokens to stuff into one prompt")
    print("    This is exactly why RAG exists.")
    print("    With 50 enterprise docs this is always the case.")
else:
    print("\nFits in context window. Try adding more PDFs to feel the limit.")
    question = "Which document discusses API design patterns?"
    # still works, but increasingly expensive as docs grow
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user",
                   "content": f"Documents:\n{combined}\n\nQuestion: {question}"}],
        temperature=0.0
    )
    print(response.choices[0].message.content)