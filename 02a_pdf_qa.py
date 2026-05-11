import PyPDF2
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        print(f"Pages: {len(reader.pages)}")
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:  # some pages are images with no extractable text
                text += f"\n--- Page {page_num+1} ---\n{page_text}"
    return text

def count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return len(text) // 4

def ask_about_document(doc_text: str, question: str) -> str:
    """Stuff the entire document into the prompt and ask a question."""
    prompt = f"""You are a travel specialist.

DOCUMENT:
{doc_text}

QUESTION: {question}

Answer based only on the document above. If the answer is not in the document,
say "Not found in this document."
"""

    approx_tokens = count_tokens_approx(prompt)
    print(f"Approximate prompt tokens: {approx_tokens:,}")
    print(f"gpt-4o-mini context limit: 128,000 tokens")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

# ── Run it ───────────────────────────────────────────────────────
PDF_PATH = r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\Hilton Beachfront Resort and Spa Hilton Head Island Reviews & Prices _ U.S. News Travel.pdf"  # change this

print("Extracting text...")
doc_text = extract_pdf_text(PDF_PATH)
print(f"Extracted {len(doc_text):,} characters")

questions = [
    "What is the main purpose of this document?",
    "What are the key features of this resort?",
    "List any limitations or constraints described.",
    "Does this resort have lazy river",
    "Does this resort have infinity pool",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask_about_document(doc_text, q)}")