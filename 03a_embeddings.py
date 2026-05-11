from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a piece of text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # cheapest, 1536-dimensional
        input=text
    )
    return response.data[0].embedding  # list of 1536 floats

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Measure how similar two vectors are. 1.0 = identical, 0.0 = unrelated."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Get embeddings for several sentences
sentences = [
    "Oracle BRM is a billing and revenue management system",
    "Oracle billing software for telecom companies",
    "Kafka is a distributed event streaming platform",
    "CDR records capture phone call details for billing",
    "The weather is nice today",
]

print("Computing embeddings...")
embeddings = {s: get_embedding(s) for s in sentences}

# Show what an embedding looks like
first = embeddings[sentences[0]]
print(f"\nEmbedding dimensions: {len(first)}")
print(f"First 5 values: {[round(x,4) for x in first[:5]]}")
print(f"Range: {min(first):.4f} to {max(first):.4f}")

# Compare similarity between all pairs
query = "What system handles billing in telecom?"
query_emb = get_embedding(query)

print(f"\nQuery: \"{query}\"\n")
results = []
for sentence, emb in embeddings.items():
    sim = cosine_similarity(query_emb, emb)
    results.append((sim, sentence))

for sim, sentence in sorted(results, reverse=True):
    bar = "#" * int(sim * 30)
    print(f"  {sim:.4f} {bar} {sentence[:60]}")