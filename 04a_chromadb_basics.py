import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

# Persistent client: data saved to disk, survives restarts
client = chromadb.PersistentClient(path=r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\chromadb")

# Embedding function: tells ChromaDB to use OpenAI for embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)

# Create a collection (like a table in a relational DB)
# get_or_create: safe to run multiple times
collection = client.get_or_create_collection(
    name="bss_knowledge",
    embedding_function=openai_ef,
    metadata={"description": "BSS platform knowledge base"}
)

# Add documents with metadata
# ChromaDB auto-generates embeddings by calling openai_ef
collection.add(
    documents=[
        "Oracle BRM is used for telecom billing. It handles CDR rating, invoicing, and account management.",
        "Kafka is used for CDR event streaming. Topics include cdr-events, billing-events, and order-events.",
        "The order-service exposes a REST API on port 8080. It uses Spring Boot 3 and JPA for Oracle.",
        "Performance tuning for large Oracle tables requires partitioning and BULK COLLECT in PL/SQL.",
        "DBMS_STATS.GATHER_TABLE_STATS should be run after every bulk data load.",
    ],
    ids=["doc1","doc2","doc3","doc4","doc5"],
    metadatas=[
        {"source": "BRM_overview.pdf",     "topic": "billing"},
        {"source": "kafka_design.pdf",     "topic": "messaging"},
        {"source": "order_service_api.pdf","topic": "api"},
        {"source": "oracle_tuning.pdf",    "topic": "database"},
        {"source": "oracle_tuning.pdf",    "topic": "database"},
    ]
)
print(f"Collection size: {collection.count()} documents")

# Query: semantic search
results = collection.query(
    query_texts=["How do I improve Oracle query performance?"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\nTop 3 results:")
for i, (doc, meta, dist) in enumerate(
    zip(results["documents"][0],
        results["metadatas"][0],
        results["distances"][0])):
    print(f"\n{i+1}. Distance={dist:.4f} | Source: {meta['source']}")
    print(f"   {doc[:150]}")