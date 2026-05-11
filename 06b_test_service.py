# Run rag_service.py first in a separate terminal:
# uvicorn rag_service:app --port 8001

import requests

BASE = "http://localhost:8001"

# Health check
health = requests.get(f"{BASE}/health").json()
print(f"Service health: {health}")

# Ask a question
questions = [
    "What Oracle partitioning strategies are recommended for large tables?",
    "How is Kafka used in the BSS architecture?",
    "What is the purpose of DBMS_STATS in Oracle?",
    "What are the performance tuning approaches for Oracle databases?",
    "How are CDR events processed in the billing pipeline?"
]

for q in questions:
    response = requests.post(
        f"{BASE}/search",
        json={"question": q}
    ).json()
    print(f"\nQ: {q}")
    print(f"A: {response['answer']}")
    print("Sources:")
    for s in response["sources"]:
        print(f"  - {s['title']} page {s['page']}")