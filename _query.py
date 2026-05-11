import requests

r = requests.post(
    "http://localhost:8002/search",
    json={"question": "What does our documentation say about Kafka consumer patterns?"},
    timeout=60,
)
data = r.json()
print("Answer:", data.get("answer", data))
print()
for s in data.get("sources", []):
    print(f"  - {s['title']} page {s['page']}")
