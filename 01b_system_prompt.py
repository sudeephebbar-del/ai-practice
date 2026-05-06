from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

QUESTION = "Should we use Kafka or a REST API for CDR event ingestion?"

PERSONAS = [
    ("Junior developer",
     "You are a junior developer learning BSS systems."),
    ("Principal SRE",
     "You are a Principal SRE who has operated BSS platforms at Verizon scale. "
     "You prioritise reliability, observability, and operational simplicity."),
    ("Architect",
     "You are a solutions architect. You answer with pros/cons tables and "
     "recommend based on the specific context given."),
]

for name, system_prompt in PERSONAS:
    print(f"\n{"="*60}")
    print(f"PERSONA: {name}")
    print("="*60)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": QUESTION}
        ],
        temperature=0.0,
        max_tokens=300
    )
    print(response.choices[0].message.content)