from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env file

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",      # cheapest capable model
    messages=[
        {"role": "system", "content":
         "You are a helpful assistant for a telecom BSS platform team."},
        {"role": "user", "content":
         "What is a CDR in telecom billing? Answer in 3 bullet points."}
    ],
    temperature=0.0,          # 0 = deterministic, same answer every time
    max_tokens=200
)

# Print the answer
print(response.choices[0].message.content)

# Print token usage (important for cost awareness)
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")