# KEY INSIGHT: the LLM has NO memory between calls.
# To simulate a conversation you pass the ENTIRE history every time.
# This is how Copilot Chat works under the hood.

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# The conversation history grows with each turn
messages = [
#    {"role": "system", "content":
#     "You are a BSS platform expert. Be concise."}
]

def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    print("\n--- PROMPT ---")
    print(json.dumps(messages, indent=2))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )
    print(f"\nTokens used: {response.usage.total_tokens}")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    answer = response.choices[0].message.content
    #messages.append({"role": "assistant", "content": answer})
    return answer

# A multi-turn conversation
print(chat("What is Oracle BRM?"))
print(chat("What are it's main limitations?"))      # refers to "its" (BRM)
print(chat("What would you replace it with?"))     # "it" still means BRM
print(f"\nTotal messages in history: {len(messages)}")

# Print token usage (important for cost awareness)

# Observe: token count grows each turn.
# Context window limit (128k for gpt-4o) is why RAG is needed for large docs.