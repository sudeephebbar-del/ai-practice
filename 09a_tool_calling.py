from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from dotenv import load_dotenv
from typing import cast
import json

load_dotenv()
client = OpenAI()

# Define tools the LLM can call
# These map to real Python functions below
TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search_documentation",
            "description":
                "Search the BSS platform documentation for technical answers. "
                "Use for questions about architecture, APIs, Oracle, Kafka, CDR processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_oracle_table_info",
            "description":
                "Get information about an Oracle database table: row count, "
                "partition info, last stats gathered. Use when asked about specific tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"}
                },
                "required": ["table_name"]
            }
        }
    }
]

# Simulated tool implementations
def search_documentation(query: str) -> str:
    """Simulates calling your FastAPI RAG service."""
    import requests
    try:
        resp = requests.post("http://localhost:8002/search",
                             json={"question": query}, timeout=10)
        if resp.ok:
            data = resp.json()
            return data["answer"]
    except:
        pass
    return f"[Simulated] Documentation answer for: {query}"

def get_oracle_table_info(table_name: str) -> str:
    """Simulates querying Oracle for table metadata."""
    # In production: query user_tables, user_tab_partitions, etc.
    return json.dumps({
        "table": table_name.upper(),
        "row_count": 1_234_567,
        "partitions": 12,
        "last_analyzed": "2024-12-01",
        "stale_stats": False
    })

def run_agent(user_question: str):
    """Run one agent turn: LLM decides which tool (if any) to call."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system",
         "content": "You are a BSS platform assistant. "
                    "Use available tools to answer technical questions accurately."},
        {"role": "user", "content": user_question}
    ]

    # First LLM call: may return a tool_call or a direct answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"   # LLM decides: call a tool or answer directly
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        # LLM decided to call a tool
        for tool_call in msg.tool_calls:
            if tool_call.type != "function":
                continue
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"  [Agent] Calling tool: {fn_name}({fn_args})")

            # Execute the tool
            if fn_name == "search_documentation":
                result = search_documentation(**fn_args)
            elif fn_name == "get_oracle_table_info":
                result = get_oracle_table_info(**fn_args)
            else:
                result = "Unknown tool"

            # Add tool result to messages and call LLM again
            messages.append(cast(ChatCompletionMessageParam, msg.model_dump()))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Second LLM call: now it has the tool result and can answer
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return final.choices[0].message.content
    else:
        # LLM answered directly without calling a tool
        print("  [Agent] Answered directly (no tool needed)")
        return msg.content

# Test: questions that should trigger different tools (or no tool)
questions = [
    "What is 2 + 2?",                                    # no tool needed
    "How does Oracle partitioning help with large tables?",  # docs tool
    "How many rows are in the CDR_EVENTS table?",        # oracle tool
]

for q in questions:
    print(f"\nQ: {q}")
    answer = run_agent(q)
    print(f"A: {answer}")