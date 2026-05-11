# ReAct: Reason + Act. The agent loops:
# Thought → Action → Observation → Thought → Action → ... → Final Answer

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import requests

load_dotenv()

# Define tools as LangChain Tool objects
def search_rag(query: str) -> str:
    try:
        resp = requests.post("http://localhost:8001/search",
                             json={"question": query}, timeout=10)
        return resp.json()["answer"] if resp.ok else "RAG service unavailable"
    except:
        return f"[Simulated] Answer for: {query}"

def get_db_stats(table_name: str) -> str:
    return f"Table {table_name}: 1.2M rows, 12 partitions, last analyzed 2024-12-01"

tools = [
    Tool(name="SearchDocumentation",
         func=search_rag,
         description="Search BSS platform docs. Input: a question string."),
    Tool(name="GetDatabaseStats",
         func=get_db_stats,
         description="Get Oracle table stats. Input: table name."),
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Inline the standard ReAct prompt (equivalent to hub: hwchase17/react)
prompt = PromptTemplate.from_template(
    "Answer the following questions as best you can. "
    "You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)

# Create the agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent, tools=tools,
    verbose=True,        # shows Thought/Action/Observation chain
    max_iterations=5     # safety limit
)

# Run a complex question that needs multiple steps
result = executor.invoke({
    "input": "I need to optimise the CDR_EVENTS table. "
             "Check its current stats AND search the docs for best practices, "
             "then give me a combined recommendation."
})
print(f"\nFinal answer:\n{result['output']}")