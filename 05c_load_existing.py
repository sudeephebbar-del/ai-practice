# In production you run indexing once (or nightly).
# At query time you LOAD the existing vectorstore.
# This is what rag_service.py does on startup in the case study.

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Load existing vectorstore (no documents needed)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=r"C:\OneDrive\Personal\Career\Learnings\2026\Verizon\AI\files\chromadb\langchain",
    embedding_function=embeddings
)
print(f"Loaded vectorstore: {vectorstore._collection.count()} chunks")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

result = chain.invoke({"query": "What is Oracle BRM used for?"})
print(result["result"])