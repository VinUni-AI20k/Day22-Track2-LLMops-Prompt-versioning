"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with Qdrant
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
from pathlib import Path

# ── 1. Environment setup ────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import *

# TODO: set LangSmith environment variables BEFORE importing LangChain
os.environ["LANGCHAIN_TRACING_V2"]  = langchain_tracing
os.environ["LANGSMITH_API_KEY"]     = langchain_api_key
os.environ["LANGSMITH_PROJECT"]     = langchain_project
os.environ["LANGSMITH_ENDPOINT"]    = langchain_endpoint

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
# TODO: import the libraries you need, for example:
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from config import * 

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
# TODO: create a ChatOpenAI instance pointing to your endpoint
llm = ChatOpenAI(
    model=alibaba_model_name,
    api_key=alibaba_key_api,
    base_url=alibaba_url,
)

# TODO: create an OpenAIEmbeddings instance
embeddings = JinaEmbeddings(
    model=jina_embedding_name,
    api_key=jina_api_key,
)


# ── 4. Build Qdrant vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with Qdrant.

    Steps:
      a) Read your dataset
      b) Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) Call Qdrant.from_texts(chunks, embeddings) to build the index
      d) Return the vectorstore
    """
    # ── 4.1. Check if collection exists ─────────────────────────────────────────
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    if client.collection_exists(collection_name="jina_embed_v1"):
        print("ℹ️ Collection 'jina_embed_v1' already exists. Skipping upload.")
        return QdrantVectorStore(
            client=client,
            collection_name="jina_embed_v1",
            embedding=embeddings
        )

    # ── 4.2. If not, build it from scratch ──────────────────────────────────────
    print("🚀 Collection not found. Initializing and uploading knowledge base...")
    # TODO: read your dataset file
    text = (Path(__file__).parent / "data" / "knowledge_base.txt").read_text()

    # TODO: create a text splitter and split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    # TODO: build and return the Qdrant vectorstore
    vectorstore = QdrantVectorStore.from_texts(
        texts=chunks, 
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="jina_embed_v1"
        )
    return vectorstore




# ── 5. RAG prompt template ──────────────────────────────────────────────────
# TODO: define a ChatPromptTemplate with:
#   - system message: instruct the LLM to answer using ONLY the provided context
#   - human message: the user's {question}
#
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Chain structure:
        {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | StrOutputParser()

    Returns: (chain, retriever)
    """
    # TODO: create a retriever from the vectorstore (k=3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # TODO: define a helper to join retrieved docs into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # TODO: build and return the LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# pass  # remove this line when done


# ── 7. Traced query function ────────────────────────────────────────────────
# TODO: decorate this function with @traceable so LangSmith captures it
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    # TODO: invoke the chain and return the answer
    return chain.invoke(question)

# pass  # remove this line when done


# ── 8. Sample questions (Imported from qa_pairs.py) ────────────────────────
from qa_pairs import SAMPLE_QUESTIONS


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # TODO: build the vectorstore
    vectorstore = build_vectorstore()

    # TODO: build the RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # TODO: loop through all SAMPLE_QUESTIONS, call ask(), print results
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}\n")

    # TODO: print confirmation that traces were sent
    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGSMITH_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")

    # pass  # remove this line when done


if __name__ == "__main__":
    main()
