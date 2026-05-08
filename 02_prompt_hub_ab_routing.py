"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import hashlib
from pathlib import Path

# ── 1. Environment / imports ────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import *
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing
os.environ["LANGSMITH_API_KEY"]    = langchain_api_key
os.environ["LANGSMITH_PROJECT"]    = langchain_project
os.environ["LANGSMITH_ENDPOINT"]   = langchain_endpoint

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from qdrant_client import QdrantClient
# ── 2. Define two prompt templates ──────────────────────────────────────────
# TODO: write PROMPT_V1 — concise, 2-4 sentence answers
SYSTEM_V1 = """
You are a helpful AI assistant.
Answer the user's question using ONLY the provided context.
Keep your answer concise (2-4 sentences).
If the context does not contain the answer, say: 'I don't have enough information.'

Context:
{context}"""

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# TODO: write PROMPT_V2 — structured, expert 3-5 sentence answers
SYSTEM_V2 = """
You are an expert AI tutor. Provide a structured, accurate answer.
Instructions:
1. Read the context carefully.
2. Identify the key facts relevant to the question.
3. Write a clear, well-organized answer (3-5 sentences).
4. State explicitly if the context lacks sufficient information.

Context:
{context}
"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# Prompt Hub names (change these to your own unique names)
PROMPT_V1_NAME = "my-rag-prompt-v1"   # TODO: choose a unique name
PROMPT_V2_NAME = "my-rag-prompt-v2"   # TODO: choose a unique name


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.

    Use: client.push_prompt(name, object=template, description="...")
    The 'object' argument must be a ChatPromptTemplate instance.
    """
    # TODO: push PROMPT_V1
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1: {e}")

    # TODO: push PROMPT_V2
    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.

    Use: client.pull_prompt(name) → returns a ChatPromptTemplate
    """
    prompts = {}

    # TODO: pull PROMPT_V1_NAME, fall back to local PROMPT_V1 on error
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    # TODO: pull PROMPT_V2_NAME, fall back to local PROMPT_V2 on error
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.

    Rules:
      even hash → PROMPT_V1_NAME
      odd  hash → PROMPT_V2_NAME

    This is DETERMINISTIC: same request_id always maps to the same version.
    """
    # TODO: compute MD5 hash of request_id, convert to integer
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # TODO: return V1 name if even, V2 name if odd
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

embeddings = JinaEmbeddings(
    model=jina_embedding_name,
    api_key=jina_api_key,
)

# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
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
    
    


# ── 7. Traced A/B query function ────────────────────────────────────────────
# TODO: add @traceable decorator with name="ab-rag-query" and tags=["ab-test"]
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}

    Steps:
      a) Retrieve top-3 docs with retriever.invoke(question)
      b) Join their page_content into a single context string
      c) Run (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Return the result dict
    """
    # TODO: retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # TODO: run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    # TODO: return result
    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # TODO: create LangSmith client
    client = Client(api_key=langchain_api_key)

    # TODO: push both prompts
    push_prompts_to_hub(client)

    # TODO: pull both prompts from Hub
    prompts = pull_prompts_from_hub(client)

    # TODO: build vectorstore, retriever, and LLM
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=alibaba_model_name,
        api_key=alibaba_key_api,
        base_url=alibaba_url,
    )

    # TODO: loop over all 50 questions with A/B routing
    from qa_pairs import SAMPLE_QUESTIONS
    results = []
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        results.append(result)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    # TODO: print routing summary
    v1_count = sum(1 for res in results if res["version"] == "v1")
    v2_count = len(results) - v1_count

    print("\nA/B Routing Summary:")
    print(f"  Prompt V1: {v1_count}")
    print(f"  Prompt V2: {v2_count}")


if __name__ == "__main__":
    main()
