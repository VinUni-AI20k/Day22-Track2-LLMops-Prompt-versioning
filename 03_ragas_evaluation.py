"""
Step 3 — RAGAS Evaluation
===========================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json file saved

⏰ NOTE: This step takes ~20-30 minutes. Start it early!
"""

import json
from pathlib import Path
import sys
import time
from tenacity import retry, stop_after_attempt, wait_exponential
sys.path.append(str(Path(__file__).parent))

from config import *

os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing
os.environ["LANGSMITH_API_KEY"]    = langchain_api_key
os.environ["LANGSMITH_PROJECT"]    = langchain_project
os.environ["LANGSMITH_ENDPOINT"]   = langchain_endpoint

# ── 1. Imports ───────────────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig

# Import the 4 metric classes (from the stable path) and instantiate them
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()
context_recall = ContextRecall()
context_precision = ContextPrecision()

# TODO: import LangChain components (same as steps 1 & 2)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from qdrant_client import QdrantClient

import numpy as np


# ── 2. QA pairs with ground-truth answers (Imported) ────────────────────────
from qa_pairs import QA_PAIRS


# ── 3. Prompt templates (same as step 2) ────────────────────────────────────
# TODO: define PROMPT_V1 and PROMPT_V2 (copy from step 2)
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
3. Write a clear, well-organized answer (3-5 sentences) using ONLY the provided context.
4. DO NOT include any outside knowledge.
5. State explicitly if the context lacks sufficient information.

Context:
{context}
"""

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 4. Build vectorstore (reuse logic from step 1) ───────────────────────────
def build_vectorstore():
    # TODO: copy from step 1    
    embeddings = JinaEmbeddings(
        model=jina_embedding_name,
        api_key=jina_api_key,
    )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
   
    if client.collection_exists(collection_name="jina_embed_v1"):
        return QdrantVectorStore(
            client=client,
            collection_name="jina_embed_v1",
            embedding=embeddings
        )
    else:
        print("🚀 Collection not found. Initializing and uploading knowledge base...")
        text = (Path(__file__).parent / "data" / "knowledge_base.txt").read_text()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        vectorstore = QdrantVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name="jina_embed_v1",
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
    return vectorstore


# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@traceable(name="rag_run", tags=["eval"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.

    IMPORTANT: return contexts as a LIST of strings, not a joined string!
    RAGAS needs individual passage strings to compute context_recall.

    Returns: {"answer": str, "contexts": list[str]}
    """
    # TODO: retrieve documents
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # ← list of strings!
    ctx_str  = "\n\n".join(contexts)

    # TODO: run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})

    # TODO: return both answer and contexts list
    return {"answer": answer, "contexts": contexts}



def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    # TODO: create retriever, llm, and select the right prompt
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = ChatOpenAI(
        base_url = alibaba_url,
        model=alibaba_model_name,
        api_key=alibaba_key_api,
        temperature=0.1
    )
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        # TODO: call run_rag() and collect results
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],   # must be list[str]
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")
        time.sleep(1)  # Small delay to avoid overwhelming the connection

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    """
    Convert a list of RAG result dicts into a RAGAS EvaluationDataset.

    Each SingleTurnSample needs:
      user_input         → the question
      response           → the generated answer
      retrieved_contexts → list[str] of retrieved passages
      reference          → the ground-truth answer
    """
    # TODO: build the dataset
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)

    # pass  # remove this line when done


# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    Returns a dict: {metric_name: mean_score}
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    # TODO: create the EvaluationDataset
    dataset = build_ragas_dataset(rag_results)

    # TODO: create LLM and embeddings for RAGAS to use
    llm_eval = ChatOpenAI(
        base_url = alibaba_url,
        model=alibaba_model_name,
        api_key=alibaba_key_api,
        temperature=0.1
    )
    emb_eval = JinaEmbeddings(
        model=jina_embedding_name,
        api_key=jina_api_key,
    )

    # Reduce concurrency to avoid API rate limits/timeouts
    run_config = RunConfig(max_workers=2, timeout=60)

    # TODO: run evaluate() — this makes many LLM calls!
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=run_config,
    )

    # TODO: extract mean scores
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]           # list of floats
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    # TODO: print and return scores
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
    return scores


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # TODO: build vectorstore
    vectorstore = build_vectorstore()

    # TODO: collect outputs for V1 and V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # TODO: run RAGAS evaluation on both
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # TODO: print comparison table
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")

    # TODO: check faithfulness target
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # TODO: save JSON report to data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    Path(Path(__file__).parent / "data" / "ragas_report.json").write_text(json.dumps(report, indent=2))
    print("💾 Saved data/ragas_report.json")


if __name__ == "__main__":
    main()
