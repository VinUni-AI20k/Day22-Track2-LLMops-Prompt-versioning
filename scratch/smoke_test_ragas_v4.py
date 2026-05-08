from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
import os
import sys

sys.path.append(os.getcwd())
from config import *

# Initialize LLM & Embeddings
llm_eval = ChatOpenAI(
    base_url=alibaba_url,
    model=alibaba_model_name,
    api_key=alibaba_key_api,
    temperature=0
)
emb_eval = JinaEmbeddings(
    model=jina_embedding_name,
    api_key=jina_api_key,
)

# Initialize metrics
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextRecall(),
    ContextPrecision()
]

# Mock data
sample = SingleTurnSample(
    user_input="What is ML?",
    response="Machine Learning is a field of AI.",
    retrieved_contexts=["Machine Learning (ML) is a subset of AI.", "Deep Learning is part of ML."],
    reference="Machine Learning is a branch of artificial intelligence."
)
dataset = EvaluationDataset(samples=[sample])

print("Attempting to run a 1-sample evaluation with all 4 metrics...")
try:
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm_eval,
        embeddings=emb_eval,
    )
    print("Success!")
    print(f"Result keys: {result.keys()}")
    print(result)
except Exception as e:
    print(f"Failed with error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
