from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness  # Try the "old" path
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
import os
import sys

# Add current dir to path for config
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

# Initialize metric - in the "old" path, maybe it doesn't require LLM in __init__
# or it accepts LangChain LLM
faithfulness = Faithfulness()

# Mock data
sample = SingleTurnSample(
    user_input="What is ML?",
    response="Machine Learning is a field of AI.",
    retrieved_contexts=["Machine Learning (ML) is a subset of AI."],
    reference="Machine Learning is a branch of artificial intelligence."
)
dataset = EvaluationDataset(samples=[sample])

print("Attempting to run a 1-sample evaluation...")
try:
    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=llm_eval,
        embeddings=emb_eval,
    )
    print("Success!")
    print(result)
except Exception as e:
    print(f"Failed with error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
