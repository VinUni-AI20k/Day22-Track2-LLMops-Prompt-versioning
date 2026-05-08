import os
from dotenv import load_dotenv
load_dotenv()

alibaba_key_api = os.getenv("ALIBABA_KEY_API")
alibaba_url = os.getenv("ALIBABA_URL")
alibaba_model_name = os.getenv("ALIBABA_MODEL_NAME")

qdrant_url = os.getenv("QDARNT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

jina_api_key = os.getenv("JINA_API_KEY")
jina_embedding_name = os.getenv("JINA_EMBEDDING_NAME")
jina_embedding_url = os.getenv("JINA_EMBEDDING_URL")
jina_rerank_name = os.getenv("JINA_RERANK_NAME")
jina_rerank_url = os.getenv("JINA_RERANK_URL")

langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGSMITH_API_KEY")
langchain_project = os.getenv("LANGSMITH_PROJECT")
