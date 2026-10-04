from src._5llm.llm_client import generate_response
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric, 
        ContextualRelevancyMetric
)
from deepeval.models import OllamaModel
from deepeval.evaluate import AsyncConfig
import json
import os 
import time
from rag_pipeline import ask,load_index
from src._3retrieval.retriever import INDEX_FILE, METADATA_FILE
from Evals.bedrok_model import BedrockJudge
from dotenv import load_dotenv
load_dotenv()


GOLDEN_PATH = "GoldenDataSet/golden_dataset.json"
# JUDGE_MODEL = OllamaModel(model="qwen2.5:7b", base_url="http://10.10.2.73:11434")
JUDGE_MODEL = BedrockJudge(region="us-east-1",)
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
CACHE_PATH = "Evals/pipeline_cache.json"

index, chunks = load_index(INDEX_FILE, METADATA_FILE)
cache = {}


if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

# ---------- phase 1: run pipeline (cached) ----------
test_case = []
for i, golden in enumerate(goldens, 1):
    query = golden["question"]

    if query in cache:
        print(f"[{i}/{len(goldens)}] CACHE HIT: {query[:50]}")
        answer = cache[query]["answer"]
        context = cache[query]["context"]
    else:
        print(f"[{i}/{len(goldens)}] API CALL: {query[:50]}")
        result = ask(query, index, chunks)
        # print("REsssultttttttttttttttt" , result)
        answer = result["answer"]
        context = [c["text"] for c in result.get("retrieved_chunks", [])]# ← must return BOTH now
        cache[query] = {"answer": answer, "context": context}
        with open(CACHE_PATH, "w") as f:                    # save after EVERY question
            json.dump(cache, f, indent=2)
        time.sleep(3)                                      # rerank throttle protection

    test_case.append(LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context,     
    ))

###evaluatoins Metric 
metrics = [
    FaithfulnessMetric(
        threshold = 0.7,    
        model = JUDGE_MODEL,
        include_reason=False
    ),
    AnswerRelevancyMetric(
        threshold = 0.7,
        model = JUDGE_MODEL,
        include_reason=False
    ),
    ContextualRelevancyMetric(
        threshold = 0.7,
        model = JUDGE_MODEL,
        include_reason=False
    )

]
aysnc_config = AsyncConfig(
run_async=False,
max_concurrent=2
)

evaluate(
    test_cases = test_case, 
    metrics = metrics,
    async_config = aysnc_config
)
