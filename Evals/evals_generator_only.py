import json
import os
import time

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src._5llm.llm_client import generate_response_from_context

from .grok_model import GrokJudge

api_groq=os.getenv("GROQ_API_KEY")

GOLDEN_DATA = "GoldenDataSet/golden_dataset.json"
CACHE_FILE = "Evals/generator_cache.json"
RETRIEVAL_CACHE = "Evals/retrieval_cache.json"

JUDGE_MODEL = GrokJudge(model_id="qwen/qwen3.8-27b", api_key=api_groq)
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
THRESHOLD = 0.7

with open(GOLDEN_DATA) as f:
    goldens = json.load(f)

with open(RETRIEVAL_CACHE, "r", encoding="utf-8") as f:
    retrieval_cache = json.load(f)

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

test_case = []
for i , data in enumerate(goldens, start=1):
    # print(data)
    query = data['question']
    chunks = retrieval_cache.get(query)
    if not chunks:
        print(f"[{i}] NO RETRIEVAL CACHE: {query[:50]}")
        continue

    if query in cache:
        response = cache[query]
        print(f"[{i}/{len(goldens)}] CACHE HIT: {query[:50]}")

    else:
        print(f"[{i}/{len(goldens)}] API CALL: {query[:50]}")
        response = generate_response_from_context(
            query=query, 
            context = chunks,
        )
        # print("response -->" , response)
        # # break
        cache[query] = response
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
        time.sleep(10)                              # proactive, avoids backoff storms
        
    # print(response)
    
    test_case.append(
        LLMTestCase(
            input = data['question'], 
            actual_output = response, 
            expected_output=data["ground_truth"],
            retrieval_context = chunks,
        )
    )

metrics = [
        FaithfulnessMetric(
            threshold = THRESHOLD,
            model = JUDGE_MODEL,
            include_reason = True 
        ),
        AnswerRelevancyMetric(
            threshold = THRESHOLD, 
            model = JUDGE_MODEL,
            include_reason = True
        )
    ]
async_mode = AsyncConfig(
            run_async=False,
            max_concurrent=1
        )
cache_config = CacheConfig(
    use_cache=True,
    write_cache=True
)

evaluate(test_cases=test_case, 
        metrics=metrics,
        async_config = async_mode ,
        cache_config=cache_config
        #  hyperparameters={
        #          "embedding_model": "Quwn 9B Ollama ",
        #          "chunk_size": 1500,
        #          "chunk_overlap": 200,
        #          "top_k": 5,
        #          "judge_model": "qwen2.5:3b",
        #          # "golden_set": GOLDEN_PATH,
        #      }
        )