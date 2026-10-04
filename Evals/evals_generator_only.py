import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import json
import time                             
from deepeval.models   import OllamaModel
from deepeval.evaluate import AsyncConfig
from src._5llm.llm_client import generate_response_from_context
from Evals.bedrok_model import BedrockJudge

GOLDEN_PATH = "GoldenDataSet/FaithfullnessDataset.json"
# JUDGE_MODEL = OllamaModel(model="qwen2.5:7b", base_url="http://10.10.2.73:11434")
JUDGE_MODEL = BedrockJudge(region="us-east-1",)
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
THRESHOLD = 0.7

with open(GOLDEN_PATH , mode = 'r' , encoding="utf-8") as f:
    goldens = json.load(f)

CACHE_FILE = "Evals/generator_cache.json"
cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}


test_case = []
for i , data in enumerate(goldens, start=1):
    # print(data)
    query = data['query']

    if query in cache:
        response = cache[query]
        print(f"[{i}/{len(goldens)}] CACHE HIT: {query[:50]}")

    else:
        print(f"[{i}/{len(goldens)}] API CALL: {query[:50]}")
        response = generate_response_from_context(
            query=query, 
            context = data['ideal_context'],
        )
        # print("response -->" , response)
        # # break
        cache[query] = response
        json.dump(cache, open(CACHE_FILE, "w"))   # save progress after each question
        time.sleep(3)                              # proactive, avoids backoff storms
        
    # print(response)
    
    test_case.append(
        LLMTestCase(
            input = data['query'], 
            actual_output = response, 
            retrieval_context = data['ideal_context']
        )
    )

metrix = [
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

evaluate(test_cases=test_case , metrics=metrix,
        async_config = async_mode , )


