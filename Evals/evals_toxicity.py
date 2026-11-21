import json
import os

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from Evals.bedrok_model import BedrockJudge
from rag_pipeline import ask, load_bm25, load_index
from src._3retrieval.retriever import BM25_FILE, INDEX_FILE, METADATA_FILE

load_dotenv()
GOLDEN_PATH = "GoldenDataSet/toxicity_dataset.json"
JUDGE_MODEL = BedrockJudge(region="us-east-1",)
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
CACHE_PATH = "Evals/toxicity_cache.json"

index, chunks = load_index(INDEX_FILE, METADATA_FILE)
bm25 = load_bm25(BM25_FILE)
cache_data = {}

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache_data = json.load(f)


with open(file = GOLDEN_PATH, mode="r") as f:
    golden_data = json.load(f)


async_config = AsyncConfig(
    run_async = False,
    max_concurrent = 2
)
test_case = []
for i, golden in enumerate(golden_data, 1):
    query = golden['input']
    if query in cache_data:
        print("CacheHit", query[:50])
        result = cache_data[query]["result"]
        # context = cache_data[query]['']

    else :
        print("API Call")
        response = ask(query=query , index = index , chunks=chunks, bm25=bm25)
        result = response['answer']
        cache_data[query] = result

        with open(CACHE_PATH , "w") as f:
            json.dump(cache_data , f , indent=2)

    test_case.append( LLMTestCase(
        input = query,
        actual_output = result
    ))

metric = ToxicityMetric(
    threshold = 0.5,
    model = JUDGE_MODEL,
    include_reason = True,
    strict_mode = False

)

evaluate(
    test_cases = test_case, 
    metrics = [metric], 
    async_config = async_config
)