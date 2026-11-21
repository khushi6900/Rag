## Failur Point Can be miss actual data and also retrive nois with right data 
import json
import os
import time

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from src._3retrieval.retriever import load_bm25, load_index, retrieve_and_rerank

from .grok_model import GrokJudge

load_dotenv()
GOLDEN_DATA = "GoldenDataSet/golden_dataset.json"
INDEX_FILE  = "faiss_index.bin"
BM25_FILE = "bm25_index.pkl"
MetaData = "chunk_metadata.json"
api_groq=os.getenv("GROQ_API_KEY")
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
# os.environ["DEEPEVAL_DISABLE_CACHE"] = "YES"

with open(GOLDEN_DATA) as f:
    goldens = json.load(f)

JUDGE_MODEL = GrokJudge(model_id="openai/gpt-oss-20b", api_key=api_groq)
index , chunks = load_index(index_file = INDEX_FILE, metadata_file = MetaData)
bm25 = load_bm25(BM25_FILE)
CACHE_FILE = "Evals/retrieval_cache.json"
cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

test_case = []
for i, data in enumerate(goldens, 1):
    query = data['question']

    if query in cache:
        text_result = cache[query]
        print(f"[{i}/{len(goldens)}] CACHE HIT: {query[:50]}")
    else:
        print(f"[{i}/{len(goldens)}] API CALL: {query[:50]}")
        final_chunks = retrieve_and_rerank(
            query=query,
            index=index,
            chunks=chunks,
            bm25=bm25,
            retrieve_k=30, 
            rerank_top_k=5
        )
        text_result = [t['text'] for t in final_chunks]
        cache[query] = text_result
        json.dump(cache, open(CACHE_FILE, "w"))   # save progress after each question
        time.sleep(5)                              # proactive, avoids backoff storms

    test_case.append(LLMTestCase(
        input=query,    
        expected_output=data['ground_truth'],
        actual_output=data['ground_truth'], ## this is not used for precision and recall 
        retrieval_context=text_result,
    ))

cache_config = CacheConfig(
    use_cache=True,
    write_cache=True
)

##Define the matric 
metric = [ContextualRecallMetric(
    threshold = 0.7, 
    model = JUDGE_MODEL
            ),
ContextualPrecisionMetric(
    threshold =0.7, 
    model = JUDGE_MODEL   )]

async_config = AsyncConfig(
    run_async = False, 
    max_concurrent=1    
)
evaluate(
    test_cases = test_case, 
    metrics = metric,
    async_config = async_config,
    cache_config=cache_config
)