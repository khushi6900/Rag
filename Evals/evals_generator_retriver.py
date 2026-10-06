import os

# Must be set BEFORE deepeval is imported (it may be read at import time)
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"

import json
import re
import time

import boto3
from pydantic import BaseModel
from botocore.config import Config
from botocore.exceptions import ClientError
from src._3retrieval.retriever import load_index, load_bm25
from rag_pipeline import ask
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig


def _sanitize_json(s: str) -> str:
    """Extract the outermost JSON object and fix invalid backslash escapes."""
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end + 1]
    # Escape any backslash not followed by a valid JSON escape character
    s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
    return s


class BedrockJudge(DeepEvalBaseLLM):
    def __init__(self, model_id="amazon.nova-pro-v1:0", region="us-east-1"):
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                read_timeout=120,
            ),
        )
    def load_model(self):
        return self.client

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        for attempt in range(6):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 2048, "temperature": temperature},
                )
                return response["output"]["message"]["content"][0]["text"]
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("ThrottlingException", "ServiceUnavailableException") and attempt < 5:
                    wait = 2 ** attempt * 5   # 5, 10, 20, 40, 80s
                    print(f"[BedrockJudge] {code}, sleeping {wait}s")
                    time.sleep(wait)
                else:
                    raise

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        json_schema_str = json.dumps(schema.model_json_schema())
        formatted_prompt = (
            f"{prompt}\n\n"
            "CRITICAL: Respond ONLY with a valid JSON object matching this schema:\n"
            f"{json_schema_str}\n"
            "Do not include any conversational text, explanations, or markdown code blocks."
        )

        last_err = None
        for attempt in range(3):
            # temp 0.0 on first try, slightly higher on retries so output can differ
            temp = 0.0 if attempt == 0 else 0.3
            raw_output = self.generate(formatted_prompt, temperature=temp).strip()
            try:
                cleaned = _sanitize_json(raw_output)
                try:
                    return schema.model_validate_json(cleaned)
                except Exception:
                    # lenient parse (handles raw newlines/control chars inside strings)
                    return schema.model_validate(json.loads(cleaned, strict=False))
            except Exception as e:
                last_err = e
                print(f"[BedrockJudge] parse failed (attempt {attempt + 1}/3): {e}")
        raise last_err

    async def a_generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self.generate_with_schema(prompt, schema)

    def get_model_name(self) -> str:
        return f"Bedrock-{self.model_id}"


INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"
BM25_FILE = "bm25_index.pkl"
GOLDEN_PATH = "GoldenDataSet/golden_dataset.json"
CACHE_PATH = "Evals/pipeline_cache.json"
JUDGE_MODEL = BedrockJudge(model_id="amazon.nova-pro-v1:0", region="us-east-1")

cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

# ---------- phase 1: run pipeline (cached) ----------
index, chunks = load_index(index_file=INDEX_FILE, metadata_file=METADATA_FILE)
bm25 = load_bm25(BM25_FILE)

test_case = []
for i, golden in enumerate(goldens, 1):
    query = golden["question"]
    expected_output = golden.get("ground_truth")

    if query in cache:
        print(f"[{i}/{len(goldens)}] CACHE HIT: {query[:50]}")
        answer = cache[query]["answer"]
        context = cache[query]["context"]
    else:
        print(f"[{i}/{len(goldens)}] API CALL: {query[:50]}")
        result = ask(query, index, chunks, bm25)
        answer = result["answer"]
        context = [c["text"] for c in result.get("retrieved_chunks", [])]
        cache[query] = {"answer": answer, "context": context}
        with open(CACHE_PATH, "w") as f:  # save after EVERY question
            json.dump(cache, f, indent=2)
        time.sleep(2)  # rerank throttle protection

    test_case.append(LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=context,
    ))

# ---------- phase 2: evaluation metrics ----------
metrics = [
    FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
    ContextualRelevancyMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
    ContextualRecallMetric(threshold=0.7, model=JUDGE_MODEL, include_reason=True),
]

async_config = AsyncConfig(
    run_async=False,
    max_concurrent=1,
)

evaluate(
    test_cases=test_case,
    metrics=metrics,
    async_config=async_config,
)