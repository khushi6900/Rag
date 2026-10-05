from src._3retrieval.retriever import load_index
from rag_pipeline import ask
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric, 
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric
)
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import json
import os 
import time

class BedrockJudge(DeepEvalBaseLLM):
    def __init__(self, model_id="amazon.nova-pro-v1:0", region="us-east-1"):
        self.model_id = model_id
        boto_config = Config(
            retries = {
                "max_attempts": 10,
                "mode": "adaptive"
            }
        )
        self.client = boto3.client("bedrock-runtime", region_name=region, config=boto_config)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        max_retries = 6
        delay = 3

        for attempt in range(max_retries):
            try:
                time.sleep(2)
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": [{"text": prompt}]
                    }],
                    inferenceConfig={
                        "maxTokens": 2048,
                        "temperature": 0.0
                    }
                )
                return response["output"]["message"]["content"][0]["text"]

            except ClientError as e:
                error_code = e.response.get("error", {}).get("code")
                if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                    wait_time = delay * (2 ** attempt)
                    print(f"\n[Bedrock Judge] Throttled by AWS Bedrock. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError(f"exceeded max retries")

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
        raw_output = self.generate(formatted_prompt).strip()
    
            # Clean markdown code fences if model returns them
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.startswith("```"):
            raw_output = raw_output[3:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
                
        return schema.model_validate_json(raw_output.strip())
    
    async def a_generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self.generate_with_schema(prompt, schema)
    
    def get_model_name(self) -> str:
        return f"Bedrock-{self.model_id}"

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json" 
GOLDEN_PATH = "GoldenDataSet/golden_dataset.json"
JUDGE_MODEL = BedrockJudge(model_id="amazon.nova-pro-v1:0", region="us-east-1")
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
CACHE_PATH = "Evals/pipeline_cache.json"

cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

# ---------- phase 1: run pipeline (cached) ----------
index, chunks = load_index(index_file=INDEX_FILE, metadata_file=METADATA_FILE)
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
        result = ask(query, index,chunks)
        answer = result["answer"]
        context = [c["text"] for c in result["retrieved_chunks"]]
        cache[query] = {"answer": answer, "context": context}
        with open(CACHE_PATH, "w") as f:                    # save after EVERY question
            json.dump(cache, f, indent=2)
        time.sleep(2)                                      # rerank throttle protection

    test_case.append(LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=context,     
    ))

###evaluatoins Metric 
metrics = [
    FaithfulnessMetric(
        threshold = 0.7, 
        model = JUDGE_MODEL,
        include_reason=True
    ),
    AnswerRelevancyMetric(
        threshold = 0.7,
        model = JUDGE_MODEL,
        include_reason=True
    ),
    ContextualRelevancyMetric(
        threshold = 0.7,
        model = JUDGE_MODEL,
        include_reason=True
    ),
    ContextualPrecisionMetric(
        threshold = 0.7,
        model = JUDGE_MODEL,
        include_reason = True
    ),
    ContextualRecallMetric(
        threshold = 0.7,
        model = JUDGE_MODEL,
        include_reason = True
    )
]
aysnc_config = AsyncConfig(
run_async=False,
max_concurrent=1
)

evaluate(
    test_cases = test_case, 
    metrics = metrics,
    async_config = aysnc_config
)
