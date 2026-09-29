from src._3retrieval.retriever import load_index
from rag_pipeline import ask
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric, 
        ContextualRelevancyMetric
)
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig
import boto3
import json
import os 
import time

class BedrockJudge(DeepEvalBaseLLM):
    def __init__(self, model_id="google.gemma-3-27b-it", region="us-east-1"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
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
JUDGE_MODEL = BedrockJudge(model_id="google.gemma-3-27b-it", region="us-east-1")
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
