import os
import json
import time
import boto3
from dotenv import load_dotenv
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig

from src._5llm.llm_client import generate_response_from_context
from Evals.bedrok_model import BedrockJudge

class BedrockJudge(DeepEvalBaseLLM):
    def __init__(self, model_id = "google.gemma-3-27b-it", region = "us-east-1"):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name = region)

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

GOLDEN_PATH = "GoldenDataSet\FaithfullnessDataset.json"
CACHE_FILE = "Evals/generator_cache.json"
JUDGE_MODEL = BedrockJudge(model_id = "google.gemma-3-27b-it", region = "us-east-1")
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
THRESHOLD = 0.7

with open(GOLDEN_PATH , mode = 'r' , encoding="utf-8") as f:
    goldens = json.load(f)

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

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
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
        time.sleep(3)                              # proactive, avoids backoff storms
        
    # print(response)
    
    test_case.append(
        LLMTestCase(
            input = data['query'], 
            actual_output = response, 
            retrieval_context = data['ideal_context']
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

evaluate(test_cases=test_case, 
        metrics=metrics,
        async_config = async_mode ,
        #  hyperparameters={
        #          "embedding_model": "Quwn 9B Ollama ",
        #          "chunk_size": 1500,
        #          "chunk_overlap": 200,
        #          "top_k": 5,
        #          "judge_model": "qwen2.5:3b",
        #          # "golden_set": GOLDEN_PATH,
        #      }
        )