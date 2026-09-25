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


GOLDEN_PATH = "GoldenDataSet/golden_dataset.json"
JUDGE_MODEL = OllamaModel(model="qwen2.5:7b", base_url="http://10.10.2.73:11434")
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

test_case = []
for i in goldens:
    Query = i['question']
    live_response = generate_response(prompt = Query)
    print(live_response)
    # break
    test_case.append(LLMTestCase(
        input = Query, 
        actual_output = live_response
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
        metrics = [metrics],
        async_config = aysnc_config
    )
