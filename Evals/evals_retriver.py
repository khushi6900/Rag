## Failur Point Can be miss actual data and also retrive nois with right data 
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import OllamaModel
from deepeval.evaluate import AsyncConfig 
from src._3retrieval.retriever import (retrieve_and_rerank , 
                                       load_index, 
                                       rerank_chunks, 
                                       )
from dotenv import load_dotenv
import json
import os

load_dotenv()
GOLDEN_DATA = "GoldenDataSet/golden_dataset.json"
INDEX_FILE = "D:/VsCode/AWS-RAG/faiss_index.bin"
MetaData = "D:/VsCode/AWS-RAG/chunk_metadata.json"
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"

os.environ["DEEPEVAL_DISABLE_CACHE"] = "YES"

with open(GOLDEN_DATA) as f:
    goldens = json.load(f)

JUDGE_MODEL = OllamaModel(model = "qwen2.5:7b" , base_url = "http://10.10.2.73:11434")
test_case = []
for data in goldens:

    query = data['question']
    index , chunks = load_index(index_file = INDEX_FILE, metadata_file = MetaData)

    final_chunls = retrieve_and_rerank( 
        query = query,
        index = index,
        chunks =chunks, 
       
    )
    text_result = [t['text'] for t in final_chunls]
    # print(text_result)
    # break

    test_case.append(LLMTestCase(
        input = query, 
        expected_output = data['ground_truth'], 
        actual_output="Generator Not generate yet",
        retrieval_context=text_result
    ))

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
        async_config = async_config
    )

    

