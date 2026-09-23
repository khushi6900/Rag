from ragas import evaluate
from ragas.metrics import _faithfulness
from langchain_aws import ChatBedrock, BedrockEmbeddings

import eval_data

judge_llm = ChatBedrock(model_id="amazon.nova-lite-v1:0", region_name="us-east-1")
judge_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")

test = eval_data.select(range(3))

results = evaluate(
    test,
    metrics=[_faithfulness],
    llm=judge_llm,
    embeddings=judge_embeddings
)
print(results)