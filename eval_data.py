import json
from datasets import Dataset
from rag_pipeline import ask, load_index, INDEX_FILE, METADATA_FILE

index, chunks = load_index(INDEX_FILE, METADATA_FILE)

with open("golden_dataset.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)

eval_records = []
for item in golden_data:
    result = ask(item["question"], index, chunks)
    eval_records.append({
        "question": item["question"],
        "answer": result["answer"],
        "contexts": [c["text"] for c in result["retrieved_chunks"]],
        "ground_truth": item["ground_truth"]
    })

eval_data = Dataset.from_list(eval_records)