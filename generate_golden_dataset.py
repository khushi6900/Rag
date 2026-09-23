import json
import random
import time
import boto3

# ---- CONFIG ----
REGION = "us-east-1"
LLM_MODEL_ID = "amazon.nova-micro-v1:0"
CHUNKS_FILE = "all_chunks.json"
OUTPUT_FILE = "golden_dataset.json"
SAMPLE_SIZE = 30   # sample this many chunks; expect a few to fail/skip

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def generate_answer(prompt, max_tokens=300, temperature=0.3, max_retries=4):
    """Calls Nova via Converse API, with basic retry on throttling."""
    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId=LLM_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            if "Throttling" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Throttled, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise

def generate_qa_from_chunk(chunk_text, source):
    """Asks the model for one question + ground-truth answer based only on this chunk."""
    prompt = f"""Based only on this text, write one specific, factual question and its correct answer.
Return ONLY valid JSON, no markdown, no code fences, no extra commentary:
{{"question": "...", "ground_truth": "..."}}

Text: {chunk_text}"""

    raw = generate_answer(prompt)

    # defensive cleanup in case the model adds markdown fences anyway
    cleaned = raw.strip()
    for prefix in ("```json", "```"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if "question" not in parsed or "ground_truth" not in parsed:
            print(f"  [skip] Missing expected fields from {source}")
            return None
        parsed["source_doc"] = source
        return parsed
    except json.JSONDecodeError:
        print(f"  [skip] Unparseable output from {source}: {raw[:80]!r}")
        return None

if __name__ == "__main__":
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    print(f"Loaded {len(all_chunks)} total chunks")

    sample = random.sample(all_chunks, min(SAMPLE_SIZE, len(all_chunks)))
    print(f"Sampling {len(sample)} chunks for golden dataset generation\n")

    golden_set = []
    for i, chunk in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] {chunk['source']}")
        qa = generate_qa_from_chunk(chunk["text"], chunk["source"])
        if qa:
            golden_set.append(qa)
        time.sleep(0.3)  # small pacing gap to reduce throttling risk

    print(f"\nGenerated {len(golden_set)} usable Q&A pairs out of {len(sample)} sampled")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}")
    print("\nNext: manually review each entry, trim to your best 15-20,")
    print("and add 2-3 out-of-scope questions by hand (the model can't generate those itself).")