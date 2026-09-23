import boto3
import json

REGION = "us-east-1" 
LLM_MODEL_ID = "amazon.nova-micro-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def generate_response(prompt, max_tokens=500, temperature=0.5):

    response = bedrock.converse(
        modelId=LLM_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    )

    return response["output"]["message"]["content"][0]["text"]

if __name__ == "__main__":
    prompt = "What is Amazon API Gateway?"
    response = generate_response(prompt)
    print(response)