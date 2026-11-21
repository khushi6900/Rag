import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq.chat_models import ChatGroq

load_dotenv()
# REGION = "us-east-1" 
# LLM_MODEL_ID = "amazon.nova-micro-v1:0"

# bedrock = boto3.client("bedrock-runtime", region_name=REGION)

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# def generate_response(prompt, max_tokens=500, temperature=0.5):

#     response = bedrock.converse(
#         modelId=LLM_MODEL_ID,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [{"text": prompt}]
#             }
#         ],
#         inferenceConfig={
#             "maxTokens": max_tokens,
#             "temperature": temperature
#         }
#     )
    
#     return response["output"]["message"]["content"][0]["text"]

def generate_response(prompt, max_tokens=500, temperature=0.5):
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    usage = response.usage_metadata
    return response.content, usage

## For the Evaluation 
def generate_response_from_context(
    query: str,
    context: list[str],
    max_tokens: int = 500,
    temperature: float = 0.0
):
    context_text = "\n\n".join(context)

    prompt = f"""
        You are a helpful assistant.

        Answer ONLY using the provided context.
        If the answer is not contained in the context, say:
        "I don't have enough information to answer that"

        Context:
        {context_text}

        Question:
        {query}

        Answer:
        """

    return generate_response(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

if __name__ == "__main__":
    prompt = "What is Amazon API Gateway?"
    response = generate_response(prompt)
    print(response)