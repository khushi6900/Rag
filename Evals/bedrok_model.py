import json, re
from typing import Optional
from pydantic import BaseModel
from langchain_aws import ChatBedrockConverse
from deepeval.models import DeepEvalBaseLLM

class BedrockJudge(DeepEvalBaseLLM):
    def __init__(self, model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
                 profile: Optional[str] = None, max_tokens=4096):
        self.model_id = model_id
        kwargs = dict(model=model_id, region_name=region,
                      temperature=0, max_tokens=max_tokens)
        if profile:
            kwargs["credentials_profile_name"] = profile
        self.model = ChatBedrockConverse(**kwargs)   # better fit for Nova than ChatBedrock

    def load_model(self):
        return self.model

    @staticmethod
    def _build_prompt(prompt: str, schema: type[BaseModel]) -> str:
        return (
            f"{prompt}\n\n"
            "Respond with ONLY a valid JSON object that follows this JSON schema. "
            "Include every required field for every item, and no extra text:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )

    @staticmethod
    def _parse(text: str, schema: type[BaseModel]):
        m = re.search(r"\{.*\}", text, re.DOTALL)   # strips ```json fences / chatter
        return schema.model_validate_json(m.group(0) if m else text)

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        if schema is None:
            return self.model.invoke(prompt).content
        res = self.model.invoke(self._build_prompt(prompt, schema)).content
        return self._parse(res, schema)

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        if schema is None:
            return (await self.model.ainvoke(prompt)).content
        res = (await self.model.ainvoke(self._build_prompt(prompt, schema))).content
        return self._parse(res, schema)

    def get_model_name(self):
        return f"bedrock:{self.model_id}"


# class BedrockJudge(DeepEvalBaseLLM):
#     def __init__(self, model_id = "google.gemma-3-27b-it", region = "us-east-1"):
#         self.model_id = model_id
#         self.client = boto3.client("bedrock-runtime", region_name = region)

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str) -> str:
#             response = self.client.converse(
#                 modelId=self.model_id,
#                 messages=[{
#                     "role": "user",
#                     "content": [{"text": prompt}]
#                 }],
#                 inferenceConfig={
#                     "maxTokens": 2048,
#                     "temperature": 0.0
#                 }
#             )
#             return response["output"]["message"]["content"][0]["text"]
    
#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
#         json_schema_str = json.dumps(schema.model_json_schema())
#         formatted_prompt = (
#             f"{prompt}\n\n"
#             "CRITICAL: Respond ONLY with a valid JSON object matching this schema:\n"
#             f"{json_schema_str}\n"
#             "Do not include any conversational text, explanations, or markdown code blocks."
#         )
#         raw_output = self.generate(formatted_prompt).strip()

#         # Clean markdown code fences if model returns them
#         if raw_output.startswith("```json"):
#             raw_output = raw_output[7:]
#         if raw_output.startswith("```"):
#             raw_output = raw_output[3:]
#         if raw_output.endswith("```"):
#             raw_output = raw_output[:-3]
            
#         return schema.model_validate_json(raw_output.strip())

#     async def a_generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
#         return self.generate_with_schema(prompt, schema)

#     def get_model_name(self) -> str:
#         return f"Bedrock-{self.model_id}"
