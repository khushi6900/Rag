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