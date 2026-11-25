import json
import os
import re

import groq
import httpcore
import httpx
from deepeval.models import DeepEvalBaseLLM
from groq import RateLimitError
from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel
from pydantic_core import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class GrokJudge(DeepEvalBaseLLM):
    def __init__(self, model_id="openai/gpt-oss-20b",
                api_key: str | None = None, max_tokens=2048):
        self.model_id = model_id
        self.model = ChatGroq(
            model=model_id,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=max_tokens,
            # reasoning_effort="low",
        )

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
        if not text or not text.strip():
            raise ValueError("Received empty response from LLM")
        m = re.search(r"\{.*\}", text, re.DOTALL)   # strips ```json fences / chatter
        json_str = m.group(0) if m else text
        return schema.model_validate_json(json_str)

    # Automatically retry on Rate Limits (429) or Network/Socket errors with exponential backoff
    @retry(
        retry=retry_if_exception_type((
            RateLimitError, 
            httpcore.ConnectError, 
            groq.APIConnectionError,
            httpx.ConnectError, 
            ValidationError, 
            json.JSONDecodeError,
            ValueError
        )),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            return self.model.invoke(prompt).content
        res = self.model.invoke(self._build_prompt(prompt, schema)).content
        return self._parse(res, schema)

    @retry(
        retry=retry_if_exception_type((
            RateLimitError, 
            httpcore.ConnectError, 
            httpx.ConnectError, 
            ValidationError, 
            json.JSONDecodeError,
            ValueError
        )),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            return (await self.model.ainvoke(prompt)).content
        res = (await self.model.ainvoke(self._build_prompt(prompt, schema))).content
        return self._parse(res, schema)

    # @retry(
    #     retry=retry_if_exception_type((ValueError, Exception)), # Catches your empty response error
    #     wait=wait_exponential(multiplier=2, min=3, max=30),
    #     stop=stop_after_attempt(5),
    #     reraise=True
    # )
    # def generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
    #     json_schema_str = json.dumps(schema.model_json_schema())
    #     formatted_prompt = (
    #         f"{prompt}\n\n"
    #         "CRITICAL: Respond ONLY with a valid JSON object matching this schema:\n"
    #         f"{json_schema_str}\n"
    #         "Do not include any conversational text, explanations, or markdown code blocks."
    #     )
    #     raw_output = self.generate(formatted_prompt).strip()

    #     if not raw_output:
    #         raise ValueError("Received an empty response from Groq judge model. Retrying...")
    #     # Clean markdown code fences if model returns them
    #     if raw_output.startswith("```json"):
    #         raw_output = raw_output[7:]
    #     if raw_output.startswith("```"):
    #         raw_output = raw_output[3:]
    #     if raw_output.endswith("```"):
    #         raw_output = raw_output[:-3]
            
    #     return schema.model_validate_json(raw_output)

    # async def a_generate_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
    #     return self.generate_with_schema(prompt, schema)

    def get_model_name(self):
        return f"grok:{self.model_id}"