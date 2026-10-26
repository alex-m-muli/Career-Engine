"""
LLM orchestration module for the Career Engine.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Type, TypeVar, cast

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, ValidationError

Groq: Any = None
GROQ_AVAILABLE = False
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("LLMClient")

T = TypeVar("T", bound=BaseModel)

_CHARS_PER_TOKEN = 3.5

# Reduced timeouts. If it takes longer than 45s, fail fast to the other provider.
GEMINI_TIMEOUT_MS = 45000
GROQ_TIMEOUT_SEC = 45.0

GROQ_TPM_LIMIT = 8000
MAX_GEMINI_RETRIES = 2
MAX_GROQ_RETRIES = 2
BASE_BACKOFF_SEC = 2.0
MAX_BACKOFF_SEC = 16.0


class LLMClient:
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required.")
            
        self._gemini_client = genai.Client(
            api_key=self.gemini_api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
        )
        
        self._groq_client: Any = None
        if GROQ_AVAILABLE and self.groq_api_key and Groq is not None:
            self._groq_client = Groq(api_key=self.groq_api_key)
            
        self._token_history: Deque[tuple[float, int]] = deque()

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / _CHARS_PER_TOKEN) + 1

    def _wait_for_groq_capacity(self, prompt_tokens: int, max_tokens: int) -> None:
        if not self._groq_client:
            return
            
        now = time.time()
        while self._token_history and self._token_history[0][0] < now - 60:
            self._token_history.popleft()
            
        current_tokens = sum(t for _, t in self._token_history)
        estimated_total = prompt_tokens + max_tokens
        
        if current_tokens + estimated_total > GROQ_TPM_LIMIT:
            if self._token_history:
                oldest_time = self._token_history[0][0]
                sleep_time = max(0.0, (oldest_time + 60) - now) + 1.0
                logger.info(f"[Groq Pacing] Token budget exceeded ({current_tokens + estimated_total}/{GROQ_TPM_LIMIT}). Sleeping {sleep_time:.1f}s.")
                time.sleep(sleep_time)
            else:
                sleep_time = max(1.0, (estimated_total / GROQ_TPM_LIMIT) * 60)
                logger.warning(f"[Groq Pacing] Single request ({estimated_total} tokens) exceeds TPM limit. Sleeping {sleep_time:.1f}s.")
                time.sleep(sleep_time)
                
        now = time.time()
        while self._token_history and self._token_history[0][0] < now - 60:
            self._token_history.popleft()

    def _record_groq_tokens(self, tokens: int) -> None:
        self._token_history.append((time.time(), tokens))

    def _clean_json(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        text = text.strip()
        start_brace = text.find("{")
        start_bracket = text.find("[")
        
        if start_brace == -1 and start_bracket == -1:
            return text
            
        if start_brace == -1:
            start_idx = start_bracket
        elif start_bracket == -1:
            start_idx = start_brace
        else:
            start_idx = min(start_brace, start_bracket)
            
        open_char = text[start_idx]
        close_char = "}" if open_char == "{" else "]"
        
        depth = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == open_char:
                depth += 1
            elif text[i] == close_char:
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
                    
        if end_idx != -1:
            text = text[start_idx:end_idx+1]
            
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def _clean_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(schema, dict):
            schema.pop("additionalProperties", None)
            for key, value in schema.items():
                if isinstance(value, dict):
                    self._clean_schema_for_gemini(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._clean_schema_for_gemini(item)
        return schema

    def _generate_gemini(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> T:
        schema_dict = self._clean_schema_for_gemini(response_model.model_json_schema())

        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema_dict,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        last_exception: Optional[Exception] = None
        
        for attempt in range(1, MAX_GEMINI_RETRIES + 1):
            try:
                logger.info(f"[Gemini] Attempt {attempt}/{MAX_GEMINI_RETRIES}...")
                response = self._gemini_client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    return response_model.model_validate_json(self._clean_json(response.text))
                raise ValueError("Empty response text from Gemini.")
                
            except ServerError as e:
                last_exception = e
                code = getattr(e, "code", 500)
                if code in (500, 503, 504):
                    sleep_time = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** (attempt - 1))) + random.uniform(0, 1)
                    logger.warning(f"[Gemini] {code} ServerError. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[Gemini] ServerError (non-retryable): {e}")
                    break
            except ClientError as e:
                last_exception = e
                code = getattr(e, "code", 400)
                if code == 429:
                    sleep_time = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** (attempt - 1))) + random.uniform(0, 1)
                    logger.warning(f"[Gemini] 429 Rate Limit. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[Gemini] ClientError (non-retryable): {e}")
                    break
            except (ValidationError, ValueError) as e:
                logger.error(f"[Gemini] Parse/Validation error: {e}")
                last_exception = e
                break 
            except Exception as e:
                logger.error(f"[Gemini] Unexpected error: {e}")
                last_exception = e
                break
                
        raise last_exception or RuntimeError("Gemini generation failed.")

    def _generate_groq(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> T:
        if not self._groq_client:
            raise RuntimeError("Groq client not configured.")
            
        models_to_try = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        last_exception: Optional[Exception] = None
        
        for model_id in models_to_try:
            prompt_tokens = self._estimate_tokens(prompt) + (self._estimate_tokens(system_instruction) if system_instruction else 0)
            self._wait_for_groq_capacity(prompt_tokens, max_tokens)
            
            response_format: dict[str, Any]
            if "gpt-oss" in model_id:
                schema = response_model.model_json_schema()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    }
                }
            else:
                response_format = {"type": "json_object"}
                
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            for attempt in range(1, MAX_GROQ_RETRIES + 1):
                try:
                    logger.info(f"[Groq] Model {model_id} attempt {attempt}/{MAX_GROQ_RETRIES}...")
                    response = self._groq_client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        response_format=cast(Any, response_format),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=GROQ_TIMEOUT_SEC,
                    )
                    
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError(f"Empty content from {model_id}.")
                        
                    tokens_used = response.usage.total_tokens if response.usage else (prompt_tokens + max_tokens)
                    self._record_groq_tokens(tokens_used)
                    
                    parsed = response_model.model_validate_json(self._clean_json(content))
                    logger.info(f"[Groq] Success with {model_id}.")
                    return parsed
                    
                except Exception as e:
                    last_exception = e
                    status_code = getattr(e, "status_code", None)
                    if status_code == 429:
                        sleep_time = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** (attempt - 1))) + random.uniform(0, 1)
                        logger.warning(f"[Groq] {model_id} 429 Rate Limit. Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"[Groq] {model_id} Error: {e}")
                        break
                        
        raise last_exception or RuntimeError("All Groq fallback models failed.")

    def generate_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        prefer_groq: bool = False,
    ) -> T:
        """
        Generate structured JSON. If prefer_groq is True, Groq is tried first
        (faster for structured output), with Gemini as fallback. Otherwise,
        Gemini is tried first.
        """
        if prefer_groq and self._groq_client:
            try:
                return self._generate_groq(
                    prompt=prompt,
                    response_model=response_model,
                    system_instruction=system_instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"Groq failed: {e}. Routing to Gemini fallback...")
                
            try:
                return self._generate_gemini(
                    prompt=prompt,
                    response_model=response_model,
                    system_instruction=system_instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(f"All LLM providers failed. Final error: {e}")
                raise
        else:
            try:
                return self._generate_gemini(
                    prompt=prompt,
                    response_model=response_model,
                    system_instruction=system_instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"Gemini failed: {e}. Routing to Groq fallback...")
                
            if not self._groq_client:
                raise RuntimeError("Gemini failed and no Groq fallback is available.")
                
            try:
                return self._generate_groq(
                    prompt=prompt,
                    response_model=response_model,
                    system_instruction=system_instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(f"All LLM providers failed. Final error: {e}")
                raise