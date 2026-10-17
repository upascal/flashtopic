"""OpenRouter-compatible async LLM gateway: bounded concurrency, retry, structured output.

Secrets are library-clean: ``load_dotenv()`` searches from the cwd upward (no hard-coded
project root), then falls back to the real environment — so it works equally from a
``.env`` (copy ``.env.example``) or from CI/cloud env vars. You can also override
programmatically with :func:`configure_llm`. Only the labeling / prevalence-judge stages
need a key; the clustering core runs secret-free.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()  # searches cwd upward; harmless if absent

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_config = {
    "api_key": os.environ.get("OPENROUTER_API_KEY"),
    "base_url": os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL),
}
_client: AsyncOpenAI | None = None


def configure_llm(*, api_key: str | None = None, base_url: str | None = None) -> None:
    """Override the API key and/or base URL programmatically (resets the client)."""
    global _client
    if api_key is not None:
        _config["api_key"] = api_key
    if base_url is not None:
        _config["base_url"] = base_url
    _client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = _config["api_key"] or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Set the env var, add it to a .env "
                "(see .env.example), or call configure_llm(api_key=...). Only the LLM "
                "labeling / prevalence-judge stages need it — clustering runs without a key."
            )
        _client = AsyncOpenAI(base_url=_config["base_url"], api_key=key)
    return _client


_semaphore: asyncio.Semaphore | None = None
_semaphore_limit: int = 5


def configure_concurrency(limit: int) -> None:
    """Set the in-flight LLM-call cap (resets the semaphore)."""
    global _semaphore, _semaphore_limit
    _semaphore_limit, _semaphore = limit, None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_semaphore_limit)
    return _semaphore


# Approximate per-1M-token prices for cost telemetry only; unknown models → 0.
# OpenRouter billing is the ground truth. Override with set_pricing if you care.
_PRICING: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v3.2": (0.27, 1.10),
    "google/gemini-3-flash": (0.50, 3.00),
    "google/gemini-3-pro": (3.50, 10.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}


def set_pricing(model: str, input_per_m: float, output_per_m: float) -> None:
    _PRICING[model] = (input_per_m, output_per_m)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _PRICING.get(model, (0.0, 0.0))
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None


def _backoff(attempt: int) -> float:
    return min(2**attempt + random.random(), 30.0)


async def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_attempts: int = 6,
) -> LLMResult:
    """One LLM call returning raw text; retries on transient errors."""
    client, sem = get_client(), _get_semaphore()
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            if not text:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                return LLMResult("", model, in_tok, out_tok,
                                 estimate_cost_usd(model, in_tok, out_tok), "empty_response")
            return LLMResult(text, model, in_tok, out_tok, estimate_cost_usd(model, in_tok, out_tok))
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_err = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(_backoff(attempt))
                continue
        except openai.APIError as e:
            last_err = e
            break
    return LLMResult("", model, 0, 0, 0.0,
                     f"llm_call_failed: {str(last_err)[:200]}" if last_err else "unknown")


async def call_llm_structured(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_model: type[BaseModel],
    *,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    max_attempts: int = 6,
) -> tuple[BaseModel | None, LLMResult]:
    """One LLM call returning a parsed Pydantic instance via JSON-schema mode.

    NOTE: keep response models free of numeric ``ge``/``le`` constraints — some
    providers reject ``minimum``/``maximum`` on number types in structured output.
    """
    client, sem = get_client(), _get_semaphore()
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": response_model.__name__, "schema": response_model.model_json_schema()},
    }
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            raw = LLMResult(text, model, in_tok, out_tok, estimate_cost_usd(model, in_tok, out_tok))
            if not text:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raw.error = "empty_response"
                return None, raw
            try:
                return response_model.model_validate_json(text), raw
            except (ValidationError, json.JSONDecodeError) as ve:
                if attempt < max_attempts - 1:
                    last_err = ve
                    await asyncio.sleep(1.0)
                    continue
                raw.error = f"validation_failed: {str(ve)[:200]}"
                return None, raw
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_err = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(_backoff(attempt))
                continue
        except openai.APIError as e:
            last_err = e
            break
    return None, LLMResult("", model, 0, 0, 0.0,
                           f"llm_call_failed: {str(last_err)[:200]}" if last_err else "unknown")


def parse_json_from_text(text: str) -> Any:
    """Extract JSON from text — bare or fenced ```json blocks."""
    if not text:
        raise ValueError("empty text")
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)
