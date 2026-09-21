"""Module 7 (part 1) - pluggable LLM providers.

Groq, Gemini, OpenAI and Ollama all speak the OpenAI chat-completions wire
format, so one client class covers all four and switching provider is a base
URL plus a key. Provider choice then becomes a config decision rather than a
code change — which is what makes it cheap to run generation on a free tier
and the evaluation judge on a local model.

``ExtractiveClient`` is the floor: it needs no key, no network, and no model.
It stitches together the highest-scoring retrieved sentences. Answers are
blunt, but the pipeline, the citations, and the whole evaluation harness stay
runnable with zero credentials, which keeps the repo clonable by anyone.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.text import count_tokens

log = get_logger(__name__)

# USD per 1M tokens (prompt, completion). Free tiers are recorded as 0.0 so the
# cost-per-query metric reflects what you would actually pay.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "llama-3.1-8b-instant": (0.0, 0.0),
    "openai/gpt-oss-120b": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
    "qwen/qwen3.8-27b": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-2.0-flash": (0.0, 0.0),
}


class RateLimiter:
    """Thread-safe rolling-window throttle over both requests and tokens.

    The tokens-per-minute limit is what actually binds on free tiers. Groq's
    free allowance is 1000 requests/day but only 8000 tokens/minute, and a
    single RAG call carrying five retrieved passages costs ~3300 tokens — so
    a request-per-minute throttle would sail past the limit and collect 429s.

    Usage is tracked in a rolling 60-second window and the caller blocks until
    the window has drained enough room for the estimated cost of the next
    request. Waiting deliberately beats backing off after a rejection: the
    request still gets made, just on schedule, and no quota is burned on calls
    that get refused.
    """

    def __init__(self, requests_per_minute: int = 0, tokens_per_minute: int = 0) -> None:
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def acquire(self, estimated_tokens: int = 0) -> None:
        if self.rpm <= 0 and self.tpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                used_tokens = sum(t for _, t in self._events)
                used_requests = len(self._events)

                token_ok = self.tpm <= 0 or used_tokens + estimated_tokens <= self.tpm
                request_ok = self.rpm <= 0 or used_requests + 1 <= self.rpm

                if token_ok and request_ok:
                    self._events.append((now, estimated_tokens))
                    return

                # Sleep until the oldest event ages out of the window.
                wait = 60.0 - (now - self._events[0][0]) + 0.05 if self._events else 0.5
            log.debug("rate limit: waiting %.1fs", wait)
            time.sleep(max(wait, 0.05))

    def record_actual(self, estimated: int, actual: int) -> None:
        """Correct the window after the fact, once real usage is known."""
        if self.tpm <= 0 or not self._events:
            return
        with self._lock:
            timestamp, _ = self._events[-1]
            self._events[-1] = (timestamp, max(actual, estimated))


# Conservative free-tier defaults: (requests/min, tokens/min).
# Local and paid providers are left unmetered.
#
# Groq's documented allowance is 8000 tokens/min. The budget here is set well
# below it because the pre-request estimate comes from tiktoken while the
# provider counts with its own tokenizer; the estimate runs low, so pacing to
# the exact documented limit still collects 429s.
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    "groq": (25, 6200),
    "gemini": (12, 900_000),
    "openai": (0, 0),
    "ollama": (0, 0),
    "extractive": (0, 0),
}


def is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return any(tag in lowered for tag in ("gpt-oss", "o1", "o3", "qwen3", "deepseek-r1"))


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        prompt_rate, completion_rate = PRICING.get(self.model, (0.0, 0.0))
        return (
            self.prompt_tokens * prompt_rate + self.completion_tokens * completion_rate
        ) / 1_000_000


class LLMClient(ABC):
    provider: str = "base"
    model: str = ""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

    @property
    def available(self) -> bool:
        return True


class OpenAICompatibleClient(LLMClient):
    """One implementation for every OpenAI-wire-format provider."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 5,
        limits: tuple[int, int] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        rpm, tpm = limits if limits is not None else DEFAULT_LIMITS.get(provider, (0, 0))
        self.limiter = RateLimiter(rpm, tpm)
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else (get_settings().reasoning_effort if is_reasoning_model(model) else None)
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key or "not-needed",
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Budget prompt + worst-case completion so the window is never oversold.
        estimated = count_tokens(system) + count_tokens(user) + max_tokens
        self.limiter.acquire(estimated)

        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            # Reasoning models (gpt-oss, o-series, R1) spend the completion
            # budget on hidden chain-of-thought before emitting any content.
            # Left unbounded they return an *empty* message once max_tokens is
            # exhausted — which silently looked like a parse failure until the
            # token accounting gave it away. Capping the effort keeps the
            # visible answer inside the budget.
            extra["reasoning_effort"] = self.reasoning_effort

        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        latency = time.perf_counter() - start
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.limiter.record_actual(estimated, prompt_tokens + completion_tokens)

        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency,
        )


class ExtractiveClient(LLMClient):
    """Offline fallback: no key, no network, no model.

    Returns the leading sentences of the top-ranked passages with their citation
    markers intact. This is genuinely extractive, not generative — it cannot
    hallucinate, which makes it a useful faithfulness ceiling to compare against.
    """

    provider = "extractive"
    model = "extractive-baseline"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        from src.common.text import split_sentences

        start = time.perf_counter()
        # The prompt embeds context as "[n] <text>"; recover those blocks.
        blocks: list[tuple[int, str]] = []
        current_marker: int | None = None
        current: list[str] = []
        for line in user.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and "]" in stripped[:6]:
                if current_marker is not None:
                    blocks.append((current_marker, " ".join(current)))
                try:
                    current_marker = int(stripped[1 : stripped.index("]")])
                except ValueError:
                    current_marker = None
                current = [stripped[stripped.index("]") + 1 :].strip()]
            elif current_marker is not None:
                current.append(stripped)
        if current_marker is not None:
            blocks.append((current_marker, " ".join(current)))

        if not blocks:
            text = "The information could not be found in the provided documents."
        else:
            parts = []
            for marker, body in blocks[:3]:
                sentences = split_sentences(body)
                if sentences:
                    parts.append(" ".join(sentences[:2]).strip() + f" [{marker}]")
            text = " ".join(parts) or "The information could not be found."

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            latency=time.perf_counter() - start,
        )


# Clients are cached per (provider, model) because the rate limiter lives on
# the client and its window is only meaningful if every caller shares it.
# Keying by model matches how providers meter: Groq tracks tokens-per-minute
# per model, so generating on 120b and judging on 20b draw on separate budgets.
_CLIENTS: dict[tuple[str, str | None], LLMClient] = {}


def _build(provider: str, model: str | None = None) -> LLMClient | None:
    cache_key = (provider.lower(), model)
    if cache_key in _CLIENTS:
        return _CLIENTS[cache_key]
    client = _build_uncached(provider, model)
    if client is not None:
        _CLIENTS[cache_key] = client
    return client


def _build_uncached(provider: str, model: str | None = None) -> LLMClient | None:
    s = get_settings()
    provider = provider.lower()

    if provider == "groq":
        if not s.groq_api_key:
            return None
        return OpenAICompatibleClient(
            "groq", model or s.groq_model, s.groq_api_key, "https://api.groq.com/openai/v1"
        )
    if provider == "gemini":
        if not s.gemini_api_key:
            return None
        return OpenAICompatibleClient(
            "gemini",
            model or s.gemini_model,
            s.gemini_api_key,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == "openai":
        if not s.openai_api_key:
            return None
        return OpenAICompatibleClient("openai", model or s.openai_model, s.openai_api_key)
    if provider == "ollama":
        client = OpenAICompatibleClient(
            "ollama", model or s.ollama_model, "ollama", s.ollama_base_url
        )
        if not _ollama_reachable(s.ollama_base_url):
            return None
        return client
    if provider in {"extractive", "none", "offline"}:
        return ExtractiveClient()
    return None


def _ollama_reachable(base_url: str) -> bool:
    import httpx

    try:
        root = base_url.rstrip("/").removesuffix("/v1")
        httpx.get(f"{root}/api/tags", timeout=1.5).raise_for_status()
        return True
    except Exception:
        return False


# Preference order when LLM_PROVIDER=auto: best free quality first, then local,
# then the offline floor so the pipeline always answers.
AUTO_ORDER = ("groq", "gemini", "openai", "ollama", "extractive")


def get_llm(provider: str | None = None, model: str | None = None) -> LLMClient:
    s = get_settings()
    requested = (provider or s.llm_provider or "auto").lower()

    if requested != "auto":
        client = _build(requested, model)
        if client is None:
            log.warning(
                "LLM provider %r is configured but unavailable (missing key or server); "
                "falling back to auto-detection",
                requested,
            )
        else:
            return client

    for candidate in AUTO_ORDER:
        client = _build(candidate, model if candidate == requested else None)
        if client is not None:
            if candidate == "extractive":
                log.warning(
                    "No LLM credentials found. Using the offline extractive baseline. "
                    "Set GROQ_API_KEY in .env for generated answers."
                )
            else:
                log.info("LLM provider: %s (%s)", candidate, client.model)
            return client

    return ExtractiveClient()


def get_judge_llm() -> LLMClient:
    """Model used by the evaluation judge.

    Kept separate so you can judge with a local/cheap model while generating
    with a better one — and so the judge is never the same call as the thing
    being judged.
    """
    s = get_settings()
    return get_llm(s.judge_provider or None, s.judge_model or None)
