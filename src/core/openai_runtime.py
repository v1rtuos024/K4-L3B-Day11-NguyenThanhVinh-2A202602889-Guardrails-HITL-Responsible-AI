"""
OpenAI SDK runtime — dùng cho:

  Blue Team → configured compatible API (default: Mistral)
  Red Team  → configured compatible API (default: Cohere)

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)
    _provider_gate: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1), repr=False)

    def _client(self):
        from openai import OpenAI

        return OpenAI(max_retries=0, timeout=90, **(self.client_kwargs or {}))

    async def _complete(self, agent: OpenAIAgent, user_message: str):
        """Bound retries for transient throttling, including OpenRouter's nested hint."""
        from openai import RateLimitError

        async with self._provider_gate:
            client = self._client()
            waited = 0.0
            try:
                for attempt in range(4):
                    try:
                        return await asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.model,
                            messages=[{"role": "developer" if self.provider == "cohere" else "system",
                                       "content": agent.instruction},
                                      {"role": "user", "content": user_message}],
                            temperature=self.temperature,
                        )
                    except RateLimitError as exc:
                        # A zero request allowance cannot recover through backoff.
                        if exc.response.headers.get("x-ratelimit-limit-req-minute") == "0":
                            raise
                        body = exc.body if isinstance(exc.body, dict) else {}
                        error = body.get("error", body)
                        if error.get("code") in {"insufficient_quota", "billing_hard_limit_reached"}:
                            raise
                        metadata = error.get("metadata") or {}
                        hint = (exc.response.headers.get("Retry-After")
                                or metadata.get("retry_after_seconds")
                                or (metadata.get("headers") or {}).get("Retry-After"))
                        try:
                            delay = float(hint) if hint is not None else 5 * (2 ** attempt)
                        except (TypeError, ValueError):
                            delay = 5 * (2 ** attempt)
                        # Defer rather than retry sooner than a large provider hint.
                        if not math.isfinite(delay) or delay < 0 or delay > 60:
                            raise
                        delay += random.uniform(0.1, 0.5)
                        if attempt == 3 or waited + delay > 180:
                            raise
                        print(f"Provider throttled; retry {attempt + 1}/3 in {delay:.1f}s.", flush=True)
                        await asyncio.sleep(delay)
                        waited += delay
            finally:
                close = getattr(client, "close", None)
                if close:
                    close()

    async def chat(self, agent: OpenAIAgent, user_message: str, *,
                   user_id: str = "student", decision: dict | None = None) -> str:
        decision = decision if decision is not None else {}
        decision.update(blocked=False, layer=None, redacted=False)
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                decision.update(blocked=True, layer="input_hook")
                return blocked

        block_msg = await self._run_input_plugins(user_message, user_id=user_id, decision=decision)
        if block_msg is not None:
            return block_msg

        # Input admission happens before the provider queue. This permits real
        # burst testing without flooding the provider's shared free endpoint.
        completion = await self._complete(agent, user_message)
        text = (completion.choices[0].message.content or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text, decision=decision)
        return text

    async def _run_input_plugins(self, user_message: str, *, user_id: str = "student",
                                 decision: dict | None = None) -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext(user_id=user_id)
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            if decision is not None:
                decision.update(blocked=True, layer=plugin.name)
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str, *, decision: dict | None = None) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            old_blocked = getattr(plugin, "blocked_count", 0)
            old_redacted = getattr(plugin, "redacted_count", 0)
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
            if decision is not None:
                if getattr(plugin, "redacted_count", 0) > old_redacted:
                    decision.update(redacted=True, layer=plugin.name)
                if getattr(plugin, "blocked_count", 0) > old_blocked:
                    decision.update(blocked=True, layer=plugin.name)
        return _content_to_text(llm_response.content) or text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — the configured compatible API, with the same guardrails."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team compatible API path; both targets use the same configured model."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
