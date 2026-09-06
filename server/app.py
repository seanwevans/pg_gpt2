"""HTTP chat endpoint in front of a pg_gpt2 database.

GitHub Pages serves static files only, so the chat UI published there needs a
network-reachable endpoint to talk to.  This is that endpoint: a thin, CORS
enabled JSON API whose entire job is to turn an HTTP request into a call to
``llm_generate``/``llm_generate_stream`` and stream the result back.

Run it with::

    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import settings
from .db import BackendBusy, backend

API_VERSION = "1"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """A chat turn, or a bare completion prompt.

    GPT-2 is a completion model with no chat fine-tuning, so ``messages`` is
    flattened into a plain transcript before it reaches the database.  Sending
    ``prompt`` instead skips that framing entirely.
    """

    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=settings.default_temperature, ge=0.0, le=2.0)
    top_k: int = Field(default=settings.default_top_k, ge=0)
    top_p: float = Field(default=settings.default_top_p, ge=0.0, le=1.0)
    model: str | None = None


class ChatResponse(BaseModel):
    model: str
    prompt: str
    completion: str
    text: str
    tokens: int
    elapsed_ms: int


class _RateLimiter:
    """Fixed-window-free token bucket keyed on client address."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self._per_minute <= 0:
            return
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60.0:
            hits.popleft()
        if len(hits) >= self._per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit of {self._per_minute} requests/minute exceeded",
            )
        hits.append(now)


limiter = _RateLimiter(settings.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await backend.start()
    try:
        yield
    finally:
        await backend.stop()


app = FastAPI(
    title="pg_gpt2 chat endpoint",
    version=API_VERSION,
    description="Text generation served directly out of PostgreSQL.",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=86400,
)


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _build_prompt(payload: ChatRequest) -> str:
    if payload.prompt is not None:
        prompt = payload.prompt
    elif payload.messages:
        lines = []
        if settings.chat_system_prompt:
            lines.append(settings.chat_system_prompt)
        for message in payload.messages:
            speaker = {"system": "System", "user": "Human", "assistant": "AI"}[
                message.role
            ]
            lines.append(f"{speaker}: {message.content}")
        lines.append("AI:")
        prompt = "\n".join(lines)
    else:
        raise HTTPException(status_code=422, detail="provide either prompt or messages")

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is empty")
    if len(prompt) > settings.max_prompt_chars:
        raise HTTPException(
            status_code=413,
            detail=f"prompt exceeds {settings.max_prompt_chars} characters",
        )
    return prompt


def _clamp_tokens(requested: int) -> int:
    return max(1, min(requested, settings.max_tokens_limit))


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    try:
        version = await backend.ping()
    except Exception as exc:  # pragma: no cover - depends on the database
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "pg_llm_version": version,
        "default_model": settings.model,
        "max_tokens": settings.max_tokens_limit,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    models = await backend.list_models()
    return {
        "default": settings.model,
        "models": [
            {
                "id": info.model,
                "n_layer": info.n_layer,
                "n_head": info.n_head,
                "d_model": info.d_model,
                "n_positions": info.n_positions,
                "vocab": info.vocab,
                "param_rows": info.param_rows,
                "tokenizer_rows": info.vocab_rows,
                "ready": info.param_rows > 0 and info.vocab_rows > 0,
            }
            for info in models
        ],
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    limiter.check(_client_key(request))

    prompt = _build_prompt(payload)
    model = payload.model or settings.model
    max_tokens = _clamp_tokens(payload.max_tokens)

    started = time.monotonic()
    try:
        text = await backend.generate(
            prompt, max_tokens, payload.temperature, payload.top_k, payload.top_p, model
        )
    except BackendBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed_ms = int((time.monotonic() - started) * 1000)

    # llm_generate returns prompt + completion; the caller usually wants just
    # the new text, so hand back both.
    completion = text[len(prompt):] if text.startswith(prompt) else text

    return ChatResponse(
        model=model,
        prompt=prompt,
        completion=completion,
        text=text,
        tokens=max_tokens,
        elapsed_ms=elapsed_ms,
    )


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    limiter.check(_client_key(request))

    prompt = _build_prompt(payload)
    model = payload.model or settings.model
    max_tokens = _clamp_tokens(payload.max_tokens)

    async def events() -> AsyncIterator[str]:
        yield _sse("start", {"model": model, "prompt": prompt, "max_tokens": max_tokens})
        previous = ""
        try:
            async for item in backend.generate_stream(
                prompt,
                max_tokens,
                payload.temperature,
                payload.top_k,
                payload.top_p,
                model,
            ):
                # `text` is the full decoded sequence so far, including the
                # prompt; send the newly decoded suffix as the delta.
                full = item.text
                completion = full[len(prompt):] if full.startswith(prompt) else full
                delta = completion[len(previous):] if completion.startswith(previous) else completion
                previous = completion
                yield _sse(
                    "token",
                    {
                        "step": item.step,
                        "token_id": item.token_id,
                        "delta": delta,
                        "completion": completion,
                    },
                )
                if await request.is_disconnected():
                    return
            yield _sse("done", {"completion": previous})
        except BackendBusy as exc:
            yield _sse("error", {"message": str(exc)})
        except Exception as exc:  # pragma: no cover - depends on the database
            yield _sse("error", {"message": f"generation failed: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
