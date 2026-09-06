"""Runtime configuration for the pg_gpt2 chat endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Everything the service reads from the environment.

    The defaults are tuned for the docker-compose stack in this repository:
    a single Postgres holding one model, fronted by a public read-only HTTP
    endpoint that a static page on GitHub Pages can call.
    """

    dsn: str = os.environ.get(
        "PG_GPT2_DSN", "postgresql://postgres@localhost:5432/postgres"
    )
    model: str = os.environ.get("PG_GPT2_MODEL", "gpt2-small")

    # Generation limits. These are hard caps: a client may ask for less but
    # never for more, because every token is a full forward pass in SQL.
    max_tokens_limit: int = _env_int("PG_GPT2_MAX_TOKENS", 64)
    max_prompt_chars: int = _env_int("PG_GPT2_MAX_PROMPT_CHARS", 2000)
    default_temperature: float = _env_float("PG_GPT2_TEMPERATURE", 0.8)
    default_top_k: int = _env_int("PG_GPT2_TOP_K", 40)
    default_top_p: float = _env_float("PG_GPT2_TOP_P", 0.95)

    # A generation that overruns its budget is aborted by Postgres itself
    # rather than tying up a connection indefinitely.
    statement_timeout_ms: int = _env_int("PG_GPT2_STATEMENT_TIMEOUT_MS", 120_000)

    # Connection pool. Generation is CPU-bound inside Postgres, so a small
    # pool plus a queue is preferable to letting requests fan out.
    pool_min_size: int = _env_int("PG_GPT2_POOL_MIN", 1)
    pool_max_size: int = _env_int("PG_GPT2_POOL_MAX", 4)
    max_concurrent: int = _env_int("PG_GPT2_MAX_CONCURRENT", 4)
    queue_timeout_s: float = _env_float("PG_GPT2_QUEUE_TIMEOUT_S", 30.0)

    # Per-client token bucket, keyed on the peer address.
    rate_limit_per_minute: int = _env_int("PG_GPT2_RATE_LIMIT_PER_MINUTE", 20)

    # CORS. A GitHub Pages site is a different origin from the API host, so
    # the browser will not talk to it unless the origin is allowed here.
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("PG_GPT2_CORS_ORIGINS", ["*"])
    )

    chat_system_prompt: str = os.environ.get(
        "PG_GPT2_SYSTEM_PROMPT",
        "The following is a conversation with an AI assistant.",
    )
    enable_docs: bool = _env_bool("PG_GPT2_ENABLE_DOCS", True)


settings = Settings()
