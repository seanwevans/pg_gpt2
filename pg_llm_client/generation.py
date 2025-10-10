"""Utilities for generating text from a PostgreSQL pg_llm deployment.

This module provides a small convenience wrapper around the SQL API exposed by
:mod:`pg_llm`.  It enables client applications to tweak sampling parameters,
run lightweight beam search, and stream generated tokens as they become
available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence

import numpy as np
import psycopg


@dataclass(frozen=True)
class StreamEvent:
    """Represents a single streamed token from ``llm_generate_stream``."""

    step: int
    token_id: int
    token: str
    text: str
    is_complete: bool


@dataclass(frozen=True)
class BeamResult:
    """Represents the result of a beam search candidate."""

    text: str
    token_ids: List[int]
    score: float


class PGLLMClient:
    """High-level helper for issuing text generation queries.

    Parameters
    ----------
    conn:
        A live :class:`psycopg.Connection` to the database hosting the
        ``pg_llm`` extension.
    """

    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 1.0,
        topk: int = 50,
        topp: float = 0.95,
        model: str = "gpt2-small",
    ) -> str:
        """Return a single completion for ``prompt``.

        The method is a thin wrapper around the ``llm_generate`` SQL function
        so callers can tweak temperature/top-k/top-p directly from client code.
        """

        query = (
            "SELECT llm_generate(%s, %s, %s, %s, %s, %s)"
        )
        with self._conn.cursor() as cur:
            cur.execute(query, (prompt, max_tokens, temperature, topk, topp, model))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("llm_generate returned no rows")
        return row[0]

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 1.0,
        topk: int = 50,
        topp: float = 0.95,
        model: str = "gpt2-small",
    ) -> Generator[StreamEvent, None, None]:
        """Yield tokens from ``llm_generate_stream`` as they are produced."""

        sql = (
            "SELECT step, token_id, token, text, is_complete "
            "FROM llm_generate_stream(%s, %s, %s, %s, %s, %s)"
        )
        with self._conn.cursor() as cur:
            stream = cur.stream(
                sql,
                (prompt, max_tokens, temperature, topk, topp, model),
            )
            for step, token_id, token, text, is_complete in stream:
                yield StreamEvent(
                    step=step,
                    token_id=token_id,
                    token=token,
                    text=text,
                    is_complete=is_complete,
                )

    def beam_search(
        self,
        prompt: str,
        *,
        beam_width: int = 3,
        max_tokens: int = 64,
        temperature: float = 1.0,
        model: str = "gpt2-small",
        eos_token: int = 50256,
    ) -> List[BeamResult]:
        """Return the top ``beam_width`` completions using beam search."""

        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if max_tokens <= 0:
            return []

        base_tokens = list(self._encode(prompt, model))
        beams: List[tuple[List[int], float]] = [(base_tokens, 0.0)]
        completed: List[tuple[List[int], float]] = []

        for _ in range(max_tokens):
            candidates: List[tuple[List[int], float]] = []
            for tokens, score in beams:
                logits = self._fetch_logits(tokens, model)
                log_probs = self._log_softmax(logits, temperature)
                top_indices = self._top_indices(log_probs, beam_width)

                for token_id in top_indices:
                    idx = int(token_id)
                    next_tokens = tokens + [idx]
                    next_score = score + float(log_probs[idx])
                    if idx == eos_token:
                        completed.append((next_tokens, next_score))
                    else:
                        candidates.append((next_tokens, next_score))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[:beam_width]

        if not completed:
            completed = beams

        results: List[BeamResult] = []
        for tokens, score in sorted(completed, key=lambda item: item[1], reverse=True)[
            :beam_width
        ]:
            text = self._decode(tokens, model)
            results.append(BeamResult(text=text, token_ids=tokens, score=score))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode(self, prompt: str, model: str) -> List[int]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT llm_encode(%s, %s)", (prompt, model))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("llm_encode returned no rows")
        token_ids = row[0] or []
        return list(token_ids)

    def _decode(self, token_ids: Sequence[int], model: str) -> str:
        with self._conn.cursor() as cur:
            cur.execute("SELECT llm_decode(%s, %s)", (list(token_ids), model))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("llm_decode returned no rows")
        return row[0] or ""

    def _fetch_logits(self, token_ids: Sequence[int], model: str) -> np.ndarray:
        with self._conn.cursor(binary=True) as cur:
            cur.execute("SELECT llm_logits(%s, %s)", (list(token_ids), model))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("llm_logits returned no rows")
        blob = row[0]
        if isinstance(blob, memoryview):
            data = blob.tobytes()
        elif isinstance(blob, (bytes, bytearray)):
            data = bytes(blob)
        else:
            raise TypeError("llm_logits returned unexpected type")
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        if logits.size == 0:
            return logits
        temp = max(float(temperature), 1e-5)
        scaled = logits.astype(np.float64) / temp
        max_logit = np.max(scaled)
        stable = scaled - max_logit
        exp = np.exp(stable)
        total = np.sum(exp)
        if not np.isfinite(total) or total <= 0:
            return np.full_like(logits, fill_value=-np.inf, dtype=np.float64)
        log_probs = stable - np.log(total)
        return log_probs.astype(np.float64)

    @staticmethod
    def _top_indices(log_probs: np.ndarray, beam_width: int) -> Iterable[int]:
        if beam_width >= log_probs.size:
            return np.argsort(log_probs)[::-1]
        indices = np.argpartition(log_probs, -beam_width)[-beam_width:]
        return indices[np.argsort(log_probs[indices])[::-1]]
