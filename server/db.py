"""Database access for the chat endpoint.

Everything here is a thin wrapper over the SQL API the extension exposes;
no model logic lives in Python.  Generation runs inside Postgres, so the
Python side is only responsible for pooling, timeouts and back-pressure.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import AsyncIterator

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from .config import settings


@dataclass(frozen=True)
class ModelInfo:
    model: str
    n_layer: int
    n_head: int
    d_model: int
    n_positions: int
    vocab: int
    param_rows: int
    vocab_rows: int


@dataclass(frozen=True)
class StreamEvent:
    step: int
    token_id: int
    token: str
    text: str
    is_complete: bool


class Backend:
    """Owns the connection pool and serialises access to the model."""

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)

    async def start(self) -> None:
        self._pool = AsyncConnectionPool(
            settings.dsn,
            min_size=settings.pool_min_size,
            max_size=settings.pool_max_size,
            open=False,
            kwargs={"autocommit": True},
        )
        await self._pool.open(wait=True, timeout=30)

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @contextlib.asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        """Bound in-flight generations so a burst queues instead of thrashing."""

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=settings.queue_timeout_s
            )
        except asyncio.TimeoutError as exc:  # pragma: no cover - load dependent
            raise BackendBusy("the model is busy; try again shortly") from exc
        try:
            yield
        finally:
            self._semaphore.release()

    @contextlib.asynccontextmanager
    async def _connection(self):
        if self._pool is None:
            raise RuntimeError("backend not started")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("SET statement_timeout = {}").format(
                        sql.Literal(settings.statement_timeout_ms)
                    )
                )
            yield conn

    async def ping(self) -> str:
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'pg_llm'")
                row = await cur.fetchone()
        if row is None:
            raise RuntimeError("the pg_llm extension is not installed in this database")
        return row[0]

    async def list_models(self) -> list[ModelInfo]:
        query = """
            SELECT c.model,
                   c.n_layer,
                   c.n_head,
                   c.d_model,
                   c.n_positions,
                   c.vocab,
                   (SELECT count(*) FROM llm_param p WHERE p.model = c.model),
                   (SELECT count(*) FROM llm_bpe_vocab v WHERE v.model = c.model)
              FROM llm_model_config c
             ORDER BY c.model
        """
        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                rows = await cur.fetchall()
        return [ModelInfo(*row) for row in rows]

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        model: str,
    ) -> str:
        async with self._slot():
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT llm_generate(
                            %s::text, %s::int, %s::float4, %s::int, %s::float4, %s::text
                        )
                        """,
                        (prompt, max_tokens, temperature, top_k, top_p, model),
                    )
                    row = await cur.fetchone()
        return row[0] if row and row[0] is not None else ""

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        model: str,
        eos_token: int = 50256,
    ) -> AsyncIterator[StreamEvent]:
        """Yield one event per generated token, as it is produced.

        The loop lives here rather than in llm_generate_stream because a
        PL/pgSQL set-returning function fills its whole tuplestore before the
        first row becomes visible, which would defeat streaming.  Each
        iteration is still a single call into the database: llm_next_token runs
        the forward pass and the sampler in SQL, and llm_decode turns the
        sequence back into text.
        """

        async with self._slot():
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT llm_encode(%s::text, %s::text)", (prompt, model)
                    )
                    row = await cur.fetchone()
                    ids: list[int] = list(row[0] or []) if row else []
                    if not ids:
                        return

                    for step in range(1, max_tokens + 1):
                        await cur.execute(
                            """
                            SELECT llm_next_token(
                                %s::int[], %s::text, %s::float4, %s::int, %s::float4
                            )
                            """,
                            (ids, model, temperature, top_k, top_p),
                        )
                        row = await cur.fetchone()
                        if row is None or row[0] is None:
                            return
                        next_id = int(row[0])
                        ids.append(next_id)

                        await cur.execute(
                            "SELECT llm_decode(%s::int[], %s::text), "
                            "       coalesce((SELECT v.token FROM llm_bpe_vocab v "
                            "                  WHERE v.model = %s::text "
                            "                    AND v.token_id = %s::int), '')",
                            (ids, model, model, next_id),
                        )
                        row = await cur.fetchone()
                        text = (row[0] if row else "") or ""
                        token = (row[1] if row else "") or ""

                        complete = next_id == eos_token or step >= max_tokens
                        yield StreamEvent(
                            step=step,
                            token_id=next_id,
                            token=token,
                            text=text,
                            is_complete=complete,
                        )
                        if next_id == eos_token:
                            return


class BackendBusy(RuntimeError):
    """Raised when the generation queue is saturated."""


backend = Backend()
