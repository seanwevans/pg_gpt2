import sys
from pathlib import Path

import numpy as np
import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pg_llm_client import PGLLMClient


@pytest.fixture
def client(postgres_dsn: str) -> PGLLMClient:
    conn = psycopg.connect(postgres_dsn)
    try:
        yield PGLLMClient(conn)
    finally:
        conn.close()


def test_generate_with_temperature_controls(postgres_dsn: str) -> None:
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute(
            """
            CREATE OR REPLACE FUNCTION llm_generate(
                prompt TEXT,
                max_tokens INT,
                temperature FLOAT4,
                topk INT,
                topp FLOAT4,
                model_name TEXT,
                eos_token INT DEFAULT 50256)
            RETURNS TEXT
            LANGUAGE SQL
            AS $$ SELECT prompt || ' :: ' || temperature::TEXT $$;
            """
        )

    with psycopg.connect(postgres_dsn) as conn:
        generated = PGLLMClient(conn).generate("Hello", temperature=0.8)
    assert generated == "Hello :: 0.8"


def test_streaming_generation(postgres_dsn: str) -> None:
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute(
            """
            CREATE OR REPLACE FUNCTION llm_generate_stream(
                prompt TEXT,
                max_tokens INT,
                temperature FLOAT4,
                topk INT,
                topp FLOAT4,
                model_name TEXT,
                eos_token INT DEFAULT 50256)
            RETURNS TABLE(step INT, token_id INT, token TEXT, text TEXT, is_complete BOOLEAN)
            LANGUAGE SQL
            AS $$
                SELECT 1, 10, 'A', prompt || 'A', false
                UNION ALL
                SELECT 2, 11, 'B', prompt || 'AB', true
            $$;
            """
        )

    with psycopg.connect(postgres_dsn) as conn:
        events = list(PGLLMClient(conn).stream("Hi"))

    assert [event.token for event in events] == ["A", "B"]
    assert events[-1].is_complete is True
    assert events[-1].text.endswith("AB")


def test_beam_search_ranks_candidates(monkeypatch: pytest.MonkeyPatch, client: PGLLMClient) -> None:
    monkeypatch.setattr(client, "_encode", lambda prompt, model: [0])

    vocab = {1: "A", 2: "B", 3: ""}

    def fake_decode(tokens, model):
        return "".join(vocab.get(tok, "") for tok in tokens if tok in vocab)

    monkeypatch.setattr(client, "_decode", fake_decode)

    logits = iter(
        [
            np.array([0.0, 3.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.5, 0.2, 3.0], dtype=np.float32),
            np.array([0.0, 0.1, 3.5, 0.0], dtype=np.float32),
        ]
    )

    def fake_fetch(ids, model):
        try:
            return next(logits)
        except StopIteration:  # pragma: no cover - safety for unexpected calls
            return np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)

    monkeypatch.setattr(client, "_fetch_logits", fake_fetch)

    results = client.beam_search("Hi", beam_width=2, max_tokens=2, eos_token=3)

    assert results
    assert results[0].text.startswith("A")
    assert results[0].token_ids[-1] == 3
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))
