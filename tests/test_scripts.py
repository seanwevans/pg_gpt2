import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psycopg
import pytest

convert_gpt2_checkpoint = importlib.import_module("scripts.convert_gpt2_checkpoint")
ingest_tokenizer = importlib.import_module("scripts.ingest_tokenizer")
prepare_dataset = importlib.import_module("scripts.prepare_dataset")


@pytest.fixture
def schema_setup(postgres_dsn: str) -> None:
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS llm_bpe_vocab")
        conn.execute("DROP TABLE IF EXISTS llm_bpe_merges")
        conn.execute(
            """
            CREATE TABLE llm_bpe_vocab (
                model TEXT,
                token_id INTEGER PRIMARY KEY,
                token TEXT,
                score DOUBLE PRECISION,
                bytes BYTEA
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE llm_bpe_merges (
                model TEXT,
                rank INTEGER PRIMARY KEY,
                "left" TEXT,
                "right" TEXT,
                pair TEXT
            )
            """
        )
        conn.execute("DROP TABLE IF EXISTS llm_dataset")
        conn.execute(
            """
            CREATE TABLE llm_dataset (
                id SERIAL PRIMARY KEY,
                tokens INTEGER[],
                target INTEGER[]
            )
            """
        )


def test_ingest_tokenizer_cli(postgres_dsn: str, tmp_path: Path, schema_setup: None) -> None:
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"

    vocab_path.write_text(json.dumps({"hello": 0, "world": 1}), encoding="utf-8")
    merges_path.write_text("# merges\nhe llo\nwo rld\n", encoding="utf-8")

    argv = [
        "ingest_tokenizer.py",
        "--dsn",
        postgres_dsn,
        "--model",
        "test-model",
        "--vocab",
        str(vocab_path),
        "--merges",
        str(merges_path),
        "--truncate",
    ]

    original_argv = sys.argv
    sys.argv = argv
    try:
        ingest_tokenizer.main()
    finally:
        sys.argv = original_argv

    with psycopg.connect(postgres_dsn) as conn:
        vocab_rows = conn.execute(
            "SELECT token_id, token, bytes FROM llm_bpe_vocab ORDER BY token_id"
        ).fetchall()
        merge_rows = conn.execute(
            'SELECT rank, "left", "right" FROM llm_bpe_merges ORDER BY rank'
        ).fetchall()

    assert [(row[0], row[1]) for row in vocab_rows] == [(0, "hello"), (1, "world")]
    assert [bytes(row[2]) for row in vocab_rows] == [b"hello", b"world"]
    assert merge_rows == [(0, "he", "llo"), (1, "wo", "rld")]


def test_prepare_dataset_cli(postgres_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema_setup: None) -> None:
    text_path = tmp_path / "input.txt"
    text_path.write_text("Once upon a time", encoding="utf-8")

    class DummyTokenizer:
        pad_token = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, name: str):
            return cls()

        def encode(self, text: str):
            return [ord(ch) % 97 for ch in text]

    monkeypatch.setattr(prepare_dataset, "_lazy_import_tokenizer", lambda: DummyTokenizer)

    argv = [
        "prepare_dataset.py",
        "--dsn",
        postgres_dsn,
        "--tokenizer",
        "dummy",
        "--input",
        str(text_path),
        "--block-size",
        "4",
        "--batch-size",
        "2",
        "--truncate",
    ]

    original_argv = sys.argv
    sys.argv = argv
    try:
        prepare_dataset.main()
    finally:
        sys.argv = original_argv

    with psycopg.connect(postgres_dsn) as conn:
        rows = conn.execute("SELECT tokens, target FROM llm_dataset ORDER BY id").fetchall()

    assert rows, "expected rows in llm_dataset"
    for tokens, target in rows:
        assert len(tokens) == len(target)
        assert all(isinstance(t, int) for t in tokens)
        assert all(isinstance(t, int) for t in target)


def test_convert_checkpoint_generates_expected_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel:
        def state_dict(self):
            return {
                "transformer.wte.weight": np.ones((2, 2), dtype=np.float32),
                "transformer.wpe.weight": np.ones((2, 2), dtype=np.float32) * 2,
                "transformer.h.0.attn.c_attn.weight": np.ones((2, 2), dtype=np.float32) * 3,
                "transformer.h.0.ln_1.weight": np.ones((2,), dtype=np.float32) * 4,
                "transformer.ln_f.bias": np.ones((2,), dtype=np.float32) * 5,
            }

    class DummyAutoModel:
        @classmethod
        def from_pretrained(cls, source: str, revision=None, torch_dtype=None):
            return DummyModel()

    monkeypatch.setattr(
        convert_gpt2_checkpoint, "_lazy_import_transformers", lambda: DummyAutoModel
    )

    output_path = tmp_path / "dummy.npz"
    convert_gpt2_checkpoint.convert_checkpoint("dummy", output_path)

    tensors = {}
    with output_path.open("rb") as raw:
        import gzip

        with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            while True:
                name_len_bytes = gz.read(2)
                if not name_len_bytes:
                    break
                name_len = int.from_bytes(name_len_bytes, "little")
                name = gz.read(name_len).decode("utf-8")
                array = np.load(gz, allow_pickle=False)
                tensors[name] = array

    assert set(tensors) == {
        "h.0.attn.c_attn.weight",
        "h.0.ln_1.weight",
        "ln_f.bias",
        "wpe",
        "wte",
    }
    assert tensors["wte"].dtype == np.float32
    assert tensors["h.0.attn.c_attn.weight"].shape == (2, 2)
