import sys
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _extract_function(sql_text: str, name: str) -> str:
    marker = f"CREATE OR REPLACE FUNCTION {name}"
    try:
        start = sql_text.index(marker)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise AssertionError(f"could not find definition for {name}") from exc
    terminator = "$$ LANGUAGE plpgsql;"
    try:
        end = sql_text.index(terminator, start) + len(terminator)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise AssertionError(f"could not find terminator for {name}") from exc
    return sql_text[start:end]


def _install_checkpoint_helpers(conn: psycopg.Connection) -> None:
    sql_path = ROOT / "sql" / "pg_llm--0.1.0.sql"
    sql_text = sql_path.read_text(encoding="utf-8")
    save_fn = _extract_function(sql_text, "llm_checkpoint_save")
    load_fn = _extract_function(sql_text, "llm_checkpoint_load")

    conn.execute("DROP FUNCTION IF EXISTS llm_checkpoint_save(TEXT, TEXT)")
    conn.execute("DROP FUNCTION IF EXISTS llm_checkpoint_load(TEXT, INT)")
    conn.execute(save_fn)
    conn.execute(load_fn)


def _prepare_checkpoint_schema(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS checkpoint_calls")
    conn.execute("DROP TABLE IF EXISTS llm_checkpoint CASCADE")
    conn.execute("DROP TABLE IF EXISTS llm_param_resolved")
    conn.execute("DROP FUNCTION IF EXISTS pg_llm_export_npz(TEXT, TEXT)")
    conn.execute("DROP FUNCTION IF EXISTS pg_llm_import_npz(TEXT, TEXT)")

    conn.execute(
        """
        CREATE TABLE llm_param_resolved (
            model TEXT,
            step INT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE llm_checkpoint (
            id SERIAL PRIMARY KEY,
            model TEXT,
            step INT,
            created_at TIMESTAMPTZ DEFAULT now(),
            note TEXT,
            n_params BIGINT,
            file_path TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE checkpoint_calls (
            kind TEXT,
            path TEXT,
            model TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE FUNCTION pg_llm_export_npz(path TEXT, model TEXT)
        RETURNS void
        LANGUAGE plpgsql
        AS $$
        BEGIN
            INSERT INTO checkpoint_calls(kind, path, model)
            VALUES ('export', path, model);
        END;
        $$
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE FUNCTION pg_llm_import_npz(path TEXT, model TEXT)
        RETURNS void
        LANGUAGE plpgsql
        AS $$
        BEGIN
            INSERT INTO checkpoint_calls(kind, path, model)
            VALUES ('import', path, model);
        END;
        $$
        """
    )
    _install_checkpoint_helpers(conn)


@pytest.mark.usefixtures("postgres_dsn")
class TestCheckpointHelpers:
    def test_llm_checkpoint_save_invokes_export(self, postgres_dsn: str) -> None:
        with psycopg.connect(postgres_dsn, autocommit=True) as conn:
            _prepare_checkpoint_schema(conn)
            conn.execute(
                "INSERT INTO llm_param_resolved(model, step) VALUES (%s, %s)",
                ("demo", 0),
            )
            conn.execute(
                "INSERT INTO llm_param_resolved(model, step) VALUES (%s, %s)",
                ("demo", 1),
            )
            conn.execute(
                "INSERT INTO llm_param_resolved(model, step) VALUES (%s, %s)",
                ("demo", 2),
            )

            conn.execute(
                "SELECT llm_checkpoint_save(%s, %s)",
                ("demo", "initial snapshot"),
            )

            calls = conn.execute(
                "SELECT kind, path, model FROM checkpoint_calls ORDER BY kind"
            ).fetchall()
            assert calls == [
                ("export", "/mnt/checkpoints/demo-step2.npz", "demo")
            ]

            rows = conn.execute(
                """
                SELECT model, step, n_params, file_path, note
                FROM llm_checkpoint
                ORDER BY id
                """
            ).fetchall()
            assert rows == [
                ("demo", 2, 3, "/mnt/checkpoints/demo-step2.npz", "initial snapshot")
            ]

    def test_llm_checkpoint_load_invokes_import(self, postgres_dsn: str) -> None:
        with psycopg.connect(postgres_dsn, autocommit=True) as conn:
            _prepare_checkpoint_schema(conn)
            conn.execute(
                "INSERT INTO llm_param_resolved(model, step) VALUES (%s, %s)",
                ("demo", 5),
            )
            conn.execute(
                "SELECT llm_checkpoint_save(%s, %s)",
                ("demo", "before load"),
            )
            checkpoint_id, path = conn.execute(
                "SELECT id, file_path FROM llm_checkpoint ORDER BY id DESC LIMIT 1"
            ).fetchone()

            conn.execute("DELETE FROM checkpoint_calls")
            conn.execute(
                "SELECT llm_checkpoint_load(%s, %s)",
                ("demo", checkpoint_id),
            )

            calls = conn.execute(
                "SELECT kind, path, model FROM checkpoint_calls ORDER BY kind"
            ).fetchall()
            assert calls == [("import", path, "demo")]
