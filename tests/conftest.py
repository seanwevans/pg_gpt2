import os
import pwd
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import psycopg
import pytest


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    def _find_binary(name: str) -> str | None:
        found = shutil.which(name)
        if found:
            return found
        pg_root = Path("/usr/lib/postgresql")
        if pg_root.exists():
            for candidate in pg_root.glob(f"*/bin/{name}"):
                if candidate.exists():
                    return str(candidate)
        return None

    initdb_path = _find_binary("initdb")
    pg_ctl_path = _find_binary("pg_ctl")
    if not initdb_path or not pg_ctl_path:
        pytest.skip("PostgreSQL binaries are not available in the environment")

    data_dir = Path(tempfile.mkdtemp(prefix="pg_gpt2_pgdata_"))
    pg_user = pwd.getpwnam("postgres")
    os.chown(str(data_dir), pg_user.pw_uid, pg_user.pw_gid)

    runuser_path = shutil.which("runuser")
    if not runuser_path:
        pytest.skip("runuser is required to start PostgreSQL as the postgres user")

    subprocess.run(
        [
            runuser_path,
            "-u",
            "postgres",
            "--",
            initdb_path,
            "-D",
            str(data_dir),
            "-A",
            "trust",
            "--username",
            "postgres",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    pg_hba_path = Path(data_dir) / "pg_hba.conf"
    with pg_hba_path.open("a", encoding="utf-8") as handle:
        handle.write("host    all             all             127.0.0.1/32            trust\n")
    os.chown(str(pg_hba_path), pg_user.pw_uid, pg_user.pw_gid)

    port = _find_free_port()
    log_file = Path(data_dir) / "postgresql.log"

    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    env["PGUSER"] = "postgres"

    subprocess.run(
        [
            runuser_path,
            "-u",
            "postgres",
            "--",
            pg_ctl_path,
            "-D",
            str(data_dir),
            "-l",
            str(log_file),
            "-o",
            f"-F -p {port} -h 127.0.0.1",
            "-w",
            "start",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    dsn = f"postgresql://postgres@127.0.0.1:{port}/postgres"

    try:
        with psycopg.connect(dsn) as conn:
            conn.execute("SELECT 1")
        yield dsn
    finally:
        subprocess.run(
            [
                runuser_path,
                "-u",
                "postgres",
                "--",
                pg_ctl_path,
                "-D",
                str(data_dir),
                "-m",
                "fast",
                "-w",
                "stop",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        shutil.rmtree(data_dir, ignore_errors=True)
