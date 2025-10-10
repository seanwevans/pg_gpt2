#!/usr/bin/env python3
"""Benchmark pg_llm kernels against native PyTorch implementations.

This script connects to a PostgreSQL database that has the `pg_llm` extension
installed and compares the runtime of core tensor primitives with their
PyTorch equivalents on representative GPT-2 model dimensions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import psycopg
from psycopg import errors
import torch
import torch.nn.functional as F

MODEL_CONFIGS: List[Dict[str, int]] = [
    {"name": "gpt2-small", "hidden": 768, "heads": 12},
    {"name": "gpt2-medium", "hidden": 1024, "heads": 16},
    {"name": "gpt2-large", "hidden": 1280, "heads": 20},
]


@dataclass
class BenchmarkResult:
    model: str
    op: str
    pg_ms: float
    torch_ms: float

    @property
    def ratio(self) -> float:
        return self.pg_ms / self.torch_ms if self.torch_ms else float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="PostgreSQL DSN for benchmarking")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=[cfg["name"] for cfg in MODEL_CONFIGS],
        help="Subset of model configurations to benchmark",
    )
    parser.add_argument("--trials", type=int, default=25, help="Timed repetitions per benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup repetitions before timing")
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=64,
        help="Number of tokens per batch for matmul benchmarks",
    )
    parser.add_argument(
        "--layernorm-eps",
        type=float,
        default=1e-5,
        help="LayerNorm epsilon used by both implementations",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to emit benchmark results as JSON",
    )
    parser.add_argument(
        "--disable-jit",
        action="store_true",
        help="Disable PostgreSQL LLVM JIT during benchmarking",
    )
    return parser.parse_args()


def _binary(array: np.ndarray) -> psycopg.Binary:
    return psycopg.Binary(array.astype(np.float32, copy=False).tobytes())


def _time_torch(fn, warmup: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(trials):
        fn()
    end = time.perf_counter()
    return (end - start) / max(trials, 1)


def _time_pg(
    conn: psycopg.Connection,
    query: str,
    params: Tuple,
    warmup: int,
    trials: int,
) -> Tuple[float, memoryview]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        reference = cur.fetchone()[0]
        for _ in range(warmup):
            cur.execute(query, params)
            cur.fetchone()
        start = time.perf_counter()
        for _ in range(trials):
            cur.execute(query, params)
            cur.fetchone()
        end = time.perf_counter()
    avg = (end - start) / max(trials, 1)
    return avg, reference


def _ensure_autograd_disabled(conn: psycopg.Connection) -> None:
    try:
        conn.execute("UPDATE llm_autograd_mode SET flag = false")
    except errors.UndefinedTable:
        # Extension tables may not be present if the extension has not been
        # created. In that case the query simply isn't needed.
        pass


def _maybe_configure_jit(conn: psycopg.Connection, enable: bool) -> None:
    try:
        conn.execute(f"SET jit = {'on' if enable else 'off'}")
        conn.execute("SET jit_above_cost = 0")
    except psycopg.Error:
        # Older PostgreSQL builds might not have JIT support compiled in.
        pass


def benchmark_add(
    conn: psycopg.Connection,
    hidden: int,
    warmup: int,
    trials: int,
) -> BenchmarkResult:
    with torch.no_grad():
        lhs = torch.randn(hidden, dtype=torch.float32)
        rhs = torch.randn(hidden, dtype=torch.float32)
        torch_out = torch.empty_like(lhs)

        def torch_run() -> None:
            torch.add(lhs, rhs, out=torch_out)

    torch_ms = _time_torch(torch_run, warmup, trials) * 1000.0

    params = (
        _binary(lhs.numpy()),
        _binary(rhs.numpy()),
    )
    pg_ms, result = _time_pg(
        conn,
        "SELECT pg_llm_add(%s,%s)",
        params,
        warmup,
        trials,
    )
    pg_ms *= 1000.0

    pg_array = np.frombuffer(result, dtype=np.float32)
    if not np.allclose(pg_array, torch_out.numpy(), atol=1e-5):
        raise AssertionError("pg_llm_add result mismatch versus PyTorch")

    return BenchmarkResult("", "add", pg_ms, torch_ms)


def benchmark_layernorm(
    conn: psycopg.Connection,
    hidden: int,
    eps: float,
    warmup: int,
    trials: int,
) -> BenchmarkResult:
    with torch.no_grad():
        x = torch.randn(hidden, dtype=torch.float32)
        gamma = torch.randn(hidden, dtype=torch.float32)
        beta = torch.randn(hidden, dtype=torch.float32)

        def torch_run() -> torch.Tensor:
            return F.layer_norm(x, (hidden,), gamma, beta, eps)

    # layer_norm allocates, so cache the output for verification separately.
    torch_out = torch_run()
    torch_ms = _time_torch(lambda: torch_run(), warmup, trials) * 1000.0

    params = (
        _binary(x.numpy()),
        _binary(gamma.numpy()),
        _binary(beta.numpy()),
        eps,
    )
    pg_ms, result = _time_pg(
        conn,
        "SELECT pg_llm_layernorm(%s,%s,%s,%s)",
        params,
        warmup,
        trials,
    )
    pg_ms *= 1000.0

    pg_array = np.frombuffer(result, dtype=np.float32)
    if not np.allclose(pg_array, torch_out.numpy(), atol=3e-4):
        raise AssertionError("pg_llm_layernorm result mismatch versus PyTorch")

    return BenchmarkResult("", "layernorm", pg_ms, torch_ms)


def benchmark_matmul(
    conn: psycopg.Connection,
    batch_tokens: int,
    hidden: int,
    warmup: int,
    trials: int,
) -> BenchmarkResult:
    n = hidden
    with torch.no_grad():
        lhs = torch.randn(batch_tokens, hidden, dtype=torch.float32)
        rhs = torch.randn(hidden, n, dtype=torch.float32)
        torch_out = torch.empty(batch_tokens, n, dtype=torch.float32)

        def torch_run() -> None:
            torch.mm(lhs, rhs, out=torch_out)

    torch_ms = _time_torch(torch_run, warmup, trials) * 1000.0

    params = (
        _binary(lhs.numpy()),
        _binary(rhs.numpy()),
        batch_tokens,
        hidden,
        n,
    )
    pg_ms, result = _time_pg(
        conn,
        "SELECT pg_llm_matmul(%s,%s,%s,%s,%s)",
        params,
        warmup,
        trials,
    )
    pg_ms *= 1000.0

    pg_array = np.frombuffer(result, dtype=np.float32).reshape(batch_tokens, n)
    if not np.allclose(pg_array, torch_out.numpy(), atol=2e-4):
        raise AssertionError("pg_llm_matmul result mismatch versus PyTorch")

    return BenchmarkResult("", "matmul", pg_ms, torch_ms)


def run_benchmarks(conn: psycopg.Connection, args: argparse.Namespace) -> List[BenchmarkResult]:
    selected_names = set(args.models) if args.models else {cfg["name"] for cfg in MODEL_CONFIGS}
    results: List[BenchmarkResult] = []

    for cfg in MODEL_CONFIGS:
        if cfg["name"] not in selected_names:
            continue

        matmul_res = benchmark_matmul(conn, args.batch_tokens, cfg["hidden"], args.warmup, args.trials)
        matmul_res.model = cfg["name"]
        results.append(matmul_res)

        add_res = benchmark_add(conn, cfg["hidden"], args.warmup, args.trials)
        add_res.model = cfg["name"]
        results.append(add_res)

        layernorm_res = benchmark_layernorm(conn, cfg["hidden"], args.layernorm_eps, args.warmup, args.trials)
        layernorm_res.model = cfg["name"]
        results.append(layernorm_res)

    return results


def emit_results(results: Iterable[BenchmarkResult]) -> None:
    header = f"{'Model':<12} {'Operation':<12} {'pg_llm (ms)':>12} {'PyTorch (ms)':>12} {'Ratio':>9}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.model:<12} {res.op:<12} {res.pg_ms:12.3f} {res.torch_ms:12.3f} {res.ratio:9.2f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        _maybe_configure_jit(conn, enable=not args.disable_jit)
        _ensure_autograd_disabled(conn)
        results = run_benchmarks(conn, args)

    emit_results(results)

    if args.json:
        payload = [res.__dict__ | {"ratio": res.ratio} for res in results]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
