#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 checkpoints into pg_llm_import_npz archives.

The pg_llm_import_npz function expects a gzip-compressed stream of NumPy
arrays, each preceded by a uint16 name length and UTF-8 encoded tensor name.
This script downloads (or reads) an official GPT-2 checkpoint via
``transformers`` and writes the tensors into that container so that the
extension can import them.

Example
-------

```
python scripts/convert_gpt2_checkpoint.py \
    --source gpt2 \
    --output /mnt/models/gpt2-small.npz
```

Use ``--source`` to point at either a HuggingFace model id (requires network
access) or a local directory containing the checkpoint files.
"""

from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def _lazy_import_transformers():
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "transformers must be installed to load GPT-2 checkpoints. "
            "Install it with `pip install transformers torch`."
        ) from exc
    return AutoModelForCausalLM


def _tensor_name_map(state_dict: Dict[str, "np.ndarray"]) -> Iterable[Tuple[str, str]]:
    """Yield (state_dict_key, npz_name) pairs for tensors to export.

    Parameters that are not required for inference/training inside the
    database are skipped (e.g., causal masks from the PyTorch checkpoint).
    """

    for key in state_dict:
        if not key.startswith("transformer."):
            # Skip heads like lm_head.weight – logits layer is tied to wte
            continue

        short = key[len("transformer.") :]

        if short in {"wte.weight", "wpe.weight"}:
            yield key, short.split(".")[0]
            continue

        if short.startswith("h."):
            parts = short.split(".")
            if len(parts) < 3:
                continue
            layer = parts[1]
            remainder = ".".join(parts[2:])

            # Ignore cached biases/masks that are not trainable weights
            if remainder in {"attn.bias", "attn.masked_bias"}:
                continue

            allowed = {
                "attn.c_attn.weight",
                "attn.c_attn.bias",
                "attn.c_proj.weight",
                "attn.c_proj.bias",
                "mlp.c_fc.weight",
                "mlp.c_fc.bias",
                "mlp.c_proj.weight",
                "mlp.c_proj.bias",
                "ln_1.weight",
                "ln_1.bias",
                "ln_2.weight",
                "ln_2.bias",
            }

            if remainder in allowed:
                yield key, f"h.{layer}.{remainder}"
            continue

        if short in {"ln_f.weight", "ln_f.bias"}:
            yield key, short


def _write_tensor_entry(fp: gzip.GzipFile, name: str, array: np.ndarray) -> None:
    name_bytes = name.encode("utf-8")
    if len(name_bytes) > 0xFFFF:
        raise ValueError(f"Tensor name {name!r} exceeds 65535 bytes")

    # Ensure float32 numpy array in C order
    np_array = np.asarray(array, dtype=np.float32)
    if not np_array.flags["C_CONTIGUOUS"]:
        np_array = np.ascontiguousarray(np_array)

    buffer = io.BytesIO()
    # Use numpy.lib.format to emit a canonical .npy header/payload pair
    np.lib.format.write_array(buffer, np_array, allow_pickle=False)
    payload = buffer.getvalue()

    fp.write(len(name_bytes).to_bytes(2, "little"))
    fp.write(name_bytes)
    fp.write(payload)


def convert_checkpoint(source: str, output: Path, revision: str | None = None) -> None:
    AutoModelForCausalLM = _lazy_import_transformers()

    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        torch_dtype="float32",  # ensure tensors load on CPU as float32
    )
    state_dict = model.state_dict()

    tensors: Dict[str, np.ndarray] = {}
    for ckpt_key, export_name in _tensor_name_map(state_dict):
        tensor = state_dict[ckpt_key]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        tensors[export_name] = np.asarray(tensor, dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as gz:
        for name in sorted(tensors):
            _write_tensor_entry(gz, name, tensors[name])

    print(f"Wrote {len(tensors)} tensors to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="HuggingFace model id or local path to a GPT-2 checkpoint",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/tag when loading from HuggingFace",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .npz path for pg_llm_import_npz",
    )

    args = parser.parse_args()
    convert_checkpoint(args.source, args.output, revision=args.revision)


if __name__ == "__main__":
    main()
