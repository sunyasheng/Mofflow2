import argparse
import json
import os
from typing import Any, Dict

import torch


def is_tensorlike(x: Any) -> bool:
    try:
        return hasattr(x, "shape") and hasattr(x, "dtype")
    except Exception:
        return False


def summarize_tensor(name: str, t: torch.Tensor) -> None:
    print(f"- {name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    with torch.no_grad():
        flat = t.float().reshape(-1)
        if flat.numel() > 0:
            print(
                f"  stats -> min={flat.min().item():.4g}, max={flat.max().item():.4g}, mean={flat.mean().item():.4g}"
            )


def try_sequence_preview(obj: Any) -> bool:
    """Return True if looks like a list of sequences and shows a brief preview."""
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], str):
        print(f"Detected list of sequences: count={len(obj)}")
        preview = obj[:5]
        for i, s in enumerate(preview):
            print(f"  [{i}] {s[:120]}{'...' if len(s) > 120 else ''}")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect predictions_*.pt content")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to predictions_*.pt (e.g., /ibex/.../inference/predictions_0.pt)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(args.path)

    print(f"Loading: {args.path}")
    data = torch.load(args.path, map_location="cpu")

    # If it's a dict of tensors typical for FlowModule predict.py
    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))

        # Common keys for 3D predictions
        for key in [
            "cart_coords",
            "num_atoms",
            "atom_types",
            "lattices",
            "gt_data",
            "gt_data_batch",
        ]:
            if key in data:
                val = data[key]
                if is_tensorlike(val):
                    summarize_tensor(key, val)
                else:
                    print(f"- {key}: type={type(val)}")

        # Quick geometry sanity checks if coordinates are present
        if "cart_coords" in data and is_tensorlike(data["cart_coords"]):
            coords = data["cart_coords"]
            # If shaped [k, total_atoms, 3], take first k
            if coords.ndim == 3 and coords.shape[-1] == 3:
                first = coords[0]
                d = torch.cdist(first, first)
                min_nonzero = d[d > 0].min().item() if (d.numel() and (d > 0).any()) else float("nan")
                print(f"Min inter-atom distance (first sample): {min_nonzero:.4g} Å")

        # Some runs may store decoded sequences under a key; try to detect
        for key in ["sequences", "decoded", "all_sequences", "seqs", "preds"]:
            if key in data and try_sequence_preview(data[key]):
                break

    else:
        # Not a dict — could be sequences serialized directly
        if not try_sequence_preview(data):
            print(f"Unknown data type: {type(data)}. Showing repr() preview:")
            print(repr(data)[:500])


if __name__ == "__main__":
    main()


