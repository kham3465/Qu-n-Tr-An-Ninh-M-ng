"""File-level hold-out on Curated (never mix SolidiFI into F1)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def holdout_sources(labels: list[dict[str, Any]], *, seed: int = 0, frac: float = 0.3) -> list[str]:
    srcs = sorted({str(r.get("source") or "") for r in labels if r.get("source")})
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n = max(1, int(round(len(srcs) * frac)))
    return sorted(srcs[:n])


def write_holdout(path: Path, sources: list[str], *, seed: int, frac: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": seed, "frac": frac, "n": len(sources), "sources": sources}, indent=2),
        encoding="utf-8",
    )


def load_or_make_holdout(path: Path, labels: list[dict[str, Any]], *, seed: int = 0, frac: float = 0.3) -> list[str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            srcs = [str(s) for s in (data.get("sources") or [])]
            if srcs:
                return srcs
        except json.JSONDecodeError:
            pass
    srcs = holdout_sources(labels, seed=seed, frac=frac)
    write_holdout(path, srcs, seed=seed, frac=frac)
    return srcs
