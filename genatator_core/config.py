from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def local_or_remote(ref: str) -> str:
    """Return a string accepted by HF APIs.

    The repository has no explicit `source` field. A value is treated as local
    when the expanded path exists; otherwise it is passed to Hugging Face as a
    repo id. This function is used for datasets, model backbones, tokenizers,
    and checkpoints.
    """
    p = Path(ref).expanduser()
    return str(p) if p.exists() else ref


def is_local(ref: str | None) -> bool:
    if not ref:
        return False
    return Path(ref).expanduser().exists()
