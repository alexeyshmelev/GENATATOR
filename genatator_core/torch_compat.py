from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _torch_version_tuple() -> tuple[int, int, int]:
    text = str(torch.__version__).split("+", 1)[0]
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        return (0, 0, 0)
    return tuple(int(value) for value in match.groups())


def allow_transformers_torch_load_on_legacy_torch(
    enabled: bool = True,
    *,
    context: str = "",
) -> None:
    """Permit trusted ``pytorch_model.bin`` loading with torch < 2.6.

    Recent Transformers releases call ``check_torch_load_is_safe`` from more than
    one module. In particular, ``Trainer._load_best_model`` keeps a module-level
    reference in ``transformers.trainer``. Patching only ``modeling_utils`` is not
    enough and causes training to fail after the last epoch when
    ``load_best_model_at_end=True``.

    This function patches every Transformers location used by model loading,
    checkpoint resume, and best-checkpoint restoration. The compatibility mode is
    explicit, logged, and intended only for trusted local/Hugging Face model
    repositories when the runtime is pinned to torch 2.2.2+cu121.
    """
    if not enabled or _torch_version_tuple() >= (2, 6, 0):
        return

    def _disabled_check_torch_load_is_safe() -> None:
        return None

    module_names = (
        "transformers.utils.import_utils",
        "transformers.modeling_utils",
        "transformers.trainer",
    )
    patched: list[str] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not import {module_name!r} while enabling legacy torch checkpoint "
                f"compatibility for {context!r}: {exc}"
            ) from exc
        if not hasattr(module, "check_torch_load_is_safe"):
            raise RuntimeError(
                f"Installed Transformers module {module_name!r} does not expose "
                "check_torch_load_is_safe; refusing to apply an incomplete compatibility patch."
            )
        setattr(module, "check_torch_load_is_safe", _disabled_check_torch_load_is_safe)
        patched.append(module_name)

    logger.warning(
        "[torch_compat] Enabled explicit trusted-checkpoint torch.load compatibility "
        "for torch=%s context=%s patched_modules=%s. This bypass is required for "
        "legacy .bin checkpoints on torch<2.6; use trusted checkpoints only.",
        torch.__version__,
        context,
        patched,
    )


def trusted_torch_load(path: str | Path, *, map_location: str = "cpu") -> Any:
    """Load a trusted tensor checkpoint using the safest API available in torch 2.2.

    ``weights_only=True`` is requested explicitly. A clear exception is propagated
    if the checkpoint is not a plain tensor state dictionary.
    """
    checkpoint_path = Path(path).expanduser()
    logger.warning(
        "[torch_compat] Loading trusted PyTorch checkpoint with torch=%s path=%s "
        "weights_only=True",
        torch.__version__,
        checkpoint_path,
    )
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
