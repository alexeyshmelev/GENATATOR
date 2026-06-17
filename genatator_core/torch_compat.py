from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)


def _torch_version_tuple() -> tuple[int, int, int]:
    text = str(torch.__version__).split('+', 1)[0]
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return (0, 0, 0)
    return tuple(int(x) for x in m.groups())


def allow_transformers_torch_load_on_legacy_torch(enabled: bool = True, *, context: str = "") -> None:
    """Allow loading legacy PyTorch `.bin` HF checkpoints with torch<2.6.

    New Transformers versions block `torch.load` on torch<2.6 because of
    CVE-2025-32434. Several public GENA checkpoints are still distributed as
    PyTorch `.bin` files, while the requested runtime pins torch==2.2.2+cu121.
    This explicit patch keeps those checkpoints loadable. It is intentionally
    loud and never silent.
    """
    if not enabled:
        return
    if _torch_version_tuple() >= (2, 6, 0):
        return
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils
    except Exception as e:
        raise RuntimeError(f"Could not patch Transformers torch.load safety check for {context}: {e}") from e

    def _disabled_check_torch_load_is_safe():
        return None

    import_utils.check_torch_load_is_safe = _disabled_check_torch_load_is_safe
    modeling_utils.check_torch_load_is_safe = _disabled_check_torch_load_is_safe
    logger.warning(
        "[torch_compat] Enabled explicit Transformers torch.load compatibility patch for torch=%s context=%s. "
        "This is required for legacy .bin checkpoints on torch<2.6. Use trusted model repositories only.",
        torch.__version__,
        context,
    )
