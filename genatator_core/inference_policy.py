from __future__ import annotations

import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


def is_gpt_segmentation(model_cfg: Dict[str, Any], task: str) -> bool:
    return task == "segmentation" and model_cfg.get("family") == "gpt"


def inference_uses_reverse_complement(cfg: Dict[str, Any], task: str) -> bool:
    """Resolve reverse-complement averaging without allowing it for GPT heads."""

    requested = bool(cfg.get("inference", {}).get("use_reverse_complement", True))
    if is_gpt_segmentation(cfg.get("model", {}), task):
        if requested:
            logger.info(
                "[infer.policy] ignoring use_reverse_complement=true for the GPT "
                "segmentation family; GPT inference is forward-only"
            )
        return False
    return requested


def segmentation_uses_cds_heuristic(cfg: Dict[str, Any]) -> bool:
    """Resolve CDS post-processing, which is mandatory for exon-only GPT heads."""

    requested = bool(cfg.get("inference", {}).get("use_cds_heuristic", True))
    if cfg.get("model", {}).get("family") == "gpt":
        if not requested:
            logger.warning(
                "[infer.policy] ignoring use_cds_heuristic=false for the GPT "
                "segmentation family; GPT does not predict a CDS track"
            )
        return True
    return requested
