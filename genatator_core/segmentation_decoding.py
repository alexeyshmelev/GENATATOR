from __future__ import annotations

import numpy as np


SEGMENTATION_CLASS_NAMES: tuple[str, ...] = (
    "5UTR",
    "exon",
    "intron",
    "3UTR",
    "CDS",
)
SEGMENTATION_CLASS_INDEX = {
    class_name: SEGMENTATION_CLASS_NAMES.index(class_name)
    for class_name in ("exon", "CDS")
}
SEGMENTATION_INTERVAL_COMPARISON_GROUPS = {
    "exon": tuple(
        SEGMENTATION_CLASS_NAMES.index(name)
        for name in ("exon", "intron")
    ),
    "CDS": tuple(
        SEGMENTATION_CLASS_NAMES.index(name)
        for name in ("CDS", "intron", "5UTR", "3UTR")
    ),
}


def segmentation_competition_predictions(
    scores: np.ndarray,
    class_name: str,
) -> np.ndarray:
    """Decode one multilabel track with the requested class competition set."""
    try:
        comparison_channels = SEGMENTATION_INTERVAL_COMPARISON_GROUPS[class_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported segmentation interval class: {class_name!r}"
        ) from exc
    chosen_scores = np.asarray(scores)[..., list(comparison_channels)]
    max_scores = np.max(chosen_scores, axis=-1)
    return (chosen_scores[..., 0] == max_scores).astype(np.int8)
