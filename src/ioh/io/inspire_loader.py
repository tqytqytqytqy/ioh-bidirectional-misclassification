from __future__ import annotations

FORBIDDEN_TARGET_DATASET_METRICS = {"true_auc", "hidden_auc", "hdr", "HDR", "episode_sensitivity"}


def guard_inspire_metric(metric: str) -> str:
    if metric in FORBIDDEN_TARGET_DATASET_METRICS or str(metric).lower() in {m.lower() for m in FORBIDDEN_TARGET_DATASET_METRICS}:
        raise ValueError(
            "This target-population dataset lacks continuous arterial waveform truth for hidden-burden estimation. "
            "Use only resource/transportability modules."
        )
    return metric
