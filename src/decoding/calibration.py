"""Warmup calibration helpers for risk thresholds and normalization ranges."""
from __future__ import annotations

import json
from typing import Iterable, Tuple

import numpy as np

from .config import DecoderConfig


def _to_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def quantile_pair(
    values: Iterable[float],
    low_q: float,
    high_q: float,
    min_gap: float = 1e-6,
) -> Tuple[float, float]:
    arr = _to_array(values)
    if arr.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(arr, low_q))
    high = float(np.quantile(arr, high_q))
    min_gap = max(float(min_gap), 1e-6)
    if high <= low + min_gap:
        high = low + min_gap
    return low, high


def apply_warmup_calibration(
    cfg: DecoderConfig,
    risk_values: Iterable[float],
    entropy_values: Iterable[float],
    influence_values: Iterable[float],
    low_q: float | None = None,
    high_q: float | None = None,
) -> dict:
    """
    Update cfg in-place using warmup quantiles and return a serializable payload.
    """
    low_q = cfg.risk_low_quantile if low_q is None else low_q
    high_q = cfg.risk_high_quantile if high_q is None else high_q
    low_q = float(np.clip(low_q, 0.0, 1.0))
    high_q = float(np.clip(high_q, 0.0, 1.0))
    if low_q > high_q:
        low_q, high_q = high_q, low_q

    ent_arr = _to_array(entropy_values)
    inf_arr = _to_array(influence_values)
    ent_low, ent_high = quantile_pair(ent_arr, 0.1, 0.9, min_gap=1e-3)
    inf_low, inf_high = quantile_pair(inf_arr, 0.1, 0.9, min_gap=1e-6)

    risk_source = _to_array(risk_values)
    if cfg.risk_fusion_mode in ("independent_triggers", "entropy_only"):
        # For independent-trigger fusion, use entropy-normalized risk for robust thresholds.
        if ent_arr.size > 0 and ent_high > ent_low:
            risk_source = np.clip((ent_arr - ent_low) / (ent_high - ent_low + 1e-8), 0.0, 1.0)
        else:
            risk_source = np.array([0.0, 1.0], dtype=np.float64)
    risk_low, risk_high = quantile_pair(
        risk_source,
        low_q,
        high_q,
        min_gap=cfg.risk_threshold_min_gap,
    )
    if cfg.risk_fusion_mode in ("independent_triggers", "entropy_only"):
        risk_low = float(np.clip(risk_low, 0.0, 1.0))
        risk_high = float(np.clip(risk_high, 0.0, 1.0))
        if risk_high <= risk_low:
            risk_high = float(min(1.0, risk_low + cfg.risk_threshold_min_gap))
            if risk_high <= risk_low:
                risk_low = float(max(0.0, risk_high - cfg.risk_threshold_min_gap))

    cfg.risk_low_threshold = risk_low
    cfg.risk_high_threshold = risk_high
    cfg.entropy_norm_range = (ent_low, ent_high)
    cfg.influence_norm_range = (inf_low, inf_high)

    return {
        "risk_low_threshold": risk_low,
        "risk_high_threshold": risk_high,
        "risk_low_quantile": low_q,
        "risk_high_quantile": high_q,
        "entropy_norm_range": [ent_low, ent_high],
        "influence_norm_range": [inf_low, inf_high],
        "cfg": {
            "risk_threshold": cfg.risk_threshold,
            "risk_weights": list(cfg.risk_weights),
            "entropy_windowing": list(cfg.entropy_windowing),
        },
    }


def save_calibration(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_calibration(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
