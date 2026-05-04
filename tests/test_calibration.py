"""Tests for warmup calibration helpers and dual-threshold config updates."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding.calibration import apply_warmup_calibration, quantile_pair
from decoding.config import DecoderConfig


def test_quantile_pair_handles_empty():
    low, high = quantile_pair([], 0.4, 0.75)
    assert low == 0.0
    assert high == pytest.approx(1.0)


def test_apply_warmup_calibration_updates_cfg_ranges():
    cfg = DecoderConfig()
    payload = apply_warmup_calibration(
        cfg=cfg,
        risk_values=[0.1, 0.2, 0.4, 0.8, 0.9],
        entropy_values=[0.3, 0.4, 0.6, 1.2],
        influence_values=[0.0, 0.05, 0.1, 0.3],
        low_q=0.4,
        high_q=0.75,
    )
    assert cfg.risk_low_threshold is not None
    assert cfg.risk_high_threshold is not None
    assert cfg.risk_low_threshold <= cfg.risk_high_threshold
    assert cfg.entropy_norm_range is not None
    assert cfg.influence_norm_range is not None
    assert payload["risk_low_threshold"] == pytest.approx(cfg.risk_low_threshold)
    assert payload["risk_high_threshold"] == pytest.approx(cfg.risk_high_threshold)


def test_apply_warmup_calibration_enforces_threshold_gap():
    cfg = DecoderConfig(risk_threshold_min_gap=0.05)
    payload = apply_warmup_calibration(
        cfg=cfg,
        risk_values=[0.2, 0.2, 0.2, 0.2],
        entropy_values=[0.4, 0.4, 0.4],
        influence_values=[0.0, 0.0, 0.0],
        low_q=0.75,
        high_q=0.9,
    )
    assert (payload["risk_high_threshold"] - payload["risk_low_threshold"]) >= 0.05 - 1e-9
