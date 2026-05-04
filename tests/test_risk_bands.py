"""Tests for risk band classification and adaptive tier fallback."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding.config import BudgetTier, DecoderConfig
from decoding.risk_pf_decoder import (
    budget_tier_for_band,
    classify_risk_band,
    classify_risk_band_decoupled,
)


def test_classify_risk_band_dual_thresholds():
    cfg = DecoderConfig(risk_low_threshold=0.2, risk_high_threshold=0.7)
    assert classify_risk_band(0.1, cfg) == "low"
    assert classify_risk_band(0.5, cfg) == "mid"
    assert classify_risk_band(0.9, cfg) == "high"


def test_budget_tier_degrades_when_overhead_exceeds_cap():
    cfg = DecoderConfig(
        adaptive_budget_enabled=True,
        overhead_cap_ratio=3.0,
        budget_tier_low=BudgetTier(1, 1, 0),
        budget_tier_mid=BudgetTier(2, 2, 1),
        budget_tier_high=BudgetTier(3, 3, 1),
    )
    # Normal overhead keeps high tier.
    tier_high = budget_tier_for_band("high", overhead_ratio=2.0, cfg=cfg)
    assert tier_high.pf_top_k == 3

    # Overhead guard demotes high->mid.
    tier_demoted = budget_tier_for_band("high", overhead_ratio=5.0, cfg=cfg)
    assert tier_demoted.pf_top_k == 2


def test_classify_risk_band_decoupled_requires_influence_for_high():
    cfg = DecoderConfig(
        risk_low_threshold=0.5,
        risk_high_threshold=0.8,
        influence_trigger_floor=0.01,
    )
    assert classify_risk_band_decoupled(0.9, 0.0, False, cfg) == "mid"
    assert classify_risk_band_decoupled(0.9, 0.02, True, cfg) == "high"
