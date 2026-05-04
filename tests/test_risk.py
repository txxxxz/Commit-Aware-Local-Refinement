"""Unit tests: entropy, attention influence proxy, and joint risk."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding.config import DecoderConfig
from decoding.risk import (
    attention_influence_proxy,
    compute_influence,
    dynamic_risk_thresholds,
    dynamic_valid_positions,
    entropy_at_position,
    normalize_influence_scores,
    risk_score,
)


def test_entropy_onehot():
    logits = np.zeros(10)
    logits[0] = 100.0
    ent = entropy_at_position(logits.reshape(1, -1), 0)
    assert ent == pytest.approx(0.0, abs=1e-5)


def test_entropy_uniform():
    logits = np.ones(10)
    p = 1.0 / 10
    expected = -10 * (p * np.log(p))
    ent = entropy_at_position(logits.reshape(1, -1), 0)
    assert ent == pytest.approx(expected, abs=1e-5)


def test_entropy_known():
    p = np.array([0.5, 0.3, 0.2])
    expected = -np.sum(p * np.log(p))
    logits = np.log(p + 1e-12)
    ent = entropy_at_position(logits.reshape(1, -1), 0)
    assert ent == pytest.approx(expected, abs=1e-4)


def test_dynamic_valid_positions_filters_committed_confident_and_eot():
    logits = np.array(
        [
            [10.0, -10.0, -10.0],  # top1 prob ~1.0, should be filtered
            [-1.0, 0.0, 2.0],  # argmax=2 (EOT), should be filtered
            [0.1, 0.1, 0.1],  # committed, should be filtered
            [0.2, 0.3, 0.1],  # valid
        ],
        dtype=np.float64,
    )
    committed = np.array([False, False, True, False], dtype=bool)
    valid, meta = dynamic_valid_positions(
        logits=logits,
        candidate_positions=[0, 1, 2, 3],
        committed_mask=committed,
        max_prob_threshold=0.99,
        eot_token_id=2,
        exclude_eot=True,
    )
    assert valid == [3]
    assert 3 in meta


def test_attention_influence_proxy_uses_committed_targets_only():
    attention = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.1, 0.6],
            [0.4, 0.4, 0.2],
        ],
        dtype=np.float64,
    )
    committed = np.array([True, False, True], dtype=bool)
    scores = attention_influence_proxy(attention, committed, [0, 1, 2])
    # For i=1: A[1,0] + A[1,2] = 0.3 + 0.6
    assert scores[1] == pytest.approx(0.9, abs=1e-6)


def test_normalize_influence_scores_minmax():
    raw = {1: 2.0, 2: 4.0, 3: 6.0}
    normed = normalize_influence_scores(raw)
    assert normed[1] == pytest.approx(0.0, abs=1e-6)
    assert normed[2] == pytest.approx(0.5, abs=1e-6)
    assert normed[3] == pytest.approx(1.0, abs=1e-6)


def test_dynamic_risk_thresholds_quantiles():
    tau_low, tau_high = dynamic_risk_thresholds([0.1, 0.2, 0.4, 0.8], low_q=0.25, high_q=0.75)
    assert tau_low <= tau_high
    assert tau_low == pytest.approx(0.175, abs=1e-6)
    assert tau_high == pytest.approx(0.5, abs=1e-6)


def test_risk_score_joint_gate_with_beta():
    cfg = DecoderConfig(risk_beta=2.0)
    r = risk_score(entropy=1.5, influence=0.5, cfg=cfg, running_stats=None)
    assert r == pytest.approx(1.5 * (1.0 + 2.0 * 0.5), abs=1e-6)


def test_legacy_compute_influence_shim_is_stable_and_non_negative():
    cfg = DecoderConfig()
    rng = np.random.default_rng(123)
    logits = rng.standard_normal((8, 32)).astype(np.float64)
    committed = np.zeros(8, dtype=bool)
    forced = np.zeros(8, dtype=np.int64)

    def mock_step(latents, cm, ft):
        del cm, ft
        return latents

    inf1, _ = compute_influence(logits, 2, 0.1, mock_step, committed, forced, cfg)
    inf2, _ = compute_influence(logits, 2, 0.1, mock_step, committed, forced, cfg)
    assert inf1 >= 0
    assert inf2 >= 0
    assert inf1 == pytest.approx(inf2, rel=1e-9, abs=1e-12)
