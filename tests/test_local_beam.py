"""Tests for commit-timing-aware local beam utilities."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding.config import DecoderConfig
from decoding.local_beam import (
    LocalBeamRiskCandidate,
    compute_local_beam_risks,
    run_commit_timing_local_beam,
    select_local_beam_trigger,
)
from decoding.sampler_adapter import SamplerAdapter


class StaticSampler(SamplerAdapter):
    def __init__(self, seq_len=3, vocab_size=5):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def step(self, latents, committed_mask, forced_tokens):
        del latents, committed_mask, forced_tokens
        return np.zeros((self.seq_len, self.vocab_size), dtype=np.float32)


def token_ids_to_code(ids):
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, int):
        ids = [ids]
    token_text = {0: "", 1: "value", 2: ":", 3: "return", 4: "("}
    return "".join(token_text.get(int(tok), "x") for tok in ids)


def test_temporal_kl_ranks_changed_position():
    cfg = DecoderConfig(local_beam_struct_weight=0.5)
    prev = np.zeros((3, 5), dtype=np.float32)
    cur = np.zeros((3, 5), dtype=np.float32)
    prev[0, 1] = 6.0
    cur[0, 1] = 6.0
    prev[1, 1] = 6.0
    cur[1, 2] = 6.0

    risks = compute_local_beam_risks(
        logits=cur,
        prev_logits=prev,
        candidate_positions=[0, 1],
        committed_mask=np.zeros(3, dtype=bool),
        cfg=cfg,
        token_ids_to_code=token_ids_to_code,
        sampler=StaticSampler(),
    )
    by_pos = {risk.pos: risk for risk in risks}

    assert by_pos[1].kl_raw > by_pos[0].kl_raw
    assert by_pos[1].kl_norm > by_pos[0].kl_norm
    assert by_pos[1].structure_score > 0.0


def test_select_local_beam_trigger_is_conservative():
    cfg = DecoderConfig(
        local_beam_tau_entropy=0.4,
        local_beam_tau_kl=0.7,
        local_beam_tau_risk=0.35,
    )
    weak = LocalBeamRiskCandidate(
        pos=0,
        entropy_norm=0.8,
        kl_raw=0.1,
        kl_norm=0.2,
        structure_score=1.0,
        structure_weight=1.5,
        risk=0.24,
    )
    strong = LocalBeamRiskCandidate(
        pos=1,
        entropy_norm=0.7,
        kl_raw=2.0,
        kl_norm=0.9,
        structure_score=1.0,
        structure_weight=1.5,
        risk=0.945,
    )

    assert select_local_beam_trigger([weak], cfg=cfg, branch_events=0) is None
    assert select_local_beam_trigger([strong], cfg=cfg, branch_events=0).pos == 1
    assert select_local_beam_trigger([strong], cfg=cfg, branch_events=1) is None


def test_local_beam_keeps_baseline_without_clear_margin():
    cfg = DecoderConfig(
        local_beam_size=3,
        local_beam_horizon=0,
        local_beam_margin_base=10.0,
        local_beam_margin_branch=0.0,
        parsing_checks_enabled=False,
    )
    logits = np.zeros((3, 5), dtype=np.float32)
    logits[0, 1] = 4.0
    logits[0, 2] = 3.9
    risk = LocalBeamRiskCandidate(
        pos=0,
        entropy_norm=0.9,
        kl_raw=1.0,
        kl_norm=1.0,
        structure_score=1.0,
        structure_weight=1.5,
        risk=1.0,
    )

    result = run_commit_timing_local_beam(
        logits=logits,
        prev_logits=logits,
        risk=risk,
        committed_mask=np.zeros(3, dtype=bool),
        forced_tokens=np.zeros(3, dtype=np.int64),
        cfg=cfg,
        sampler=StaticSampler(),
        latents=None,
        token_ids_to_code=None,
    )

    assert result.selected_kind == "baseline"
    assert result.baseline_preserved is True
    assert result.accepted_alternative is False


def test_local_beam_delay_only_selects_delay_particle():
    cfg = DecoderConfig(
        local_beam_mode="delay_only",
        local_beam_size=3,
        local_beam_horizon=0,
        parsing_checks_enabled=False,
    )
    logits = np.zeros((3, 5), dtype=np.float32)
    risk = LocalBeamRiskCandidate(
        pos=0,
        entropy_norm=0.9,
        kl_raw=1.0,
        kl_norm=1.0,
        structure_score=1.0,
        structure_weight=1.5,
        risk=1.0,
    )

    result = run_commit_timing_local_beam(
        logits=logits,
        prev_logits=logits,
        risk=risk,
        committed_mask=np.zeros(3, dtype=bool),
        forced_tokens=np.zeros(3, dtype=np.int64),
        cfg=cfg,
        sampler=StaticSampler(),
        latents=None,
        token_ids_to_code=None,
    )

    assert result.selected_kind == "delay"
    assert result.selected_token is None
    assert result.delay_selected is True
