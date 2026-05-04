"""Unit tests for time-aware routing behavior."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding import DecoderConfig, RiskAwarePFDecoder


class UniformLogitSampler:
    def __init__(self, seq_len: int = 8, vocab_size: int = 16):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def step(self, latents, committed_mask, forced_tokens):
        del latents, committed_mask, forced_tokens
        return np.zeros((self.seq_len, self.vocab_size), dtype=np.float64)


class MixedRiskSampler:
    """Position 0 is high-entropy; others are highly confident."""

    def __init__(self, seq_len: int = 8, vocab_size: int = 16):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def step(self, latents, committed_mask, forced_tokens):
        del latents
        logits = np.zeros((self.seq_len, self.vocab_size), dtype=np.float64)
        # One medium-entropy position to keep quantile spread among valid tokens.
        logits[1, :] = 0.0
        logits[1, 0] = 1.0
        for i in range(2, self.seq_len):
            logits[i, :] = -8.0
            logits[i, 0] = 8.0
        if committed_mask is not None and forced_tokens is not None:
            for i in range(self.seq_len):
                if bool(committed_mask[i]):
                    tok = int(forced_tokens[i])
                    logits[i, :] = -10.0
                    logits[i, tok] = 10.0
        return logits


def test_early_stage_keeps_uncertain_tokens_masked():
    cfg = DecoderConfig(
        pf_enabled=False,
        diffusion_mid_step=1,
        random_seed=0,
    )
    sampler = UniformLogitSampler(seq_len=8, vocab_size=16)
    decoder = RiskAwarePFDecoder(cfg, sampler=sampler)

    _, stats = decoder.generate_with_stats(prompt="x", sampler=sampler, max_steps=2)
    step0 = stats["step_logs"][0]
    assert step0["t"] == 2
    assert step0["action_counts"]["commit_argmax"] == 0
    assert step0["action_counts"]["freeze_delay"] == 8


def test_low_risk_positions_freeze_with_argmax():
    cfg = DecoderConfig(
        pf_enabled=False,
        diffusion_mid_step=3,
        random_seed=0,
    )
    sampler = MixedRiskSampler(seq_len=8, vocab_size=16)
    decoder = RiskAwarePFDecoder(cfg, sampler=sampler)

    _, stats = decoder.generate_with_stats(prompt="x", sampler=sampler, max_steps=1)
    step0 = stats["step_logs"][0]
    assert step0["action_counts"]["commit_argmax"] >= 1
    assert step0["low_risk_mask_size"] >= 1


def test_late_high_risk_triggers_local_pf():
    cfg = DecoderConfig(
        pf_enabled=True,
        pf_top_k=2,
        pf_particles=2,
        pf_horizon_steps=1,
        diffusion_mid_step=3,
        random_seed=0,
    )
    sampler = MixedRiskSampler(seq_len=8, vocab_size=16)
    decoder = RiskAwarePFDecoder(cfg, sampler=sampler)

    _, stats = decoder.generate_with_stats(prompt="x", sampler=sampler, max_steps=1)
    step0 = stats["step_logs"][0]
    assert stats["pf_trigger_count"] >= 1
    assert step0["action_counts"]["commit_pf"] >= 1
