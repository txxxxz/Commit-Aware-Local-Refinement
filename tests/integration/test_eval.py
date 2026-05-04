"""Lightweight integration: deterministic smoke test with decoder stats."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from decoding import DecoderConfig, PlaceholderDiffusionSampler, RiskAwarePFDecoder


def test_two_samples_deterministic_and_stats():
    cfg = DecoderConfig(random_seed=42, pf_enabled=True, max_delay_steps=3)
    sampler = PlaceholderDiffusionSampler(seq_len=32, vocab_size=128, random_seed=42, deterministic=True)
    decoder = RiskAwarePFDecoder(cfg)
    prompts = ["def add(a, b):", "def mul(x, y):"]
    for prompt in prompts:
        out, stats = decoder.generate_with_stats(
            prompt,
            cfg,
            sampler,
            max_steps=10,
            token_ids_to_code=None,
        )
        assert isinstance(out, str)
        assert len(out) > 0
        assert len(out) <= 32
        assert "pf_trigger_count" in stats
        assert "extra_forwards" in stats
        assert stats["pf_trigger_count"] >= 0
        assert stats["extra_forwards"] >= 0
