"""Unit tests for PF scoring and fallback behavior."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding.config import DecoderConfig
from decoding.pf import badness_particle, parser_feedback_from_source, run_local_pf
from decoding.sampler_adapter import PlaceholderDiffusionSampler, SamplerAdapter


def token_ids_to_code(ids):
    # Force parsable code when token 1 is chosen at position 0.
    if int(ids[0]) == 1:
        return "def f():\n    return 1\n"
    return "def f( "


def token_ids_to_always_parsable_code(ids):
    return f"def f():\n    return {int(ids[0])}\n"


class LookaheadPreferenceSampler(SamplerAdapter):
    def step(self, latents, committed_mask, forced_tokens):
        del latents
        committed = np.asarray(committed_mask, dtype=bool)
        forced = np.asarray(forced_tokens, dtype=np.int64)
        seq_len = int(committed.shape[0])
        vocab_size = 8
        logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
        if seq_len > 1 and committed[0] and int(forced[0]) == 1:
            logits[1:, 0] = 8.0
        return logits


class BadSyntaxLookaheadSampler(SamplerAdapter):
    def step(self, latents, committed_mask, forced_tokens):
        del latents
        committed = np.asarray(committed_mask, dtype=bool)
        forced = np.asarray(forced_tokens, dtype=np.int64)
        seq_len = int(committed.shape[0])
        vocab_size = 8
        logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
        if seq_len > 1 and committed[0] and int(forced[0]) == 2:
            logits[1:, 0] = 8.0
        return logits


class _DummyTokenizer:
    eos_token_id = 2
    mask_token_id = 6
    all_special_ids = [2, 6, 7]


class SpecialTokenFilteringSampler(SamplerAdapter):
    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.eot_token_id = 2

    def step(self, latents, committed_mask, forced_tokens):
        del latents, committed_mask, forced_tokens
        return np.zeros((4, 8), dtype=np.float32)


def test_pf_prefers_parsable_candidate():
    cfg = DecoderConfig(
        pf_particles=3,
        pf_top_k=3,
        pf_horizon_steps=2,
        random_seed=42,
        parsing_checks_enabled=True,
    )
    sampler = PlaceholderDiffusionSampler(seq_len=8, vocab_size=16, random_seed=42, deterministic=True)

    logits = np.zeros((8, 16), dtype=np.float64)
    logits[0, 1] = 5.0  # parsable branch
    logits[0, 2] = 4.0
    logits[0, 3] = 3.0

    committed = np.zeros(8, dtype=bool)
    forced = np.zeros(8, dtype=np.int64)
    chosen, log_info = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code,
    )
    assert chosen is not None
    assert isinstance(chosen, (int, np.integer))
    assert len(log_info) <= cfg.pf_horizon_steps


def test_badness_parse_checks():
    cfg = DecoderConfig(parsing_checks_enabled=True, language="python")
    forced = np.array([1, 2, 3], dtype=np.int64)
    b1, ok1 = badness_particle(forced, cfg, lambda x: "def f(): pass")
    assert ok1 is True
    assert b1 == 0.0

    b2, ok2 = badness_particle(forced, cfg, lambda x: "def f( ")
    assert ok2 is False
    assert b2 > 0


def test_pf_force_fallback_returns_best_particle():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=2,
        pf_horizon_steps=1,
        random_seed=0,
        parsing_checks_enabled=False,
    )
    sampler = PlaceholderDiffusionSampler(seq_len=4, vocab_size=8, random_seed=0, deterministic=True)
    logits = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float64)
    committed = np.zeros(4, dtype=bool)
    forced = np.zeros(4, dtype=np.int64)

    chosen, _ = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=None,
        force_fallback=True,
    )
    assert chosen is not None


def test_pf_lookahead_bonus_can_override_local_argmax():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=2,
        pf_horizon_steps=1,
        pf_stability_weight=3.0,
        parsing_checks_enabled=False,
    )
    sampler = LookaheadPreferenceSampler()
    logits = np.zeros((4, 8), dtype=np.float64)
    logits[0, 1] = 4.0
    logits[0, 2] = 4.2  # local argmax without lookahead
    committed = np.zeros(4, dtype=bool)
    forced = np.zeros(4, dtype=np.int64)

    chosen, log_info = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=None,
        current_t=50,
        pf_window=(30, 70),
    )

    assert chosen == 1
    assert log_info[0]["particle_tokens"][0] == 1
    assert log_info[0]["particle_lookahead_bonus"][0] > 0.0


def test_pf_do_no_harm_rejects_parse_failed_replacement():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=2,
        pf_horizon_steps=1,
        pf_stability_weight=3.0,
        pf_syntax_reward=0.0,
        pf_parse_fail_penalty=0.0,
        parsing_checks_enabled=True,
        pf_do_no_harm_enabled=True,
    )
    sampler = BadSyntaxLookaheadSampler()
    logits = np.zeros((4, 8), dtype=np.float64)
    logits[0, 1] = 4.2  # original argmax, parses
    logits[0, 2] = 4.0  # lookahead-favored, syntax-bad
    committed = np.zeros(4, dtype=bool)
    forced = np.zeros(4, dtype=np.int64)

    chosen, log_info = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=token_ids_to_code,
        current_t=50,
        pf_window=(30, 70),
    )

    assert chosen is None
    assert log_info[0]["chosen_token_before_gate"] == 2
    assert log_info[0]["pf_do_no_harm_rejected"] is True
    assert log_info[0]["pf_do_no_harm_reason"] == "candidate_parse_failed"


def test_pf_do_no_harm_rejects_syntax_tied_replacement():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=2,
        pf_horizon_steps=1,
        pf_stability_weight=3.0,
        pf_syntax_reward=0.0,
        parsing_checks_enabled=True,
        pf_do_no_harm_enabled=True,
        pf_do_no_harm_min_quality_gain=0.05,
    )
    sampler = BadSyntaxLookaheadSampler()
    logits = np.zeros((4, 8), dtype=np.float64)
    logits[0, 1] = 4.2  # original argmax
    logits[0, 2] = 4.0  # lookahead-favored, but no syntax improvement
    committed = np.zeros(4, dtype=bool)
    forced = np.zeros(4, dtype=np.int64)

    chosen, log_info = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=token_ids_to_always_parsable_code,
        current_t=50,
        pf_window=(30, 70),
    )

    assert chosen is None
    assert log_info[0]["chosen_token_before_gate"] == 2
    assert log_info[0]["pf_do_no_harm_rejected"] is True
    assert log_info[0]["pf_do_no_harm_reason"] == "candidate_syntax_not_improved"


def test_pf_syntax_reward_is_active_inside_pf_window():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=2,
        pf_horizon_steps=0,
        pf_syntax_reward=5.0,
        parsing_checks_enabled=True,
        pf_time_window_mode="absolute",
        pf_time_window_start=30,
        pf_time_window_end=70,
    )
    sampler = PlaceholderDiffusionSampler(seq_len=4, vocab_size=8, random_seed=0, deterministic=True)
    logits = np.zeros((4, 8), dtype=np.float64)
    logits[0, 1] = 4.0
    logits[0, 2] = 4.2  # local argmax without syntax reward
    committed = np.zeros(4, dtype=bool)
    forced = np.zeros(4, dtype=np.int64)

    chosen_in_window, log_info_in_window = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=token_ids_to_code,
        current_t=50,
        pf_window=(30, 70),
    )
    chosen_outside_window, _ = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=token_ids_to_code,
        current_t=80,
        pf_window=(30, 70),
    )

    assert chosen_in_window == 1
    assert log_info_in_window[0]["particle_syntax_reward"][0] > 0.0
    assert chosen_outside_window == 2


def test_pf_filters_special_tokens_from_candidates():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=3,
        pf_horizon_steps=0,
        parsing_checks_enabled=False,
    )
    sampler = SpecialTokenFilteringSampler()
    logits = np.zeros((4, 8), dtype=np.float64)
    logits[0, 2] = 6.0  # eos special token, should be excluded
    logits[0, 1] = 5.0
    logits[0, 3] = 4.0
    committed = np.zeros(4, dtype=bool)
    forced = np.zeros(4, dtype=np.int64)

    chosen, log_info = run_local_pf(
        logits,
        0,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=None,
        current_t=50,
        pf_window=(30, 70),
    )

    assert chosen == 1
    assert 2 in log_info[0]["forbidden_token_ids"]
    assert 2 not in log_info[0]["particle_tokens"]


def test_pf_syntax_signal_discriminates_projected_candidates():
    cfg = DecoderConfig(
        pf_particles=2,
        pf_top_k=2,
        pf_horizon_steps=0,
        pf_syntax_reward=6.0,
        parsing_checks_enabled=True,
        pf_time_window_mode="absolute",
        pf_time_window_start=30,
        pf_time_window_end=70,
    )
    sampler = PlaceholderDiffusionSampler(seq_len=4, vocab_size=8, random_seed=0, deterministic=True)
    logits = np.zeros((4, 8), dtype=np.float64)
    logits[2, 2] = 4.2  # local argmax
    logits[2, 3] = 4.0
    committed = np.zeros(4, dtype=bool)
    committed[0] = True
    forced = np.zeros(4, dtype=np.int64)
    forced[0] = 1

    def projected_token_ids_to_code(ids):
        vals = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if vals == [1]:
            return "def f():\n    "
        if vals == [1, 2]:
            return "def f():\n    x ="
        if vals == [1, 3]:
            return "def f():\n    x"
        return "def f():\n    pass"

    chosen, log_info = run_local_pf(
        logits,
        2,
        committed,
        forced,
        cfg,
        sampler,
        None,
        token_ids_to_code=projected_token_ids_to_code,
        current_t=50,
        pf_window=(30, 70),
    )

    assert chosen == 3
    assert log_info[0]["particle_syntax_candidate_quality"][0] > log_info[0]["particle_syntax_candidate_quality"][1]


def test_parser_feedback_detects_bracket_and_indent_failures():
    bracket_feedback = parser_feedback_from_source("def f(x:\n    return x\n", min_prefix_chars=1)
    indent_feedback = parser_feedback_from_source("def f():\nreturn 1\n", min_prefix_chars=1)

    assert bracket_feedback["observed"] is True
    assert bracket_feedback["bracket_issue"] is True
    assert "bracket" in bracket_feedback["issue_types"]
    assert 0.0 <= bracket_feedback["quality_score"] < 1.0
    assert bracket_feedback["severity_score"] > 0.0

    assert indent_feedback["observed"] is True
    assert indent_feedback["indent_issue"] is True
    assert "indent" in indent_feedback["issue_types"]
    assert 0.0 <= indent_feedback["quality_score"] < 1.0
    assert indent_feedback["severity_score"] > 0.0
