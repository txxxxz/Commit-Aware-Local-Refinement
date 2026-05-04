"""Tests for run_eval result directory and output path helpers."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.run_eval import (
    _DEFAULT_PF_EXTRA_FORWARD_BUDGET,
    _DreamBranchState,
    _advance_dream_branch_trials,
    _apply_completion_token_overrides,
    _build_branch_observe_rollout_specs,
    _build_joint_branch_candidates,
    _constraint_identifier_signal,
    _extract_identifier_counts,
    _first_text_diff,
    _first_token_diff,
    _bump_parser_issue_hist,
    _bump_parser_score_hist,
    _build_dream_noop_stats,
    _build_eb_transfer_allowed_mask,
    _compose_branch_completion_state,
    _dream_apply_sampling_step,
    _parser_gradient_metrics,
    _parser_feedback_after_forced_token,
    _parser_hotspot_score,
    _resolve_branch_observe_trigger_mode,
    _select_branch_observe_candidate_indices,
    _shadow_trigger_score_t,
    _level0_selection_score,
    _level1_branch_candidate_decision,
    _obvious_truncation_for_eval,
    _visible_mbpp_tests,
    _visible_test_score,
    parse_success,
    parse_success_for_eval,
    postprocess_completion_for_eval,
    _prune_dream_branch_bank,
    _prepare_result_dirs,
    _resolve_pf_persistence_steps,
    _resolve_result_root,
    _select_best_dream_branch_entry,
    _select_pf_candidate_positions,
    _resolve_budgeted_entropy_pf_controls,
    _resolve_dream_repair_policy,
    _is_dream_noop_config,
    _repair_policy_uses_baseline_sampling,
    _route_rollback_only_action,
    _route_pf_rb_action,
    _select_rdd_rollback_positions,
    _apply_completion_remask,
    _selected_pf_particle_parser_delta,
    _should_accept_budgeted_pf_choice,
    _structure_sensitive_token_score,
    _should_extend_dream_branches,
    _summarize_branch_select_records,
    _target_function_present_for_eval,
    _top_parser_issue_timesteps,
    _top_parser_score_timesteps,
    unit_test_pass,
)
from decoding.config import DecoderConfig
from decoding.pf import ParticleState


def test_resolve_result_root_priority(tmp_path):
    project_root = str(tmp_path / "project")
    result_root = str(tmp_path / "explicit_results")
    log_dir = str(tmp_path / "legacy_log_dir")

    assert _resolve_result_root(project_root, result_root, log_dir) == str(Path(result_root).resolve())
    assert _resolve_result_root(project_root, "", log_dir) == str(Path(log_dir).resolve())
    assert _resolve_result_root(project_root, "", "") == str(Path(project_root, "results_remote").resolve())


def test_prepare_result_dirs_creates_expected_layout(tmp_path):
    paths = _prepare_result_dirs(str(tmp_path), timestamp="20260304_101530")

    run_dir = Path(paths["run_dir"])
    assert run_dir.exists()
    assert run_dir.name == "20260304_101530"
    assert (run_dir / "rawdata").is_dir()
    assert (run_dir / "json").is_dir()
    assert paths["timestamp"] == "20260304_101530"


def test_prepare_result_dirs_handles_timestamp_collision(tmp_path):
    first = _prepare_result_dirs(str(tmp_path), timestamp="20260304_101530")
    second = _prepare_result_dirs(str(tmp_path), timestamp="20260304_101530")

    assert Path(first["run_dir"]).name == "20260304_101530"
    assert Path(second["run_dir"]).name == "20260304_101530_01"


def test_dream_noop_config_requires_all_interventions_disabled():
    cfg = DecoderConfig(
        pf_enabled=False,
        delay_commit_enabled=False,
        influence_enabled=False,
        rdd_rollback_enabled=False,
        local_beam_enabled=False,
        eb_sampler_enabled=False,
    )

    assert _is_dream_noop_config(cfg) is True

    cfg.local_beam_enabled = True
    assert _is_dream_noop_config(cfg) is False

    cfg.local_beam_enabled = False
    cfg.shadow_mode_enabled = True
    assert _is_dream_noop_config(cfg) is False

    cfg.shadow_mode_enabled = False
    cfg.branch_observe_enabled = True
    assert _is_dream_noop_config(cfg) is False

    cfg.branch_observe_enabled = False
    cfg.branch_select_enabled = True
    assert _is_dream_noop_config(cfg) is False


def test_dream_noop_stats_are_zero_intervention():
    cfg = DecoderConfig(
        pf_enabled=False,
        delay_commit_enabled=False,
        influence_enabled=False,
    )
    stats = _build_dream_noop_stats(diffusion_steps=4, cfg=cfg)

    assert stats["dream_noop_fast_path"] is True
    assert stats["extra_forwards"] == 0
    assert stats["latency_ratio_vs_baseline"] == 1.0
    assert stats["branch_width_trace"] == [1, 1, 1, 1]
    assert stats["local_beam_branch_events"] == 0
    assert stats["branch_observe_branch_events"] == 0


def test_level0_selection_score_prefers_parseable_nontruncated_code():
    good = "def f():\n    return 1\n"
    bad = "def f():\n    return 1 +"

    assert _obvious_truncation_for_eval(bad) is True
    assert _obvious_truncation_for_eval(good) is False
    assert _level0_selection_score(True, True, good) > _level0_selection_score(True, False, bad)


def test_visible_mbpp_score_counts_explicit_public_asserts():
    item = {
        "prompt": "Write a python function to add one.",
        "public_tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
    }

    score = _visible_test_score(
        item=item,
        prompt=item["prompt"],
        completion="def add_one(x):\n    return x + 1\n",
        dataset="mbpp",
        timeout=1,
    )

    assert score["enabled"] is True
    assert score["passed"] == 2
    assert score["total"] == 2
    assert score["all_passed"] is True


def test_visible_mbpp_score_uses_prompt_asserts_but_not_oracle_test_list():
    hidden_only = {
        "prompt": "Write a python function to add one.",
        "test_list": ["assert add_one(1) == 2"],
    }
    prompt_visible = {
        "prompt": "Write a python function to add one.\nassert add_one(1) == 2",
        "test_list": ["assert add_one(2) == 3"],
    }

    assert _visible_mbpp_tests(hidden_only) == []
    assert _visible_mbpp_tests(prompt_visible) == ["assert add_one(1) == 2"]

    score = _visible_test_score(
        item=hidden_only,
        prompt=hidden_only["prompt"],
        completion="def add_one(x):\n    return x + 1\n",
        dataset="mbpp",
        timeout=1,
    )
    assert score["enabled"] is False
    assert score["error"] == "missing_visible_tests"


def test_level1_selector_routes_level0_and_visible_repairs_separately():
    no_visible = {"enabled": False, "passed": 0, "total": 0, "all_passed": False}
    visible_fail = {"enabled": True, "passed": 0, "total": 2, "all_passed": False, "pass_rate": 0.0}
    visible_pass = {"enabled": True, "passed": 2, "total": 2, "all_passed": True, "pass_rate": 1.0}
    cand_visible_better = {"enabled": True, "passed": 1, "total": 2, "all_passed": False, "pass_rate": 0.5}

    level0_repair = _level1_branch_candidate_decision(
        baseline_level0_pass=False,
        baseline_level0_score=1.25,
        baseline_visible=no_visible,
        cand_format_ok=True,
        cand_parse_ok=True,
        cand_obvious_truncation=False,
        cand_level0_score=3.25,
        cand_visible=no_visible,
        model_score=0.0,
        min_score_gain=1.0,
        visible_min_pass_gain=1,
        visible_require_level0=True,
    )
    assert level0_repair["eligible"] is True
    assert level0_repair["repair_mode"] == "level0"
    assert level0_repair["level0_gain_ok"] is True
    assert level0_repair["visible_gain_ok"] is False

    visible_repair = _level1_branch_candidate_decision(
        baseline_level0_pass=True,
        baseline_level0_score=3.25,
        baseline_visible=visible_fail,
        cand_format_ok=True,
        cand_parse_ok=True,
        cand_obvious_truncation=False,
        cand_level0_score=3.25,
        cand_visible=cand_visible_better,
        model_score=0.0,
        min_score_gain=1.0,
        visible_min_pass_gain=1,
        visible_require_level0=True,
    )
    assert visible_repair["eligible"] is True
    assert visible_repair["repair_mode"] == "level1"
    assert visible_repair["level0_gain_ok"] is False
    assert visible_repair["visible_gain_ok"] is True

    locked_baseline = _level1_branch_candidate_decision(
        baseline_level0_pass=True,
        baseline_level0_score=3.25,
        baseline_visible=visible_pass,
        cand_format_ok=True,
        cand_parse_ok=True,
        cand_obvious_truncation=False,
        cand_level0_score=3.25,
        cand_visible=visible_pass,
        model_score=0.0,
        min_score_gain=1.0,
        visible_min_pass_gain=1,
        visible_require_level0=True,
    )
    assert locked_baseline["eligible"] is False
    assert locked_baseline["repair_mode"] == "none"


def test_branch_observe_trigger_mode_auto_uses_local_beam_mode():
    cfg = DecoderConfig(local_beam_mode="kl_only", branch_observe_trigger_mode="auto")
    assert _resolve_branch_observe_trigger_mode(cfg) == "kl_only"

    cfg = DecoderConfig(local_beam_mode="entropy_kl_struct", branch_observe_trigger_mode="auto")
    assert _resolve_branch_observe_trigger_mode(cfg) == "legacy_entropy_kl_struct"

    cfg = DecoderConfig(local_beam_mode="delay_only", branch_observe_trigger_mode="auto")
    assert _resolve_branch_observe_trigger_mode(cfg) == "legacy_entropy_kl_struct"


def test_shadow_trigger_score_modes_change_formula():
    entropy_t = torch.tensor([0.6], dtype=torch.float32)
    kl_t = torch.tensor([0.5], dtype=torch.float32)
    struct_weight_t = torch.tensor([1.5], dtype=torch.float32)

    assert torch.allclose(_shadow_trigger_score_t("entropy_only", entropy_t, kl_t, struct_weight_t), torch.tensor([0.6]))
    assert torch.allclose(_shadow_trigger_score_t("kl_only", entropy_t, kl_t, struct_weight_t), torch.tensor([0.5]))
    assert torch.allclose(_shadow_trigger_score_t("entropy_kl", entropy_t, kl_t, struct_weight_t), torch.tensor([0.3]))
    assert torch.allclose(_shadow_trigger_score_t("entropy_kl_struct", entropy_t, kl_t, struct_weight_t), torch.tensor([0.45]))


def test_random_structural_branch_policy_prefers_structural_positions():
    rng = __import__("random").Random(7)
    selection_score_t = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
    structure_score_t = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    conf_proxy_t = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    selected = _select_branch_observe_candidate_indices(
        selection_score_t=selection_score_t,
        structure_score_t=structure_score_t,
        conf_proxy_t=conf_proxy_t,
        max_count=2,
        event_policy="random_structural",
        rng=rng,
    )

    assert set(selected).issubset({1, 3})


def test_branch_observe_rollout_specs_support_delay_only_and_top2_candidate_modes():
    delay_only = _build_branch_observe_rollout_specs(
        official_token=7,
        top_token_ids=[7, 11, 13],
        branch_observe_top_k=0,
        branch_observe_beam_size=2,
        branch_observe_include_delay=True,
    )
    assert delay_only == [("delay", None)]

    candidate_only = _build_branch_observe_rollout_specs(
        official_token=7,
        top_token_ids=[7, 11, 13],
        branch_observe_top_k=1,
        branch_observe_beam_size=2,
        branch_observe_include_delay=False,
    )
    assert candidate_only == [("candidate", 11)]

    top2 = _build_branch_observe_rollout_specs(
        official_token=7,
        top_token_ids=[7, 11, 13],
        branch_observe_top_k=3,
        branch_observe_beam_size=3,
        branch_observe_include_delay=False,
    )
    assert top2 == [("candidate", 11), ("candidate", 13)]


def test_target_function_present_for_eval_detects_humaneval_entrypoint():
    item = {"entry_point": "truncate_number"}
    prompt = "def truncate_number(number: float) -> float:\n    \"\"\"Return decimal part.\"\"\"\n"
    completion = "    return number % 1.0\n"

    assert _target_function_present_for_eval(item=item, prompt=prompt, completion=completion, dataset="humaneval") is True
    assert _target_function_present_for_eval(item={"entry_point": "other"}, prompt=prompt, completion=completion, dataset="humaneval") is False


def test_summarize_branch_select_records_tracks_parse_and_particle_breakdown():
    records = [
        {
            "selected": True,
            "reason": "selected_level0_strict_gain",
            "selected_kind": "candidate",
            "baseline": {
                "format_ok": True,
                "parse_ok": False,
                "obvious_truncation": False,
                "target_function_present": False,
            },
            "selected_format_ok": True,
            "selected_parse_ok": True,
            "selected_obvious_truncation": False,
            "selected_target_function_present": True,
        },
        {
            "selected": True,
            "reason": "selected_level0_strict_gain",
            "selected_kind": "delay",
            "baseline": {
                "format_ok": False,
                "parse_ok": False,
                "obvious_truncation": True,
                "target_function_present": True,
            },
            "selected_format_ok": True,
            "selected_parse_ok": True,
            "selected_obvious_truncation": False,
            "selected_target_function_present": True,
        },
        {
            "selected": False,
            "reason": "keep_baseline_level0_pass",
            "selected_kind": "baseline",
            "baseline": {},
        },
    ]

    breakdown = _summarize_branch_select_records(records)

    assert breakdown["reason_counts"]["selected_level0_strict_gain"] == 2
    assert breakdown["reason_counts"]["keep_baseline_level0_pass"] == 1
    assert breakdown["selected_candidate_count"] == 1
    assert breakdown["selected_delay_count"] == 1
    assert breakdown["selected_level0_parse_repair_count"] == 2
    assert breakdown["selected_level0_format_repair_count"] == 1
    assert breakdown["selected_level0_truncation_repair_count"] == 1
    assert breakdown["selected_level0_target_func_repair_count"] == 1


def test_visible_humaneval_score_uses_prompt_doctests():
    prompt = (
        "def add_one(x):\n"
        "    \"\"\"Return x plus one.\n"
        "    >>> add_one(1)\n"
        "    2\n"
        "    >>> add_one(-1)\n"
        "    0\n"
        "    \"\"\"\n"
    )

    score = _visible_test_score(
        item={"prompt": prompt},
        prompt=prompt,
        completion="    return x + 1\n",
        dataset="humaneval",
        timeout=1,
    )

    assert score["enabled"] is True
    assert score["passed"] == 2
    assert score["total"] == 2
    assert score["all_passed"] is True


def test_first_diff_helpers_report_exact_and_divergent_outputs():
    assert _first_text_diff("abc", "abc")["match"] is True
    diff = _first_text_diff("abc", "axc")
    assert diff["match"] is False
    assert diff["index"] == 1
    assert diff["left_char"] == "b"
    assert diff["right_char"] == "x"

    class ToyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(ch) for ch in text]

    token_diff = _first_token_diff(ToyTokenizer(), "abc", "abz")
    assert token_diff["available"] is True
    assert token_diff["match"] is False
    assert token_diff["index"] == 2
    assert token_diff["left_token_id"] == ord("c")
    assert token_diff["right_token_id"] == ord("z")


def test_unit_test_pass_falls_back_to_local_humaneval_runner():
    item = {
        "task_id": "HumanEval/local",
        "prompt": "def add_one(x):\n",
        "entry_point": "add_one",
        "canonical_solution": "    return x + 1\n",
        "test": "\n\ndef check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n",
    }

    assert unit_test_pass(item, item["canonical_solution"], dataset="humaneval", timeout=2.0) is True
    assert unit_test_pass(item, "    return x\n", dataset="humaneval", timeout=2.0) is False


def test_mbpp_unit_test_pass_runs_tests_without_prompt_prefix():
    item = {
        "task_id": "MBPP/local",
        "prompt": "Write a python function to add one.",
        "test": "assert add_one(1) == 2\nassert add_one(-1) == 0",
        "test_list": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
        "test_imports": [],
    }
    good = "def add_one(x):\n    return x + 1\n"
    bad = "def add_one(x):\n    return x\n"

    assert parse_success_for_eval(item, item["prompt"], good, dataset="mbpp") is True
    assert unit_test_pass(item, good, dataset="mbpp", timeout=2.0) is True
    assert unit_test_pass(item, bad, dataset="mbpp", timeout=2.0) is False


def test_mbpp_postprocess_extracts_code_from_fence():
    item = {"task_id": "MBPP/local", "prompt": "Write a function."}
    raw = "Here is the solution:\n```python\ndef f(x):\n    return x + 1\n```\nExplanation: done."

    processed, meta = postprocess_completion_for_eval(item=item, completion=raw, dataset="mbpp")

    assert processed == "def f(x):\n    return x + 1\n"
    assert meta["mode"] == "strip_markdown_fence"


def test_humaneval_postprocess_strips_markdown_and_examples():
    item = {
        "task_id": "HumanEval/local",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return decimal part.\"\"\"\n",
        "entry_point": "truncate_number",
    }
    raw = (
        "    return number % 1.0\n\n"
        "# Example usage\n"
        "print(truncate_number(3.5))\n"
        "```\n\n"
        "### Explanation\n"
        "Use modulo."
    )

    processed, meta = postprocess_completion_for_eval(item=item, completion=raw, dataset="humaneval")

    assert processed == "    return number % 1.0\n"
    assert meta["mode"] == "humaneval_function_body"
    assert meta["changed"] is True
    assert parse_success(item["prompt"], processed) is True


def test_humaneval_postprocess_extracts_body_from_full_function_fence():
    item = {
        "task_id": "HumanEval/local",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return decimal part.\"\"\"\n",
        "entry_point": "truncate_number",
    }
    raw = (
        "Here is the solution:\n"
        "```python\n"
        "def truncate_number(number: float) -> float:\n"
        "    return number % 1.0\n"
        "```\n"
    )

    processed, _ = postprocess_completion_for_eval(item=item, completion=raw, dataset="humaneval")

    assert processed == "    return number % 1.0\n"
    assert parse_success(item["prompt"], processed) is True


def test_apply_completion_token_overrides_updates_completion_slice_only():
    x = torch.tensor([[101, 102, 0, 0, 0]], dtype=torch.long)
    override_mask = torch.tensor([False, True, True], dtype=torch.bool)
    override_tokens = torch.tensor([0, 7, 8], dtype=torch.long)

    out = _apply_completion_token_overrides(
        x=x,
        prompt_len=2,
        completion_len=3,
        override_mask=override_mask,
        override_tokens=override_tokens,
    )

    assert torch.equal(out[0, :2], torch.tensor([101, 102], dtype=torch.long))
    assert torch.equal(out[0, 2:], torch.tensor([0, 7, 8], dtype=torch.long))


def test_apply_completion_remask_masks_only_selected_completion_positions():
    x = torch.tensor([[101, 102, 5, 6, 7, 8]], dtype=torch.long)

    out = _apply_completion_remask(
        x=x,
        prompt_len=2,
        completion_len=4,
        remask_positions=[1, 3],
        mask_token_id=0,
    )

    assert torch.equal(out[0, :2], torch.tensor([101, 102], dtype=torch.long))
    assert torch.equal(out[0, 2:], torch.tensor([5, 0, 7, 0], dtype=torch.long))


def test_select_rdd_rollback_positions_targets_error_progress_neighborhood():
    committed = torch.tensor([True, True, True, True, True, True, True, True], dtype=torch.bool)

    positions = _select_rdd_rollback_positions(
        committed_mask_t=committed,
        syntax_error_progress=0.5,
        rollback_window=3,
    )

    assert positions == [3, 4, 5]


def test_resolve_dream_repair_policy_keeps_experiment_modes_disjoint():
    cfg = DecoderConfig(pf_enabled=True, rdd_rollback_enabled=False)
    assert _resolve_dream_repair_policy(cfg) == "none"

    cfg = DecoderConfig(pf_enabled=False, rdd_rollback_enabled=True)
    assert _resolve_dream_repair_policy(cfg) == "rollback_only"

    cfg = DecoderConfig(pf_enabled=True, rdd_rollback_enabled=True)
    assert _resolve_dream_repair_policy(cfg) == "pf_rb"


def test_only_rollback_only_policy_uses_baseline_sampling():
    assert _repair_policy_uses_baseline_sampling("none") is False
    assert _repair_policy_uses_baseline_sampling("pf_rb") is False
    assert _repair_policy_uses_baseline_sampling("rollback_only") is True


def test_rollback_only_route_ignores_projected_token_damage():
    committed = {
        "observed": True,
        "parse_ok": True,
        "severity_score": 0.0,
        "syntax_error_progress": 0.0,
    }

    route = _route_rollback_only_action(
        committed_feedback=committed,
        prompt="def f():\n",
        committed_completion="    return ",
        committed_token_count=3,
        rollback_available=True,
        rollback_min_severity=0.25,
        repair_cooldown_active=False,
    )

    assert route["action"] == "normal"
    assert route["state"] == "no_confirmed_damage"


def test_pf_rb_route_rolls_back_confirmed_committed_damage_before_pf():
    committed = {
        "observed": True,
        "parse_ok": False,
        "severity_score": 0.6,
        "syntax_error_progress": 0.95,
        "primary_issue": "indent",
    }
    projected = {
        "observed": True,
        "parse_ok": False,
        "severity_score": 0.7,
        "syntax_error_progress": 0.95,
    }

    route = _route_pf_rb_action(
        committed_feedback=committed,
        projected_feedback=projected,
        prompt="def f():\n",
        committed_completion="    if x:\n",
        projected_completion="    if x:\n        return x",
        committed_token_count=4,
        projected_token_count=8,
        masked_positions_t=torch.tensor([4, 5, 6], dtype=torch.long),
        entropy_all_t=torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.7, 0.1], dtype=torch.float32),
        rollback_available=True,
        pf_available=True,
        high_risk_masked_position=True,
        rollback_min_severity=0.25,
        near_radius=2,
    )

    assert route["action"] == "rollback"
    assert route["state"] == "context_level_damage"


def test_pf_rb_route_uses_pf_for_projected_damage_near_masked_entropy():
    committed = {
        "observed": True,
        "parse_ok": True,
        "severity_score": 0.0,
        "syntax_error_progress": 0.0,
    }
    projected = {
        "observed": True,
        "parse_ok": False,
        "severity_score": 0.55,
        "syntax_error_progress": 0.93,
        "primary_issue": "bracket",
    }

    route = _route_pf_rb_action(
        committed_feedback=committed,
        projected_feedback=projected,
        prompt="def f():\n",
        committed_completion="    return ",
        projected_completion="    return values[",
        committed_token_count=3,
        projected_token_count=8,
        masked_positions_t=torch.tensor([5, 6], dtype=torch.long),
        entropy_all_t=torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.95, 0.7, 0.1], dtype=torch.float32),
        rollback_available=True,
        pf_available=True,
        high_risk_masked_position=True,
        rollback_min_severity=0.25,
        near_radius=2,
    )

    assert route["action"] == "pf"
    assert route["state"] == "token_level_risk"
    assert route["projected_error_near_masked"] is True


def test_pf_rb_route_does_not_pf_parse_ok_projected_candidate():
    feedback = {
        "observed": True,
        "parse_ok": True,
        "severity_score": 0.0,
        "syntax_error_progress": 0.0,
        "primary_issue": "none",
    }

    route = _route_pf_rb_action(
        committed_feedback=feedback,
        projected_feedback=feedback,
        prompt="def f():\n",
        committed_completion="    return ",
        projected_completion="    return 1",
        committed_token_count=3,
        projected_token_count=6,
        masked_positions_t=torch.tensor([4], dtype=torch.long),
        entropy_all_t=torch.tensor([0.1, 0.1, 0.1, 0.1, 0.95, 0.1], dtype=torch.float32),
        rollback_available=True,
        pf_available=True,
        high_risk_masked_position=True,
        rollback_min_severity=0.25,
        near_radius=2,
    )

    assert route["action"] == "normal"
    assert route["projected_parse_ok"] is True


def test_pf_rb_route_blocks_pf_during_rollback_cooldown():
    committed = {
        "observed": True,
        "parse_ok": True,
        "severity_score": 0.0,
        "syntax_error_progress": 0.0,
    }

    route = _route_pf_rb_action(
        committed_feedback=committed,
        projected_feedback=committed,
        prompt="def f():\n",
        committed_completion="    return x",
        projected_completion="    return x",
        committed_token_count=4,
        projected_token_count=4,
        masked_positions_t=torch.tensor([2], dtype=torch.long),
        entropy_all_t=torch.tensor([0.1, 0.1, 0.9, 0.1], dtype=torch.float32),
        rollback_available=False,
        pf_available=True,
        high_risk_masked_position=True,
        rollback_min_severity=0.25,
        near_radius=2,
        repair_cooldown_active=True,
    )

    assert route["action"] == "normal"
    assert route["state"] == "cooldown"


def test_pf_rb_route_does_not_rollback_low_severity_gradient_noise():
    committed = {
        "observed": True,
        "parse_ok": False,
        "severity_score": 0.05,
        "syntax_error_progress": 0.95,
        "primary_issue": "syntax",
    }

    route = _route_pf_rb_action(
        committed_feedback=committed,
        projected_feedback=committed,
        prompt="def f():\n",
        committed_completion="    return",
        projected_completion="    return",
        committed_token_count=3,
        projected_token_count=3,
        masked_positions_t=torch.tensor([2], dtype=torch.long),
        entropy_all_t=torch.tensor([0.1, 0.1, 0.2], dtype=torch.float32),
        rollback_available=True,
        pf_available=False,
        high_risk_masked_position=False,
        rollback_min_severity=0.25,
        near_radius=2,
        parser_gradient_hotspot=True,
    )

    assert route["action"] == "normal"
    assert route["state"] == "no_confirmed_damage"


def test_rollback_only_route_ignores_trailing_incomplete_prefix():
    committed = {
        "observed": True,
        "parse_ok": False,
        "severity_score": 0.55,
        "syntax_error_progress": 0.99,
        "primary_issue": "syntax",
        "issue_types": ["syntax"],
        "syntax_error_message": "invalid syntax",
    }

    route = _route_rollback_only_action(
        committed_feedback=committed,
        prompt="def f():\n",
        committed_completion="    return x +",
        committed_token_count=3,
        rollback_available=True,
        rollback_min_severity=0.25,
    )

    assert route["action"] == "normal"
    assert route["committed_prefix_ast_risk"] is False


def test_resolve_pf_persistence_steps_is_short_early_and_long_late():
    assert _resolve_pf_persistence_steps(t_remaining=70, t_start=30, t_end=70, base_steps=3) == 3
    assert _resolve_pf_persistence_steps(t_remaining=30, t_start=30, t_end=70, base_steps=3) == 10
    mid = _resolve_pf_persistence_steps(t_remaining=50, t_start=30, t_end=70, base_steps=3)
    assert 3 < mid < 10


def test_structure_sensitive_token_score_prefers_code_structure_tokens():
    assert _structure_sensitive_token_score("return") > _structure_sensitive_token_score("value")
    assert _structure_sensitive_token_score("):") > _structure_sensitive_token_score("foo")


def test_select_pf_candidate_positions_keeps_top_entropy_and_adds_structure_hotspot():
    masked_positions_t = torch.tensor([1, 3, 5], dtype=torch.long)
    entropy_all_t = torch.tensor([0.1, 0.95, 0.2, 0.72, 0.1, 0.68], dtype=torch.float32)
    argmax_all_t = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    token_map = {
        11: "name",
        13: "return",
        15: "value",
    }

    def token_ids_to_code(ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return "".join(token_map.get(int(tok), "") for tok in ids)

    selected, meta = _select_pf_candidate_positions(
        masked_positions_t=masked_positions_t,
        entropy_all_t=entropy_all_t,
        argmax_all_t=argmax_all_t,
        token_ids_to_code=token_ids_to_code,
        budget=2,
        parser_hotspot_active=True,
    )

    assert selected == [1, 3]
    assert meta[1]["entropy"] == float(entropy_all_t[1].item())
    assert meta[3]["structure_score"] > meta[1]["structure_score"]


def test_extract_identifier_counts_tracks_reuse():
    counts = _extract_identifier_counts("def solve(nums): return nums")
    assert counts["solve"] == 1
    assert counts["nums"] == 2


def test_constraint_identifier_signal_rewards_structure_and_identifier_reuse():
    score, meta = _constraint_identifier_signal(
        pos=8,
        token_text="return",
        seq_len=16,
        parser_feedback={"primary_issue": "indent"},
        parser_hotspot_active=True,
        prompt_identifier_counts=_extract_identifier_counts("def solve(nums):"),
        source_identifier_counts=_extract_identifier_counts("total = nums"),
    )

    assert score > 0.3
    assert meta["constraint_score"] > 0.8


def test_budgeted_entropy_pf_controls_cap_particles_and_stop_at_budget():
    controls = _resolve_budgeted_entropy_pf_controls(
        mode="budgeted_entropy",
        base_allow_pf_step=True,
        pf_positions_cap=3,
        pf_particles_step=4,
        pf_horizon_steps=3,
        extra_forwards_used=6,
        extra_forward_budget=12,
        parser_hotspot_active=False,
        current_gradient_hotspot=False,
    )

    assert controls["allow_pf_step"] is True
    assert controls["pf_positions_cap"] == 1
    assert controls["pf_particles_step"] == 2
    assert controls["remaining_extra_forward_budget"] == 6

    blocked = _resolve_budgeted_entropy_pf_controls(
        mode="budgeted_entropy",
        base_allow_pf_step=True,
        pf_positions_cap=1,
        pf_particles_step=4,
        pf_horizon_steps=3,
        extra_forwards_used=12,
        extra_forward_budget=12,
        parser_hotspot_active=True,
        current_gradient_hotspot=False,
    )

    assert blocked["allow_pf_step"] is False
    assert blocked["budget_blocked"] is True


def test_budgeted_entropy_default_budget_is_unlimited():
    assert _DEFAULT_PF_EXTRA_FORWARD_BUDGET == 0

    controls = _resolve_budgeted_entropy_pf_controls(
        mode="budgeted_entropy",
        base_allow_pf_step=True,
        pf_positions_cap=1,
        pf_particles_step=4,
        pf_horizon_steps=3,
        extra_forwards_used=999,
        extra_forward_budget=_DEFAULT_PF_EXTRA_FORWARD_BUDGET,
        parser_hotspot_active=True,
        current_gradient_hotspot=False,
    )

    assert controls["allow_pf_step"] is True
    assert controls["budget_active"] is False
    assert controls["budget_blocked"] is False


def test_budgeted_entropy_pf_controls_uses_wider_hotspot_particles():
    controls = _resolve_budgeted_entropy_pf_controls(
        mode="budgeted_entropy",
        base_allow_pf_step=True,
        pf_positions_cap=1,
        pf_particles_step=4,
        pf_horizon_steps=3,
        extra_forwards_used=0,
        extra_forward_budget=12,
        parser_hotspot_active=True,
        current_gradient_hotspot=False,
    )

    assert controls["pf_particles_step"] == 3
    assert controls["syntax_repair_mode"] is True


def test_budgeted_pf_acceptance_rejects_parser_quality_regression():
    assert _should_accept_budgeted_pf_choice(
        base_quality=0.7,
        candidate_quality=0.5,
        candidate_parse_ok=False,
        tolerance=0.02,
    ) is False
    assert _should_accept_budgeted_pf_choice(
        base_quality=0.7,
        candidate_quality=0.69,
        candidate_parse_ok=True,
        tolerance=0.02,
    ) is True
    assert _should_accept_budgeted_pf_choice(
        base_quality=0.2,
        candidate_quality=0.21,
        candidate_parse_ok=False,
        tolerance=0.02,
    ) is True
    assert _should_accept_budgeted_pf_choice(
        base_quality=1.0,
        candidate_quality=0.9,
        candidate_parse_ok=False,
        tolerance=0.02,
    ) is False


def test_selected_pf_particle_parser_delta_reads_chosen_particle():
    meta = _selected_pf_particle_parser_delta(
        [
            {
                "particle_tokens": [10, 11],
                "particle_syntax_base_quality": [0.25, 0.25],
                "particle_syntax_candidate_quality": [0.5, 0.1],
                "particle_parse_ok": [True, False],
            }
        ],
        chosen_token=11,
    )

    assert meta["observed"] is True
    assert meta["base_quality"] == 0.25
    assert meta["candidate_quality"] == 0.1
    assert meta["candidate_parse_ok"] is False


def test_parser_feedback_after_forced_token_reparses_projected_code():
    current = torch.tensor([0, 0], dtype=torch.long)
    committed = torch.tensor([False, False], dtype=torch.bool)
    predicted = torch.tensor([2, 3], dtype=torch.long)
    token_map = {
        1: "1",
        2: "",
        3: "\n",
    }

    def token_ids_to_code(ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(token_map.get(int(tok), "") for tok in ids)

    feedback, source = _parser_feedback_after_forced_token(
        current_tokens_t=current,
        committed_mask_t=committed,
        predicted_tokens_t=predicted,
        pos=0,
        token=1,
        token_ids_to_code=token_ids_to_code,
        source_prefix="def f():\n    return ",
    )

    assert source == "1\n"
    assert feedback["parse_ok"] is True


def test_parser_hotspot_helpers_weight_nearby_failures():
    hist = {"bracket": {}, "indent": {}}
    _bump_parser_issue_hist(hist["bracket"], 60)
    _bump_parser_issue_hist(hist["indent"], 59)

    score_at_59 = _parser_hotspot_score(hist, t_remaining=59, radius=2)
    score_at_55 = _parser_hotspot_score(hist, t_remaining=55, radius=2)
    top = _top_parser_issue_timesteps(hist, limit=2)

    assert score_at_59 > 1.0
    assert score_at_55 == 0.0
    assert top["bracket"][0]["t"] == 60
    assert top["indent"][0]["t"] == 59


def test_parser_hotspot_score_uses_gradient_events_not_static_bad_state():
    hist = {"bracket": {}, "indent": {}}
    _bump_parser_score_hist(hist["bracket"], 60, 0.24)
    score_at_60 = _parser_hotspot_score(hist, t_remaining=60, radius=2)
    score_at_61 = _parser_hotspot_score(hist, t_remaining=61, radius=2)
    top = _top_parser_score_timesteps(hist, limit=1)

    assert abs(score_at_60 - 0.24) < 1e-9
    assert 0.11 < score_at_61 < 0.13
    assert top["bracket"][0]["t"] == 60
    assert abs(top["bracket"][0]["score"] - 0.24) < 1e-9


def test_parser_gradient_metrics_only_fire_on_worsening():
    feedback = {
        "observed": True,
        "parse_ok": False,
        "primary_issue": "bracket",
        "bracket_issue": True,
        "indent_issue": False,
        "quality_score": 0.72,
        "severity_score": 0.28,
    }
    first = _parser_gradient_metrics(feedback, prev_severity=0.05, prev_delta=0.02)
    second = _parser_gradient_metrics(feedback, prev_severity=0.28, prev_delta=0.23)

    assert first["score"] > 0.2
    assert first["delta"] > 0.2
    assert second["score"] == 0.0
    assert second["delta"] == 0.0


def test_compose_branch_completion_state_overlays_only_owned_tokens():
    base_tokens = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    base_committed = torch.tensor([True, False, False, True], dtype=torch.bool)
    branch = _DreamBranchState(
        branch_id=1,
        owned_mask=torch.tensor([False, True, False, False], dtype=torch.bool),
        owned_tokens=torch.tensor([0, 42, 0, 0], dtype=torch.long),
        score=1.0,
        ttl=3,
        origin_t=50,
        source_pos=1,
        chosen_token=42,
    )

    branch_tokens, branch_committed = _compose_branch_completion_state(
        base_tokens=base_tokens,
        base_committed_mask=base_committed,
        branch=branch,
    )

    assert torch.equal(branch_tokens, torch.tensor([10, 42, 12, 13], dtype=torch.long))
    assert torch.equal(branch_committed, torch.tensor([True, True, False, True], dtype=torch.bool))


def test_prune_dream_branch_bank_keeps_best_unique_branches():
    branches = [
        _DreamBranchState(
            branch_id=1,
            owned_mask=torch.tensor([False, True, False], dtype=torch.bool),
            owned_tokens=torch.tensor([0, 7, 0], dtype=torch.long),
            score=1.2,
            eval_score=1.2,
            ttl=3,
            origin_t=60,
            source_pos=1,
            chosen_token=7,
        ),
        _DreamBranchState(
            branch_id=2,
            owned_mask=torch.tensor([False, True, False], dtype=torch.bool),
            owned_tokens=torch.tensor([0, 7, 0], dtype=torch.long),
            score=0.8,
            eval_score=0.8,
            ttl=2,
            origin_t=59,
            source_pos=1,
            chosen_token=7,
        ),
        _DreamBranchState(
            branch_id=3,
            owned_mask=torch.tensor([True, False, False], dtype=torch.bool),
            owned_tokens=torch.tensor([5, 0, 0], dtype=torch.long),
            score=1.0,
            eval_score=1.0,
            ttl=2,
            origin_t=58,
            source_pos=0,
            chosen_token=5,
        ),
    ]

    kept, pruned = _prune_dream_branch_bank(branches, beam_width=2)

    assert [branch.branch_id for branch in kept] == [1, 3]
    assert pruned == 1


def test_advance_dream_branch_trials_triggers_merge_when_trial_expires():
    branches = [
        _DreamBranchState(
            branch_id=11,
            owned_mask=torch.tensor([True, False], dtype=torch.bool),
            owned_tokens=torch.tensor([5, 0], dtype=torch.long),
            score=1.04,
            eval_score=1.04,
            ttl=2,
            origin_t=50,
            source_pos=0,
            chosen_token=5,
        ),
        _DreamBranchState(
            branch_id=12,
            owned_mask=torch.tensor([False, True], dtype=torch.bool),
            owned_tokens=torch.tensor([0, 7], dtype=torch.long),
            score=0.96,
            eval_score=0.96,
            ttl=1,
            origin_t=50,
            source_pos=1,
            chosen_token=7,
        ),
    ]

    advanced, merge_due = _advance_dream_branch_trials(branches)

    assert [branch.ttl for branch in advanced] == [1, 0]
    assert merge_due is True


def test_select_best_dream_branch_entry_prefers_highest_eval_score():
    entries = [
        (
            _DreamBranchState(
                branch_id=21,
                owned_mask=torch.tensor([True, False], dtype=torch.bool),
                owned_tokens=torch.tensor([9, 0], dtype=torch.long),
                score=1.25,
                eval_score=1.25,
                ttl=3,
                origin_t=40,
                source_pos=0,
                chosen_token=9,
            ),
            torch.tensor([9, 0], dtype=torch.long),
            torch.tensor([True, False], dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.float32),
        ),
        (
            _DreamBranchState(
                branch_id=22,
                owned_mask=torch.tensor([False, True], dtype=torch.bool),
                owned_tokens=torch.tensor([0, 4], dtype=torch.long),
                score=0.8,
                eval_score=0.8,
                ttl=3,
                origin_t=40,
                source_pos=1,
                chosen_token=4,
            ),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([False, True], dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.float32),
        ),
    ]

    selection = _select_best_dream_branch_entry(entries)

    assert selection is not None
    assert selection[0].branch_id == 21


def test_dream_apply_sampling_step_updates_each_batch_row_independently():
    x = torch.tensor(
        [
            [99, 0, 0, 0],
            [99, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    logits = torch.full((2, 4, 6), -8.0, dtype=torch.float32)
    logits[0, 1, 2] = 8.0
    logits[0, 2, 3] = 8.0
    logits[0, 3, 4] = 8.0
    logits[1, 1, 5] = 8.0
    logits[1, 2, 1] = 8.0
    logits[1, 3, 2] = 8.0

    out = _dream_apply_sampling_step(
        x_t=x,
        logits_t=logits,
        mask_token_id=0,
        t_value=1.0,
        s_value=0.5,
        alg="maskgit_plus",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        alg_temp=0.0,
        final_step=False,
    )

    assert int((out[0] != 0).sum().item()) == 2
    assert int((out[1] != 0).sum().item()) == 2
    assert set(out[0, 1:].tolist()) <= {0, 2, 3, 4}
    assert set(out[1, 1:].tolist()) <= {0, 1, 2, 5}


def test_dream_apply_sampling_step_commits_all_masks_on_final_step():
    x = torch.tensor([[77, 0, 0]], dtype=torch.long)
    logits = torch.full((1, 3, 5), -9.0, dtype=torch.float32)
    logits[0, 1, 3] = 9.0
    logits[0, 2, 4] = 9.0

    out = _dream_apply_sampling_step(
        x_t=x,
        logits_t=logits,
        mask_token_id=0,
        t_value=0.2,
        s_value=0.1,
        alg="entropy",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        alg_temp=0.0,
        final_step=True,
    )

    assert torch.equal(out, torch.tensor([[77, 3, 4]], dtype=torch.long))


def test_dream_apply_sampling_step_respects_eb_transfer_mask_before_final_step():
    x = torch.tensor([[77, 0, 0]], dtype=torch.long)
    logits = torch.full((1, 3, 5), -9.0, dtype=torch.float32)
    logits[0, 1, 3] = 9.0
    logits[0, 2, 4] = 9.0
    allowed = torch.tensor([[True, False, True]], dtype=torch.bool)

    out = _dream_apply_sampling_step(
        x_t=x,
        logits_t=logits,
        mask_token_id=0,
        t_value=1.0,
        s_value=0.1,
        alg="entropy",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        alg_temp=0.0,
        final_step=False,
        transfer_allowed_mask_t=allowed,
    )

    assert int(out[0, 1].item()) == 0
    assert int(out[0, 2].item()) == 4


def test_eb_transfer_mask_keeps_low_entropy_and_blocks_uncertain_structure():
    cfg = DecoderConfig(
        eb_sampler_enabled=True,
        eb_entropy_quantile=0.7,
        eb_min_commit_per_step=0,
        eb_structure_entropy_scale=0.25,
    )
    x = torch.tensor([[101, 0, 0, 0]], dtype=torch.long)
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    logits[0, 1, 1] = 8.0  # very low entropy plain token
    logits[0, 2, 2] = 8.0
    logits[0, 2, 3] = 7.5  # structural token with enough uncertainty to be blocked
    logits[0, 3, :] = 1.0  # high entropy
    token_text = {1: "value", 2: "(", 3: "name"}

    def token_ids_to_code(ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(token_text.get(int(tok), "") for tok in ids)

    allowed, meta = _build_eb_transfer_allowed_mask(
        x_t=x,
        logits_t=logits,
        comp_start=1,
        comp_end=4,
        mask_token_id=0,
        cfg=cfg,
        token_ids_to_code=token_ids_to_code,
        source_prefix="",
        parser_feedback={"parse_ok": True},
    )

    assert bool(allowed[0, 1].item()) is True
    assert bool(allowed[0, 2].item()) is False
    assert bool(allowed[0, 3].item()) is False
    assert meta["allowed_count"] == 1
    assert meta["structure_blocked_count"] >= 1


def test_should_extend_dream_branches_when_gap_is_small():
    cfg = DecoderConfig(
        branch_extension_enabled=True,
        branch_extension_margin=0.4,
        branch_max_extensions=1,
    )
    entries = [
        (
            _DreamBranchState(
                branch_id=31,
                owned_mask=torch.tensor([True, False], dtype=torch.bool),
                owned_tokens=torch.tensor([9, 0], dtype=torch.long),
                score=1.0,
                eval_score=1.0,
                ttl=1,
                origin_t=30,
                source_pos=0,
                chosen_token=9,
            ),
            None,
            None,
            None,
        ),
        (
            _DreamBranchState(
                branch_id=32,
                owned_mask=torch.tensor([False, True], dtype=torch.bool),
                owned_tokens=torch.tensor([0, 4], dtype=torch.long),
                score=0.75,
                eval_score=0.75,
                ttl=1,
                origin_t=30,
                source_pos=1,
                chosen_token=4,
            ),
            None,
            None,
            None,
        ),
    ]

    assert _should_extend_dream_branches(entries, cfg) is True


def test_build_joint_branch_candidates_combines_two_positions():
    base_tokens = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    base_committed = torch.tensor([True, False, False, True], dtype=torch.bool)
    base_owned_mask = torch.zeros(4, dtype=torch.bool)
    base_owned_tokens = torch.zeros(4, dtype=torch.long)
    particle_groups = [
        (
            1,
            [
                ParticleState(
                    token_at_pos=21,
                    source_pos=1,
                    forced_tokens=torch.tensor([10, 21, 12, 13], dtype=torch.long),
                    committed_mask=torch.tensor([True, True, False, True], dtype=torch.bool),
                    score=1.2,
                    syntax_candidate_quality=0.9,
                    lookahead_confidence=0.3,
                )
            ],
        ),
        (
            2,
            [
                ParticleState(
                    token_at_pos=22,
                    source_pos=2,
                    forced_tokens=torch.tensor([10, 11, 22, 13], dtype=torch.long),
                    committed_mask=torch.tensor([True, False, True, True], dtype=torch.bool),
                    score=0.8,
                    syntax_candidate_quality=0.8,
                    lookahead_confidence=0.2,
                )
            ],
        ),
    ]

    candidates = _build_joint_branch_candidates(
        particle_groups=particle_groups,
        base_owned_mask_t=base_owned_mask,
        base_owned_tokens_t=base_owned_tokens,
        base_tokens_t=base_tokens,
        base_committed_t=base_committed,
        parent_score=0.5,
        next_branch_id=100,
        t_remaining=50,
        beam_width=3,
        trial_steps=3,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.branch_id == 100
    assert torch.equal(candidate.owned_mask, torch.tensor([False, True, True, False], dtype=torch.bool))
    assert torch.equal(candidate.owned_tokens, torch.tensor([0, 21, 22, 0], dtype=torch.long))
    assert abs(candidate.score - 2.5) < 1e-9
