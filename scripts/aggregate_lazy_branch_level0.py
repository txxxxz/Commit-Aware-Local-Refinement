#!/usr/bin/env python3
"""Aggregate lazy-branch Level0 experiments from baseline/shadow/eager-branch runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path.parent.parent
    if (path / "json" / "sample_metrics.json").exists():
        return path
    raise FileNotFoundError(f"Cannot find json/sample_metrics.json under {path}")


def _obvious_truncation(text: str) -> bool:
    text = str(text or "").rstrip()
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith("\\"):
        return True
    trailing_ops = (
        "+",
        "-",
        "*",
        "/",
        "%",
        "//",
        "**",
        "=",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        ",",
        ".",
        ":",
        "(",
        "[",
        "{",
    )
    return bool(stripped.endswith(trailing_ops))


def _level0_pass(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("format_ok", False)
        and row.get("parse_ok", False)
        and not bool(row.get("obvious_truncation", _obvious_truncation(str(row.get("completion", "")))))
    )


def _level0_score(format_ok: bool, parse_ok: bool, completion: str) -> float:
    score = 0.0
    score += 1.0 if bool(format_ok) else 0.0
    score += 2.0 if bool(parse_ok) else 0.0
    if str(completion or "").strip():
        score += 0.25
    if _obvious_truncation(completion):
        score -= 1.0
    return float(score)


def _first_diff(left: str, right: str) -> Dict[str, Any]:
    left = str(left or "")
    right = str(right or "")
    if left == right:
        return {"match": True, "index": -1}
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return {
                "match": False,
                "index": idx,
                "left_char": left[idx],
                "right_char": right[idx],
            }
    return {
        "match": False,
        "index": limit,
        "left_char": left[limit] if limit < len(left) else "",
        "right_char": right[limit] if limit < len(right) else "",
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    q = min(max(float(q), 0.0), 1.0)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _iter_nonbaseline_particles(branch_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    particles: List[Dict[str, Any]] = []
    branch_observe = branch_row.get("branch_observe", {})
    for event_idx, event in enumerate(branch_observe.get("event_logs", []) or []):
        for particle_idx, particle in enumerate(event.get("particle_logs", []) or []):
            if str(particle.get("kind", "")) == "baseline":
                continue
            particles.append(
                {
                    "event_index": int(event_idx),
                    "particle_index": int(particle_idx),
                    "event": event,
                    "particle": particle,
                }
            )
    return particles


def _select_best_level0_candidate(
    *,
    baseline_row: Dict[str, Any],
    branch_row: Dict[str, Any],
    min_score_gain: float,
    allow_baseline_level0_pass: bool,
) -> Tuple[Optional[Dict[str, Any]], int, int]:
    baseline_level0_pass = _level0_pass(baseline_row)
    if baseline_level0_pass and not bool(allow_baseline_level0_pass):
        return None, 0, 0
    baseline_score = _level0_score(
        bool(baseline_row.get("format_ok", False)),
        bool(baseline_row.get("parse_ok", False)),
        str(baseline_row.get("completion", "")),
    )
    eligible: List[Dict[str, Any]] = []
    total_candidates = 0
    for item in _iter_nonbaseline_particles(branch_row):
        particle = item["particle"]
        completion = str(particle.get("completion", particle.get("raw_output", "")))
        fmt_ok = bool(particle.get("format_ok_eval", particle.get("format_ok", False)))
        parse_ok = bool(particle.get("parse_ok_eval", particle.get("parse_ok", False)))
        trunc = bool(_obvious_truncation(completion))
        level0_score = _level0_score(fmt_ok, parse_ok, completion)
        total_candidates += 1
        if fmt_ok and parse_ok and (not trunc) and level0_score >= baseline_score + float(min_score_gain):
            eligible.append(
                {
                    **item,
                    "completion": completion,
                    "format_ok": fmt_ok,
                    "parse_ok": parse_ok,
                    "unit_pass": bool(particle.get("unit_pass", False)),
                    "obvious_truncation": trunc,
                    "level0_score": float(level0_score),
                    "select_score": float(level0_score + 0.01 * float(particle.get("score", 0.0))),
                }
            )
    if not eligible:
        return None, total_candidates, 0
    best = max(
        eligible,
        key=lambda item: (
            float(item["select_score"]),
            float(item["level0_score"]),
            bool(item["unit_pass"]),
        ),
    )
    return best, total_candidates, len(eligible)


def _trigger_decision(
    *,
    mode: str,
    baseline_row: Dict[str, Any],
    shadow_row: Optional[Dict[str, Any]],
    severe_risk_threshold: Optional[float],
) -> Tuple[bool, str]:
    baseline_parse_fail = not bool(baseline_row.get("parse_ok", False))
    baseline_level0_fail = not _level0_pass(baseline_row)
    if mode == "baseline_level0_fail":
        return (True, "baseline_level0_fail") if baseline_level0_fail else (False, "no_trigger")
    if mode == "baseline_parse_fail":
        return (True, "baseline_parse_fail") if baseline_parse_fail else (False, "no_trigger")
    if mode == "baseline_level0_fail_or_severe_risk":
        shadow_max_risk = float((shadow_row or {}).get("shadow_max_risk", 0.0))
        severe = severe_risk_threshold is not None and shadow_max_risk >= float(severe_risk_threshold)
        if baseline_level0_fail:
            return True, "baseline_level0_fail"
        if severe:
            return True, "severe_risk"
        return False, "no_trigger"
    raise ValueError(f"Unsupported mode: {mode}")


def aggregate_lazy_branch(
    *,
    baseline_run_dir: Path,
    branch_run_dir: Path,
    shadow_run_dir: Optional[Path],
    mode: str,
    min_score_gain: float,
    severe_risk_quantile: float,
    severe_risk_threshold: Optional[float],
    allow_level0_pass_selection_on_severe_risk: bool,
) -> Dict[str, Any]:
    baseline_summary = _load_json(baseline_run_dir / "json" / "summary.json")
    baseline_rows: List[Dict[str, Any]] = _load_json(baseline_run_dir / "json" / "sample_metrics.json")
    branch_summary = _load_json(branch_run_dir / "json" / "summary.json")
    branch_rows: List[Dict[str, Any]] = _load_json(branch_run_dir / "json" / "sample_metrics.json")
    shadow_rows: Optional[List[Dict[str, Any]]] = None
    if shadow_run_dir is not None:
        shadow_rows = _load_json(shadow_run_dir / "json" / "sample_metrics.json")
    n_samples = len(baseline_rows)
    if len(branch_rows) != n_samples:
        raise ValueError("Baseline and branch runs have different sample counts.")
    if shadow_rows is not None and len(shadow_rows) != n_samples:
        raise ValueError("Shadow and baseline runs have different sample counts.")

    if severe_risk_threshold is None and shadow_rows is not None:
        severe_risk_threshold = _quantile(
            [float(row.get("shadow_max_risk", 0.0)) for row in shadow_rows],
            q=float(severe_risk_quantile),
        )

    selected_rows: List[Dict[str, Any]] = []
    trigger_reason_counts: Dict[str, int] = {}
    selected_count = 0
    eligible_candidate_count = 0
    total_candidates_scored = 0
    recovery = 0
    damage = 0
    parse_recovery = 0
    parse_damage = 0
    exact_output_match_count = 0
    branch_trigger_count = 0
    branch_event_count = 0
    branch_rollout_count = 0
    branch_extra_forwards = 0
    selected_candidate_count = 0
    selected_delay_count = 0
    selected_parse_repair_count = 0
    selected_format_repair_count = 0
    selected_truncation_repair_count = 0

    for idx, baseline_row in enumerate(baseline_rows):
        task_id = str(baseline_row.get("task_id", idx))
        branch_row = branch_rows[idx]
        shadow_row = shadow_rows[idx] if shadow_rows is not None else None
        triggered, trigger_reason = _trigger_decision(
            mode=mode,
            baseline_row=baseline_row,
            shadow_row=shadow_row,
            severe_risk_threshold=severe_risk_threshold,
        )
        trigger_reason_counts[trigger_reason] = int(trigger_reason_counts.get(trigger_reason, 0)) + 1
        final_row = dict(baseline_row)
        lazy_meta: Dict[str, Any] = {
            "task_id": task_id,
            "mode": str(mode),
            "triggered": bool(triggered),
            "trigger_reason": str(trigger_reason),
            "selected": False,
            "selected_kind": "baseline",
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "min_score_gain": float(min_score_gain),
            "allow_level0_pass_selection_on_severe_risk": bool(allow_level0_pass_selection_on_severe_risk),
        }
        if triggered:
            branch_trigger_count += 1
            branch_event_count += int(branch_row.get("branch_observe", {}).get("branch_events", 0))
            branch_rollout_count += int(branch_row.get("branch_observe", {}).get("rollout_count", 0))
            branch_extra_forwards += int(branch_row.get("branch_observe", {}).get("extra_forwards", 0))
            allow_level0_pass = bool(
                mode == "baseline_level0_fail_or_severe_risk"
                and trigger_reason == "severe_risk"
                and allow_level0_pass_selection_on_severe_risk
            )
            best, total_candidates, eligible_candidates = _select_best_level0_candidate(
                baseline_row=baseline_row,
                branch_row=branch_row,
                min_score_gain=float(min_score_gain),
                allow_baseline_level0_pass=allow_level0_pass,
            )
            total_candidates_scored += int(total_candidates)
            lazy_meta["candidate_count"] = int(total_candidates)
            lazy_meta["eligible_candidate_count"] = int(eligible_candidates)
            eligible_candidate_count += int(eligible_candidates)
            if best is not None:
                final_row = dict(branch_row)
                final_row["raw_completion"] = str(best["particle"].get("raw_output", best["completion"]))
                final_row["completion"] = str(best["completion"])
                final_row["format_ok"] = bool(best["format_ok"])
                final_row["parse_ok"] = bool(best["parse_ok"])
                final_row["unit_pass"] = bool(best["unit_pass"])
                final_row["obvious_truncation"] = bool(best["obvious_truncation"])
                final_row["raw_format_ok"] = bool(best["particle"].get("format_ok", best["format_ok"]))
                final_row["raw_parse_ok"] = bool(best["particle"].get("parse_ok", best["parse_ok"]))
                final_row["branch_select"] = {
                    "enabled": True,
                    "verifier": "level0",
                    "selected": True,
                    "reason": "selected_level0_lazy_strict_gain",
                    "selected_kind": str(best["particle"].get("kind", "")),
                    "selected_event_index": int(best["event_index"]),
                    "selected_particle_index": int(best["particle_index"]),
                    "selected_token_id": best["particle"].get("token_id"),
                    "selected_level0_score": float(best["level0_score"]),
                    "selected_select_score": float(best["select_score"]),
                    "selected_format_ok": bool(best["format_ok"]),
                    "selected_parse_ok": bool(best["parse_ok"]),
                    "selected_unit_pass": bool(best["unit_pass"]),
                    "selected_obvious_truncation": bool(best["obvious_truncation"]),
                    "baseline": {
                        "format_ok": bool(baseline_row.get("format_ok", False)),
                        "parse_ok": bool(baseline_row.get("parse_ok", False)),
                        "unit_pass": bool(baseline_row.get("unit_pass", False)),
                        "obvious_truncation": bool(
                            baseline_row.get("obvious_truncation", _obvious_truncation(str(baseline_row.get("completion", ""))))
                        ),
                    },
                }
                lazy_meta.update(
                    {
                        "selected": True,
                        "selected_kind": str(best["particle"].get("kind", "")),
                        "selected_event_index": int(best["event_index"]),
                        "selected_particle_index": int(best["particle_index"]),
                        "eligible_candidate_count": 1,
                        "selected_level0_score": float(best["level0_score"]),
                    }
                )
                selected_count += 1
                if str(best["particle"].get("kind", "")) == "candidate":
                    selected_candidate_count += 1
                elif str(best["particle"].get("kind", "")) == "delay":
                    selected_delay_count += 1
                if (not bool(baseline_row.get("parse_ok", False))) and bool(best["parse_ok"]):
                    selected_parse_repair_count += 1
                if (not bool(baseline_row.get("format_ok", False))) and bool(best["format_ok"]):
                    selected_format_repair_count += 1
                if bool(baseline_row.get("obvious_truncation", _obvious_truncation(str(baseline_row.get("completion", ""))))) and (
                    not bool(best["obvious_truncation"])
                ):
                    selected_truncation_repair_count += 1
        baseline_unit = bool(baseline_row.get("unit_pass", False))
        final_unit = bool(final_row.get("unit_pass", False))
        baseline_parse = bool(baseline_row.get("parse_ok", False))
        final_parse = bool(final_row.get("parse_ok", False))
        recovery += int((not baseline_unit) and final_unit)
        damage += int(baseline_unit and (not final_unit))
        parse_recovery += int((not baseline_parse) and final_parse)
        parse_damage += int(baseline_parse and (not final_parse))
        exact_output_match_count += int(str(final_row.get("completion", "")) == str(baseline_row.get("completion", "")))
        latency_sec = float(branch_row.get("latency_sec", 0.0)) if triggered else float(baseline_row.get("latency_sec", 0.0))
        final_row["latency_sec"] = float(latency_sec)
        final_row["lazy_branch"] = lazy_meta
        final_row["baseline_compare"] = {
            "enabled": True,
            "baseline_completion": baseline_row.get("completion", ""),
            "method_completion": final_row.get("completion", ""),
            "exact_output_match": bool(str(final_row.get("completion", "")) == str(baseline_row.get("completion", ""))),
            "recovered_baseline_failure": bool((not baseline_unit) and final_unit),
            "damaged_baseline_success": bool(baseline_unit and (not final_unit)),
            "parse_recovered_baseline_failure": bool((not baseline_parse) and final_parse),
            "parse_damaged_baseline_success": bool(baseline_parse and (not final_parse)),
            "first_eval_text_diff": _first_diff(
                str(baseline_row.get("completion", "")),
                str(final_row.get("completion", "")),
            ),
            "baseline_latency_sec": float(baseline_row.get("latency_sec", 0.0)),
        }
        selected_rows.append(final_row)

    pass_count = sum(1 for row in selected_rows if bool(row.get("unit_pass", False)))
    parse_count = sum(1 for row in selected_rows if bool(row.get("parse_ok", False)))
    format_count = sum(1 for row in selected_rows if bool(row.get("format_ok", False)))
    avg_latency_sec = sum(float(row.get("latency_sec", 0.0)) for row in selected_rows) / max(len(selected_rows), 1)
    avg_baseline_latency_sec = float(baseline_summary.get("avg_latency_sec", 0.0))
    latency_ratio = float(avg_latency_sec / avg_baseline_latency_sec) if avg_baseline_latency_sec > 0 else 0.0
    oracle_union = 0
    oracle_union_parse = 0
    for idx, baseline_row in enumerate(baseline_rows):
        triggered = bool(selected_rows[idx].get("lazy_branch", {}).get("triggered", False))
        branch_row = branch_rows[idx]
        any_particle_pass = bool(branch_row.get("branch_observe", {}).get("any_particle_unit_pass", baseline_row.get("unit_pass", False)))
        any_particle_parse = bool(branch_row.get("branch_observe", {}).get("any_particle_parse_ok", baseline_row.get("parse_ok", False)))
        oracle_union += int(any_particle_pass if triggered else bool(baseline_row.get("unit_pass", False)))
        oracle_union_parse += int(any_particle_parse if triggered else bool(baseline_row.get("parse_ok", False)))

    return {
        "method": "lazy_branch_level0",
        "mode": str(mode),
        "n_samples": int(n_samples),
        "pass_count": int(pass_count),
        "parse_count": int(parse_count),
        "format_count": int(format_count),
        "selected_count": int(selected_count),
        "eligible_candidate_count": int(eligible_candidate_count),
        "candidate_count_scored": int(total_candidates_scored),
        "recovery": int(recovery),
        "damage": int(damage),
        "net": int(recovery - damage),
        "parse_recovery": int(parse_recovery),
        "parse_damage": int(parse_damage),
        "oracle_union": int(oracle_union),
        "oracle_union_parse": int(oracle_union_parse),
        "exact_output_match_count": int(exact_output_match_count),
        "avg_latency_sec": float(avg_latency_sec),
        "avg_baseline_latency_sec": float(avg_baseline_latency_sec),
        "latency_ratio": float(latency_ratio),
        "avg_extra_forwards": float(branch_extra_forwards / max(n_samples, 1)),
        "extra_forward_ratio": float(
            (branch_extra_forwards / max(n_samples, 1))
            / max(float(branch_summary.get("avg_extra_forwards", 0.0)), 1.0)
        ),
        "branch_trigger_count": int(branch_trigger_count),
        "branch_event_count": int(branch_event_count),
        "branch_rollout_count": int(branch_rollout_count),
        "branch_extra_forwards_total": int(branch_extra_forwards),
        "avg_branch_particles": float(branch_rollout_count / max(branch_event_count, 1)) if branch_event_count else 0.0,
        "avg_branch_rollout_steps": float(branch_extra_forwards / max(branch_event_count, 1)) if branch_event_count else 0.0,
        "trigger_reason_counts": dict(sorted(trigger_reason_counts.items())),
        "selected_candidate_count": int(selected_candidate_count),
        "selected_delay_count": int(selected_delay_count),
        "selected_level0_parse_repair_count": int(selected_parse_repair_count),
        "selected_level0_format_repair_count": int(selected_format_repair_count),
        "selected_level0_truncation_repair_count": int(selected_truncation_repair_count),
        "severe_risk_quantile": float(severe_risk_quantile),
        "severe_risk_threshold": None if severe_risk_threshold is None else float(severe_risk_threshold),
        "allow_level0_pass_selection_on_severe_risk": bool(allow_level0_pass_selection_on_severe_risk),
        "source_runs": {
            "baseline": str(baseline_run_dir),
            "branch": str(branch_run_dir),
            "shadow": str(shadow_run_dir) if shadow_run_dir is not None else "",
        },
        "selected_rows": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate lazy-branch Level0 experiments.")
    parser.add_argument("--baseline_run_dir", required=True)
    parser.add_argument("--branch_run_dir", required=True)
    parser.add_argument("--shadow_run_dir", default="")
    parser.add_argument(
        "--mode",
        choices=("baseline_level0_fail", "baseline_parse_fail", "baseline_level0_fail_or_severe_risk"),
        required=True,
    )
    parser.add_argument("--min_score_gain", type=float, default=1.0)
    parser.add_argument("--severe_risk_quantile", type=float, default=0.8)
    parser.add_argument("--severe_risk_threshold", type=float, default=None)
    parser.add_argument(
        "--allow_level0_pass_selection_on_severe_risk",
        action="store_true",
        default=False,
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = aggregate_lazy_branch(
        baseline_run_dir=_resolve_run_dir(args.baseline_run_dir),
        branch_run_dir=_resolve_run_dir(args.branch_run_dir),
        shadow_run_dir=_resolve_run_dir(args.shadow_run_dir) if str(args.shadow_run_dir).strip() else None,
        mode=str(args.mode),
        min_score_gain=float(args.min_score_gain),
        severe_risk_quantile=float(args.severe_risk_quantile),
        severe_risk_threshold=args.severe_risk_threshold,
        allow_level0_pass_selection_on_severe_risk=bool(args.allow_level0_pass_selection_on_severe_risk),
    )
    if str(args.output).strip():
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
