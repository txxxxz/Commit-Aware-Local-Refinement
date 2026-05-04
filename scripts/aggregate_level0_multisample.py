#!/usr/bin/env python3
"""Aggregate multiple baseline runs with a baseline-preserving Level0 selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def _level0_pass(row: Dict[str, Any]) -> bool:
    return bool(row.get("format_ok", False) and row.get("parse_ok", False) and not row.get("obvious_truncation", False))


def _level0_score(row: Dict[str, Any]) -> float:
    if "level0_score" in row:
        return float(row.get("level0_score", 0.0))
    score = 0.0
    score += 1.0 if bool(row.get("format_ok", False)) else 0.0
    score += 2.0 if bool(row.get("parse_ok", False)) else 0.0
    if str(row.get("completion", "") or "").strip():
        score += 0.25
    if bool(row.get("obvious_truncation", False)):
        score -= 1.0
    return float(score)


def _first_diff(left: str, right: str) -> Dict[str, Any]:
    left = str(left or "")
    right = str(right or "")
    if left == right:
        return {"match": True, "index": -1}
    lim = min(len(left), len(right))
    for idx in range(lim):
        if left[idx] != right[idx]:
            return {
                "match": False,
                "index": idx,
                "left_char": left[idx],
                "right_char": right[idx],
            }
    return {
        "match": False,
        "index": lim,
        "left_char": left[lim] if lim < len(left) else "",
        "right_char": right[lim] if lim < len(right) else "",
    }


def aggregate_runs(run_dirs: Sequence[Path], min_score_gain: float) -> Dict[str, Any]:
    summaries = [_load_json(run_dir / "json" / "summary.json") for run_dir in run_dirs]
    sample_lists: List[List[Dict[str, Any]]] = [_load_json(run_dir / "json" / "sample_metrics.json") for run_dir in run_dirs]
    if not sample_lists or not sample_lists[0]:
        raise ValueError("No samples found.")
    n_samples = len(sample_lists[0])
    for idx, rows in enumerate(sample_lists[1:], start=1):
        if len(rows) != n_samples:
            raise ValueError(f"Run {run_dirs[idx]} has {len(rows)} samples, expected {n_samples}.")

    selected_rows: List[Dict[str, Any]] = []
    selected_count = 0
    eligible_count = 0
    unit_recoveries = 0
    unit_damages = 0
    parse_recoveries = 0
    parse_damages = 0
    exact_output_match_count = 0

    for sample_idx in range(n_samples):
        baseline = sample_lists[0][sample_idx]
        candidates = [rows[sample_idx] for rows in sample_lists[1:]]
        task_id = str(baseline.get("task_id", sample_idx))
        baseline_score = _level0_score(baseline)
        baseline_pass = _level0_pass(baseline)
        chosen = baseline
        chosen_run = 0
        reason = "keep_baseline_level0_pass" if baseline_pass else "keep_baseline_no_eligible_candidate"

        eligible_candidates: List[Dict[str, Any]] = []
        if not baseline_pass:
            for run_idx, cand in enumerate(candidates, start=1):
                cand_score = _level0_score(cand)
                cand_ok = _level0_pass(cand)
                if cand_ok and cand_score >= baseline_score + float(min_score_gain):
                    eligible_candidates.append(
                        {
                            "run_index": run_idx,
                            "row": cand,
                            "level0_score": cand_score,
                        }
                    )
            eligible_count += len(eligible_candidates)
            if eligible_candidates:
                best = max(
                    eligible_candidates,
                    key=lambda item: (
                        float(item["level0_score"]),
                        bool(item["row"].get("unit_pass", False)),
                        not bool(item["row"].get("obvious_truncation", False)),
                    ),
                )
                chosen = best["row"]
                chosen_run = int(best["run_index"])
                selected_count += 1
                reason = "selected_level0_independent_sample"

        baseline_unit = bool(baseline.get("unit_pass", False))
        chosen_unit = bool(chosen.get("unit_pass", False))
        baseline_parse = bool(baseline.get("parse_ok", False))
        chosen_parse = bool(chosen.get("parse_ok", False))
        unit_recoveries += int((not baseline_unit) and chosen_unit)
        unit_damages += int(baseline_unit and (not chosen_unit))
        parse_recoveries += int((not baseline_parse) and chosen_parse)
        parse_damages += int(baseline_parse and (not chosen_parse))
        exact_output_match_count += int(str(baseline.get("completion", "")) == str(chosen.get("completion", "")))

        row_out = dict(chosen)
        row_out["multisample_level0"] = {
            "task_id": task_id,
            "selected": bool(chosen_run != 0),
            "reason": reason,
            "baseline_run_index": 0,
            "selected_run_index": int(chosen_run),
            "candidate_run_count": int(len(candidates)),
            "eligible_candidate_count": int(len(eligible_candidates)),
            "baseline_level0_score": float(baseline_score),
            "selected_level0_score": float(_level0_score(chosen)),
            "baseline_level0_pass": bool(baseline_pass),
            "selected_level0_pass": bool(_level0_pass(chosen)),
            "unit_recovered_baseline": bool((not baseline_unit) and chosen_unit),
            "unit_damaged_baseline": bool(baseline_unit and (not chosen_unit)),
            "parse_recovered_baseline": bool((not baseline_parse) and chosen_parse),
            "parse_damaged_baseline": bool(baseline_parse and (not chosen_parse)),
            "first_eval_text_diff_from_baseline": _first_diff(
                str(baseline.get("completion", "")),
                str(chosen.get("completion", "")),
            ),
        }
        selected_rows.append(row_out)

    pass_count = sum(1 for row in selected_rows if bool(row.get("unit_pass", False)))
    parse_count = sum(1 for row in selected_rows if bool(row.get("parse_ok", False)))
    format_count = sum(1 for row in selected_rows if bool(row.get("format_ok", False)))
    baseline_latency = float(summaries[0].get("avg_latency_sec", 0.0))
    aggregate_latency = sum(float(summary.get("avg_latency_sec", 0.0)) for summary in summaries)
    latency_ratio = float(aggregate_latency / baseline_latency) if baseline_latency > 0 else 0.0
    baseline_extra = float(summaries[0].get("avg_extra_forwards", 0.0))
    aggregate_extra = sum(float(summary.get("avg_extra_forwards", 0.0)) for summary in summaries)
    extra_forward_ratio = float(aggregate_extra / max(baseline_extra, 1.0)) if baseline_extra > 0 else float(len(run_dirs))

    return {
        "method": "independent_multisample_level0",
        "n_runs": int(len(run_dirs)),
        "n_samples": int(n_samples),
        "pass_count": int(pass_count),
        "parse_count": int(parse_count),
        "format_count": int(format_count),
        "selected_count": int(selected_count),
        "eligible_candidate_count": int(eligible_count),
        "recovery": int(unit_recoveries),
        "damage": int(unit_damages),
        "net": int(unit_recoveries - unit_damages),
        "parse_recovery": int(parse_recoveries),
        "parse_damage": int(parse_damages),
        "oracle_union": int(
            sum(
                1
                for sample_idx in range(n_samples)
                if any(bool(rows[sample_idx].get("unit_pass", False)) for rows in sample_lists)
            )
        ),
        "latency_ratio": float(latency_ratio),
        "extra_forward_ratio": float(extra_forward_ratio),
        "avg_latency_sec_total": float(aggregate_latency),
        "avg_latency_sec_baseline": float(baseline_latency),
        "exact_output_match_count": int(exact_output_match_count),
        "source_run_dirs": [str(path) for path in run_dirs],
        "selected_rows": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple baseline runs with Level0 selection.")
    parser.add_argument("--run_dir", action="append", required=True, help="Run directory containing json/summary.json and json/sample_metrics.json.")
    parser.add_argument("--min_score_gain", type=float, default=1.0)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    run_dirs = [_resolve_run_dir(path_str) for path_str in args.run_dir]
    result = aggregate_runs(run_dirs=run_dirs, min_score_gain=float(args.min_score_gain))
    if str(args.output).strip():
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
