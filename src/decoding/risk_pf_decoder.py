"""Risk-aware time-scheduled decoder with attention influence proxy + local PF."""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    BudgetTier,
    DecoderConfig,
    is_in_pf_time_window,
    resolve_pf_particles_for_t,
    resolve_pf_time_window,
)
from .local_beam import (
    compute_local_beam_risks,
    run_commit_timing_local_beam,
    select_local_beam_trigger,
)
from .pf import run_local_pf
from .risk import (
    attention_influence_proxy,
    dynamic_risk_thresholds,
    dynamic_valid_positions,
    entropy_at_position,
    normalize_influence_scores,
    risk_score,
    softmax,
)
from .sampler_adapter import SamplerAdapter

logger = logging.getLogger(__name__)


@dataclass
class BranchState:
    """Single sequence state maintained across diffusion steps."""

    forced_tokens: np.ndarray
    committed_mask: np.ndarray
    delay_count: np.ndarray
    latents: Any = None
    score: float = 0.0


def _entropy_histogram(entropies: Dict[int, float]) -> Dict[str, int]:
    """Fixed bins for quick per-step entropy shape checks."""
    values = np.array(list(entropies.values()), dtype=np.float64)
    hist = {
        "lt_0_5": 0,
        "0_5_to_1_0": 0,
        "1_0_to_1_5": 0,
        "1_5_to_2_0": 0,
        "2_0_to_2_5": 0,
        "2_5_to_3_0": 0,
        "ge_3_0": 0,
    }
    if values.size == 0:
        return hist
    hist["lt_0_5"] = int((values < 0.5).sum())
    hist["0_5_to_1_0"] = int(((values >= 0.5) & (values < 1.0)).sum())
    hist["1_0_to_1_5"] = int(((values >= 1.0) & (values < 1.5)).sum())
    hist["1_5_to_2_0"] = int(((values >= 1.5) & (values < 2.0)).sum())
    hist["2_0_to_2_5"] = int(((values >= 2.0) & (values < 2.5)).sum())
    hist["2_5_to_3_0"] = int(((values >= 2.5) & (values < 3.0)).sum())
    hist["ge_3_0"] = int((values >= 3.0).sum())
    return hist


def _clone_logits_for_temporal_kl(logits: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(logits):
            return logits.detach().clone()
    except Exception:
        pass
    return np.array(logits, copy=True)


def classify_risk_band(risk: float, cfg: DecoderConfig) -> str:
    low = float(cfg.risk_low_threshold if cfg.risk_low_threshold is not None else cfg.risk_threshold * 0.5)
    high = float(cfg.risk_high_threshold if cfg.risk_high_threshold is not None else cfg.risk_threshold)
    if risk < low:
        return "low"
    if risk >= high:
        return "high"
    return "mid"


def classify_risk_band_decoupled(
    risk: float,
    influence: float,
    has_influence: bool,
    cfg: DecoderConfig,
) -> str:
    influence_trigger = bool(has_influence and influence >= float(cfg.influence_trigger_floor))
    low = float(cfg.risk_low_threshold if cfg.risk_low_threshold is not None else cfg.risk_threshold * 0.5)
    high = float(cfg.risk_high_threshold if cfg.risk_high_threshold is not None else cfg.risk_threshold)
    if risk >= high and influence_trigger:
        return "high"
    if risk >= low or influence_trigger:
        return "mid"
    return "low"


def _degrade_band_for_overhead(band: str, overhead_ratio: float, cfg: DecoderConfig) -> str:
    if not cfg.adaptive_budget_enabled:
        return band
    if overhead_ratio <= cfg.overhead_cap_ratio:
        return band
    if band == "high":
        return "mid"
    if band == "mid":
        return "low"
    return "low"


def budget_tier_for_band(band: str, overhead_ratio: float, cfg: DecoderConfig) -> BudgetTier:
    band = _degrade_band_for_overhead(band, overhead_ratio, cfg)
    if band == "high":
        return cfg.budget_tier_high
    if band == "mid":
        return cfg.budget_tier_mid
    return cfg.budget_tier_low


class RiskAwarePFDecoder:
    """Wrap a diffusion sampler and apply time-aware risk routing."""

    def __init__(self, cfg: DecoderConfig, sampler: Optional[SamplerAdapter] = None) -> None:
        self.cfg = cfg
        self.sampler = sampler

    def _decode_tokens(
        self,
        token_ids: np.ndarray,
        token_ids_to_code: Optional[Callable[[np.ndarray], str]],
    ) -> str:
        if token_ids_to_code is not None:
            return token_ids_to_code(token_ids)
        return "".join(chr(ord("a") + int(t) % 26) for t in token_ids)

    def _sampler_step_with_aux(
        self,
        sampler: SamplerAdapter,
        latents: Any,
        committed_mask: np.ndarray,
        forced_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if hasattr(sampler, "step_with_aux"):
            out = sampler.step_with_aux(latents, committed_mask, forced_tokens)
            if isinstance(out, tuple) and len(out) == 2:
                logits, aux = out
                return logits, (aux or {})
        logits = sampler.step(latents, committed_mask, forced_tokens)
        return logits, {}

    def _apply_commit(
        self,
        state: BranchState,
        idx: int,
        token: int,
        logits: np.ndarray,
    ) -> None:
        probs = softmax(logits[idx])
        log_prob = float(np.log(max(float(probs[token]), 1e-12)))
        state.forced_tokens[idx] = int(token)
        state.committed_mask[idx] = True
        state.delay_count[idx] = 0
        state.score += log_prob

    def generate(
        self,
        prompt: str,
        cfg: Optional[DecoderConfig] = None,
        sampler: Optional[SamplerAdapter] = None,
        max_steps: int = 20,
        token_ids_to_code: Optional[Callable[[np.ndarray], str]] = None,
    ) -> str:
        """Primary API."""
        text, _ = self.generate_with_stats(
            prompt=prompt,
            cfg=cfg,
            sampler=sampler,
            max_steps=max_steps,
            token_ids_to_code=token_ids_to_code,
        )
        return text

    def generate_with_stats(
        self,
        prompt: str,
        cfg: Optional[DecoderConfig] = None,
        sampler: Optional[SamplerAdapter] = None,
        max_steps: int = 20,
        token_ids_to_code: Optional[Callable[[np.ndarray], str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Decode and return (text, rich_stats)."""
        source_prefix = str(prompt or "")
        cfg = cfg or self.cfg
        sampler = sampler or self.sampler
        if sampler is None:
            raise ValueError("SamplerAdapter is required for decoding.")

        np.random.seed(cfg.random_seed)
        seq_len = getattr(sampler, "seq_len", 64)
        eot_token_id = getattr(sampler, "eot_token_id", None)
        if eot_token_id is None:
            eot_token_id = getattr(sampler, "eos_token_id", None)

        state = BranchState(
            forced_tokens=np.zeros(seq_len, dtype=np.int64),
            committed_mask=np.zeros(seq_len, dtype=bool),
            delay_count=np.zeros(seq_len, dtype=np.int64),
            latents=None,
            score=0.0,
        )

        total_steps = max(int(max_steps), 1)
        t_mid = min(max(int(cfg.diffusion_mid_step), 1), total_steps)
        pf_t_start, pf_t_end = resolve_pf_time_window(total_steps=total_steps, cfg=cfg)
        pf_cooldown_steps = max(int(cfg.pf_cooldown_steps), 3)
        pf_risk_gradient_sigma = max(float(getattr(cfg, "pf_risk_gradient_sigma", 1.5)), 0.0)
        pf_trigger_limit = max(int(getattr(cfg, "pf_max_triggers_per_sample", 0)), 0)
        pf_cooldown_remaining = 0

        pf_trigger_count = 0
        risk_trigger_count = 0
        extra_forwards_influence = 0
        extra_forwards_pf = 0
        influence_compute_total = 0
        baseline_forwards = 0
        prev_logits_for_kl: Any = None
        local_beam_branch_events = 0
        local_beam_accepted_alternatives = 0
        local_beam_delay_count = 0
        local_beam_extra_forwards = 0
        local_beam_beam_sizes: List[int] = []
        local_beam_event_logs: List[Dict[str, Any]] = []

        action_counts_total = {
            "commit_argmax": 0,
            "commit_pf": 0,
            "commit_fallback_max_prob": 0,
            "freeze_delay": 0,
            "freeze_cooldown": 0,
            "freeze_budget": 0,
        }
        commit_events: List[Dict[str, Any]] = []
        step_logs: List[Dict[str, Any]] = []
        branch_width_trace: List[int] = []

        for step in range(total_steps):
            t = total_steps - step
            in_pf_time_window = is_in_pf_time_window(t_remaining=t, total_steps=total_steps, cfg=cfg)
            cooldown_active = bool(pf_cooldown_remaining > 0)
            allow_pf_step = bool(in_pf_time_window and not cooldown_active)
            allow_kl_step = bool(cfg.influence_enabled and in_pf_time_window)
            pf_particles_step = resolve_pf_particles_for_t(
                t_remaining=t,
                total_steps=total_steps,
                cfg=cfg,
                t_start=pf_t_start,
                t_end=pf_t_end,
            )
            branch_width_trace.append(1)
            step_extra_before_pf = extra_forwards_pf

            logits, aux = self._sampler_step_with_aux(
                sampler=sampler,
                latents=state.latents,
                committed_mask=state.committed_mask,
                forced_tokens=state.forced_tokens,
            )
            baseline_forwards += 1
            state.latents = logits

            uncommitted = np.where(~state.committed_mask)[0]
            if len(uncommitted) == 0:
                step_logs.append(
                    {
                        "step": step,
                        "t": t,
                        "allow_pf_phase": bool(in_pf_time_window),
                        "allow_pf_step": bool(allow_pf_step),
                        "allow_kl_step": bool(allow_kl_step),
                        "cooldown_active": bool(cooldown_active),
                        "cooldown_remaining_start": int(pf_cooldown_remaining),
                        "cooldown_remaining_end": int(pf_cooldown_remaining),
                        "pf_triggered_this_step": False,
                        "pf_phase_ratio_active": False,
                        "pf_time_window_mode": str(cfg.pf_time_window_mode),
                        "pf_time_window_start_t": int(pf_t_start),
                        "pf_time_window_end_t": int(pf_t_end),
                        "in_pf_time_window": bool(in_pf_time_window),
                        "pf_particles_step": int(pf_particles_step),
                        "tau_low": 0.0,
                        "tau_high": 0.0,
                        "entropy_min": 0.0,
                        "entropy_mean": 0.0,
                        "entropy_max": 0.0,
                        "entropy_histogram": _entropy_histogram({}),
                        "uncommitted_mask_size": 0,
                        "valid_mask_size": 0,
                        "invalid_mask_size": 0,
                        "influence_target_mask_size": 0,
                        "influence_compute_count": 0,
                        "high_risk_mask_size": 0,
                        "mid_risk_mask_size": 0,
                        "low_risk_mask_size": 0,
                        "risk_band_histogram": {"low": 0, "mid": 0, "high": 0},
                        "risk_triggers": [],
                        "step_commit_budget": 0,
                        "influence_targets": [],
                        "influence_target_tokens": [],
                        "influence_top_positions": [],
                        "influence_top_affected": {},
                        "pf_decisions": {},
                        "action_counts": {
                            "commit_argmax": 0,
                            "commit_pf": 0,
                            "commit_fallback_max_prob": 0,
                            "freeze_delay": 0,
                            "freeze_cooldown": 0,
                            "freeze_budget": 0,
                        },
                        "token_replacement_count": 0,
                        "resample_count": 0,
                        "branch_count_before": 1,
                        "branch_count_after": 1,
                        "branch_pruned_count": 0,
                        "raw_entropy": [],
                        "raw_influence": [],
                        "extra_forwards_influence_step": 0,
                        "extra_forwards_pf_step": 0,
                        "extra_forwards_step_total": 0,
                    }
                )
                break

            if cfg.dynamic_valid_mask_enabled:
                valid_positions, _ = dynamic_valid_positions(
                    logits=logits,
                    candidate_positions=(int(idx) for idx in uncommitted),
                    committed_mask=state.committed_mask,
                    max_prob_threshold=cfg.valid_mask_max_prob,
                    eot_token_id=eot_token_id,
                    exclude_eot=cfg.valid_mask_exclude_eot,
                )
            else:
                valid_positions = [int(idx) for idx in uncommitted]
            valid_set = set(valid_positions)

            entropies: Dict[int, float] = {
                int(idx): entropy_at_position(logits, int(idx)) for idx in valid_positions
            }

            attention = aux.get("last_attention") if isinstance(aux, dict) else None
            if allow_kl_step and cfg.use_attention_influence_proxy:
                influences_raw = attention_influence_proxy(
                    attention=attention,
                    committed_mask=state.committed_mask,
                    candidate_positions=valid_positions,
                )
            else:
                influences_raw = {int(idx): 0.0 for idx in valid_positions}
            influences = normalize_influence_scores(influences_raw)

            influence_compute_count = int(len(valid_positions)) if allow_kl_step else 0
            influence_compute_total += influence_compute_count

            local_beam_risks = []
            local_beam_trigger = None
            if (
                bool(getattr(cfg, "local_beam_enabled", False))
                and prev_logits_for_kl is not None
                and in_pf_time_window
                and valid_positions
            ):
                try:
                    local_beam_risks = compute_local_beam_risks(
                        logits=logits,
                        prev_logits=prev_logits_for_kl,
                        candidate_positions=valid_positions,
                        committed_mask=state.committed_mask,
                        cfg=cfg,
                        token_ids_to_code=token_ids_to_code,
                        forced_tokens=state.forced_tokens,
                        sampler=sampler,
                    )
                    local_beam_trigger = select_local_beam_trigger(
                        risks=local_beam_risks,
                        cfg=cfg,
                        branch_events=local_beam_branch_events,
                    )
                except Exception as exc:
                    local_beam_event_logs.append(
                        {
                            "step": int(step),
                            "t": int(t),
                            "event": "risk_compute_error",
                            "error": f"{exc}",
                        }
                    )
                    local_beam_risks = []
                    local_beam_trigger = None

            risks: Dict[int, float] = {}
            for idx in uncommitted:
                idx = int(idx)
                if idx in valid_set:
                    risks[idx] = risk_score(
                        entropy=entropies[idx],
                        influence=influences.get(idx, 0.0),
                        cfg=cfg,
                        running_stats=None,
                        use_entropy=True,
                        use_influence=cfg.influence_enabled,
                    )
                else:
                    risks[idx] = 0.0

            tau_low, tau_high = dynamic_risk_thresholds(
                (risks[int(i)] for i in valid_positions),
                low_q=cfg.risk_low_quantile,
                high_q=cfg.risk_high_quantile,
                min_gap=cfg.risk_threshold_min_gap,
            )
            if len(uncommitted) > 0:
                masked_risk_vals = np.asarray([float(risks.get(int(i), 0.0)) for i in uncommitted], dtype=np.float64)
                risk_mean = float(masked_risk_vals.mean())
                risk_std = float(masked_risk_vals.std())
                risk_gradient_gate = float(risk_mean + pf_risk_gradient_sigma * risk_std)
            else:
                risk_mean = 0.0
                risk_std = 0.0
                risk_gradient_gate = 0.0
            pf_trigger_gate = float(max(float(tau_high), float(risk_gradient_gate)))

            ranked_positions = sorted((int(i) for i in uncommitted), key=lambda i: risks.get(i, 0.0), reverse=True)
            pf_decisions: Dict[int, Dict[str, Any]] = {}
            action_counts = {
                "commit_argmax": 0,
                "commit_pf": 0,
                "commit_fallback_max_prob": 0,
                "freeze_delay": 0,
                "freeze_cooldown": 0,
                "freeze_budget": 0,
            }
            token_replacement_count = 0
            resample_count = 0
            high_risk_positions: List[int] = []
            pf_triggered_this_step = False

            for idx in ranked_positions:
                idx = int(idx)
                argmax_token = int(np.argmax(logits[idx]))
                risk_val = float(risks.get(idx, 0.0))

                if local_beam_trigger is not None and idx == int(local_beam_trigger.pos):
                    try:
                        lb_result = run_commit_timing_local_beam(
                            logits=logits,
                            prev_logits=prev_logits_for_kl,
                            risk=local_beam_trigger,
                            committed_mask=state.committed_mask,
                            forced_tokens=state.forced_tokens,
                            cfg=cfg,
                            sampler=sampler,
                            latents=state.latents,
                            token_ids_to_code=token_ids_to_code,
                            source_prefix=source_prefix,
                        )
                        local_beam_branch_events += 1
                        local_beam_extra_forwards += int(lb_result.extra_forwards)
                        extra_forwards_pf += int(lb_result.extra_forwards)
                        local_beam_beam_sizes.append(int(len(lb_result.particle_logs)))
                        if bool(lb_result.accepted_alternative):
                            local_beam_accepted_alternatives += 1
                        if bool(lb_result.delay_selected):
                            local_beam_delay_count += 1
                        pf_triggered_this_step = True
                        risk_trigger_count += 1
                        high_risk_positions.append(idx)

                        selected_token = lb_result.selected_token
                        if selected_token is None:
                            state.delay_count[idx] += 1
                            action_counts["freeze_delay"] += 1
                        else:
                            self._apply_commit(state=state, idx=idx, token=int(selected_token), logits=logits)
                            if bool(lb_result.accepted_alternative):
                                action_counts["commit_pf"] += 1
                            else:
                                action_counts["commit_argmax"] += 1
                            if int(selected_token) != argmax_token:
                                token_replacement_count += 1
                            commit_events.append(
                                {
                                    "step": step,
                                    "t": t,
                                    "pos": idx,
                                    "token": int(selected_token),
                                    "reason": f"local_beam_{lb_result.reason}",
                                    "risk": float(local_beam_trigger.risk),
                                    "entropy": float(local_beam_trigger.entropy_norm),
                                    "influence": float(local_beam_trigger.kl_norm),
                                    "temporal_kl": float(local_beam_trigger.kl_raw),
                                    "structure_score": float(local_beam_trigger.structure_score),
                                    "argmax_token": int(argmax_token),
                                    "token_replaced": bool(int(selected_token) != argmax_token),
                                    "branch_idx": int(lb_result.selected_particle_index),
                                }
                            )
                            resample_count += 1

                        event_log = {
                            "step": int(step),
                            "t": int(t),
                            "pos": int(idx),
                            "entropy_norm": float(local_beam_trigger.entropy_norm),
                            "kl_raw": float(local_beam_trigger.kl_raw),
                            "kl_norm": float(local_beam_trigger.kl_norm),
                            "risk": float(local_beam_trigger.risk),
                            "structure_score": float(local_beam_trigger.structure_score),
                            "top_tokens": [int(tok) for tok in local_beam_trigger.top_tokens],
                            "top_token_texts": list(local_beam_trigger.top_token_texts),
                            "particle_scores": [
                                float(log.get("score", 0.0)) for log in lb_result.particle_logs
                            ],
                            "particle_logs": lb_result.particle_logs,
                            "selected_particle": int(lb_result.selected_particle_index),
                            "selected_kind": str(lb_result.selected_kind),
                            "selected_token": int(selected_token) if selected_token is not None else None,
                            "baseline_preserved": bool(lb_result.baseline_preserved),
                            "alternative_replaced_baseline": bool(lb_result.accepted_alternative),
                            "extra_forwards": int(lb_result.extra_forwards),
                            "reason": str(lb_result.reason),
                        }
                        local_beam_event_logs.append(event_log)
                        pf_decisions[idx] = {
                            "reason": f"local_beam_{lb_result.reason}",
                            "argmax_token": int(argmax_token),
                            "chosen_token": int(selected_token) if selected_token is not None else None,
                            "token_replaced": bool(selected_token is not None and int(selected_token) != argmax_token),
                            "risk": float(local_beam_trigger.risk),
                            "entropy_norm": float(local_beam_trigger.entropy_norm),
                            "temporal_kl_norm": float(local_beam_trigger.kl_norm),
                            "structure_score": float(local_beam_trigger.structure_score),
                            "candidate_count": int(len(lb_result.particle_logs)),
                            "local_beam": event_log,
                        }
                    except Exception as exc:
                        state.delay_count[idx] += 1
                        action_counts["freeze_delay"] += 1
                        pf_decisions[idx] = {
                            "reason": "local_beam_error_remask",
                            "argmax_token": int(argmax_token),
                            "chosen_token": None,
                            "token_replaced": False,
                            "risk": risk_val,
                            "local_beam_error": f"{exc}",
                        }
                    continue

                if idx not in valid_set or risk_val < tau_low:
                    self._apply_commit(state=state, idx=idx, token=argmax_token, logits=logits)
                    action_counts["commit_argmax"] += 1
                    commit_events.append(
                        {
                            "step": step,
                            "t": t,
                            "pos": idx,
                            "token": int(argmax_token),
                            "reason": "freeze_low_risk",
                            "risk": risk_val,
                            "entropy": float(entropies.get(idx, 0.0)),
                            "influence": float(influences.get(idx, 0.0)),
                            "argmax_token": int(argmax_token),
                            "token_replaced": False,
                            "branch_idx": 0,
                        }
                    )
                    pf_decisions[idx] = {
                        "reason": "freeze_low_risk",
                        "argmax_token": int(argmax_token),
                        "chosen_token": int(argmax_token),
                        "token_replaced": False,
                        "risk": risk_val,
                    }
                    continue

                if not in_pf_time_window:
                    state.delay_count[idx] += 1
                    action_counts["freeze_delay"] += 1
                    pf_decisions[idx] = {
                        "reason": "remask_outside_pf_time_window",
                        "argmax_token": int(argmax_token),
                        "chosen_token": None,
                        "token_replaced": False,
                        "risk": risk_val,
                    }
                    continue

                if cooldown_active:
                    state.delay_count[idx] += 1
                    action_counts["freeze_cooldown"] += 1
                    pf_decisions[idx] = {
                        "reason": "remask_pf_cooldown",
                        "argmax_token": int(argmax_token),
                        "chosen_token": None,
                        "token_replaced": False,
                        "risk": risk_val,
                    }
                    continue

                if risk_val >= tau_high and cfg.pf_enabled:
                    if pf_trigger_limit > 0 and pf_trigger_count >= pf_trigger_limit:
                        state.delay_count[idx] += 1
                        action_counts["freeze_budget"] += 1
                        pf_decisions[idx] = {
                            "reason": "remask_pf_trigger_cap",
                            "argmax_token": int(argmax_token),
                            "chosen_token": None,
                            "token_replaced": False,
                            "risk": risk_val,
                            "pf_trigger_limit": int(pf_trigger_limit),
                        }
                        continue
                    if risk_val < risk_gradient_gate:
                        state.delay_count[idx] += 1
                        action_counts["freeze_delay"] += 1
                        pf_decisions[idx] = {
                            "reason": "remask_below_risk_gradient_gate",
                            "argmax_token": int(argmax_token),
                            "chosen_token": None,
                            "token_replaced": False,
                            "risk": risk_val,
                        }
                        continue
                    if risk_val < pf_trigger_gate:
                        state.delay_count[idx] += 1
                        action_counts["freeze_delay"] += 1
                        pf_decisions[idx] = {
                            "reason": "remask_below_pf_quantile",
                            "argmax_token": int(argmax_token),
                            "chosen_token": None,
                            "token_replaced": False,
                            "risk": risk_val,
                        }
                        continue
                    chosen, pf_logs = run_local_pf(
                        logits=logits,
                        pos=idx,
                        committed_mask=state.committed_mask,
                        forced_tokens=state.forced_tokens,
                        cfg=cfg,
                        sampler=sampler,
                        latents=state.latents,
                        token_ids_to_code=token_ids_to_code,
                        force_fallback=False,
                        pf_particles_override=int(pf_particles_step),
                        current_t=t,
                        pf_window=(pf_t_start, pf_t_end),
                        source_prefix=source_prefix,
                    )
                    pf_forward_calls = int(
                        sum(int(log.get("lookahead_forward_calls_total", 0)) for log in pf_logs)
                    )
                    extra_forwards_pf += pf_forward_calls

                    pf_trigger_count += 1
                    risk_trigger_count += 1
                    high_risk_positions.append(idx)
                    pf_triggered_this_step = True

                    if chosen is not None:
                        self._apply_commit(state=state, idx=idx, token=int(chosen), logits=logits)
                        action_counts["commit_pf"] += 1
                        if int(chosen) != argmax_token:
                            token_replacement_count += 1
                        commit_events.append(
                            {
                                "step": step,
                                "t": t,
                                "pos": idx,
                                "token": int(chosen),
                                "reason": "trigger_pf",
                                "risk": risk_val,
                                "entropy": float(entropies.get(idx, 0.0)),
                                "influence": float(influences.get(idx, 0.0)),
                                "argmax_token": int(argmax_token),
                                "token_replaced": bool(int(chosen) != argmax_token),
                                "branch_idx": 0,
                            }
                        )
                        resample_count += 1
                    else:
                        action_counts["freeze_delay"] += 1

                    pf_decisions[idx] = {
                        "reason": "trigger_pf",
                        "argmax_token": int(argmax_token),
                        "chosen_token": int(chosen) if chosen is not None else None,
                        "token_replaced": bool(chosen is not None and int(chosen) != argmax_token),
                        "risk": risk_val,
                        "candidate_count": int(pf_particles_step),
                    }
                    continue

                state.delay_count[idx] += 1
                action_counts["freeze_delay"] += 1
                pf_decisions[idx] = {
                    "reason": "remask_late_mid_risk",
                    "argmax_token": int(argmax_token),
                    "chosen_token": None,
                    "token_replaced": False,
                    "risk": risk_val,
                }

            low_count = 0
            mid_count = 0
            high_count = 0
            for idx in uncommitted:
                idx = int(idx)
                rv = float(risks.get(idx, 0.0))
                if rv < tau_low:
                    low_count += 1
                elif rv >= tau_high:
                    high_count += 1
                else:
                    mid_count += 1

            cooldown_start = int(pf_cooldown_remaining)
            if pf_triggered_this_step and pf_cooldown_steps > 0:
                pf_cooldown_remaining = int(pf_cooldown_steps)
            elif pf_cooldown_remaining > 0:
                pf_cooldown_remaining = int(pf_cooldown_remaining - 1)
            cooldown_end = int(pf_cooldown_remaining)

            influence_sorted = sorted(influences_raw.items(), key=lambda item: float(item[1]), reverse=True)
            influence_top_positions = [
                {
                    "pos": int(i),
                    "influence": float(v),
                    "argmax_token": int(np.argmax(logits[int(i)])),
                }
                for i, v in influence_sorted[:10]
            ]
            influence_target_tokens = [
                {"pos": int(i), "argmax_token": int(np.argmax(logits[int(i)]))}
                for i in valid_positions
            ]

            step_log = {
                "step": step,
                "t": t,
                "allow_pf_phase": bool(in_pf_time_window),
                "allow_pf_step": bool(allow_pf_step),
                "allow_kl_step": bool(allow_kl_step),
                "cooldown_active": bool(cooldown_active),
                "cooldown_remaining_start": int(cooldown_start),
                "cooldown_remaining_end": int(cooldown_end),
                "pf_triggered_this_step": bool(pf_triggered_this_step),
                "pf_phase_ratio_active": False,
                "pf_time_window_mode": str(cfg.pf_time_window_mode),
                "pf_time_window_start_t": int(pf_t_start),
                "pf_time_window_end_t": int(pf_t_end),
                "in_pf_time_window": bool(in_pf_time_window),
                "pf_particles_step": int(pf_particles_step),
                "tau_low": float(tau_low),
                "tau_high": float(tau_high),
                "risk_mean": float(risk_mean),
                "risk_std": float(risk_std),
                "risk_gradient_sigma": float(pf_risk_gradient_sigma),
                "risk_gradient_gate": float(risk_gradient_gate),
                "pf_trigger_gate": float(pf_trigger_gate),
                "entropy_min": float(min(entropies.values())) if entropies else 0.0,
                "entropy_mean": float(np.mean(list(entropies.values()))) if entropies else 0.0,
                "entropy_max": float(max(entropies.values())) if entropies else 0.0,
                "entropy_histogram": _entropy_histogram(entropies),
                "uncommitted_mask_size": int(len(uncommitted)),
                "valid_mask_size": int(len(valid_positions)),
                "invalid_mask_size": int(len(uncommitted) - len(valid_positions)),
                "influence_target_mask_size": int(len(valid_positions)),
                "influence_compute_count": int(influence_compute_count),
                "high_risk_mask_size": int(high_count),
                "mid_risk_mask_size": int(mid_count),
                "low_risk_mask_size": int(low_count),
                "risk_band_histogram": {"low": int(low_count), "mid": int(mid_count), "high": int(high_count)},
                "risk_triggers": [int(i) for i in high_risk_positions],
                "step_commit_budget": int(len(uncommitted)),
                "influence_targets": [int(i) for i in valid_positions],
                "influence_target_tokens": influence_target_tokens,
                "influence_top_positions": influence_top_positions,
                "influence_top_affected": {},
                "pf_decisions": pf_decisions,
                "action_counts": action_counts,
                "token_replacement_count": int(token_replacement_count),
                "resample_count": int(resample_count),
                "branch_count_before": 1,
                "branch_count_after": 1,
                "branch_pruned_count": 0,
                "raw_entropy": [
                    {
                        "pos": int(i),
                        "value": float(entropies[int(i)]),
                        "argmax_token": int(np.argmax(logits[int(i)])),
                    }
                    for i in sorted(entropies.keys())
                ],
                "raw_influence": [
                    {
                        "pos": int(i),
                        "value": float(influences_raw.get(int(i), 0.0)),
                        "argmax_token": int(np.argmax(logits[int(i)])),
                    }
                    for i in sorted(entropies.keys())
                ],
                "local_beam_enabled": bool(getattr(cfg, "local_beam_enabled", False)),
                "local_beam_triggered_this_step": bool(
                    local_beam_trigger is not None and any(int(p) == int(local_beam_trigger.pos) for p in high_risk_positions)
                ),
                "local_beam_risk_top": [
                    {
                        "pos": int(cand.pos),
                        "entropy_norm": float(cand.entropy_norm),
                        "kl_raw": float(cand.kl_raw),
                        "kl_norm": float(cand.kl_norm),
                        "structure_score": float(cand.structure_score),
                        "risk": float(cand.risk),
                        "top_tokens": [int(tok) for tok in cand.top_tokens],
                        "top_token_texts": list(cand.top_token_texts),
                    }
                    for cand in local_beam_risks[:5]
                ],
                "extra_forwards_influence_step": 0,
                "extra_forwards_pf_step": int(extra_forwards_pf - step_extra_before_pf),
                "extra_forwards_step_total": int(extra_forwards_pf - step_extra_before_pf),
            }
            step_logs.append(step_log)

            for action_name in action_counts_total:
                action_counts_total[action_name] += int(action_counts[action_name])

            if cfg.logging_level == "DEBUG":
                logger.debug(
                    "step=%s t=%s uncommitted=%s tau_low=%.4f tau_high=%.4f commits(argmax=%s,pf=%s)",
                    step,
                    t,
                    len(uncommitted),
                    tau_low,
                    tau_high,
                    action_counts["commit_argmax"],
                    action_counts["commit_pf"],
                )

            if bool(np.all(state.committed_mask)):
                break
            prev_logits_for_kl = _clone_logits_for_temporal_kl(logits)

        # Final safety collapse: commit remaining masks with current argmax.
        remaining = np.where(~state.committed_mask)[0]
        if remaining.size > 0:
            logits, _ = self._sampler_step_with_aux(
                sampler=sampler,
                latents=state.latents,
                committed_mask=state.committed_mask,
                forced_tokens=state.forced_tokens,
            )
            baseline_forwards += 1
            state.latents = logits
            for idx in remaining.tolist():
                tok = int(np.argmax(logits[int(idx)]))
                self._apply_commit(state=state, idx=int(idx), token=tok, logits=logits)
                action_counts_total["commit_fallback_max_prob"] += 1
                commit_events.append(
                    {
                        "step": int(total_steps),
                        "t": 0,
                        "pos": int(idx),
                        "token": int(tok),
                        "reason": "final_fallback_argmax",
                        "risk": 0.0,
                        "entropy": 0.0,
                        "influence": 0.0,
                        "argmax_token": int(tok),
                        "token_replaced": False,
                        "branch_idx": 0,
                    }
                )

        text = self._decode_tokens(state.forced_tokens, token_ids_to_code)
        extra_forwards_total = int(extra_forwards_influence + extra_forwards_pf)
        avg_branch_width = 1.0
        latency_ratio_vs_baseline = float(extra_forwards_total / max(baseline_forwards, 1) + 1.0)

        stats = {
            "pf_trigger_count": int(pf_trigger_count),
            "extra_forwards": int(extra_forwards_total),
            "extra_forwards_influence": int(extra_forwards_influence),
            "extra_forwards_pf": int(extra_forwards_pf),
            "extra_forwards_local_beam": int(local_beam_extra_forwards),
            "risk_trigger_count": int(risk_trigger_count),
            "influence_compute_count": int(influence_compute_total),
            "n_commits": int(len(commit_events)),
            "action_counts_total": action_counts_total,
            "step_logs": step_logs,
            "commit_events": commit_events,
            "avg_branch_width": avg_branch_width,
            "max_branch_width": 1,
            "baseline_forwards": int(baseline_forwards),
            "latency_ratio_vs_baseline": latency_ratio_vs_baseline,
            "branch_width_trace": [1 for _ in step_logs] if step_logs else [1],
            "diffusion_mid_step": int(t_mid),
            "pf_time_window_mode": str(cfg.pf_time_window_mode),
            "pf_time_window_start_t": int(pf_t_start),
            "pf_time_window_end_t": int(pf_t_end),
            "local_beam_enabled": bool(getattr(cfg, "local_beam_enabled", False)),
            "local_beam_branch_events": int(local_beam_branch_events),
            "local_beam_accepted_alternatives": int(local_beam_accepted_alternatives),
            "local_beam_delay_count": int(local_beam_delay_count),
            "local_beam_avg_beam_size": float(np.mean(local_beam_beam_sizes)) if local_beam_beam_sizes else 0.0,
            "local_beam_event_logs": local_beam_event_logs,
        }
        return text, stats
