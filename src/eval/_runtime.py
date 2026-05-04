"""Evaluation runtime implementation shared by the CLI and compatibility exports."""
import argparse
import ast
from collections import Counter
import copy
import random
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as script from repo root or from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from decoding import (
    DecoderConfig,
    DreamSamplerAdapter,
    LLaDASamplerAdapter,
    PlaceholderDiffusionSampler,
    RiskAwarePFDecoder,
    load_dream_model,
    load_llada_model,
)
from decoding.calibration import apply_warmup_calibration, save_calibration
from decoding.config import (
    ablation_delay_no_pf,
    ablation_entropy_only,
    ablation_influence_only,
    ablation_local_beam,
    ablation_no_pf,
    ablation_pf_no_delay,
    fast_baseline,
    resolve_pf_particles_for_t,
    resolve_pf_time_window,
)
from decoding.pf import parser_feedback_from_source, parser_feedback_from_tokens, run_local_pf
from decoding.local_beam import (
    compute_local_beam_risks,
    run_commit_timing_local_beam,
    select_local_beam_trigger,
)
from decoding.risk import (
    attention_influence_proxy,
    compute_influence,
    dynamic_risk_thresholds,
    dynamic_valid_positions,
    entropy_at_position,
    normalize_influence_scores,
    risk_score,
)

logger = logging.getLogger(__name__)

_DEFAULT_PF_EXTRA_FORWARD_BUDGET = 0
_REPAIR_POLICY_NONE = "none"
_REPAIR_POLICY_ROLLBACK_ONLY = "rollback_only"
_REPAIR_POLICY_PF_RB = "pf_rb"


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
    except Exception:
        pass
    try:
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        cuda_states = state.get("torch_cuda_all")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception:
        pass


def _first_text_diff(a: str, b: str, context: int = 24) -> Dict[str, Any]:
    left = str(a or "")
    right = str(b or "")
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    if idx == limit and len(left) == len(right):
        return {"match": True, "index": -1}
    start = max(idx - int(context), 0)
    end_left = min(idx + int(context), len(left))
    end_right = min(idx + int(context), len(right))
    return {
        "match": False,
        "index": int(idx),
        "left_char": left[idx] if idx < len(left) else "",
        "right_char": right[idx] if idx < len(right) else "",
        "left_context": left[start:end_left],
        "right_context": right[start:end_right],
        "left_len": int(len(left)),
        "right_len": int(len(right)),
    }


def _first_token_diff(
    tokenizer: Any,
    a: str,
    b: str,
    context: int = 8,
) -> Dict[str, Any]:
    if tokenizer is None:
        return {"available": False, "reason": "missing_tokenizer"}
    try:
        left_ids = tokenizer.encode(str(a or ""), add_special_tokens=False)
        right_ids = tokenizer.encode(str(b or ""), add_special_tokens=False)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    limit = min(len(left_ids), len(right_ids))
    idx = 0
    while idx < limit and int(left_ids[idx]) == int(right_ids[idx]):
        idx += 1
    if idx == limit and len(left_ids) == len(right_ids):
        return {"available": True, "match": True, "index": -1}
    start = max(idx - int(context), 0)
    end_left = min(idx + int(context), len(left_ids))
    end_right = min(idx + int(context), len(right_ids))
    return {
        "available": True,
        "match": False,
        "index": int(idx),
        "left_token_id": int(left_ids[idx]) if idx < len(left_ids) else None,
        "right_token_id": int(right_ids[idx]) if idx < len(right_ids) else None,
        "left_context_ids": [int(v) for v in left_ids[start:end_left]],
        "right_context_ids": [int(v) for v in right_ids[start:end_right]],
        "left_len": int(len(left_ids)),
        "right_len": int(len(right_ids)),
    }


@dataclass
class _DreamBranchState:
    branch_id: int
    owned_mask: Any
    owned_tokens: Any
    score: float
    ttl: int
    origin_t: int
    source_pos: int
    chosen_token: int
    parse_quality: float = 1.0
    severity: float = 0.0
    future_entropy: float = 0.0
    lookahead_confidence: float = 0.0
    eval_score: float = 0.0
    extensions_used: int = 0


def _compose_branch_completion_state(
    base_tokens: Any,
    base_committed_mask: Any,
    branch: _DreamBranchState,
) -> Tuple[Any, Any]:
    torch = __import__("torch")
    comp_ids_t = base_tokens.clone()
    committed_t = base_committed_mask.clone()
    owned_mask_t = branch.owned_mask.to(device=comp_ids_t.device, dtype=torch.bool).reshape(-1)
    owned_tokens_t = branch.owned_tokens.to(device=comp_ids_t.device, dtype=torch.long).reshape(-1)
    lim = min(int(comp_ids_t.numel()), int(committed_t.numel()), int(owned_mask_t.numel()), int(owned_tokens_t.numel()))
    if lim <= 0:
        return comp_ids_t, committed_t
    mask_t = owned_mask_t[:lim]
    if bool(torch.any(mask_t).item()):
        comp_ids_t[:lim] = torch.where(mask_t, owned_tokens_t[:lim], comp_ids_t[:lim])
        committed_t[:lim] = torch.where(mask_t, torch.ones_like(committed_t[:lim]), committed_t[:lim])
    return comp_ids_t, committed_t


def _dream_branch_state_key(branch: _DreamBranchState) -> Tuple[Tuple[int, int], ...]:
    torch = __import__("torch")
    owned_mask_t = branch.owned_mask.to(dtype=torch.bool).reshape(-1)
    owned_tokens_t = branch.owned_tokens.to(dtype=torch.long).reshape(-1)
    lim = min(int(owned_mask_t.numel()), int(owned_tokens_t.numel()))
    if lim <= 0:
        return tuple()
    idx_t = torch.nonzero(owned_mask_t[:lim], as_tuple=False).flatten()
    if int(idx_t.numel()) <= 0:
        return tuple()
    return tuple((int(i.item()), int(owned_tokens_t[int(i.item())].item())) for i in idx_t)


def _prune_dream_branch_bank(
    branches: List[_DreamBranchState],
    beam_width: int,
) -> Tuple[List[_DreamBranchState], int]:
    if not branches:
        return [], 0
    kept: List[_DreamBranchState] = []
    seen_keys = set()
    ordered = sorted(branches, key=lambda b: float(max(b.eval_score, b.score)), reverse=True)
    for branch in ordered:
        key = _dream_branch_state_key(branch)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(branch)
        if len(kept) >= max(int(beam_width), 1):
            break
    pruned = max(len(branches) - len(kept), 0)
    return kept, pruned


def _advance_dream_branch_trials(branches: List[_DreamBranchState]) -> Tuple[List[_DreamBranchState], bool]:
    if not branches:
        return [], False
    active: List[_DreamBranchState] = []
    merge_due = False
    for branch in branches:
        branch.ttl = max(int(branch.ttl) - 1, 0)
        if int(branch.ttl) <= 0:
            merge_due = True
        active.append(branch)
    return active, bool(merge_due)


def _select_best_dream_branch_entry(
    evaluated_branches: List[Tuple[_DreamBranchState, Any, Any, Any]],
) -> Optional[Tuple[_DreamBranchState, Any, Any, Any]]:
    if not evaluated_branches:
        return None
    return max(
        evaluated_branches,
        key=lambda entry: float(max(entry[0].eval_score, entry[0].score)),
    )


def _should_extend_dream_branches(
    ranked_entries: List[Tuple[_DreamBranchState, Any, Any, Any]],
    cfg: DecoderConfig,
) -> bool:
    if not bool(getattr(cfg, "branch_extension_enabled", False)):
        return False
    if len(ranked_entries) < 2:
        return False
    max_ext = int(getattr(cfg, "branch_max_extensions", 0))
    if max_ext <= 0:
        return False
    if any(int(entry[0].extensions_used) >= max_ext for entry in ranked_entries[:2]):
        return False
    best_score = float(max(ranked_entries[0][0].eval_score, ranked_entries[0][0].score))
    second_score = float(max(ranked_entries[1][0].eval_score, ranked_entries[1][0].score))
    gap = float(best_score - second_score)
    return gap < float(getattr(cfg, "branch_extension_margin", 0.0))


def _build_joint_branch_candidates(
    particle_groups: List[Tuple[int, List[Any]]],
    base_owned_mask_t: Any,
    base_owned_tokens_t: Any,
    base_tokens_t: Any,
    base_committed_t: Any,
    parent_score: float,
    next_branch_id: int,
    t_remaining: int,
    beam_width: int,
    trial_steps: int,
) -> List[_DreamBranchState]:
    import itertools
    torch = __import__("torch")

    if not particle_groups:
        return []

    combos = list(itertools.product(*[group[1] for group in particle_groups if group[1]]))
    if not combos:
        return []

    candidates: List[_DreamBranchState] = []
    sorted_combos = sorted(
        combos,
        key=lambda combo: float(parent_score + sum(float(p.score) for p in combo)),
        reverse=True,
    )
    for combo_idx, combo in enumerate(sorted_combos[: max(int(beam_width) * 2, int(beam_width), 1)]):
        owned_mask_t = base_owned_mask_t.clone()
        owned_tokens_t = base_owned_tokens_t.clone()
        combo_score = float(parent_score)
        parse_quality = 1.0
        severity = 0.0
        lookahead_conf = 0.0
        source_positions: List[int] = []
        for particle in combo:
            particle_tokens_t = particle.forced_tokens.to(device=base_tokens_t.device, dtype=torch.long).reshape(-1)
            particle_committed_t = particle.committed_mask.to(device=base_committed_t.device, dtype=torch.bool).reshape(-1)
            lim = min(
                int(owned_mask_t.numel()),
                int(owned_tokens_t.numel()),
                int(particle_tokens_t.numel()),
                int(particle_committed_t.numel()),
                int(base_tokens_t.numel()),
                int(base_committed_t.numel()),
            )
            if lim <= 0:
                continue
            delta_mask_t = particle_committed_t[:lim] & (
                (~base_committed_t[:lim]) | (particle_tokens_t[:lim] != base_tokens_t[:lim])
            )
            owned_mask_t[:lim] = owned_mask_t[:lim] | delta_mask_t
            owned_tokens_t[:lim] = torch.where(delta_mask_t, particle_tokens_t[:lim], owned_tokens_t[:lim])
            combo_score += float(particle.score)
            parse_quality = min(parse_quality, float(particle.syntax_candidate_quality))
            severity = max(severity, float(max(1.0 - float(particle.syntax_candidate_quality), 0.0)))
            lookahead_conf += float(particle.lookahead_confidence)
            source_positions.append(int(particle.source_pos) if hasattr(particle, "source_pos") else int(-1))

        if not bool(torch.any(owned_mask_t).item()):
            continue

        source_pos = next((pos for pos in source_positions if pos >= 0), -1)
        chosen_token = int(combo[0].token_at_pos) if combo else -1
        candidates.append(
            _DreamBranchState(
                branch_id=int(next_branch_id + combo_idx),
                owned_mask=owned_mask_t.clone(),
                owned_tokens=owned_tokens_t.clone(),
                score=float(combo_score),
                ttl=int(trial_steps),
                origin_t=int(t_remaining),
                source_pos=int(source_pos),
                chosen_token=int(chosen_token),
                parse_quality=float(parse_quality),
                severity=float(severity),
                future_entropy=0.0,
                lookahead_confidence=float(lookahead_conf),
                eval_score=float(combo_score),
            )
        )
    return candidates[: max(int(beam_width), 1)]


def _dream_top_p_logits(logits_t: Any, top_p: Optional[float]) -> Any:
    torch = __import__("torch")
    if top_p is None or float(top_p) >= 1.0:
        return logits_t
    sorted_logits_t, sorted_indices_t = torch.sort(logits_t, descending=True, dim=-1)
    cumulative_probs_t = torch.cumsum(torch.softmax(sorted_logits_t, dim=-1), dim=-1)
    sorted_remove_t = cumulative_probs_t > float(top_p)
    if int(sorted_remove_t.shape[-1]) > 0:
        sorted_remove_t[..., 1:] = sorted_remove_t[..., :-1].clone()
        sorted_remove_t[..., 0] = False
    remove_mask_t = torch.zeros_like(logits_t, dtype=torch.bool, device=logits_t.device)
    remove_mask_t.scatter_(-1, sorted_indices_t, sorted_remove_t)
    return logits_t.masked_fill(remove_mask_t, torch.finfo(logits_t.dtype).min)


def _dream_top_k_logits(logits_t: Any, top_k: Optional[int]) -> Any:
    torch = __import__("torch")
    if top_k is None:
        return logits_t
    k = min(max(int(top_k), 1), int(logits_t.shape[-1]))
    cutoff_t = torch.topk(logits_t, k, dim=-1).values[..., -1, None]
    return logits_t.masked_fill(logits_t < cutoff_t, torch.finfo(logits_t.dtype).min)


def _dream_sample_mask_logits(
    logits_t: Any,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> Tuple[Any, Any]:
    torch = __import__("torch")
    dists = torch.distributions
    logits_work_t = logits_t
    if float(temperature) > 0.0:
        logits_work_t = logits_work_t / float(temperature)
    logits_work_t = _dream_top_p_logits(logits_work_t, top_p=top_p)
    logits_work_t = _dream_top_k_logits(logits_work_t, top_k=top_k)
    probs_t = torch.softmax(logits_work_t, dim=-1)

    if float(temperature) > 0.0:
        try:
            x0_t = dists.Categorical(probs=probs_t).sample()
            confidence_t = torch.gather(probs_t, -1, x0_t.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence_t, x0_t = probs_t.max(dim=-1)
    else:
        confidence_t, x0_t = probs_t.max(dim=-1)

    if margin_confidence:
        sorted_probs_t, _ = torch.sort(probs_t, dim=-1, descending=True)
        confidence_t = sorted_probs_t[..., 0] - sorted_probs_t[..., 1]
    elif neg_entropy:
        log_probs_t = torch.log(probs_t.clamp_min(1e-10))
        confidence_t = torch.sum(probs_t * log_probs_t, dim=-1)

    return confidence_t, x0_t


def _dream_sample_mask_logits_chunked(
    logits_t: Any,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    chunk_size: int = 64,
) -> Tuple[Any, Any]:
    torch = __import__("torch")
    if int(getattr(logits_t, "ndim", 0)) != 2:
        return _dream_sample_mask_logits(
            logits_t,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=margin_confidence,
            neg_entropy=neg_entropy,
        )
    total_rows = int(logits_t.shape[0])
    if total_rows <= max(int(chunk_size), 1):
        return _dream_sample_mask_logits(
            logits_t,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=margin_confidence,
            neg_entropy=neg_entropy,
        )

    conf_chunks: List[Any] = []
    token_chunks: List[Any] = []
    for start in range(0, total_rows, max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), total_rows)
        conf_t, tok_t = _dream_sample_mask_logits(
            logits_t[start:stop],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=margin_confidence,
            neg_entropy=neg_entropy,
        )
        conf_chunks.append(conf_t)
        token_chunks.append(tok_t)
    return torch.cat(conf_chunks, dim=0), torch.cat(token_chunks, dim=0)


def _prepare_dream_diffusion_state(
    input_ids_t: Any,
    attention_mask_t: Optional[Any],
    max_new_tokens: int,
    mask_token_id: int,
) -> Tuple[Any, Any, Optional[Any]]:
    torch = __import__("torch")
    F = torch.nn.functional
    max_length = int(input_ids_t.shape[1]) + int(max_new_tokens)
    x_t = F.pad(input_ids_t, (0, max_length - int(input_ids_t.shape[1])), value=int(mask_token_id))
    if attention_mask_t is not None and bool(torch.any(attention_mask_t == 0).item()):
        loop_attn_t = F.pad(attention_mask_t, (0, max_length - int(attention_mask_t.shape[1])), value=1.0)
        tok_idx_t = loop_attn_t.long().cumsum(-1) - 1
        tok_idx_t.masked_fill_(loop_attn_t == 0, 1)
        loop_attn_t = torch.logical_and(
            loop_attn_t.unsqueeze(1).unsqueeze(-2),
            loop_attn_t.unsqueeze(1).unsqueeze(-1),
        )
        return x_t, loop_attn_t, tok_idx_t
    return x_t, "full", None


def _expand_dream_loop_context(
    attention_mask_t: Any,
    tok_idx_t: Optional[Any],
    batch_size: int,
) -> Tuple[Any, Optional[Any]]:
    if isinstance(attention_mask_t, str):
        expanded_attn = attention_mask_t
    else:
        expanded_attn = attention_mask_t
        if int(getattr(expanded_attn, "shape", [0])[0]) == 1 and int(batch_size) > 1:
            expand_shape = (int(batch_size),) + tuple(int(v) for v in expanded_attn.shape[1:])
            expanded_attn = expanded_attn.expand(expand_shape)
    expanded_tok_idx = tok_idx_t
    if tok_idx_t is not None and int(getattr(tok_idx_t, "shape", [0])[0]) == 1 and int(batch_size) > 1:
        expanded_tok_idx = tok_idx_t.expand(int(batch_size), -1)
    return expanded_attn, expanded_tok_idx


def _dream_forward_step_logits(
    model: Any,
    x_t: Any,
    attention_mask_t: Any,
    tok_idx_t: Optional[Any],
) -> Any:
    torch = __import__("torch")
    out_t = model(x_t, attention_mask_t, tok_idx_t)
    logits_t = out_t.logits if hasattr(out_t, "logits") else out_t[0]
    return torch.cat([logits_t[:, :1], logits_t[:, :-1]], dim=1).float()


def _dream_apply_sampling_step(
    x_t: Any,
    logits_t: Any,
    mask_token_id: int,
    t_value: float,
    s_value: float,
    alg: str,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    alg_temp: float,
    final_step: bool,
    transfer_allowed_mask_t: Optional[Any] = None,
) -> Any:
    torch = __import__("torch")
    x_next_t = x_t.clone()
    mask_index_t = x_t.eq(int(mask_token_id))
    if int(mask_index_t.sum().item()) <= 0:
        return x_next_t

    for row_idx in range(int(x_t.shape[0])):
        row_mask_pos_t = torch.nonzero(mask_index_t[row_idx], as_tuple=False).flatten()
        masked_count = int(row_mask_pos_t.numel())
        if masked_count <= 0:
            continue
        allowed_rel_t = None
        if transfer_allowed_mask_t is not None and not bool(final_step):
            allowed_t = transfer_allowed_mask_t
            if not torch.is_tensor(allowed_t):
                allowed_t = torch.as_tensor(allowed_t, device=x_t.device, dtype=torch.bool)
            allowed_t = allowed_t.to(device=x_t.device, dtype=torch.bool)
            if int(getattr(allowed_t, "ndim", 0)) == 1:
                row_allowed_full_t = allowed_t
            elif int(getattr(allowed_t, "ndim", 0)) == 2 and int(allowed_t.shape[0]) > row_idx:
                row_allowed_full_t = allowed_t[row_idx]
            else:
                row_allowed_full_t = None
            if row_allowed_full_t is not None and int(row_allowed_full_t.numel()) >= int(x_t.shape[1]):
                allowed_at_mask_t = row_allowed_full_t[row_mask_pos_t]
                allowed_rel_t = torch.nonzero(allowed_at_mask_t, as_tuple=False).flatten()
                if int(allowed_rel_t.numel()) <= 0:
                    continue
        row_mask_logits_t = logits_t[row_idx, row_mask_pos_t]
        if str(alg) == "origin":
            p_transfer = 1.0 if bool(final_step) else float(max(1.0 - float(s_value) / max(float(t_value), 1e-12), 0.0))
            _, row_sampled_tokens_t = _dream_sample_mask_logits_chunked(
                row_mask_logits_t,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            transfer_draw_t = torch.rand(masked_count, device=x_t.device, dtype=torch.float32)
            transfer_rel_t = torch.nonzero(transfer_draw_t < float(p_transfer), as_tuple=False).flatten()
            if allowed_rel_t is not None and int(transfer_rel_t.numel()) > 0:
                allowed_set = set(int(v.item()) for v in allowed_rel_t)
                transfer_rel_t = torch.as_tensor(
                    [int(v.item()) for v in transfer_rel_t if int(v.item()) in allowed_set],
                    device=x_t.device,
                    dtype=torch.long,
                )
            if int(transfer_rel_t.numel()) <= 0:
                continue
            transfer_idx_t = row_mask_pos_t[transfer_rel_t]
            x_next_t[row_idx, transfer_idx_t] = row_sampled_tokens_t[transfer_rel_t].to(device=x_t.device, dtype=torch.long)
            continue

        if str(alg) == "maskgit_plus":
            row_confidence_t, row_sampled_tokens_t = _dream_sample_mask_logits_chunked(
                row_mask_logits_t,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        elif str(alg) == "topk_margin":
            row_confidence_t, row_sampled_tokens_t = _dream_sample_mask_logits_chunked(
                row_mask_logits_t,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=True,
            )
        elif str(alg) == "entropy":
            row_confidence_t, row_sampled_tokens_t = _dream_sample_mask_logits_chunked(
                row_mask_logits_t,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                neg_entropy=True,
            )
        else:
            raise RuntimeError(f"Unknown Dream diffusion algorithm: {alg}")

        num_transfer = masked_count if bool(final_step) else int(masked_count * (1.0 - float(s_value) / max(float(t_value), 1e-12)))
        num_transfer = min(max(int(num_transfer), 0), masked_count)
        if allowed_rel_t is not None:
            num_transfer = min(num_transfer, int(allowed_rel_t.numel()))
        if num_transfer <= 0:
            continue
        confidence_for_selection_t = row_confidence_t
        if allowed_rel_t is not None:
            confidence_for_selection_t = row_confidence_t[allowed_rel_t]
        if alg_temp is None or float(alg_temp) == 0.0:
            selected_rel_t = torch.topk(confidence_for_selection_t, k=num_transfer, dim=-1).indices
        else:
            conf_row_t = torch.softmax(confidence_for_selection_t / float(alg_temp), dim=-1)
            if float(conf_row_t.sum().item()) <= 0.0:
                continue
            selected_rel_t = torch.multinomial(conf_row_t, num_samples=num_transfer, replacement=False)
        transfer_rel_t = allowed_rel_t[selected_rel_t] if allowed_rel_t is not None else selected_rel_t
        transfer_idx_t = row_mask_pos_t[transfer_rel_t]
        x_next_t[row_idx, transfer_idx_t] = row_sampled_tokens_t[transfer_rel_t].to(device=x_t.device, dtype=torch.long)
    return x_next_t


def _eb_token_structure_meta(token_text: str, prefix_text: str) -> Dict[str, bool]:
    text = str(token_text or "")
    stripped = text.strip()
    lowered = stripped.lower()
    current_line = str(prefix_text or "").split("\n")[-1]
    current_line_l = current_line.lstrip().lower()
    bracket_or_quote = any(ch in text for ch in "()[]{}'\"")
    indentation_like = bool("\n" in text or "\t" in text or (text.startswith(" ") and len(text) != len(text.lstrip(" "))))
    syntax_keyword = lowered in {
        "def",
        "class",
        "lambda",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "return",
        "yield",
        "from",
        "import",
    }
    signature_line = bool(current_line_l.startswith(("def ", "async def ", "class ")))
    return {
        "structure": bool(bracket_or_quote or indentation_like or syntax_keyword),
        "bracket_or_quote": bool(bracket_or_quote),
        "indentation": bool(indentation_like),
        "signature_line": bool(signature_line),
    }


def _build_eb_transfer_allowed_mask(
    x_t: Any,
    logits_t: Any,
    comp_start: int,
    comp_end: int,
    mask_token_id: int,
    cfg: DecoderConfig,
    token_ids_to_code: Optional[Callable[[Any], str]] = None,
    source_prefix: str = "",
    parser_feedback: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    torch = __import__("torch")
    x_work_t = x_t if torch.is_tensor(x_t) else torch.as_tensor(x_t)
    logits_work_t = logits_t if torch.is_tensor(logits_t) else torch.as_tensor(logits_t)
    if int(getattr(x_work_t, "ndim", 0)) == 1:
        x_work_t = x_work_t.unsqueeze(0)
    if int(getattr(logits_work_t, "ndim", 0)) == 2:
        logits_work_t = logits_work_t.unsqueeze(0)

    allowed_t = torch.ones_like(x_work_t, dtype=torch.bool, device=x_work_t.device)
    seq_len = int(x_work_t.shape[-1])
    start = max(min(int(comp_start), seq_len), 0)
    end = max(min(int(comp_end), seq_len), start)
    if end > start:
        allowed_t[:, start:end] = False

    q = float(np.clip(getattr(cfg, "eb_entropy_quantile", 0.35), 0.0, 1.0))
    min_commit = max(int(getattr(cfg, "eb_min_commit_per_step", 1)), 0)
    structure_scale = float(np.clip(getattr(cfg, "eb_structure_entropy_scale", 0.65), 0.05, 1.0))
    signature_scale = float(np.clip(getattr(cfg, "eb_signature_entropy_scale", 0.5), 0.05, 1.0))
    syntax_near_scale = float(np.clip(getattr(cfg, "eb_syntax_near_entropy_scale", 0.5), 0.05, 1.0))
    near_radius = max(int(getattr(cfg, "eb_near_radius", 2)), 0)

    meta: Dict[str, Any] = {
        "enabled": True,
        "entropy_quantile": float(q),
        "candidate_count": 0,
        "allowed_count": 0,
        "blocked_count": 0,
        "structure_blocked_count": 0,
        "signature_blocked_count": 0,
        "syntax_near_blocked_count": 0,
        "min_fallback_count": 0,
        "thresholds": [],
    }

    for row_idx in range(int(x_work_t.shape[0])):
        comp_ids_t = x_work_t[row_idx, start:end].to(dtype=torch.long)
        comp_logits_t = logits_work_t[row_idx, start:end].float()
        masked_rel_t = torch.nonzero(comp_ids_t.eq(int(mask_token_id)), as_tuple=False).flatten()
        if int(masked_rel_t.numel()) <= 0:
            continue

        row_logits_t = comp_logits_t[masked_rel_t]
        probs_t = torch.softmax(row_logits_t, dim=-1)
        entropy_t = -(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum(dim=-1)
        argmax_t = torch.argmax(row_logits_t, dim=-1)
        base_threshold = float(torch.quantile(entropy_t, q).item()) if int(entropy_t.numel()) > 0 else 0.0
        projected_comp_t = comp_ids_t.clone()
        projected_comp_t[masked_rel_t] = argmax_t.to(device=projected_comp_t.device, dtype=torch.long)
        projected_completion = ""
        syntax_error_pos = -1
        if token_ids_to_code is not None:
            try:
                projected_completion = _decode_token_sequence_text(projected_comp_t, token_ids_to_code)
            except Exception:
                projected_completion = ""
        signature_line_end_pos = -1
        projected_lstrip = projected_completion.lstrip()
        if token_ids_to_code is not None and projected_lstrip.startswith(("def ", "async def ", "class ")):
            signature_line_end_pos = int(projected_comp_t.numel()) - 1
            for sig_pos, sig_tok in enumerate(projected_comp_t.detach().cpu().reshape(-1).tolist()):
                token_piece = _decode_single_token_text(int(sig_tok), token_ids_to_code)
                if "\n" in token_piece:
                    signature_line_end_pos = int(sig_pos)
                    break
        if isinstance(parser_feedback, dict) and not bool(parser_feedback.get("parse_ok", True)):
            progress_meta = _syntax_error_completion_progress(
                feedback=parser_feedback,
                prompt=str(source_prefix or ""),
                completion=projected_completion,
            )
            if bool(progress_meta.get("in_completion", False)):
                syntax_error_pos = _progress_to_token_index(
                    progress=float(progress_meta.get("completion_progress", 0.0)),
                    token_count=int(max(end - start, 1)),
                )

        row_allowed_rel: List[int] = []
        fallback_rows: List[Tuple[float, bool, int]] = []
        token_text_cache: Dict[int, str] = {}
        for local_idx, pos_t in enumerate(masked_rel_t):
            pos = int(pos_t.item())
            entropy_val = float(entropy_t[local_idx].item())
            token_id = int(argmax_t[local_idx].item())
            if token_ids_to_code is not None and token_id not in token_text_cache:
                token_text_cache[token_id] = _decode_single_token_text(token_id, token_ids_to_code)
            token_text = token_text_cache.get(token_id, "")
            prefix_text = "def " if 0 <= int(pos) <= int(signature_line_end_pos) else ""
            structure_meta = _eb_token_structure_meta(token_text=token_text, prefix_text=prefix_text)
            syntax_near = bool(syntax_error_pos >= 0 and abs(int(pos) - int(syntax_error_pos)) <= near_radius)
            scale = 1.0
            if bool(structure_meta["structure"]):
                scale = min(scale, structure_scale)
            if bool(structure_meta["signature_line"]):
                scale = min(scale, signature_scale)
            if syntax_near:
                scale = min(scale, syntax_near_scale)
            threshold = float(base_threshold * scale)
            allowed = bool(entropy_val <= threshold + 1e-12)
            if allowed:
                row_allowed_rel.append(pos)
            else:
                if bool(structure_meta["structure"]):
                    meta["structure_blocked_count"] += 1
                if bool(structure_meta["signature_line"]):
                    meta["signature_blocked_count"] += 1
                if syntax_near:
                    meta["syntax_near_blocked_count"] += 1
            fallback_rows.append((entropy_val, bool(structure_meta["signature_line"] or syntax_near), pos))
            meta["thresholds"].append(
                {
                    "pos": int(pos),
                    "entropy": float(entropy_val),
                    "threshold": float(threshold),
                    "allowed": bool(allowed),
                    "structure": bool(structure_meta["structure"]),
                    "signature_line": bool(structure_meta["signature_line"]),
                    "syntax_near": bool(syntax_near),
                }
            )

        if min_commit > 0 and len(row_allowed_rel) < min(min_commit, int(masked_rel_t.numel())):
            already = set(row_allowed_rel)
            for _, _, pos in sorted(fallback_rows, key=lambda item: (item[1], item[0])):
                if pos in already:
                    continue
                row_allowed_rel.append(int(pos))
                already.add(int(pos))
                meta["min_fallback_count"] += 1
                if len(row_allowed_rel) >= min(min_commit, int(masked_rel_t.numel())):
                    break

        for pos in row_allowed_rel:
            allowed_t[row_idx, start + int(pos)] = True
        meta["candidate_count"] += int(masked_rel_t.numel())
        meta["allowed_count"] += int(len(row_allowed_rel))

    meta["blocked_count"] = int(max(int(meta["candidate_count"]) - int(meta["allowed_count"]), 0))
    return allowed_t, meta


def _branch_future_entropy(logits_comp_t: Any, committed_mask_t: Any) -> float:
    torch = __import__("torch")
    logits_t = logits_comp_t.to(dtype=torch.float32)
    cm_t = committed_mask_t.to(device=logits_t.device, dtype=torch.bool).reshape(-1)
    lim = min(int(logits_t.shape[0]), int(cm_t.numel()))
    if lim <= 0:
        return 0.0
    unresolved_t = ~cm_t[:lim]
    if int(unresolved_t.sum().item()) <= 0:
        return 0.0
    probs_t = torch.softmax(logits_t[:lim][unresolved_t], dim=-1)
    ent_t = -(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum(dim=-1)
    return float(ent_t.mean().item()) if int(ent_t.numel()) > 0 else 0.0


def _score_dream_branch_state(
    branch: _DreamBranchState,
    branch_tokens_t: Any,
    logits_comp_t: Any,
    committed_mask_t: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
    cfg: DecoderConfig,
    source_prefix: str = "",
) -> Tuple[float, Dict[str, Any], float]:
    if source_prefix and token_ids_to_code is not None:
        completion_prefix, _ = _decode_committed_completion_prefix(
            token_ids=branch_tokens_t,
            committed_mask=committed_mask_t,
            token_ids_to_code=token_ids_to_code,
        )
        feedback = parser_feedback_from_source(
            source=str(source_prefix) + completion_prefix,
            min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
        )
    else:
        feedback = parser_feedback_from_tokens(
            forced_tokens=branch_tokens_t,
            committed_mask=committed_mask_t,
            token_ids_to_code=token_ids_to_code,
            min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
        )
    future_entropy = _branch_future_entropy(logits_comp_t, committed_mask_t)
    parse_quality = float(feedback.get("quality_score", 1.0))
    branch_score = float(branch.score)
    branch_score += float(getattr(cfg, "branch_parse_weight", 2.0)) * parse_quality
    branch_score -= float(getattr(cfg, "branch_entropy_weight", 0.2)) * future_entropy
    return branch_score, feedback, future_entropy


def _run_baseline_with_sampler(prompt: str, sampler, max_steps: int = 20, seq_len: int = 64):
    del prompt
    committed_mask = np.zeros(seq_len, dtype=bool)
    forced_tokens = np.zeros(seq_len, dtype=np.int64)
    latents = None
    for _ in range(max_steps):
        logits = sampler.step(latents, committed_mask, forced_tokens)
        latents = logits
        for i in range(seq_len):
            if not committed_mask[i]:
                forced_tokens[i] = np.argmax(logits[i])
                committed_mask[i] = True
        if committed_mask.all():
            break
    return forced_tokens, seq_len


def load_json_items(path: str, max_samples: int, offset: int = 0) -> list:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else list(data.values())
    if offset >= len(items):
        return []
    return items[offset : offset + max_samples]


def _resolve_result_root(project_root: str, result_root: str, log_dir: str) -> str:
    if result_root:
        return os.path.abspath(os.path.expanduser(result_root))
    if log_dir:
        return os.path.abspath(os.path.expanduser(log_dir))
    return os.path.join(project_root, "results_remote")


def _prepare_result_dirs(result_root: str, timestamp: str = "") -> Dict[str, str]:
    base_dir = Path(result_root).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    base_name = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{base_name}_{suffix:02d}"
        suffix += 1

    rawdata_dir = run_dir / "rawdata"
    json_dir = run_dir / "json"
    rawdata_dir.mkdir(parents=True, exist_ok=False)
    json_dir.mkdir(parents=True, exist_ok=False)
    return {
        "result_root": str(base_dir),
        "run_dir": str(run_dir),
        "rawdata_dir": str(rawdata_dir),
        "json_dir": str(json_dir),
        "timestamp": run_dir.name,
    }


def _write_json(path: Path, payload: Any) -> None:
    def _to_jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_jsonable(v) for v in value]
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_completion_token_overrides(
    x: Any,
    prompt_len: int,
    completion_len: int,
    override_mask: Any,
    override_tokens: Any,
) -> Any:
    try:
        import torch
    except ImportError:
        return x

    if x is None or not torch.is_tensor(x):
        return x
    if override_mask is None or override_tokens is None:
        return x

    if int(getattr(x, "ndim", 0)) == 2:
        if int(x.shape[0]) != 1:
            return x
        x_out = x.clone()
        seq_view = x_out[0]
    elif int(getattr(x, "ndim", 0)) == 1:
        x_out = x.clone()
        seq_view = x_out
    else:
        return x

    comp_start = min(int(prompt_len), int(seq_view.shape[0]))
    comp_end = min(int(prompt_len + completion_len), int(seq_view.shape[0]))
    if comp_end <= comp_start:
        return x_out

    comp_view = seq_view[comp_start:comp_end]
    mask_t = override_mask if torch.is_tensor(override_mask) else torch.as_tensor(override_mask, device=comp_view.device)
    tok_t = override_tokens if torch.is_tensor(override_tokens) else torch.as_tensor(override_tokens, device=comp_view.device)
    mask_t = mask_t.to(device=comp_view.device, dtype=torch.bool).reshape(-1)
    tok_t = tok_t.to(device=comp_view.device, dtype=torch.long).reshape(-1)
    lim = min(int(comp_view.shape[0]), int(mask_t.numel()), int(tok_t.numel()))
    if lim <= 0:
        return x_out

    comp_view[:lim] = torch.where(mask_t[:lim], tok_t[:lim], comp_view[:lim])
    return x_out


def _apply_completion_remask(
    x: Any,
    prompt_len: int,
    completion_len: int,
    remask_positions: Sequence[int],
    mask_token_id: int,
) -> Any:
    try:
        import torch
    except ImportError:
        return x

    if x is None or not torch.is_tensor(x) or not remask_positions:
        return x

    if int(getattr(x, "ndim", 0)) == 2:
        if int(x.shape[0]) != 1:
            return x
        x_out = x.clone()
        seq_view = x_out[0]
    elif int(getattr(x, "ndim", 0)) == 1:
        x_out = x.clone()
        seq_view = x_out
    else:
        return x

    comp_start = min(int(prompt_len), int(seq_view.shape[0]))
    comp_end = min(int(prompt_len + completion_len), int(seq_view.shape[0]))
    if comp_end <= comp_start:
        return x_out

    comp_view = seq_view[comp_start:comp_end]
    for pos in remask_positions:
        pos_i = int(pos)
        if 0 <= pos_i < int(comp_view.shape[0]):
            comp_view[pos_i] = int(mask_token_id)
    return x_out


def _merge_completion_transfer_block_mask(
    allowed_t: Optional[Any],
    x_t: Any,
    prompt_len: int,
    completion_len: int,
    blocked_completion_mask_t: Any,
) -> Optional[Any]:
    try:
        import torch
    except ImportError:
        return allowed_t
    if x_t is None or not torch.is_tensor(x_t) or blocked_completion_mask_t is None:
        return allowed_t
    blocked_t = (
        blocked_completion_mask_t
        if torch.is_tensor(blocked_completion_mask_t)
        else torch.as_tensor(blocked_completion_mask_t)
    )
    blocked_t = blocked_t.to(device=x_t.device, dtype=torch.bool).reshape(-1)
    if int(blocked_t.numel()) <= 0 or not bool(torch.any(blocked_t).item()):
        return allowed_t

    if allowed_t is None:
        merged_t = torch.ones_like(x_t, dtype=torch.bool, device=x_t.device)
    else:
        merged_t = allowed_t if torch.is_tensor(allowed_t) else torch.as_tensor(allowed_t, device=x_t.device)
        merged_t = merged_t.to(device=x_t.device, dtype=torch.bool).clone()
        if int(getattr(merged_t, "ndim", 0)) == 1:
            merged_t = merged_t.unsqueeze(0)
    if int(getattr(merged_t, "ndim", 0)) != 2:
        return allowed_t

    comp_start_i = min(int(prompt_len), int(merged_t.shape[-1]))
    comp_end_i = min(int(prompt_len + completion_len), int(merged_t.shape[-1]))
    if comp_end_i <= comp_start_i:
        return merged_t
    lim = min(int(comp_end_i - comp_start_i), int(blocked_t.numel()))
    if lim > 0:
        merged_t[:, comp_start_i : comp_start_i + lim] = torch.where(
            blocked_t[:lim].unsqueeze(0),
            torch.zeros_like(merged_t[:, comp_start_i : comp_start_i + lim]),
            merged_t[:, comp_start_i : comp_start_i + lim],
        )
    return merged_t


def _select_rdd_rollback_positions(
    committed_mask_t: Any,
    syntax_error_progress: float,
    rollback_window: int,
) -> List[int]:
    try:
        import torch
    except ImportError:
        return []

    if committed_mask_t is None:
        return []
    mask_t = committed_mask_t if torch.is_tensor(committed_mask_t) else torch.as_tensor(committed_mask_t)
    mask_t = mask_t.to(dtype=torch.bool).reshape(-1)
    committed_positions_t = torch.nonzero(mask_t, as_tuple=False).flatten()
    committed_count = int(committed_positions_t.numel())
    if committed_count <= 0:
        return []

    window = min(max(int(rollback_window), 1), committed_count)
    progress = float(np.clip(float(syntax_error_progress), 0.0, 1.0))
    center_rank = int(progress * float(max(committed_count - 1, 0)) + 0.5)
    start_rank = int(max(min(center_rank - window // 2, committed_count - window), 0))
    selected = committed_positions_t[start_rank : start_rank + window]
    return [int(pos.item()) for pos in selected]


def _default_parser_feedback() -> Dict[str, Any]:
    return {
        "observed": False,
        "parse_ok": True,
        "primary_issue": "none",
        "issue_types": [],
        "bracket_issue": False,
        "indent_issue": False,
        "syntax_error_message": "",
        "syntax_error_progress": 0.0,
        "quality_score": 1.0,
        "severity_score": 0.0,
    }


def _default_repair_route() -> Dict[str, Any]:
    return {
        "action": "normal",
        "state": "no_confirmed_damage",
        "reason": "normal_diffusion",
        "committed_parse_ok": True,
        "projected_parse_ok": True,
        "committed_severity": 0.0,
        "projected_severity": 0.0,
        "committed_error_completion_progress": 0.0,
        "committed_error_is_committed": False,
        "committed_prefix_ast_risk": False,
        "projected_error_completion_progress": 0.0,
        "projected_error_pos": -1,
        "projected_error_near_masked": False,
        "projected_nearest_masked_distance": None,
        "projected_nearest_entropy": 0.0,
        "projected_entropy_gate": 0.0,
    }


def _resolve_dream_repair_policy(cfg: DecoderConfig) -> str:
    if not bool(getattr(cfg, "rdd_rollback_enabled", False)):
        return _REPAIR_POLICY_NONE
    if bool(getattr(cfg, "pf_enabled", False)):
        return _REPAIR_POLICY_PF_RB
    return _REPAIR_POLICY_ROLLBACK_ONLY


def _repair_policy_uses_baseline_sampling(repair_policy: str) -> bool:
    return str(repair_policy) == _REPAIR_POLICY_ROLLBACK_ONLY


def _is_dream_noop_config(cfg: DecoderConfig) -> bool:
    """Return true when the Dream hook path must be exactly official baseline."""
    return bool(
        not getattr(cfg, "pf_enabled", False)
        and not getattr(cfg, "delay_commit_enabled", False)
        and not getattr(cfg, "rdd_rollback_enabled", False)
        and not getattr(cfg, "eb_sampler_enabled", False)
        and not getattr(cfg, "local_beam_enabled", False)
        and not getattr(cfg, "shadow_mode_enabled", False)
        and not getattr(cfg, "branch_observe_enabled", False)
        and not getattr(cfg, "branch_select_enabled", False)
        and not getattr(cfg, "influence_enabled", False)
        and not getattr(cfg, "entropy_kl_logging_enabled", False)
        and str(getattr(cfg, "correctness_signal_mode", "none") or "none").lower() == "none"
        and _resolve_dream_repair_policy(cfg) == _REPAIR_POLICY_NONE
    )


def _build_dream_noop_stats(diffusion_steps: int, cfg: DecoderConfig) -> Dict[str, Any]:
    steps = max(int(diffusion_steps), 1)
    return {
        "dream_noop_fast_path": True,
        "pf_trigger_count": 0,
        "extra_forwards": 0,
        "extra_forwards_influence": 0,
        "extra_forwards_pf": 0,
        "extra_forwards_local_beam": 0,
        "risk_trigger_count": 0,
        "influence_compute_count": 0,
        "n_commits": 0,
        "action_counts_total": {
            "commit_argmax": 0,
            "commit_pf": 0,
            "commit_fallback_max_prob": 0,
            "freeze_delay": 0,
            "freeze_cooldown": 0,
            "freeze_budget": 0,
        },
        "parser_feedback_counts": {"bracket": 0, "indent": 0},
        "parser_hotspot_counts": {"bracket": 0, "indent": 0},
        "parser_feedback_histograms": {"bracket": {}, "indent": {}},
        "parser_hotspot_histograms": {"bracket": {}, "indent": {}},
        "parser_feedback_top_timesteps": {},
        "parser_hotspot_top_timesteps": {},
        "rdd_rollback_enabled": False,
        "rdd_rollback_count": 0,
        "rdd_remasked_tokens_total": 0,
        "rdd_rollback_cleared_pf_count": 0,
        "rdd_rollback_events": [],
        "repair_policy": _REPAIR_POLICY_NONE,
        "repair_route_counts": {"normal": 0, "pf": 0, "rollback": 0},
        "pf_rb_route_counts": {"normal": 0, "pf": 0, "rollback": 0},
        "step_logs": [],
        "risk_trace_lines": [],
        "commit_events": [],
        "avg_branch_width": 1.0,
        "max_branch_width": 1,
        "baseline_forwards": steps,
        "latency_ratio_vs_baseline": 1.0,
        "branch_width_trace": [1 for _ in range(steps)],
        "pf_budget_mode": str(getattr(cfg, "pf_budget_mode", "legacy") or "legacy"),
        "pf_extra_forward_budget": int(max(getattr(cfg, "pf_extra_forward_budget", 0), 0)),
        "pf_trigger_limit": int(max(getattr(cfg, "pf_max_triggers_per_sample", 0), 0)),
        "eb_sampler_enabled": False,
        "eb_step_count": 0,
        "eb_candidate_count": 0,
        "eb_allowed_count": 0,
        "eb_blocked_count": 0,
        "eb_min_fallback_count": 0,
        "eb_structure_blocked_count": 0,
        "eb_signature_blocked_count": 0,
        "eb_syntax_near_blocked_count": 0,
        "eb_allowed_per_step": 0.0,
        "eb_blocked_per_step": 0.0,
        "local_beam_enabled": False,
        "local_beam_branch_events": 0,
        "local_beam_accepted_alternatives": 0,
        "local_beam_delay_count": 0,
        "local_beam_avg_beam_size": 0.0,
        "local_beam_event_logs": [],
        "shadow_mode_enabled": False,
        "shadow_num_steps": 0,
        "shadow_num_risk_events": 0,
        "shadow_max_risk": 0.0,
        "shadow_mean_risk": 0.0,
        "shadow_high_risk_early_commit_count": 0,
        "shadow_top_risk_events": [],
        "shadow_step_commit_stats": [],
        "branch_observe_enabled": False,
        "branch_observe_force_baseline_output": True,
        "branch_observe_branch_events": 0,
        "branch_observe_rollout_count": 0,
        "branch_observe_extra_forwards": 0,
        "branch_observe_avg_beam_size": 0.0,
        "branch_observe_event_logs": [],
        "branch_observe_errors": [],
        "branch_select_enabled": False,
        "branch_select_verifier": "level0",
        "branch_select_selected": False,
        "branch_select_meta": {},
    }


def _feedback_severity(feedback: Dict[str, Any]) -> float:
    if not isinstance(feedback, dict):
        return 0.0
    if feedback.get("severity_score") is not None:
        return float(np.clip(float(feedback.get("severity_score", 0.0)), 0.0, 1.0))
    quality = float(feedback.get("quality_score", 1.0 if feedback.get("parse_ok", True) else 0.0))
    return float(np.clip(1.0 - quality, 0.0, 1.0))


def _feedback_parse_bad(feedback: Dict[str, Any], min_severity: float = 0.0) -> bool:
    if not isinstance(feedback, dict):
        return False
    return bool(
        feedback.get("observed", False)
        and not bool(feedback.get("parse_ok", True))
        and _feedback_severity(feedback) >= float(min_severity)
    )


def _feedback_tokenize_bad(feedback: Dict[str, Any]) -> bool:
    if not isinstance(feedback, dict):
        return False
    primary_issue = str(feedback.get("primary_issue", "") or "").lower()
    issue_types = {str(v).lower() for v in (feedback.get("issue_types", []) or [])}
    message = str(feedback.get("syntax_error_message", "") or "").lower()
    return bool(
        primary_issue in {"decode_error", "tokenize_error", "tokenizer_error"}
        or issue_types.intersection({"decode_error", "tokenize_error", "tokenizer_error"})
        or "decode_error" in message
        or "tokenize" in message
    )


def _feedback_obvious_syntax_risk(feedback: Dict[str, Any], min_severity: float = 0.0) -> bool:
    if not _feedback_parse_bad(feedback, min_severity=float(min_severity)) and not _feedback_tokenize_bad(feedback):
        return False
    primary_issue = str(feedback.get("primary_issue", "") or "").lower()
    issue_types = {str(v).lower() for v in (feedback.get("issue_types", []) or [])}
    if _feedback_tokenize_bad(feedback):
        return True
    if bool(feedback.get("bracket_issue", False)) or bool(feedback.get("indent_issue", False)):
        return True
    if primary_issue in {"bracket", "indent", "syntax"}:
        return True
    return bool(issue_types.intersection({"bracket", "indent", "syntax"}))


def _committed_prefix_ast_risk(
    feedback: Dict[str, Any],
    committed_error: Dict[str, Any],
    committed_completion: str,
    committed_token_count: int,
    min_severity: float,
) -> bool:
    if not _feedback_parse_bad(feedback, min_severity=float(min_severity)):
        return False
    if not bool(committed_error.get("in_completion", False)):
        return False
    error_token = _progress_to_token_index(
        progress=float(committed_error.get("completion_progress", 0.0)),
        token_count=int(committed_token_count),
    )
    if error_token < 0:
        return False

    issue_types = {str(v) for v in (feedback.get("issue_types", []) or [])}
    primary_issue = str(feedback.get("primary_issue", ""))
    if bool(feedback.get("bracket_issue", False)) or bool(feedback.get("indent_issue", False)):
        return True
    if primary_issue in {"bracket", "indent"} or issue_types.intersection({"bracket", "indent"}):
        return True

    completion_tail = str(committed_completion or "").rstrip()
    trailing_bad_suffixes = ("=", "+", "-", "*", "/", "%", ".", ",", ":", "\\", "(", "[", "{")
    trailing_bad_words = {"and", "or", "not", "is", "in", "return", "raise", "from", "lambda"}
    if completion_tail.endswith(trailing_bad_suffixes):
        return False
    if completion_tail.endswith("#"):
        return False
    last_word = completion_tail.split()[-1] if completion_tail.split() else ""
    if last_word in trailing_bad_words:
        return False

    message = str(feedback.get("syntax_error_message", "") or "").lower()
    error_at_tail = error_token >= max(int(committed_token_count) - 1, 0)
    tail_noise_hints = (
        "unexpected eof",
        "eof while scanning",
        "unterminated string",
        "invalid syntax",
    )
    if error_at_tail and any(hint in message for hint in tail_noise_hints):
        return False
    return True


def _syntax_error_completion_progress(
    feedback: Dict[str, Any],
    prompt: str,
    completion: str,
) -> Dict[str, Any]:
    prompt_text = str(prompt or "")
    completion_text = str(completion or "")
    full_len = max(len(prompt_text) + len(completion_text), 1)
    completion_len = max(len(completion_text), 1)
    progress = float(np.clip(float((feedback or {}).get("syntax_error_progress", 0.0)), 0.0, 1.0))
    error_char = progress * float(full_len)
    rel = (error_char - float(len(prompt_text))) / float(completion_len)
    in_completion = bool(completion_text and error_char >= float(len(prompt_text)) and rel <= 1.0)
    return {
        "full_progress": float(progress),
        "error_char": float(error_char),
        "completion_progress": float(np.clip(rel, 0.0, 1.0)),
        "in_completion": bool(in_completion),
    }


def _progress_to_token_index(progress: float, token_count: int) -> int:
    count = int(token_count)
    if count <= 0:
        return -1
    return int(float(np.clip(float(progress), 0.0, 1.0)) * float(max(count - 1, 0)) + 0.5)


def _flatten_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    try:
        import torch

        if torch.is_tensor(values):
            return [int(v) for v in values.detach().cpu().reshape(-1).tolist()]
    except Exception:
        pass
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        try:
            return [int(v) for v in values.reshape(-1).tolist()]
        except Exception:
            pass
    if isinstance(values, (list, tuple, set)):
        return [int(v) for v in values]
    try:
        return [int(values)]
    except Exception:
        return []


def _flatten_float_list(values: Any) -> List[float]:
    if values is None:
        return []
    try:
        import torch

        if torch.is_tensor(values):
            return [float(v) for v in values.detach().cpu().reshape(-1).tolist()]
    except Exception:
        pass
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        try:
            return [float(v) for v in values.reshape(-1).tolist()]
        except Exception:
            pass
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    try:
        return [float(values)]
    except Exception:
        return []


def _projected_error_near_masked_high_entropy(
    projected_feedback: Dict[str, Any],
    prompt: str,
    projected_completion: str,
    projected_token_count: int,
    masked_positions_t: Any,
    entropy_all_t: Any,
    near_radius: int,
) -> Dict[str, Any]:
    progress_meta = _syntax_error_completion_progress(
        feedback=projected_feedback,
        prompt=prompt,
        completion=projected_completion,
    )
    error_pos = _progress_to_token_index(
        progress=float(progress_meta["completion_progress"]),
        token_count=int(projected_token_count),
    )
    masked_positions = _flatten_int_list(masked_positions_t)
    entropy_values = _flatten_float_list(entropy_all_t)
    masked_entropies = [
        float(entropy_values[pos])
        for pos in masked_positions
        if 0 <= int(pos) < len(entropy_values)
    ]
    entropy_gate = float(np.quantile(masked_entropies, 0.75)) if masked_entropies else 0.0
    radius = max(int(near_radius), 0)
    nearest_distance: Optional[int] = None
    nearest_entropy = 0.0
    near = False
    for pos in masked_positions:
        distance = abs(int(pos) - int(error_pos))
        entropy_val = float(entropy_values[pos]) if 0 <= int(pos) < len(entropy_values) else 0.0
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = int(distance)
            nearest_entropy = float(entropy_val)
        if bool(progress_meta["in_completion"]) and distance <= radius and entropy_val + 1e-12 >= entropy_gate:
            near = True
    return {
        "near": bool(near),
        "error_pos": int(error_pos),
        "nearest_distance": int(nearest_distance) if nearest_distance is not None else None,
        "nearest_entropy": float(nearest_entropy),
        "entropy_gate": float(entropy_gate),
        "completion_progress": float(progress_meta["completion_progress"]),
        "in_completion": bool(progress_meta["in_completion"]),
    }


def _route_rollback_only_action(
    committed_feedback: Dict[str, Any],
    prompt: str,
    committed_completion: str,
    committed_token_count: int,
    rollback_available: bool,
    rollback_min_severity: float,
    parser_gradient_hotspot: bool = False,
    repair_cooldown_active: bool = False,
) -> Dict[str, Any]:
    committed_feedback = committed_feedback if isinstance(committed_feedback, dict) else _default_parser_feedback()
    committed_severity = _feedback_severity(committed_feedback)
    committed_error = _syntax_error_completion_progress(
        feedback=committed_feedback,
        prompt=prompt,
        completion=committed_completion,
    )
    committed_error_is_committed = bool(
        committed_error["in_completion"] and _progress_to_token_index(committed_error["completion_progress"], committed_token_count) >= 0
    )
    committed_prefix_ast_risk = _committed_prefix_ast_risk(
        feedback=committed_feedback,
        committed_error=committed_error,
        committed_completion=committed_completion,
        committed_token_count=int(committed_token_count),
        min_severity=float(rollback_min_severity),
    )
    base = _default_repair_route()
    base.update(
        {
            "committed_parse_ok": bool(committed_feedback.get("parse_ok", True)),
            "committed_severity": float(committed_severity),
            "committed_error_completion_progress": float(committed_error["completion_progress"]),
            "committed_error_is_committed": bool(committed_error_is_committed),
            "committed_prefix_ast_risk": bool(committed_prefix_ast_risk),
        }
    )

    if repair_cooldown_active:
        base.update({"state": "cooldown", "reason": "rollback_cooldown"})
        return base

    committed_context_bad = bool(
        rollback_available
        and committed_prefix_ast_risk
    )
    if committed_context_bad:
        reason = "parser_gradient_committed_error" if bool(parser_gradient_hotspot) else "committed_parse_failure"
        base.update({"action": "rollback", "state": "context_level_damage", "reason": reason})
        return base

    return base


def _route_pf_rb_action(
    committed_feedback: Dict[str, Any],
    projected_feedback: Dict[str, Any],
    prompt: str,
    committed_completion: str,
    projected_completion: str,
    committed_token_count: int,
    projected_token_count: int,
    masked_positions_t: Any,
    entropy_all_t: Any,
    rollback_available: bool,
    pf_available: bool,
    high_risk_masked_position: bool,
    rollback_min_severity: float,
    near_radius: int,
    parser_gradient_hotspot: bool = False,
    repair_cooldown_active: bool = False,
) -> Dict[str, Any]:
    committed_feedback = committed_feedback if isinstance(committed_feedback, dict) else _default_parser_feedback()
    projected_feedback = projected_feedback if isinstance(projected_feedback, dict) else _default_parser_feedback()
    committed_severity = _feedback_severity(committed_feedback)
    projected_severity = _feedback_severity(projected_feedback)
    committed_error = _syntax_error_completion_progress(
        feedback=committed_feedback,
        prompt=prompt,
        completion=committed_completion,
    )
    committed_error_is_committed = bool(
        committed_error["in_completion"] and _progress_to_token_index(committed_error["completion_progress"], committed_token_count) >= 0
    )
    committed_prefix_ast_risk = _committed_prefix_ast_risk(
        feedback=committed_feedback,
        committed_error=committed_error,
        committed_completion=committed_completion,
        committed_token_count=int(committed_token_count),
        min_severity=float(rollback_min_severity),
    )
    projected_near = _projected_error_near_masked_high_entropy(
        projected_feedback=projected_feedback,
        prompt=prompt,
        projected_completion=projected_completion,
        projected_token_count=int(projected_token_count),
        masked_positions_t=masked_positions_t,
        entropy_all_t=entropy_all_t,
        near_radius=int(near_radius),
    )
    base = _default_repair_route()
    base.update(
        {
            "committed_parse_ok": bool(committed_feedback.get("parse_ok", True)),
            "projected_parse_ok": bool(projected_feedback.get("parse_ok", True)),
            "committed_severity": float(committed_severity),
            "projected_severity": float(projected_severity),
            "committed_error_completion_progress": float(committed_error["completion_progress"]),
            "committed_error_is_committed": bool(committed_error_is_committed),
            "committed_prefix_ast_risk": bool(committed_prefix_ast_risk),
            "projected_error_completion_progress": float(projected_near["completion_progress"]),
            "projected_error_pos": int(projected_near["error_pos"]),
            "projected_error_near_masked": bool(projected_near["near"]),
            "projected_nearest_masked_distance": projected_near["nearest_distance"],
            "projected_nearest_entropy": float(projected_near["nearest_entropy"]),
            "projected_entropy_gate": float(projected_near["entropy_gate"]),
        }
    )

    if repair_cooldown_active:
        base.update({"state": "cooldown", "reason": "rollback_cooldown"})
        return base

    committed_context_bad = bool(
        rollback_available
        and committed_prefix_ast_risk
    )
    if committed_context_bad:
        reason = "parser_gradient_committed_error" if bool(parser_gradient_hotspot) else "committed_parse_failure"
        base.update({"action": "rollback", "state": "context_level_damage", "reason": reason})
        return base

    projected_degrades = bool(
        _feedback_parse_bad(projected_feedback, min_severity=0.0)
        and projected_severity > committed_severity + 0.05
    )
    projected_tokenize_bad = _feedback_tokenize_bad(projected_feedback)
    projected_syntax_risk = _feedback_obvious_syntax_risk(projected_feedback, min_severity=0.0)
    projected_pf_repairable = bool(
        pf_available
        and not bool(projected_feedback.get("parse_ok", True))
        and (
            bool(projected_tokenize_bad)
            or (bool(projected_syntax_risk) and bool(projected_near["near"]))
            or (bool(projected_degrades) and bool(projected_near["near"]))
        )
    )
    if projected_pf_repairable:
        if projected_tokenize_bad:
            base.update({"action": "pf", "state": "token_level_risk", "reason": "projected_tokenize_failure"})
            return base
        base.update({"action": "pf", "state": "token_level_risk", "reason": "projected_parser_degradation"})
        return base

    return base


def _resolve_pf_persistence_steps(
    t_remaining: int,
    t_start: int,
    t_end: int,
    base_steps: int,
) -> int:
    early_steps = max(1, min(int(base_steps), 3))
    late_steps = max(int(base_steps), 10)
    lo = min(int(t_start), int(t_end))
    hi = max(int(t_start), int(t_end))
    if hi <= lo:
        return int(late_steps)
    progress = float(hi - int(t_remaining)) / float(max(hi - lo, 1))
    progress = float(np.clip(progress, 0.0, 1.0))
    return int(round(early_steps + progress * float(late_steps - early_steps)))


_STRUCTURE_KEYWORD_HINTS = (
    "def",
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "class",
    "with",
    "import",
    "from",
    "lambda",
)
_STRUCTURE_CHAR_HINTS = ("(", ")", "[", "]", "{", "}", ":", ".", ",", "=")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _decode_single_token_text(
    token_id: int,
    token_ids_to_code: Optional[Callable[[Any], str]],
) -> str:
    if token_ids_to_code is None:
        return ""
    token_arr = np.asarray([int(token_id)], dtype=np.int64)
    try:
        text = token_ids_to_code(token_arr)
    except Exception:
        try:
            text = token_ids_to_code(token_arr.tolist())
        except Exception:
            return ""
    return text if isinstance(text, str) else str(text)


def _decode_token_sequence_text(
    token_ids: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
) -> str:
    if token_ids_to_code is None:
        return ""
    try:
        import torch
    except Exception:
        torch = None

    try:
        return str(token_ids_to_code(token_ids))
    except Exception:
        pass

    if torch is not None and torch.is_tensor(token_ids):
        token_cpu = token_ids.detach()
        if token_cpu.device.type != "cpu":
            token_cpu = token_cpu.to(device="cpu")
        try:
            return str(token_ids_to_code(token_cpu))
        except Exception:
            try:
                return str(token_ids_to_code(token_cpu.tolist()))
            except Exception:
                return ""

    if hasattr(token_ids, "tolist"):
        try:
            return str(token_ids_to_code(token_ids.tolist()))
        except Exception:
            return ""
    return ""


def _decode_committed_completion_prefix(
    token_ids: Any,
    committed_mask: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
) -> Tuple[str, int]:
    if token_ids_to_code is None:
        return "", 0
    try:
        import torch
    except Exception:
        return "", 0

    tok_t = token_ids if torch.is_tensor(token_ids) else torch.as_tensor(token_ids)
    cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    tok_t = tok_t.to(dtype=torch.long).reshape(-1)
    cm_t = cm_t.to(device=tok_t.device, dtype=torch.bool).reshape(-1)
    lim = min(int(tok_t.numel()), int(cm_t.numel()))
    if lim <= 0:
        return "", 0

    cm_t = cm_t[:lim]
    tok_t = tok_t[:lim]
    first_uncommitted_t = torch.nonzero(~cm_t, as_tuple=False)
    prefix_len = int(first_uncommitted_t[0].item()) if int(first_uncommitted_t.numel()) > 0 else int(lim)
    if prefix_len <= 0:
        return "", 0
    return _decode_token_sequence_text(tok_t[:prefix_len], token_ids_to_code), int(prefix_len)


def _structure_sensitive_token_score(token_text: str) -> float:
    if not token_text:
        return 0.0
    normalized = (
        token_text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\n")
        .strip()
        .lower()
    )
    if not normalized:
        return 0.0

    score = 0.0
    if any(ch in normalized for ch in "()[]{}"):
        score += 1.0
    if ":" in normalized:
        score += 0.8
    if any(ch in normalized for ch in ".,="):
        score += 0.35
    if "\n" in normalized:
        score += 0.15
    if any(hint in normalized for hint in _STRUCTURE_KEYWORD_HINTS):
        score += 0.65
    return float(score)


def _extract_identifier_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ident in _IDENTIFIER_RE.findall(str(text or "")):
        counts[ident] = int(counts.get(ident, 0)) + 1
    return counts


def _project_completion_tokens(
    current_tokens_t: Any,
    committed_mask_t: Any,
    predicted_tokens_t: Any,
) -> Any:
    torch = __import__("torch")
    current_t = current_tokens_t if torch.is_tensor(current_tokens_t) else torch.as_tensor(current_tokens_t)
    committed_t = committed_mask_t if torch.is_tensor(committed_mask_t) else torch.as_tensor(committed_mask_t)
    predicted_t = predicted_tokens_t if torch.is_tensor(predicted_tokens_t) else torch.as_tensor(predicted_tokens_t)
    current_t = current_t.to(dtype=torch.long).reshape(-1)
    committed_t = committed_t.to(device=current_t.device, dtype=torch.bool).reshape(-1)
    predicted_t = predicted_t.to(device=current_t.device, dtype=torch.long).reshape(-1)
    lim = min(int(current_t.numel()), int(committed_t.numel()), int(predicted_t.numel()))
    if lim <= 0:
        return current_t[:0]
    projected_t = current_t[:lim].clone()
    projected_t[:lim] = torch.where(committed_t[:lim], projected_t[:lim], predicted_t[:lim])
    return projected_t


def _project_completion_source(
    current_tokens_t: Any,
    committed_mask_t: Any,
    predicted_tokens_t: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
) -> str:
    projected_t = _project_completion_tokens(
        current_tokens_t=current_tokens_t,
        committed_mask_t=committed_mask_t,
        predicted_tokens_t=predicted_tokens_t,
    )
    return _decode_token_sequence_text(projected_t, token_ids_to_code)


def _source_quality_score(
    source: str,
    min_prefix_chars: int,
) -> Tuple[float, Dict[str, Any]]:
    feedback = parser_feedback_from_source(source=source, min_prefix_chars=int(min_prefix_chars))
    quality = float(feedback.get("quality_score", 1.0 if feedback.get("parse_ok", True) else 0.0))
    return float(np.clip(quality, 0.0, 1.0)), feedback


def _counterfactual_repair_gain(
    pos: int,
    current_tokens_t: Any,
    committed_mask_t: Any,
    argmax_all_t: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
    min_prefix_chars: int,
    sampler_step_fn: Callable[[Any, Any, Optional[Any]], Any],
    rollout_steps: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    torch = __import__("torch")
    current_t = current_tokens_t if torch.is_tensor(current_tokens_t) else torch.as_tensor(current_tokens_t)
    committed_t = committed_mask_t if torch.is_tensor(committed_mask_t) else torch.as_tensor(committed_mask_t)
    predicted_t = argmax_all_t if torch.is_tensor(argmax_all_t) else torch.as_tensor(argmax_all_t)
    current_t = current_t.to(dtype=torch.long).reshape(-1)
    committed_t = committed_t.to(device=current_t.device, dtype=torch.bool).reshape(-1)
    predicted_t = predicted_t.to(device=current_t.device, dtype=torch.long).reshape(-1)
    lim = min(int(current_t.numel()), int(committed_t.numel()), int(predicted_t.numel()))
    pos = int(pos)
    if lim <= 0 or pos < 0 or pos >= lim:
        return 0.0, {"base_quality": 0.0, "after_quality": 0.0}

    base_source = _project_completion_source(
        current_tokens_t=current_t[:lim],
        committed_mask_t=committed_t[:lim],
        predicted_tokens_t=predicted_t[:lim],
        token_ids_to_code=token_ids_to_code,
    )
    base_quality, _ = _source_quality_score(base_source, min_prefix_chars=min_prefix_chars)

    forced_t = current_t[:lim].clone()
    force_mask_t = committed_t[:lim].clone()
    forced_t[pos] = int(predicted_t[pos].item())
    force_mask_t[pos] = True

    rollout_pred_t = predicted_t[:lim].clone()
    rollout_steps = max(int(rollout_steps), 1)
    try:
        rollout_logits_t = None
        for _ in range(rollout_steps):
            rollout_logits_t = sampler_step_fn(rollout_pred_t, force_mask_t, forced_t)
            if not torch.is_tensor(rollout_logits_t):
                rollout_logits_t = torch.as_tensor(rollout_logits_t, device=forced_t.device)
            rollout_logits_t = rollout_logits_t.to(device=forced_t.device, dtype=torch.float32)
            if int(getattr(rollout_logits_t, "ndim", 0)) != 2 or int(rollout_logits_t.shape[0]) != lim:
                return 0.0, {"base_quality": float(base_quality), "after_quality": float(base_quality)}
            rollout_pred_t = torch.argmax(rollout_logits_t, dim=-1)
    except Exception:
        return 0.0, {"base_quality": float(base_quality), "after_quality": float(base_quality)}

    after_source = _project_completion_source(
        current_tokens_t=forced_t,
        committed_mask_t=force_mask_t,
        predicted_tokens_t=rollout_pred_t,
        token_ids_to_code=token_ids_to_code,
    )
    after_quality, after_feedback = _source_quality_score(after_source, min_prefix_chars=min_prefix_chars)
    gain = float(max(after_quality - base_quality, 0.0))
    return float(np.clip(gain, 0.0, 1.0)), {
        "base_quality": float(base_quality),
        "after_quality": float(after_quality),
        "after_parse_ok": bool(after_feedback.get("parse_ok", False)),
    }


def _constraint_identifier_signal(
    pos: int,
    token_text: str,
    seq_len: int,
    parser_feedback: Dict[str, Any],
    parser_hotspot_active: bool,
    prompt_identifier_counts: Dict[str, int],
    source_identifier_counts: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    normalized = (
        str(token_text or "")
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\n")
        .strip()
    )
    tail_bias = float(pos) / float(max(int(seq_len) - 1, 1))

    constraint_score = float(_structure_sensitive_token_score(normalized))
    issue_type = str(parser_feedback.get("primary_issue", "none"))
    lowered = normalized.lower()
    if issue_type == "bracket" and any(ch in normalized for ch in "()[]{}"):
        constraint_score += 0.9
    if issue_type == "indent" and (
        ":" in normalized
        or "\n" in normalized
        or any(keyword in lowered for keyword in ("return", "if", "elif", "else", "for", "while", "try", "except", "finally"))
    ):
        constraint_score += 0.8
    if parser_hotspot_active:
        constraint_score += float(0.25 * tail_bias)

    identifier_score = 0.0
    matched_identifiers = 0
    for ident in _IDENTIFIER_RE.findall(normalized):
        if ident in source_identifier_counts:
            identifier_score += 0.45 + 0.1 * float(min(source_identifier_counts[ident], 3))
            matched_identifiers += 1
        elif ident in prompt_identifier_counts:
            identifier_score += 0.35 + 0.08 * float(min(prompt_identifier_counts[ident], 3))
            matched_identifiers += 1

    raw_score = float(constraint_score + identifier_score)
    scaled_score = float(np.clip(raw_score / 3.0, 0.0, 1.0))
    return scaled_score, {
        "constraint_score": float(constraint_score),
        "identifier_score": float(identifier_score),
        "matched_identifiers": float(matched_identifiers),
        "raw_score": float(raw_score),
    }


def _resolve_budgeted_entropy_pf_controls(
    mode: str,
    base_allow_pf_step: bool,
    pf_positions_cap: int,
    pf_particles_step: int,
    pf_horizon_steps: int,
    extra_forwards_used: int,
    extra_forward_budget: int,
    parser_hotspot_active: bool,
    current_gradient_hotspot: bool,
) -> Dict[str, Any]:
    mode_norm = str(mode or "legacy").lower()
    positions_cap = max(int(pf_positions_cap), 1)
    particles_step = max(int(pf_particles_step), 1)
    horizon_steps = max(int(pf_horizon_steps), 0)
    budget_cap = max(int(extra_forward_budget), 0)
    extra_used = max(int(extra_forwards_used), 0)
    remaining_budget = max(int(budget_cap - extra_used), 0) if budget_cap > 0 else -1

    if mode_norm != "budgeted_entropy":
        return {
            "allow_pf_step": bool(base_allow_pf_step),
            "budget_blocked": False,
            "budget_active": False,
            "syntax_repair_mode": bool(parser_hotspot_active or current_gradient_hotspot),
            "pf_positions_cap": int(positions_cap),
            "pf_particles_step": int(particles_step),
            "remaining_extra_forward_budget": int(remaining_budget),
            "estimated_pf_forward_cost": int(particles_step * horizon_steps),
        }

    syntax_repair_mode = bool(parser_hotspot_active or current_gradient_hotspot)
    positions_cap = 1
    particle_ceiling = 3 if syntax_repair_mode else 2
    particles_step = min(int(particles_step), int(particle_ceiling))

    budget_blocked = False
    if budget_cap > 0:
        if remaining_budget <= 0:
            budget_blocked = True
        elif horizon_steps > 0:
            particles_by_budget = int(remaining_budget // horizon_steps)
            if particles_by_budget <= 0:
                budget_blocked = True
            else:
                particles_step = min(int(particles_step), int(particles_by_budget))

    return {
        "allow_pf_step": bool(base_allow_pf_step and not budget_blocked),
        "budget_blocked": bool(budget_blocked),
        "budget_active": bool(budget_cap > 0),
        "syntax_repair_mode": bool(syntax_repair_mode),
        "pf_positions_cap": int(max(particles_step and positions_cap, 1)),
        "pf_particles_step": int(max(particles_step, 1)),
        "remaining_extra_forward_budget": int(remaining_budget),
        "estimated_pf_forward_cost": int(max(particles_step, 1) * horizon_steps),
    }


def _selected_pf_particle_parser_delta(
    pf_logs: Sequence[Dict[str, Any]],
    chosen_token: int,
) -> Dict[str, Any]:
    if not pf_logs:
        return {"observed": False}
    log0 = pf_logs[0] if isinstance(pf_logs[0], dict) else {}
    tokens = list(log0.get("particle_tokens", []) or [])
    try:
        chosen_idx = [int(tok) for tok in tokens].index(int(chosen_token))
    except ValueError:
        return {"observed": False}

    base_values = list(log0.get("particle_syntax_base_quality", []) or [])
    candidate_values = list(log0.get("particle_syntax_candidate_quality", []) or [])
    parse_values = list(log0.get("particle_parse_ok", []) or [])
    if chosen_idx >= len(base_values) or chosen_idx >= len(candidate_values):
        return {"observed": False}

    base_quality = float(base_values[chosen_idx])
    candidate_quality = float(candidate_values[chosen_idx])
    candidate_parse_ok = bool(parse_values[chosen_idx]) if chosen_idx < len(parse_values) else True
    observed = bool(base_quality != 0.0 or candidate_quality != 0.0 or not candidate_parse_ok)
    return {
        "observed": bool(observed),
        "base_quality": float(base_quality),
        "candidate_quality": float(candidate_quality),
        "candidate_parse_ok": bool(candidate_parse_ok),
    }


def _parser_feedback_after_forced_token(
    current_tokens_t: Any,
    committed_mask_t: Any,
    predicted_tokens_t: Any,
    pos: int,
    token: int,
    token_ids_to_code: Optional[Callable[[Any], str]],
    source_prefix: str,
) -> Tuple[Dict[str, Any], str]:
    if token_ids_to_code is None:
        return _default_parser_feedback(), ""
    try:
        import torch

        current_t = current_tokens_t if torch.is_tensor(current_tokens_t) else torch.as_tensor(current_tokens_t)
        committed_t = committed_mask_t if torch.is_tensor(committed_mask_t) else torch.as_tensor(committed_mask_t)
        predicted_t = predicted_tokens_t if torch.is_tensor(predicted_tokens_t) else torch.as_tensor(predicted_tokens_t)
        current_t = current_t.to(dtype=torch.long).reshape(-1).clone()
        committed_t = committed_t.to(device=current_t.device, dtype=torch.bool).reshape(-1).clone()
        predicted_t = predicted_t.to(device=current_t.device, dtype=torch.long).reshape(-1)
        lim = min(int(current_t.numel()), int(committed_t.numel()), int(predicted_t.numel()))
        pos = int(pos)
        if lim <= 0 or pos < 0 or pos >= lim:
            feedback = _default_parser_feedback()
            feedback.update(
                {
                    "observed": True,
                    "parse_ok": False,
                    "primary_issue": "invalid_pf_position",
                    "issue_types": ["invalid_pf_position"],
                    "syntax_error_message": "invalid_pf_position",
                    "quality_score": 0.0,
                    "severity_score": 1.0,
                }
            )
            return feedback, ""
        current_t[pos] = int(token)
        committed_t[pos] = True
        completion_source = _project_completion_source(
            current_tokens_t=current_t[:lim],
            committed_mask_t=committed_t[:lim],
            predicted_tokens_t=predicted_t[:lim],
            token_ids_to_code=token_ids_to_code,
        )
        feedback = parser_feedback_from_source(
            source=str(source_prefix or "") + completion_source,
            min_prefix_chars=0,
        )
        return feedback, completion_source
    except Exception as exc:
        feedback = _default_parser_feedback()
        feedback.update(
            {
                "observed": True,
                "parse_ok": False,
                "primary_issue": "decode_error",
                "issue_types": ["decode_error"],
                "syntax_error_message": f"decode_error: {exc}",
                "quality_score": 0.0,
                "severity_score": 1.0,
            }
        )
        return feedback, ""


def _should_accept_budgeted_pf_choice(
    base_quality: float,
    candidate_quality: float,
    candidate_parse_ok: bool,
    tolerance: float = 0.02,
) -> bool:
    tolerance = max(float(tolerance), 0.0)
    if float(candidate_quality) + tolerance < float(base_quality):
        return False
    if not bool(candidate_parse_ok) and float(base_quality) >= 1.0 - tolerance:
        return False
    return True


def _select_pf_candidate_positions(
    masked_positions_t: Any,
    entropy_all_t: Any,
    argmax_all_t: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
    budget: int,
    parser_hotspot_active: bool,
) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is required for Dream path
        raise ImportError("PF candidate selection requires torch.") from exc

    if not torch.is_tensor(masked_positions_t):
        masked_positions_t = torch.as_tensor(masked_positions_t, dtype=torch.long)
    if not torch.is_tensor(entropy_all_t):
        entropy_all_t = torch.as_tensor(entropy_all_t, dtype=torch.float32)
    if not torch.is_tensor(argmax_all_t):
        argmax_all_t = torch.as_tensor(argmax_all_t, dtype=torch.long)

    masked_positions_t = masked_positions_t.reshape(-1)
    if int(masked_positions_t.numel()) <= 0:
        return [], {}

    budget = min(max(int(budget), 1), 2, int(masked_positions_t.numel()))
    masked_entropy_t = entropy_all_t[masked_positions_t]
    top_entropy_rel = int(torch.argmax(masked_entropy_t).item())
    top_entropy_pos = int(masked_positions_t[top_entropy_rel].item())
    seq_len = max(int(entropy_all_t.numel()), 1)

    selected: List[int] = [top_entropy_pos]
    meta: Dict[int, Dict[str, float]] = {}
    top_entropy_val = float(entropy_all_t[top_entropy_pos].item())
    meta[top_entropy_pos] = {
        "entropy": float(top_entropy_val),
        "structure_score": float(
            _structure_sensitive_token_score(
                _decode_single_token_text(int(argmax_all_t[top_entropy_pos].item()), token_ids_to_code)
            )
        ),
        "priority_score": float(top_entropy_val),
    }

    if budget == 1:
        return selected, meta

    candidate_rows: List[Tuple[float, float, float, float, int]] = []
    for pos_t in masked_positions_t:
        pos = int(pos_t.item())
        if pos == top_entropy_pos:
            continue
        entropy_val = float(entropy_all_t[pos].item())
        token_text = _decode_single_token_text(int(argmax_all_t[pos].item()), token_ids_to_code)
        structure_score = _structure_sensitive_token_score(token_text)
        tail_bias = float(pos) / float(max(seq_len - 1, 1))
        priority = float(entropy_val + 0.35 * structure_score)
        if parser_hotspot_active:
            priority += float(0.2 * tail_bias + 0.25 * structure_score)
        candidate_rows.append((priority, entropy_val, structure_score, tail_bias, pos))

    if not candidate_rows:
        return selected, meta

    candidate_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]), reverse=True)
    best_priority, entropy_val, structure_score, _, pos = candidate_rows[0]
    selected.append(int(pos))
    meta[int(pos)] = {
        "entropy": float(entropy_val),
        "structure_score": float(structure_score),
        "priority_score": float(best_priority),
    }
    return selected, meta


def _bump_parser_issue_hist(hist: Dict[int, int], t_remaining: int) -> None:
    t_now = int(t_remaining)
    hist[t_now] = int(hist.get(t_now, 0)) + 1


def _bump_parser_score_hist(hist: Dict[int, float], t_remaining: int, score: float) -> None:
    t_now = int(t_remaining)
    hist[t_now] = float(hist.get(t_now, 0.0)) + float(max(score, 0.0))


def _parser_hotspot_score(
    issue_histograms: Dict[str, Dict[int, float]],
    t_remaining: int,
    radius: int,
) -> float:
    t_now = int(t_remaining)
    radius = max(int(radius), 0)
    score = 0.0
    for hist in issue_histograms.values():
        if not isinstance(hist, dict):
            continue
        for delta in range(-radius, radius + 1):
            score_at_t = float(hist.get(int(t_now + delta), 0.0))
            if score_at_t <= 0.0:
                continue
            score += float(score_at_t) / float(1 + abs(delta))
    return float(score)


def _top_parser_issue_timesteps(
    issue_histograms: Dict[str, Dict[int, int]],
    limit: int = 5,
) -> Dict[str, List[Dict[str, int]]]:
    top_steps: Dict[str, List[Dict[str, int]]] = {}
    for issue_type, hist in issue_histograms.items():
        if not isinstance(hist, dict):
            top_steps[str(issue_type)] = []
            continue
        ranked = sorted(hist.items(), key=lambda item: (-int(item[1]), -int(item[0])))
        top_steps[str(issue_type)] = [
            {"t": int(t), "count": int(count)}
            for t, count in ranked[: max(int(limit), 0)]
        ]
    return top_steps


def _top_parser_score_timesteps(
    issue_histograms: Dict[str, Dict[int, float]],
    limit: int = 5,
) -> Dict[str, List[Dict[str, float]]]:
    top_steps: Dict[str, List[Dict[str, float]]] = {}
    for issue_type, hist in issue_histograms.items():
        if not isinstance(hist, dict):
            top_steps[str(issue_type)] = []
            continue
        ranked = sorted(hist.items(), key=lambda item: (-float(item[1]), -int(item[0])))
        top_steps[str(issue_type)] = [
            {"t": int(t), "score": float(score)}
            for t, score in ranked[: max(int(limit), 0)]
        ]
    return top_steps


def _parser_gradient_metrics(
    parser_feedback: Dict[str, Any],
    prev_severity: float,
    prev_delta: float,
) -> Dict[str, float]:
    quality_score = float(parser_feedback.get("quality_score", 1.0 if parser_feedback.get("parse_ok", True) else 0.0))
    quality_score = float(np.clip(quality_score, 0.0, 1.0))
    severity = float(parser_feedback.get("severity_score", 1.0 - quality_score))
    severity = float(np.clip(severity, 0.0, 1.0))
    severity_delta = float(max(severity - float(prev_severity), 0.0))
    severity_accel = float(max(severity_delta - float(prev_delta), 0.0))
    issue_weight = 1.0
    if bool(parser_feedback.get("indent_issue")):
        issue_weight = 1.15
    elif bool(parser_feedback.get("bracket_issue")):
        issue_weight = 1.0
    score = float(issue_weight * max(severity_delta, severity_accel))
    return {
        "quality": float(quality_score),
        "severity": float(severity),
        "delta": float(severity_delta),
        "accel": float(severity_accel),
        "score": float(score),
    }


class _RiskTraceLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return False
        return "risk_trace step=" in msg and "max_risk=" in msg and "allow_pf=" in msg


def _attach_risk_trace_log_handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(_RiskTraceLogFilter())
    logging.getLogger().addHandler(handler)
    return handler


_POSTPROCESS_STOP_PREFIXES = (
    "###",
    "explanation",
    "example usage",
    "examples",
    "output:",
    "note:",
    "here is",
    "here's",
    "sure",
)


def _line_indent_width(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _strip_markdown_fence_region(text: str) -> str:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    fence_positions = [idx for idx, line in enumerate(lines) if line.strip().startswith("```")]
    if fence_positions:
        first_fence = int(fence_positions[0])
        first_nonblank = next((idx for idx, line in enumerate(lines) if line.strip()), first_fence)
        preamble = [line.strip().lower() for line in lines[first_nonblank:first_fence] if line.strip()]
        preamble_is_prose = bool(preamble) and all(
            not line.startswith(("def ", "class ", "from ", "import ", "return ", "if ", "for ", "while ", "try:"))
            and not line.startswith("#")
            and _line_indent_width(line) == 0
            for line in preamble
        )
        if first_fence == first_nonblank or preamble_is_prose:
            end_fence = next((idx for idx in fence_positions[1:] if idx > first_fence), len(lines))
            return "\n".join(lines[first_fence + 1 : end_fence])

    out: List[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if started:
                break
            started = True
            continue
        if not started and not stripped:
            continue
        started = True
        out.append(line)
    return "\n".join(out)


def _is_postprocess_stop_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return True
    return any(lowered.startswith(prefix) for prefix in _POSTPROCESS_STOP_PREFIXES)


def _normalize_function_body_lines(lines: Sequence[str]) -> str:
    body = list(lines)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return ""

    nonblank = [line for line in body if line.strip()]
    has_indented = any(_line_indent_width(line) > 0 for line in nonblank)
    if not has_indented:
        body = [("    " + line if line.strip() else line) for line in body]

    return "\n".join(body).rstrip() + "\n"


def _extract_body_from_full_function(text: str, entry_point: str) -> str:
    if not entry_point:
        return ""
    pattern = re.compile(rf"^(\s*)def\s+{re.escape(str(entry_point))}\s*\(", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""

    lines = text[match.start() :].splitlines()
    if not lines:
        return ""
    def_indent = _line_indent_width(lines[0])
    body_lines: List[str] = []
    body_started = False
    for line in lines[1:]:
        stripped = line.strip()
        if _is_postprocess_stop_line(line):
            break
        if not stripped:
            if body_started:
                body_lines.append(line)
            continue
        indent = _line_indent_width(line)
        if body_started and indent <= def_indent:
            break
        if not body_started and indent <= def_indent:
            continue
        body_started = True
        body_lines.append(line)
    return _normalize_function_body_lines(body_lines)


def _extract_humaneval_body_completion(item: Dict[str, Any], completion: str) -> str:
    prompt = str(item.get("prompt", ""))
    entry_point = str(item.get("entry_point", ""))
    text = _strip_markdown_fence_region(str(completion or ""))
    if prompt and prompt in text:
        text = text.split(prompt)[-1]

    full_function_body = _extract_body_from_full_function(text, entry_point)
    if full_function_body:
        return full_function_body

    lines = text.splitlines()
    body_lines: List[str] = []
    body_started = False
    saw_indented_body = False
    for line in lines:
        stripped = line.strip()
        if _is_postprocess_stop_line(line):
            break
        if not stripped:
            if body_started:
                body_lines.append(line)
            continue

        indent = _line_indent_width(line)
        lowered = stripped.lower()
        if not body_started:
            if lowered.startswith(("def ", "from ", "import ", "@")):
                continue
            if lowered.startswith(("here ", "sure", "the solution")):
                continue
            body_started = True

        if body_started and saw_indented_body and indent == 0:
            break
        if indent > 0:
            saw_indented_body = True
        body_lines.append(line)

    return _normalize_function_body_lines(body_lines)


def _extract_code_completion(completion: str) -> str:
    text = _strip_markdown_fence_region(str(completion or ""))
    lines = text.splitlines()
    code_lines: List[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if _is_postprocess_stop_line(line):
            break
        if not stripped:
            if started:
                code_lines.append(line)
            continue
        if not started:
            if lowered.startswith(("def ", "class ", "from ", "import ", "@")):
                started = True
            else:
                continue
        elif _line_indent_width(line) == 0 and lowered.startswith(("here ", "sure", "the solution")):
            break
        code_lines.append(line)
    processed = "\n".join(code_lines).rstrip()
    return processed + "\n" if processed else ""


def postprocess_completion_for_eval(
    item: Dict[str, Any],
    completion: str,
    dataset: str,
) -> Tuple[str, Dict[str, Any]]:
    raw = str(completion or "")
    if dataset == "humaneval":
        processed = _extract_humaneval_body_completion(item=item, completion=raw)
    else:
        processed = _extract_code_completion(raw)

    meta = {
        "enabled": True,
        "mode": "humaneval_function_body" if dataset == "humaneval" else "strip_markdown_fence",
        "changed": processed != raw,
        "raw_len": int(len(raw)),
        "processed_len": int(len(processed)),
    }
    return processed, meta


def format_success(code: str) -> bool:
    text = code.strip()
    if not text:
        return False
    if "```" in text:
        return False
    lowered = text.lower()
    if lowered.startswith("here") or lowered.startswith("sure") or lowered.startswith("explanation"):
        return False
    return True


def parse_success(prompt: str, completion: str) -> bool:
    try:
        import ast

        ast.parse(prompt + completion)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def parse_success_for_eval(item: Dict[str, Any], prompt: str, completion: str, dataset: str) -> bool:
    del item
    if dataset == "humaneval":
        return parse_success(prompt, completion)
    try:
        import ast

        ast.parse(str(completion or ""))
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def _obvious_truncation_for_eval(completion: str) -> bool:
    text = str(completion or "").rstrip()
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
    if stripped.endswith(trailing_ops):
        return True
    return False


def _level0_selection_score(format_ok: bool, parse_ok: bool, completion: str) -> float:
    score = 0.0
    score += 1.0 if bool(format_ok) else 0.0
    score += 2.0 if bool(parse_ok) else 0.0
    if str(completion or "").strip():
        score += 0.25
    if _obvious_truncation_for_eval(completion):
        score -= 1.0
    return float(score)


def _target_function_present_for_eval(item: Dict[str, Any], prompt: str, completion: str, dataset: str) -> bool:
    entry_point = str(item.get("entry_point", "") or "").strip()
    if not entry_point:
        return False
    source = str(completion or "")
    if str(dataset or "").lower() == "humaneval":
        prompt_text = str(prompt or "")
        if prompt_text and not prompt_text.endswith("\n"):
            prompt_text += "\n"
        source = f"{prompt_text}{source}"
    try:
        tree = ast.parse(source)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and str(getattr(node, "name", "")) == entry_point:
            return True
    return False


def _stable_prompt_seed(prompt: str) -> int:
    seed = 0
    for idx, ch in enumerate(str(prompt or "")[:2048]):
        seed = (seed + (idx + 1) * ord(ch)) % (2**31 - 1)
    return int(seed)


def _resolve_branch_observe_trigger_mode(cfg: DecoderConfig) -> str:
    mode = str(getattr(cfg, "branch_observe_trigger_mode", "auto") or "auto").lower()
    if mode != "auto":
        return mode
    local_mode = str(getattr(cfg, "local_beam_mode", "entropy_kl_struct") or "entropy_kl_struct").lower()
    if local_mode in {"entropy_only", "kl_only", "entropy_kl"}:
        return local_mode
    return "legacy_entropy_kl_struct"


def _shadow_trigger_score_t(trigger_mode: str, entropy_norm_t: Any, kl_norm_t: Any, struct_weight_t: Any) -> Any:
    mode = str(trigger_mode or "legacy_entropy_kl_struct").lower()
    if mode == "entropy_only":
        return entropy_norm_t
    if mode == "kl_only":
        return kl_norm_t
    if mode == "entropy_kl":
        return entropy_norm_t * kl_norm_t
    if mode == "entropy_kl_struct":
        return entropy_norm_t * kl_norm_t * struct_weight_t
    if mode == "entropy_struct":
        return entropy_norm_t * struct_weight_t
    if mode == "kl_struct":
        return kl_norm_t * struct_weight_t
    return entropy_norm_t * kl_norm_t


def _select_branch_observe_candidate_indices(
    *,
    selection_score_t: Any,
    structure_score_t: Any,
    conf_proxy_t: Any,
    max_count: int,
    event_policy: str,
    rng: random.Random,
) -> List[int]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is required in Dream path
        raise ImportError("Branch-observe candidate selection requires torch.") from exc
    total = int(getattr(selection_score_t, "numel", lambda: 0)()) if hasattr(selection_score_t, "numel") else len(selection_score_t)
    if total <= 0 or int(max_count) <= 0:
        return []
    max_count = min(int(max_count), total)
    policy = str(event_policy or "top_risk").lower()
    if policy == "top_risk":
        return [int(v) for v in torch.topk(selection_score_t, k=max_count, dim=0).indices.detach().cpu().tolist()]

    structure_vals = [float(v) for v in structure_score_t.detach().cpu().tolist()]
    candidate_ids = list(range(total))
    structural_ids = [idx for idx, score in enumerate(structure_vals) if float(score) > 0.0]
    if policy == "highest_conf_structural":
        if not structural_ids:
            return [int(v) for v in torch.topk(selection_score_t, k=max_count, dim=0).indices.detach().cpu().tolist()]
        ranked = sorted(
            structural_ids,
            key=lambda idx: (
                float(conf_proxy_t[idx].detach().cpu().item()),
                float(selection_score_t[idx].detach().cpu().item()),
                -idx,
            ),
            reverse=True,
        )
        return [int(v) for v in ranked[:max_count]]

    if policy == "random_structural" and structural_ids:
        candidate_ids = structural_ids
    sample_ids = list(candidate_ids)
    rng.shuffle(sample_ids)
    return [int(v) for v in sample_ids[:max_count]]


def _build_branch_observe_rollout_specs(
    *,
    official_token: Optional[int],
    top_token_ids: Sequence[Any],
    branch_observe_top_k: int,
    branch_observe_beam_size: int,
    branch_observe_include_delay: bool,
    allow_token_fallback: bool = True,
) -> List[Tuple[str, Optional[int]]]:
    beam_size = max(int(branch_observe_beam_size), 1)
    top_k = max(int(branch_observe_top_k), 0)
    include_delay = bool(branch_observe_include_delay)
    used_tokens = {int(official_token)} if official_token is not None else set()
    token_candidates: List[int] = []
    if top_k > 0:
        max_candidate_count = max(beam_size - 1 - int(include_delay), 0)
        for raw_token in list(top_token_ids or [])[:top_k]:
            token_i = int(raw_token)
            if token_i in used_tokens:
                continue
            used_tokens.add(token_i)
            token_candidates.append(token_i)
            if len(token_candidates) >= max_candidate_count:
                break
        if not token_candidates and top_token_ids and allow_token_fallback:
            fallback_token: Optional[int] = None
            for raw_token in list(top_token_ids or []):
                token_i = int(raw_token)
                if token_i not in used_tokens:
                    fallback_token = token_i
                    break
            if fallback_token is None:
                fallback_token = int(list(top_token_ids)[0])
            token_candidates.append(int(fallback_token))
    rollout_specs: List[Tuple[str, Optional[int]]] = [("candidate", tok) for tok in token_candidates]
    if include_delay and len(rollout_specs) + 1 < beam_size:
        rollout_specs.append(("delay", None))
    return rollout_specs[: max(beam_size - 1, 0)]


def _summarize_branch_select_records(branch_select_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    selected_kind_counts: Counter[str] = Counter()
    selected_candidate_count = 0
    selected_delay_count = 0
    selected_parse_repair_count = 0
    selected_format_repair_count = 0
    selected_truncation_repair_count = 0
    selected_target_function_repair_count = 0
    for row in branch_select_records:
        reason_counts[str(row.get("reason", ""))] += 1
        if not bool(row.get("selected", False)):
            continue
        selected_kind = str(row.get("selected_kind", "") or "")
        selected_kind_counts[selected_kind] += 1
        if selected_kind == "candidate":
            selected_candidate_count += 1
        elif selected_kind == "delay":
            selected_delay_count += 1
        baseline = row.get("baseline", {}) if isinstance(row.get("baseline", {}), dict) else {}
        if (not bool(baseline.get("parse_ok", False))) and bool(row.get("selected_parse_ok", False)):
            selected_parse_repair_count += 1
        if (not bool(baseline.get("format_ok", False))) and bool(row.get("selected_format_ok", False)):
            selected_format_repair_count += 1
        if bool(baseline.get("obvious_truncation", False)) and not bool(row.get("selected_obvious_truncation", False)):
            selected_truncation_repair_count += 1
        if (not bool(baseline.get("target_function_present", False))) and bool(
            row.get("selected_target_function_present", False)
        ):
            selected_target_function_repair_count += 1
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "selected_kind_counts": dict(sorted(selected_kind_counts.items())),
        "selected_candidate_count": int(selected_candidate_count),
        "selected_delay_count": int(selected_delay_count),
        "selected_level0_parse_repair_count": int(selected_parse_repair_count),
        "selected_level0_format_repair_count": int(selected_format_repair_count),
        "selected_level0_truncation_repair_count": int(selected_truncation_repair_count),
        "selected_level0_target_func_repair_count": int(selected_target_function_repair_count),
    }


def _normalize_visible_test_values(values: Any) -> List[str]:
    if isinstance(values, list):
        raw_values = values
    elif isinstance(values, tuple):
        raw_values = list(values)
    elif str(values or "").strip():
        raw_values = [values]
    else:
        return []
    tests: List[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        assert_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("assert ")]
        if assert_lines:
            tests.extend(assert_lines)
        else:
            tests.append(text)
    return tests


def _visible_mbpp_tests(item: Dict[str, Any], prompt: str = "") -> List[str]:
    for key in ("visible_tests", "public_tests", "visible_test_list", "public_test_list"):
        values = item.get(key)
        tests = _normalize_visible_test_values(values)
        if tests:
            return tests
    prompt_text = str(prompt or item.get("prompt", "") or "")
    tests: List[str] = []
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">>>"):
            stripped = stripped[3:].strip()
        if stripped.startswith("assert "):
            tests.append(stripped)
    return tests


def _level1_branch_candidate_decision(
    *,
    baseline_level0_pass: bool,
    baseline_level0_score: float,
    baseline_visible: Dict[str, Any],
    cand_format_ok: bool,
    cand_parse_ok: bool,
    cand_obvious_truncation: bool,
    cand_level0_score: float,
    cand_visible: Dict[str, Any],
    model_score: float,
    min_score_gain: float,
    visible_min_pass_gain: int,
    visible_require_level0: bool,
) -> Dict[str, Any]:
    baseline_visible_enabled = bool(isinstance(baseline_visible, dict) and baseline_visible.get("enabled", False))
    baseline_visible_passed = int(baseline_visible.get("passed", 0)) if baseline_visible_enabled else 0
    baseline_visible_total = int(baseline_visible.get("total", 0)) if baseline_visible_enabled else 0
    baseline_visible_pass = bool(
        baseline_level0_pass
        and baseline_visible_enabled
        and baseline_visible_total > 0
        and bool(baseline_visible.get("all_passed", False))
    )
    baseline_visible_fail = bool(
        baseline_level0_pass
        and baseline_visible_enabled
        and baseline_visible_total > 0
        and not bool(baseline_visible.get("all_passed", False))
    )

    level0_ok = bool(cand_format_ok and cand_parse_ok and not cand_obvious_truncation)
    if not bool(visible_require_level0):
        level0_ok = True

    cand_visible_enabled = bool(isinstance(cand_visible, dict) and cand_visible.get("enabled", False))
    cand_visible_passed = int(cand_visible.get("passed", 0)) if cand_visible_enabled else 0
    cand_visible_total = int(cand_visible.get("total", 0)) if cand_visible_enabled else 0
    cand_visible_gain = int(cand_visible_passed - baseline_visible_passed)

    level0_gain_ok = bool(
        (not baseline_level0_pass)
        and cand_format_ok
        and cand_parse_ok
        and not cand_obvious_truncation
        and float(cand_level0_score) >= float(baseline_level0_score) + float(min_score_gain)
    )
    visible_gain_ok = bool(
        baseline_visible_fail
        and cand_visible_enabled
        and cand_visible_total > 0
        and cand_visible_gain >= int(visible_min_pass_gain)
    )
    eligible = bool(level0_ok and not baseline_visible_pass and (level0_gain_ok or visible_gain_ok))
    repair_mode = "none"
    if eligible and visible_gain_ok:
        repair_mode = "level1"
    elif eligible and level0_gain_ok:
        repair_mode = "level0"

    select_score = float(
        (100.0 if visible_gain_ok else 0.0)
        + (20.0 if level0_gain_ok else 0.0)
        + 10.0 * cand_visible_passed
        + float(cand_visible.get("pass_rate", 0.0) if cand_visible_enabled else 0.0)
        + 0.1 * float(cand_level0_score)
        + 0.01 * float(model_score)
    )
    return {
        "eligible": bool(eligible),
        "select_score": float(select_score),
        "level0_ok": bool(level0_ok),
        "level0_gain_ok": bool(level0_gain_ok),
        "visible_gain_ok": bool(visible_gain_ok),
        "visible_gain": int(cand_visible_gain),
        "repair_mode": repair_mode,
        "baseline_visible_pass": bool(baseline_visible_pass),
        "baseline_visible_fail": bool(baseline_visible_fail),
    }


def _humaneval_visible_doctest_count(prompt: str) -> int:
    return sum(1 for line in str(prompt or "").splitlines() if line.lstrip().startswith(">>>"))


def _visible_test_score(
    item: Dict[str, Any],
    prompt: str,
    completion: str,
    dataset: str,
    timeout: float,
) -> Dict[str, Any]:
    dataset_name = str(dataset or "").lower()
    if dataset_name == "mbpp":
        return _visible_mbpp_score(item=item, prompt=prompt, completion=completion, timeout=timeout)
    if dataset_name == "humaneval":
        return _visible_humaneval_score(item=item, prompt=prompt, completion=completion, timeout=timeout)
    return {
        "enabled": False,
        "source": "none",
        "passed": 0,
        "total": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "all_passed": False,
        "error": "unsupported_dataset",
    }


def _visible_mbpp_score(item: Dict[str, Any], prompt: str, completion: str, timeout: float) -> Dict[str, Any]:
    tests = _visible_mbpp_tests(item, prompt=prompt)
    if not tests:
        return {
            "enabled": False,
            "source": "mbpp_visible_tests",
            "passed": 0,
            "total": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "all_passed": False,
            "error": "missing_visible_tests",
        }

    imports = item.get("test_imports", [])
    import_lines: List[str] = []
    if isinstance(imports, list):
        import_lines.extend(str(v) for v in imports if str(v).strip())
    elif str(imports).strip():
        import_lines.append(str(imports))
    setup = str(item.get("test_setup_code", "") or "")
    prelude = "\n".join([*import_lines, setup, str(completion or "")])
    runner = (
        "import json\n"
        f"ns = {{}}\n"
        f"tests = {repr(tests)}\n"
        f"source = {repr(prelude)}\n"
        "passed = 0\n"
        "failures = []\n"
        "try:\n"
        "    exec(source, ns)\n"
        "except Exception as exc:\n"
        "    failures = [{'index': -1, 'error': type(exc).__name__ + ': ' + str(exc)}]\n"
        "else:\n"
        "    for idx, test in enumerate(tests):\n"
        "        try:\n"
        "            exec(test, ns)\n"
        "            passed += 1\n"
        "        except Exception as exc:\n"
        "            failures.append({'index': idx, 'error': type(exc).__name__ + ': ' + str(exc)})\n"
        "print(json.dumps({'passed': passed, 'total': len(tests), 'failures': failures}))\n"
    )
    return _run_visible_score_subprocess(
        runner=runner,
        timeout=timeout,
        source="mbpp_visible_tests",
        planned_total=len(tests),
    )


def _visible_humaneval_score(
    item: Dict[str, Any],
    prompt: str,
    completion: str,
    timeout: float,
) -> Dict[str, Any]:
    del item
    planned_total = _humaneval_visible_doctest_count(prompt)
    if planned_total <= 0:
        return {
            "enabled": False,
            "source": "humaneval_prompt_doctest",
            "passed": 0,
            "total": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "all_passed": False,
            "error": "missing_prompt_doctests",
        }
    source = str(prompt or "") + str(completion or "")
    runner = (
        "import doctest, json, types\n"
        f"source = {repr(source)}\n"
        "mod = types.ModuleType('candidate')\n"
        "try:\n"
        "    exec(source, mod.__dict__)\n"
        "    result = doctest.testmod(mod, verbose=False)\n"
        "    failed = int(result.failed)\n"
        "    total = int(result.attempted)\n"
        "    passed = max(total - failed, 0)\n"
        "    error = ''\n"
        "except Exception as exc:\n"
        f"    total = {int(planned_total)}\n"
        "    passed = 0\n"
        "    failed = total\n"
        "    error = type(exc).__name__ + ': ' + str(exc)\n"
        "print(json.dumps({'passed': passed, 'total': total, 'failures': [], 'error': error}))\n"
    )
    return _run_visible_score_subprocess(
        runner=runner,
        timeout=timeout,
        source="humaneval_prompt_doctest",
        planned_total=planned_total,
    )


def _run_visible_score_subprocess(
    runner: str,
    timeout: float,
    source: str,
    planned_total: int,
) -> Dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="visible_selector_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(float(timeout), 0.1),
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "enabled": True,
            "source": source,
            "passed": 0,
            "total": int(planned_total),
            "failed": int(planned_total),
            "pass_rate": 0.0,
            "all_passed": False,
            "timeout": True,
            "error": "timed_out",
        }
    except Exception as exc:
        return {
            "enabled": True,
            "source": source,
            "passed": 0,
            "total": int(planned_total),
            "failed": int(planned_total),
            "pass_rate": 0.0,
            "all_passed": False,
            "timeout": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    payload: Dict[str, Any] = {}
    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        payload = {}
    total = int(payload.get("total", planned_total) or planned_total)
    passed = int(payload.get("passed", 0) or 0)
    failed = max(int(total - passed), 0)
    error = str(payload.get("error", "") or "")
    if int(proc.returncode) != 0 and not error:
        error = (str(proc.stderr or "").strip() or "nonzero_returncode")[-1000:]
    return {
        "enabled": True,
        "source": source,
        "passed": int(max(passed, 0)),
        "total": int(max(total, 0)),
        "failed": int(failed),
        "pass_rate": float(passed / total) if total > 0 else 0.0,
        "all_passed": bool(total > 0 and passed == total and int(proc.returncode) == 0),
        "timeout": False,
        "error": error,
        "returncode": int(proc.returncode),
        "stderr": str(proc.stderr or "")[-1000:],
        "failures": payload.get("failures", []),
    }


_CHECK_CORRECTNESS = None


def _get_check_correctness():
    global _CHECK_CORRECTNESS
    if _CHECK_CORRECTNESS is not None:
        return _CHECK_CORRECTNESS
    try:
        from human_eval.execution import check_correctness

        _CHECK_CORRECTNESS = check_correctness
    except Exception:
        _CHECK_CORRECTNESS = None
    return _CHECK_CORRECTNESS


def _run_humaneval_fallback(item: Dict[str, Any], completion: str, timeout: float) -> Dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    test = str(item.get("test", ""))
    entry_point = str(item.get("entry_point", ""))
    if not prompt or not test or not entry_point:
        return {"passed": False, "result": "missing_humaneval_fields"}

    source = prompt + str(completion or "")
    if not source.endswith("\n"):
        source += "\n"
    source += test
    if not source.endswith("\n"):
        source += "\n"
    source += f"\ncheck({entry_point})\n"

    try:
        with tempfile.TemporaryDirectory(prefix="humaneval_fallback_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", source],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(float(timeout), 0.1),
                check=False,
            )
        if int(proc.returncode) == 0:
            return {"passed": True, "result": "passed"}
        return {
            "passed": False,
            "result": "failed",
            "stderr": str(proc.stderr)[-2000:],
            "stdout": str(proc.stdout)[-2000:],
            "returncode": int(proc.returncode),
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "result": "timed_out"}
    except Exception as exc:
        return {"passed": False, "result": f"error: {type(exc).__name__}: {exc}"}


def _run_mbpp_fallback(item: Dict[str, Any], completion: str, timeout: float) -> Dict[str, Any]:
    test = str(item.get("test", ""))
    if not test and isinstance(item.get("test_list"), list):
        test = "\n".join(str(v) for v in item.get("test_list", []))
    if not test:
        return {"passed": False, "result": "missing_mbpp_tests"}

    source_parts: List[str] = []
    imports = item.get("test_imports", [])
    if isinstance(imports, list):
        source_parts.extend(str(v) for v in imports if str(v).strip())
    elif str(imports).strip():
        source_parts.append(str(imports))
    setup = str(item.get("test_setup_code", "") or "")
    if setup.strip():
        source_parts.append(setup)
    source_parts.append(str(completion or ""))
    source_parts.append(test)
    source = "\n\n".join(part.rstrip() for part in source_parts if part is not None)
    if not source.endswith("\n"):
        source += "\n"

    try:
        with tempfile.TemporaryDirectory(prefix="mbpp_fallback_") as tmpdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", source],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(float(timeout), 0.1),
                check=False,
            )
        if int(proc.returncode) == 0:
            return {"passed": True, "result": "passed"}
        return {
            "passed": False,
            "result": "failed",
            "stderr": str(proc.stderr)[-2000:],
            "stdout": str(proc.stdout)[-2000:],
            "returncode": int(proc.returncode),
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "result": "timed_out"}
    except Exception as exc:
        return {"passed": False, "result": f"error: {type(exc).__name__}: {exc}"}


def unit_test_pass(item: Dict[str, Any], completion: str, dataset: str, timeout: float) -> bool:
    if dataset == "mbpp":
        return bool(_run_mbpp_fallback(item, completion, timeout=timeout).get("passed", False))
    if dataset != "humaneval":
        return False
    if "test" not in item or "entry_point" not in item:
        return False
    checker = _get_check_correctness()
    if checker is None:
        return bool(_run_humaneval_fallback(item, completion, timeout=timeout).get("passed", False))
    try:
        result = checker(item, completion, timeout=timeout)
        return bool(result.get("passed", False))
    except Exception:
        return False


def run_baseline(prompt: str, sampler, max_steps: int = 20, seq_len: int = 64, token_ids_to_code=None) -> str:
    forced_tokens, slen = _run_baseline_with_sampler(prompt, sampler, max_steps, seq_len)
    if token_ids_to_code is not None:
        return token_ids_to_code(forced_tokens)
    return "".join(chr(ord("a") + int(t) % 26) for t in forced_tokens[:slen])


def run_dream_official_baseline(
    prompt: str,
    model,
    tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 768,
    diffusion_steps: int = 768,
    temperature: float = 0.1,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    eos_penalty: float = 3.0,
    use_chat_template: bool = False,
    max_prompt_tokens: int = 512,
) -> str:
    """Run Dream-Coder's official diffusion_generate decoding path."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when Dream backend is used
        raise ImportError("Dream official baseline requires torch.") from exc

    if use_chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)

    if max_prompt_tokens > 0 and int(input_ids.shape[-1]) > int(max_prompt_tokens):
        input_ids = input_ids[:, -int(max_prompt_tokens) :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -int(max_prompt_tokens) :]

    input_ids = input_ids.to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "output_history": False,
        "return_dict_in_generate": True,
        "steps": int(diffusion_steps),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "alg": str(alg),
        "alg_temp": float(alg_temp),
        "eos_penalty": float(eos_penalty),
    }
    with torch.no_grad():
        try:
            out = model.diffusion_generate(input_ids, **kwargs)
        except TypeError:
            # Some Dream variants do not expose eos_penalty.
            kwargs.pop("eos_penalty", None)
            out = model.diffusion_generate(input_ids, **kwargs)

    sequences = out.sequences if hasattr(out, "sequences") else out
    generated_ids = sequences[0, input_ids.shape[1] :]
    if hasattr(generated_ids, "detach"):
        generated_ids = generated_ids.detach().cpu()
    generated = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and eos_token in generated:
        generated = generated.split(eos_token)[0]
    return generated


def run_dream_official_shadow(
    prompt: str,
    model,
    tokenizer,
    cfg: DecoderConfig,
    device: str = "cuda",
    max_new_tokens: int = 768,
    diffusion_steps: int = 768,
    temperature: float = 0.1,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    eos_penalty: float = 3.0,
    use_chat_template: bool = False,
    max_prompt_tokens: int = 512,
    mask_id: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Observe entropy/KL/risk on the official Dream trajectory without modifying it."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when Dream backend is used
        raise ImportError("Dream shadow mode requires torch.") from exc

    if use_chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)

    if max_prompt_tokens > 0 and int(input_ids.shape[-1]) > int(max_prompt_tokens):
        input_ids = input_ids[:, -int(max_prompt_tokens) :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -int(max_prompt_tokens) :]

    input_ids = input_ids.to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    prompt_len = int(input_ids.shape[-1])
    effective_seq_len = int(max_new_tokens)
    comp_start = int(prompt_len)
    comp_end = int(prompt_len + effective_seq_len)
    tok_mask_id = int(mask_id) if mask_id is not None else int(getattr(tokenizer, "mask_token_id", -1))
    if tok_mask_id < 0:
        tok_mask_id = int(getattr(getattr(model, "generation_config", None), "mask_token_id", -1))
    if tok_mask_id < 0:
        raise ValueError("Dream shadow mode requires a valid mask token id.")

    entropy_top_k = max(int(getattr(cfg, "shadow_entropy_top_k", getattr(cfg, "local_beam_entropy_top_k", 64))), 2)
    kl_top_k = max(int(getattr(cfg, "shadow_kl_top_k", getattr(cfg, "local_beam_kl_top_k", 32))), 1)
    token_top_k = max(int(getattr(cfg, "shadow_token_top_k", getattr(cfg, "local_beam_top_k", 5))), 1)
    shadow_top_m = max(int(getattr(cfg, "shadow_top_m", 3)), 1)
    shadow_max_events = max(int(getattr(cfg, "shadow_max_events", 8)), 0)
    shadow_risk_threshold = max(float(getattr(cfg, "shadow_risk_threshold", getattr(cfg, "local_beam_tau_risk", 0.45))), 0.0)
    commit_lag_window = max(int(getattr(cfg, "shadow_commit_lag_window", 3)), 0)
    struct_alpha = max(float(getattr(cfg, "local_beam_struct_weight", 0.75)), 0.0)
    branch_observe_enabled = bool(getattr(cfg, "branch_observe_enabled", False))
    branch_observe_beam_size = max(int(getattr(cfg, "branch_observe_beam_size", 3)), 1)
    branch_observe_top_k = max(int(getattr(cfg, "branch_observe_top_k", 3)), 0)
    branch_observe_max_events = max(int(getattr(cfg, "branch_observe_max_events", 1)), 0)
    branch_observe_horizon = max(int(getattr(cfg, "branch_observe_horizon", getattr(cfg, "local_beam_horizon", 2))), 1)
    branch_observe_include_delay = bool(getattr(cfg, "branch_observe_include_delay", True))
    branch_observe_token_fallback = bool(getattr(cfg, "branch_observe_token_fallback", True))
    branch_observe_trigger_mode = _resolve_branch_observe_trigger_mode(cfg)
    branch_observe_event_policy = str(getattr(cfg, "branch_observe_event_policy", "top_risk") or "top_risk").lower()
    analysis_rng = random.Random(int(getattr(cfg, "random_seed", 42)) + _stable_prompt_seed(prompt))

    state: Dict[str, Any] = {
        "prev_logits_comp": None,
        "prev_x_comp": None,
        "commit_steps": {},
        "token_records": [],
        "risk_records": [],
        "risk_events": [],
        "step_commit_stats": [],
        "risk_sum_t": None,
        "risk_count": 0,
        "max_risk": 0.0,
        "num_steps": 0,
        "errors": [],
        "branch_observe_errors": [],
    }

    def _decode_shadow_token(token_id: int) -> str:
        try:
            return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except Exception:
            return ""

    def _rank_percentile(values_t: Any) -> Any:
        if int(values_t.numel()) <= 0:
            return values_t
        if int(values_t.numel()) == 1:
            return torch.where(values_t > 1e-12, torch.ones_like(values_t), torch.zeros_like(values_t))
        order_t = torch.argsort(values_t, dim=0)
        ranks_t = torch.empty_like(values_t, dtype=torch.float32)
        ranks_t[order_t] = torch.arange(int(values_t.numel()), device=values_t.device, dtype=torch.float32)
        return ranks_t / float(max(int(values_t.numel()) - 1, 1))

    def _completion_x_view(x: Any) -> Any:
        x_t = x if torch.is_tensor(x) else torch.as_tensor(x, device=device)
        if int(getattr(x_t, "ndim", 0)) == 2:
            x_t = x_t[0]
        return x_t[int(comp_start) : int(comp_end)]

    def tokens_shadow_hook(step, x, logits):
        del logits
        try:
            with torch.no_grad():
                comp_x_t = _completion_x_view(x)
                mask_t = comp_x_t == int(tok_mask_id)
                prev_x_t = state.get("prev_x_comp")
                if prev_x_t is None:
                    newly_t = torch.zeros_like(mask_t, dtype=torch.bool)
                else:
                    prev_x_t = prev_x_t.to(device=comp_x_t.device)
                    lim = min(int(prev_x_t.numel()), int(comp_x_t.numel()))
                    newly_t = torch.zeros_like(mask_t, dtype=torch.bool)
                    if lim > 0:
                        newly_t[:lim] = (prev_x_t[:lim] == int(tok_mask_id)) & (comp_x_t[:lim] != int(tok_mask_id))
                step_i = -1 if step is None else int(step)
                state["token_records"].append(
                    {
                        "step": int(step_i),
                        "num_masked_t": mask_t.sum().detach(),
                        "num_committed_t": (~mask_t).sum().detach(),
                        "newly_mask_t": newly_t.detach().clone(),
                    }
                )
                state["prev_x_comp"] = comp_x_t.detach().clone()
        except Exception as exc:  # pragma: no cover - defensive observer path
            state["errors"].append(f"tokens_hook:{type(exc).__name__}: {exc}")
        return x

    def logits_shadow_hook(step, x, logits):
        try:
            if step is None:
                return logits
            with torch.no_grad():
                if not torch.is_tensor(logits):
                    return logits
                logits_t = logits
                if int(getattr(logits_t, "ndim", 0)) != 3 or int(logits_t.shape[0]) != 1:
                    return logits
                comp_logits_t = logits_t[0, int(comp_start) : int(comp_end)]
                comp_x_t = _completion_x_view(x).to(device=comp_logits_t.device)
                seq_len = min(int(comp_logits_t.shape[0]), int(comp_x_t.numel()))
                if seq_len <= 0:
                    state["prev_logits_comp"] = comp_logits_t.detach()
                    return logits
                comp_logits_t = comp_logits_t[:seq_len]
                comp_x_t = comp_x_t[:seq_len]
                masked_pos_t = torch.nonzero(comp_x_t == int(tok_mask_id), as_tuple=False).flatten()
                state["num_steps"] = max(int(state["num_steps"]), int(step) + 1)
                if int(masked_pos_t.numel()) <= 0:
                    state["prev_logits_comp"] = comp_logits_t.detach()
                    return logits

                row_logits_t = comp_logits_t.index_select(0, masked_pos_t)
                vocab_size = int(row_logits_t.shape[-1])
                ent_k = min(max(int(entropy_top_k), 2), max(vocab_size, 2))
                ent_vals_t, ent_ids_t = torch.topk(row_logits_t, k=ent_k, dim=-1)
                ent_probs_t = torch.softmax(ent_vals_t.to(dtype=torch.float32), dim=-1)
                ent_t = -(ent_probs_t * torch.log(ent_probs_t.clamp_min(1e-12))).sum(dim=-1)
                ent_norm_t = (ent_t / float(np.log(float(ent_k)))).clamp(0.0, 1.0)

                prev_logits_t = state.get("prev_logits_comp")
                if prev_logits_t is None:
                    kl_raw_t = torch.zeros_like(ent_norm_t, dtype=torch.float32)
                    kl_norm_t = torch.zeros_like(ent_norm_t, dtype=torch.float32)
                else:
                    prev_logits_t = prev_logits_t.to(device=comp_logits_t.device)[:seq_len]
                    prev_rows_t = prev_logits_t.index_select(0, masked_pos_t)
                    kl_k = min(max(int(kl_top_k), 1), vocab_size)
                    if kl_k <= int(ent_ids_t.shape[-1]):
                        cur_top_ids_t = ent_ids_t[:, :kl_k]
                    else:
                        cur_top_ids_t = torch.topk(row_logits_t, k=kl_k, dim=-1).indices
                    prev_top_ids_t = torch.topk(prev_rows_t, k=kl_k, dim=-1).indices
                    union_ids_t = torch.cat([cur_top_ids_t, prev_top_ids_t], dim=-1)
                    cur_vals_t = row_logits_t.gather(dim=-1, index=union_ids_t).to(dtype=torch.float32)
                    prev_vals_t = prev_rows_t.gather(dim=-1, index=union_ids_t).to(dtype=torch.float32)
                    cur_p_t = torch.softmax(cur_vals_t, dim=-1)
                    prev_p_t = torch.softmax(prev_vals_t, dim=-1)
                    kl_raw_t = (
                        cur_p_t
                        * (torch.log(cur_p_t.clamp_min(1e-12)) - torch.log(prev_p_t.clamp_min(1e-12)))
                    ).sum(dim=-1)
                    kl_raw_t = torch.nan_to_num(kl_raw_t, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                    kl_norm_t = _rank_percentile(kl_raw_t)

                use_legacy_full = bool(
                    branch_observe_trigger_mode == "legacy_entropy_kl_struct"
                    and branch_observe_event_policy == "top_risk"
                )
                if use_legacy_full:
                    base_risk_t = ent_norm_t * kl_norm_t
                    risk_sum_step_t = base_risk_t.sum().detach()
                    if state.get("risk_sum_t") is None:
                        state["risk_sum_t"] = risk_sum_step_t
                    else:
                        state["risk_sum_t"] = state["risk_sum_t"] + risk_sum_step_t
                    state["risk_count"] += int(base_risk_t.numel())
                    candidate_count = min(
                        int(base_risk_t.numel()),
                        int(shadow_top_m),
                    )
                    if candidate_count <= 0:
                        state["prev_logits_comp"] = comp_logits_t.detach()
                        return logits
                    _, cand_rel_t = torch.topk(base_risk_t, k=candidate_count, dim=0)
                else:
                    top1_token_ids_t = ent_ids_t[:, 0]
                    structure_score_vals: List[float] = []
                    for tok_id in top1_token_ids_t.detach().cpu().tolist():
                        structure_score_vals.append(float(np.clip(_structure_sensitive_token_score(_decode_shadow_token(int(tok_id))), 0.0, 1.0)))
                    structure_score_t = torch.as_tensor(
                        structure_score_vals,
                        device=comp_logits_t.device,
                        dtype=torch.float32,
                    )
                    struct_weight_t = 1.0 + float(struct_alpha) * structure_score_t
                    base_risk_t = _shadow_trigger_score_t(
                        branch_observe_trigger_mode,
                        ent_norm_t,
                        kl_norm_t,
                        struct_weight_t,
                    )
                    report_risk_t = ent_norm_t * kl_norm_t * struct_weight_t
                    top2_vals_t = ent_vals_t[:, : min(int(ent_vals_t.shape[-1]), 2)]
                    if int(top2_vals_t.shape[-1]) == 1:
                        conf_proxy_t = torch.ones_like(ent_norm_t, dtype=torch.float32)
                    else:
                        conf_proxy_t = torch.sigmoid((top2_vals_t[:, 0] - top2_vals_t[:, 1]).to(dtype=torch.float32))
                    risk_sum_step_t = report_risk_t.sum().detach()
                    if state.get("risk_sum_t") is None:
                        state["risk_sum_t"] = risk_sum_step_t
                    else:
                        state["risk_sum_t"] = state["risk_sum_t"] + risk_sum_step_t
                    state["risk_count"] += int(report_risk_t.numel())
                    candidate_count = min(
                        int(base_risk_t.numel()),
                        int(shadow_top_m),
                    )
                    if candidate_count <= 0:
                        state["prev_logits_comp"] = comp_logits_t.detach()
                        return logits
                    cand_rel = _select_branch_observe_candidate_indices(
                        selection_score_t=base_risk_t,
                        structure_score_t=structure_score_t,
                        conf_proxy_t=conf_proxy_t,
                        max_count=candidate_count,
                        event_policy=branch_observe_event_policy,
                        rng=analysis_rng,
                    )
                    if not cand_rel:
                        state["prev_logits_comp"] = comp_logits_t.detach()
                        return logits
                    cand_rel_t = torch.as_tensor(cand_rel, device=comp_logits_t.device, dtype=torch.long)
                tok_k = min(max(int(token_top_k), 1), vocab_size)
                if tok_k <= int(ent_ids_t.shape[-1]):
                    top_token_ids_t = ent_ids_t.index_select(0, cand_rel_t)[:, :tok_k]
                else:
                    cand_rows_t = row_logits_t.index_select(0, cand_rel_t)
                    top_token_ids_t = torch.topk(cand_rows_t, k=tok_k, dim=-1).indices
                state["risk_records"].append(
                    {
                        "step": int(step),
                        "positions_t": masked_pos_t.index_select(0, cand_rel_t).detach().clone(),
                        "entropy_t": ent_norm_t.index_select(0, cand_rel_t).detach().clone(),
                        "kl_raw_t": kl_raw_t.index_select(0, cand_rel_t).detach().clone(),
                        "kl_norm_t": kl_norm_t.index_select(0, cand_rel_t).detach().clone(),
                        "base_risk_t": base_risk_t.index_select(0, cand_rel_t).detach().clone(),
                        "top_token_ids_t": top_token_ids_t.detach().clone(),
                        "num_masked": int(masked_pos_t.numel()),
                        "trigger_mode": str(branch_observe_trigger_mode),
                        "event_policy": str(branch_observe_event_policy),
                    }
                )
                if not use_legacy_full:
                    state["risk_records"][-1]["report_risk_t"] = report_risk_t.index_select(0, cand_rel_t).detach().clone()
                    state["risk_records"][-1]["structure_score_t"] = structure_score_t.index_select(0, cand_rel_t).detach().clone()
                    state["risk_records"][-1]["struct_weight_t"] = struct_weight_t.index_select(0, cand_rel_t).detach().clone()

                state["prev_logits_comp"] = comp_logits_t.detach()
        except Exception as exc:  # pragma: no cover - defensive observer path
            state["errors"].append(f"logits_hook:{type(exc).__name__}: {exc}")
        return logits

    kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "output_history": False,
        "return_dict_in_generate": True,
        "steps": int(diffusion_steps),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "alg": str(alg),
        "alg_temp": float(alg_temp),
        "eos_penalty": float(eos_penalty),
        "generation_logits_hook_func": logits_shadow_hook,
        "generation_tokens_hook_func": tokens_shadow_hook,
    }
    official_start_rng_state = _capture_rng_state()
    with torch.no_grad():
        try:
            out = model.diffusion_generate(input_ids, **kwargs)
        except TypeError:
            kwargs.pop("eos_penalty", None)
            out = model.diffusion_generate(input_ids, **kwargs)
    official_end_rng_state = _capture_rng_state()

    sequences = out.sequences if hasattr(out, "sequences") else out
    generated_ids = sequences[0, input_ids.shape[1] :]
    if hasattr(generated_ids, "detach"):
        generated_ids = generated_ids.detach().cpu()
    generated = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and eos_token in generated:
        generated = generated.split(eos_token)[0]

    commit_steps: Dict[int, int] = {}
    step_commit_stats: List[Dict[str, Any]] = []
    for record in state["token_records"]:
        step_i = int(record.get("step", -1))
        newly_mask_t = record.get("newly_mask_t")
        if torch.is_tensor(newly_mask_t):
            newly_positions = [int(v) for v in torch.nonzero(newly_mask_t.detach().cpu(), as_tuple=False).flatten().tolist()]
        else:
            newly_positions = []
        if step_i >= 0:
            for pos in newly_positions:
                commit_steps.setdefault(int(pos), step_i)
        num_masked_t = record.get("num_masked_t")
        num_committed_t = record.get("num_committed_t")
        num_masked = int(num_masked_t.detach().cpu().item()) if torch.is_tensor(num_masked_t) else 0
        num_committed = int(num_committed_t.detach().cpu().item()) if torch.is_tensor(num_committed_t) else 0
        step_commit_stats.append(
            {
                "step": int(step_i),
                "num_masked": int(num_masked),
                "num_committed": int(num_committed),
                "num_newly_committed": int(len(newly_positions)),
                "newly_committed_positions": newly_positions,
            }
        )

    risk_events: List[Dict[str, Any]] = []
    for record in state["risk_records"]:
        try:
            positions = [int(v) for v in record["positions_t"].detach().cpu().tolist()]
            entropy_vals = [float(v) for v in record["entropy_t"].detach().cpu().tolist()]
            kl_raw_vals = [float(v) for v in record["kl_raw_t"].detach().cpu().tolist()]
            kl_norm_vals = [float(v) for v in record["kl_norm_t"].detach().cpu().tolist()]
            base_risk_vals = [float(v) for v in record["base_risk_t"].detach().cpu().tolist()]
            report_risk_t = record.get("report_risk_t")
            structure_score_t = record.get("structure_score_t")
            struct_weight_t = record.get("struct_weight_t")
            report_risk_vals = [float(v) for v in report_risk_t.detach().cpu().tolist()] if torch.is_tensor(report_risk_t) else []
            structure_score_vals = [float(v) for v in structure_score_t.detach().cpu().tolist()] if torch.is_tensor(structure_score_t) else []
            struct_weight_vals = [float(v) for v in struct_weight_t.detach().cpu().tolist()] if torch.is_tensor(struct_weight_t) else []
            top_token_rows = record["top_token_ids_t"].detach().cpu().tolist()
        except Exception as exc:  # pragma: no cover - defensive observer path
            state["errors"].append(f"finalize_risk_record:{type(exc).__name__}: {exc}")
            continue
        event_rows: List[Dict[str, Any]] = []
        for idx, pos_i in enumerate(positions):
            top_ids = [int(v) for v in top_token_rows[idx]]
            top_texts = [_decode_shadow_token(tok) for tok in top_ids]
            if structure_score_vals and struct_weight_vals and report_risk_vals:
                struct_score = float(np.clip(structure_score_vals[idx], 0.0, 1.0))
                struct_weight = float(struct_weight_vals[idx])
                risk_val = float(report_risk_vals[idx])
            else:
                struct_score = float(
                    np.clip(max((_structure_sensitive_token_score(text) for text in top_texts), default=0.0), 0.0, 1.0)
                )
                struct_weight = float(1.0 + struct_alpha * struct_score)
                risk_val = float(base_risk_vals[idx]) * struct_weight
            event_rows.append(
                {
                    "step": int(record.get("step", -1)),
                    "position": int(pos_i),
                    "entropy_norm": float(entropy_vals[idx]),
                    "kl_raw": float(kl_raw_vals[idx]),
                    "kl_norm": float(kl_norm_vals[idx]),
                    "structure_score": float(struct_score),
                    "struct_weight": float(struct_weight),
                    "risk": float(risk_val),
                    "selection_score": float(base_risk_vals[idx]),
                    "trigger_mode": str(record.get("trigger_mode", branch_observe_trigger_mode)),
                    "event_policy": str(record.get("event_policy", branch_observe_event_policy)),
                    "top_token_ids": top_ids,
                    "top_tokens": top_texts,
                    "num_masked": int(record.get("num_masked", 0)),
                    "triggered": bool(risk_val >= shadow_risk_threshold),
                }
            )
        event_rows.sort(key=lambda row: float(row["risk"]), reverse=True)
        risk_events.extend(event_rows[:shadow_top_m])

    finalized_events: List[Dict[str, Any]] = []
    high_risk_count = 0
    early_commit_count = 0
    max_risk = 0.0
    for event in risk_events:
        event_out = dict(event)
        max_risk = max(float(max_risk), float(event_out.get("risk", 0.0)))
        commit_step = commit_steps.get(int(event_out["position"]))
        if commit_step is None:
            lag = None
            was_committed_this_step = False
            early_commit = False
        else:
            lag = int(commit_step) - int(event_out["step"])
            was_committed_this_step = bool(lag == 0)
            early_commit = bool(0 <= lag <= commit_lag_window)
        event_out["commit_step"] = commit_step
        event_out["committed_after_n_steps"] = lag
        event_out["was_committed_this_step"] = bool(was_committed_this_step)
        event_out["early_commit"] = bool(early_commit)
        if bool(event_out.get("triggered", False)):
            high_risk_count += 1
            early_commit_count += int(early_commit)
        finalized_events.append(event_out)
    finalized_events.sort(key=lambda row: float(row.get("risk", 0.0)), reverse=True)
    if shadow_max_events > 0:
        finalized_events = finalized_events[:shadow_max_events]

    risk_count = int(state["risk_count"])
    risk_sum_t = state.get("risk_sum_t")
    risk_sum = float(risk_sum_t.detach().cpu().item()) if torch.is_tensor(risk_sum_t) else 0.0
    mean_risk = float(risk_sum / max(risk_count, 1))

    def _decode_branch_output(branch_out: Any) -> str:
        branch_sequences = branch_out.sequences if hasattr(branch_out, "sequences") else branch_out
        branch_ids = branch_sequences[0, input_ids.shape[1] :]
        if hasattr(branch_ids, "detach"):
            branch_ids = branch_ids.detach().cpu()
        text = tokenizer.decode(branch_ids.tolist(), skip_special_tokens=True)
        if eos_token and eos_token in text:
            text = text.split(eos_token)[0]
        return text

    def _branch_structure_score(text: str) -> Dict[str, Any]:
        format_ok_local = format_success(text)
        parse_ok_local = True
        syntax_error = ""
        try:
            ast.parse(str(text or "").rstrip() + "\n")
        except SyntaxError as exc:
            parse_ok_local = False
            syntax_error = str(getattr(exc, "msg", "") or "")
        except Exception as exc:
            parse_ok_local = False
            syntax_error = f"{type(exc).__name__}: {exc}"
        score = float((1.0 if format_ok_local else 0.0) + (1.0 if parse_ok_local else 0.0))
        if not str(text or "").strip():
            score -= 1.0
        return {
            "format_ok": bool(format_ok_local),
            "parse_ok": bool(parse_ok_local),
            "syntax_error": syntax_error,
            "score": float(score),
        }

    def _run_one_branch_particle(event: Dict[str, Any], kind: str, token_id: Optional[int]) -> Dict[str, Any]:
        event_step = int(event.get("step", -1))
        rel_pos = int(event.get("position", -1))
        full_pos = int(comp_start + rel_pos)
        enforce_until = int(event_step + branch_observe_horizon)
        errors: List[str] = []

        def branch_tokens_hook(step, x, logits):
            del logits
            try:
                if step is None:
                    return x
                step_i = int(step)
                if step_i < event_step or step_i >= enforce_until:
                    return x
                if not torch.is_tensor(x):
                    return x
                if int(getattr(x, "ndim", 0)) == 2:
                    if int(x.shape[0]) != 1 or full_pos < 0 or full_pos >= int(x.shape[1]):
                        return x
                    x_out = x.clone()
                    x_out[0, full_pos] = int(tok_mask_id) if kind == "delay" else int(token_id)
                    return x_out
                if int(getattr(x, "ndim", 0)) == 1:
                    if full_pos < 0 or full_pos >= int(x.shape[0]):
                        return x
                    x_out = x.clone()
                    x_out[full_pos] = int(tok_mask_id) if kind == "delay" else int(token_id)
                    return x_out
            except Exception as hook_exc:  # pragma: no cover - defensive observer path
                errors.append(f"tokens_hook:{type(hook_exc).__name__}: {hook_exc}")
            return x

        branch_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "output_history": False,
            "return_dict_in_generate": True,
            "steps": int(diffusion_steps),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "alg": str(alg),
            "alg_temp": float(alg_temp),
            "eos_penalty": float(eos_penalty),
            "generation_tokens_hook_func": branch_tokens_hook,
        }
        t0_branch = time.perf_counter()
        _restore_rng_state(official_start_rng_state)
        with torch.no_grad():
            try:
                branch_out = model.diffusion_generate(input_ids, **branch_kwargs)
            except TypeError:
                branch_kwargs.pop("eos_penalty", None)
                branch_out = model.diffusion_generate(input_ids, **branch_kwargs)
        latency = time.perf_counter() - t0_branch
        branch_text = _decode_branch_output(branch_out)
        structure = _branch_structure_score(branch_text)
        exact_raw_match = bool(str(branch_text) == str(generated))
        score = float(structure.get("score", 0.0))
        if kind == "delay":
            score -= 0.05
        if not exact_raw_match:
            score += 0.05
        return {
            "kind": str(kind),
            "token_id": int(token_id) if token_id is not None else None,
            "token_text": _decode_shadow_token(int(token_id)) if token_id is not None else "",
            "raw_output": branch_text,
            "raw_output_len": int(len(branch_text)),
            "exact_raw_match_baseline": bool(exact_raw_match),
            "first_raw_text_diff": _first_text_diff(generated, branch_text),
            "format_ok": bool(structure.get("format_ok", False)),
            "parse_ok": bool(structure.get("parse_ok", False)),
            "syntax_error": str(structure.get("syntax_error", "")),
            "score": float(score),
            "latency_sec": float(latency),
            "extra_forwards_estimate": int(diffusion_steps),
            "enforced_step_start": int(event_step),
            "enforced_step_end": int(enforce_until),
            "errors": errors,
        }

    def _run_branch_particle_batch(
        event: Dict[str, Any],
        rollout_specs: Sequence[Tuple[str, Optional[int]]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not rollout_specs:
            return [], 0
        event_step = int(event.get("step", -1))
        rel_pos = int(event.get("position", -1))
        full_pos = int(comp_start + rel_pos)
        enforce_until = int(event_step + branch_observe_horizon)
        specs = [(str(kind), token_id if token_id is None else int(token_id)) for kind, token_id in rollout_specs]
        batch_size = int(len(specs))
        row_errors: List[List[str]] = [[] for _ in range(batch_size)]

        def branch_tokens_hook(step, x, logits):
            del logits
            try:
                if step is None:
                    return x
                step_i = int(step)
                if step_i < event_step or step_i >= enforce_until:
                    return x
                if not torch.is_tensor(x):
                    return x
                if int(getattr(x, "ndim", 0)) == 2:
                    if full_pos < 0 or full_pos >= int(x.shape[1]):
                        return x
                    x_out = x.clone()
                    rows = min(int(x_out.shape[0]), batch_size)
                    for row_idx in range(rows):
                        kind, token_id = specs[row_idx]
                        x_out[row_idx, full_pos] = int(tok_mask_id) if kind == "delay" else int(token_id)
                    return x_out
                if int(getattr(x, "ndim", 0)) == 1:
                    if full_pos < 0 or full_pos >= int(x.shape[0]):
                        return x
                    kind, token_id = specs[0]
                    x_out = x.clone()
                    x_out[full_pos] = int(tok_mask_id) if kind == "delay" else int(token_id)
                    return x_out
            except Exception as hook_exc:  # pragma: no cover - defensive observer path
                msg = f"tokens_hook:{type(hook_exc).__name__}: {hook_exc}"
                for row in row_errors:
                    row.append(msg)
            return x

        branch_input_ids = input_ids.repeat(batch_size, 1)
        branch_attention_mask = attention_mask.repeat(batch_size, 1) if attention_mask is not None else None
        branch_kwargs = {
            "attention_mask": branch_attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "output_history": False,
            "return_dict_in_generate": True,
            "steps": int(diffusion_steps),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "alg": str(alg),
            "alg_temp": float(alg_temp),
            "eos_penalty": float(eos_penalty),
            "generation_tokens_hook_func": branch_tokens_hook,
        }
        t0_branch = time.perf_counter()
        _restore_rng_state(official_start_rng_state)
        with torch.no_grad():
            try:
                branch_out = model.diffusion_generate(branch_input_ids, **branch_kwargs)
            except TypeError:
                branch_kwargs.pop("eos_penalty", None)
                branch_out = model.diffusion_generate(branch_input_ids, **branch_kwargs)
        latency = time.perf_counter() - t0_branch
        branch_sequences = branch_out.sequences if hasattr(branch_out, "sequences") else branch_out
        particle_logs: List[Dict[str, Any]] = []
        for row_idx, (kind, token_id) in enumerate(specs):
            branch_ids = branch_sequences[row_idx, input_ids.shape[1] :]
            if hasattr(branch_ids, "detach"):
                branch_ids = branch_ids.detach().cpu()
            branch_text = tokenizer.decode(branch_ids.tolist(), skip_special_tokens=True)
            if eos_token and eos_token in branch_text:
                branch_text = branch_text.split(eos_token)[0]
            structure = _branch_structure_score(branch_text)
            exact_raw_match = bool(str(branch_text) == str(generated))
            score = float(structure.get("score", 0.0))
            if kind == "delay":
                score -= 0.05
            if not exact_raw_match:
                score += 0.05
            particle_logs.append(
                {
                    "kind": str(kind),
                    "token_id": int(token_id) if token_id is not None else None,
                    "token_text": _decode_shadow_token(int(token_id)) if token_id is not None else "",
                    "raw_output": branch_text,
                    "raw_output_len": int(len(branch_text)),
                    "exact_raw_match_baseline": bool(exact_raw_match),
                    "first_raw_text_diff": _first_text_diff(generated, branch_text),
                    "format_ok": bool(structure.get("format_ok", False)),
                    "parse_ok": bool(structure.get("parse_ok", False)),
                    "syntax_error": str(structure.get("syntax_error", "")),
                    "score": float(score),
                    "latency_sec": float(latency / max(batch_size, 1)),
                    "batch_latency_sec": float(latency),
                    "extra_forwards_estimate": int(diffusion_steps),
                    "batched_rollout": True,
                    "enforced_step_start": int(event_step),
                    "enforced_step_end": int(enforce_until),
                    "errors": row_errors[row_idx],
                }
            )
        return particle_logs, int(diffusion_steps)

    def _run_branch_observe_rollouts(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not branch_observe_enabled or branch_observe_max_events <= 0:
            return {
                "enabled": bool(branch_observe_enabled),
                "branch_events": 0,
                "rollout_count": 0,
                "extra_forwards": 0,
                "avg_beam_size": 0.0,
                "event_logs": [],
                "errors": [],
            }
        event_logs: List[Dict[str, Any]] = []
        errors: List[str] = []
        generated_id_list = generated_ids.tolist() if hasattr(generated_ids, "tolist") else list(generated_ids)
        selected_events = [event for event in events if bool(event.get("triggered", False))]
        selected_events = selected_events[:branch_observe_max_events]
        for event in selected_events:
            rel_pos = int(event.get("position", -1))
            official_token = int(generated_id_list[rel_pos]) if 0 <= rel_pos < len(generated_id_list) else None
            particle_logs: List[Dict[str, Any]] = [
                {
                    "kind": "baseline",
                    "token_id": official_token,
                    "token_text": _decode_shadow_token(int(official_token)) if official_token is not None else "",
                    "raw_output": generated,
                    "raw_output_len": int(len(generated)),
                    "exact_raw_match_baseline": True,
                    "first_raw_text_diff": {"match": True, "index": -1},
                    "format_ok": bool(format_success(generated)),
                    "parse_ok": bool(_branch_structure_score(generated).get("parse_ok", False)),
                    "score": float(_branch_structure_score(generated).get("score", 0.0)),
                    "latency_sec": 0.0,
                    "extra_forwards_estimate": 0,
                    "selected_by_score": False,
                }
            ]
            rollout_specs = _build_branch_observe_rollout_specs(
                official_token=official_token,
                top_token_ids=list(event.get("top_token_ids", []) or []),
                branch_observe_top_k=branch_observe_top_k,
                branch_observe_beam_size=branch_observe_beam_size,
                branch_observe_include_delay=branch_observe_include_delay,
                allow_token_fallback=branch_observe_token_fallback,
            )
            event_extra_forwards = 0
            try:
                batched_logs, batch_extra_forwards = _run_branch_particle_batch(event=event, rollout_specs=rollout_specs)
                particle_logs.extend(batched_logs)
                event_extra_forwards += int(batch_extra_forwards)
            except Exception as batch_exc:  # pragma: no cover - defensive observer path
                errors.append(f"branch_batch:{type(batch_exc).__name__}: {batch_exc}")
            if not any(bool(part.get("batched_rollout", False)) for part in particle_logs if part.get("kind") != "baseline"):
                for kind, token_id in rollout_specs:
                    try:
                        particle_logs.append(_run_one_branch_particle(event=event, kind=kind, token_id=token_id))
                        event_extra_forwards += int(diffusion_steps)
                    except Exception as branch_exc:  # pragma: no cover - defensive observer path
                        errors.append(f"branch_particle:{type(branch_exc).__name__}: {branch_exc}")
            selected_idx = 0
            if particle_logs:
                selected_idx = max(range(len(particle_logs)), key=lambda idx: float(particle_logs[idx].get("score", 0.0)))
                particle_logs[selected_idx]["selected_by_score"] = True
            event_logs.append(
                {
                    "step": int(event.get("step", -1)),
                    "position": int(event.get("position", -1)),
                    "risk": float(event.get("risk", 0.0)),
                    "entropy_norm": float(event.get("entropy_norm", 0.0)),
                    "kl_norm": float(event.get("kl_norm", 0.0)),
                    "structure_score": float(event.get("structure_score", 0.0)),
                    "top_token_ids": list(event.get("top_token_ids", []) or []),
                    "top_tokens": list(event.get("top_tokens", []) or []),
                    "official_token_id": official_token,
                    "official_token_text": _decode_shadow_token(int(official_token)) if official_token is not None else "",
                    "beam_size": int(len(particle_logs)),
                    "extra_forwards_estimate": int(event_extra_forwards),
                    "selected_particle": int(selected_idx),
                    "selected_kind": str(particle_logs[selected_idx].get("kind", "")) if particle_logs else "",
                    "particle_logs": particle_logs,
                }
            )
        _restore_rng_state(official_end_rng_state)
        rollout_count = sum(1 for event in event_logs for part in event.get("particle_logs", []) if part.get("kind") != "baseline")
        extra_forwards = sum(int(event.get("extra_forwards_estimate", 0)) for event in event_logs)
        return {
            "enabled": True,
            "branch_events": int(len(event_logs)),
            "rollout_count": int(rollout_count),
            "extra_forwards": int(extra_forwards),
            "avg_beam_size": float(
                sum(int(event.get("beam_size", 0)) for event in event_logs) / max(len(event_logs), 1)
            ),
            "event_logs": event_logs,
            "errors": errors,
        }

    branch_observe_stats = _run_branch_observe_rollouts(finalized_events)
    steps = max(int(diffusion_steps), 1)
    stats = _build_dream_noop_stats(diffusion_steps=steps, cfg=cfg)
    stats.update(
        {
            "dream_noop_fast_path": False,
            "shadow_mode_enabled": True,
            "shadow_num_steps": int(state["num_steps"]),
            "shadow_num_risk_events": int(high_risk_count),
            "shadow_max_risk": float(max_risk),
            "shadow_mean_risk": float(mean_risk),
            "shadow_high_risk_early_commit_count": int(early_commit_count),
            "shadow_top_risk_events": finalized_events,
            "shadow_step_commit_stats": step_commit_stats,
            "shadow_errors": list(state["errors"]),
            "branch_observe_enabled": bool(branch_observe_stats.get("enabled", False)),
            "branch_observe_trigger_mode": str(branch_observe_trigger_mode),
            "branch_observe_event_policy": str(branch_observe_event_policy),
            "branch_observe_force_baseline_output": bool(getattr(cfg, "branch_observe_force_baseline_output", True)),
            "branch_observe_branch_events": int(branch_observe_stats.get("branch_events", 0)),
            "branch_observe_rollout_count": int(branch_observe_stats.get("rollout_count", 0)),
            "branch_observe_extra_forwards": int(branch_observe_stats.get("extra_forwards", 0)),
            "branch_observe_avg_beam_size": float(branch_observe_stats.get("avg_beam_size", 0.0)),
            "branch_observe_event_logs": branch_observe_stats.get("event_logs", []),
            "branch_observe_errors": list(branch_observe_stats.get("errors", [])) + list(state["branch_observe_errors"]),
            "branch_select_enabled": bool(getattr(cfg, "branch_select_enabled", False)),
            "branch_select_verifier": str(getattr(cfg, "branch_select_verifier", "level0") or "level0"),
            "extra_forwards": int(branch_observe_stats.get("extra_forwards", 0)),
            "extra_forwards_local_beam": int(branch_observe_stats.get("extra_forwards", 0)),
        }
    )
    return generated, stats


def run_dream_official_pf(
    prompt: str,
    model,
    tokenizer,
    cfg: DecoderConfig,
    device: str = "cuda",
    max_new_tokens: int = 768,
    diffusion_steps: int = 768,
    temperature: float = 0.1,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    eos_penalty: float = 3.0,
    use_chat_template: bool = False,
    max_prompt_tokens: int = 512,
    mask_id: Optional[int] = None,
    pf_positions_per_step: int = 1,
    pf_logit_bias: float = 100.0,
    pf_parse_checks: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Dream official diffusion loop with time-aware risk routing + local PF hooks.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when Dream backend is used
        raise ImportError("Dream official PF path requires torch.") from exc

    if _is_dream_noop_config(cfg):
        generated = run_dream_official_baseline(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            eos_penalty=eos_penalty,
            use_chat_template=use_chat_template,
            max_prompt_tokens=max_prompt_tokens,
        )
        return generated, _build_dream_noop_stats(diffusion_steps=diffusion_steps, cfg=cfg)

    if bool(getattr(cfg, "shadow_mode_enabled", False)) or bool(getattr(cfg, "branch_observe_enabled", False)):
        return run_dream_official_shadow(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            max_new_tokens=max_new_tokens,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
            eos_penalty=eos_penalty,
            use_chat_template=use_chat_template,
            max_prompt_tokens=max_prompt_tokens,
            mask_id=mask_id,
        )

    if use_chat_template:
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else getattr(inputs, "attention_mask", None)

    if max_prompt_tokens > 0 and int(input_ids.shape[-1]) > int(max_prompt_tokens):
        input_ids = input_ids[:, -int(max_prompt_tokens) :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -int(max_prompt_tokens) :]

    input_ids = input_ids.to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    prompt_len = int(input_ids.shape[-1])
    effective_seq_len = int(max_new_tokens)
    comp_start = int(prompt_len)
    comp_end = int(prompt_len + effective_seq_len)
    tok_mask_id = int(mask_id) if mask_id is not None else int(getattr(tokenizer, "mask_token_id", -1))
    if tok_mask_id < 0:
        tok_mask_id = int(getattr(getattr(model, "generation_config", None), "mask_token_id", -1))
    if tok_mask_id < 0:
        raise ValueError("Dream official PF requires a valid mask token id.")

    pf_sampler = DreamSamplerAdapter(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        seq_len=effective_seq_len,
        max_prompt_tokens=max_prompt_tokens,
        mask_id=tok_mask_id,
        use_chat_template=use_chat_template,
    )

    def _decode_completion_ids(ids: np.ndarray) -> str:
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    total_steps = max(int(diffusion_steps), 1)
    pf_phase_ratio = float(np.clip(cfg.pf_phase_ratio, 0.0, 1.0))
    pf_phase_ratio_active = False
    pf_t_start, pf_t_end = resolve_pf_time_window(total_steps=total_steps, cfg=cfg)
    pf_positions_cap = max(int(pf_positions_per_step), 1)
    pf_cooldown_steps = max(int(cfg.pf_cooldown_steps), 3)
    pf_risk_gradient_sigma = max(float(getattr(cfg, "pf_risk_gradient_sigma", 1.5)), 0.0)
    joint_gate_quantile = float(np.clip(cfg.joint_gate_quantile, 0.0, 1.0))
    pf_trigger_quantile = float(np.clip(max(cfg.pf_trigger_quantile, cfg.risk_high_quantile), 0.0, 1.0))
    parser_feedback_radius = max(int(getattr(cfg, "parser_feedback_window_radius", 2)), 0)
    parser_feedback_threshold = max(float(getattr(cfg, "parser_feedback_hotspot_threshold", 0.1)), 0.0)
    parser_feedback_gate_scale_default = float(np.clip(getattr(cfg, "parser_feedback_gate_scale", 0.85), 0.1, 1.0))
    pf_cooldown_remaining = 0
    pf_token_ids_to_code = _decode_completion_ids if (pf_parse_checks or cfg.parsing_checks_enabled) else None
    prompt_identifier_counts = _extract_identifier_counts(prompt)
    secondary_signal_mode = str(getattr(cfg, "correctness_signal_mode", "none") or "none").lower()
    if secondary_signal_mode not in {"none", "counterfactual_gain", "constraint_identifier"}:
        secondary_signal_mode = "none"
    counterfactual_rollout_steps = max(int(getattr(cfg, "counterfactual_rollout_steps", 1)), 1)
    pf_budget_mode = str(getattr(cfg, "pf_budget_mode", "legacy") or "legacy").lower()
    if pf_budget_mode not in {"legacy", "budgeted_entropy"}:
        pf_budget_mode = "legacy"
    budgeted_entropy_pf = bool(pf_budget_mode == "budgeted_entropy")
    pf_extra_forward_budget = max(int(getattr(cfg, "pf_extra_forward_budget", 0)), 0)
    pf_acceptance_tolerance = max(float(getattr(cfg, "pf_acceptance_tolerance", 0.02)), 0.0)
    pf_trigger_limit = max(int(getattr(cfg, "pf_max_triggers_per_sample", 0)), 0)
    pf_scoring_cfg = cfg
    if budgeted_entropy_pf:
        pf_scoring_cfg = copy.copy(cfg)
        pf_scoring_cfg.pf_syntax_reward = 0.0
        pf_scoring_cfg.pf_stability_weight = 0.0
    pf_persistent_tokens_t = torch.zeros(effective_seq_len, dtype=torch.long, device=device)
    pf_persistent_ttl_t = torch.zeros(effective_seq_len, dtype=torch.long, device=device)
    parser_issue_histograms: Dict[str, Dict[int, int]] = {"bracket": {}, "indent": {}}
    parser_hotspot_histograms: Dict[str, Dict[int, float]] = {"bracket": {}, "indent": {}}
    parser_feedback_state: Dict[str, float] = {"prev_severity": 0.0, "prev_delta": 0.0}
    rdd_rollback_enabled = bool(getattr(cfg, "rdd_rollback_enabled", False))
    rdd_rollback_window = max(int(getattr(cfg, "rdd_rollback_window", 8)), 1)
    rdd_rollback_max_events = max(int(getattr(cfg, "rdd_rollback_max_events", 2)), 0)
    rdd_rollback_min_severity = float(np.clip(getattr(cfg, "rdd_rollback_min_severity", 0.25), 0.0, 1.0))
    rdd_rollback_cooldown_steps = max(int(getattr(cfg, "rdd_rollback_cooldown_steps", 2)), 0)
    repair_policy = _resolve_dream_repair_policy(cfg)
    rollback_policy_enabled = repair_policy in {_REPAIR_POLICY_ROLLBACK_ONLY, _REPAIR_POLICY_PF_RB}
    pf_rb_policy_enabled = repair_policy == _REPAIR_POLICY_PF_RB
    rdd_pending_remask_t = torch.zeros(effective_seq_len, dtype=torch.bool, device=device)
    rdd_rollback_cooldown_remaining = 0
    branch_bank: List[_DreamBranchState] = []
    branch_width_trace: List[int] = []
    next_branch_id = 1
    prev_logits_comp_t: Optional["torch.Tensor"] = None
    local_beam_delay_mask_t = torch.zeros(effective_seq_len, dtype=torch.bool, device=device)

    stats_state: Dict[str, Any] = {
        "pf_trigger_count": 0,
        "risk_trigger_count": 0,
        "extra_forwards_influence": 0,
        "extra_forwards_pf": 0,
        "influence_compute_count": 0,
        "n_commits": 0,
        "action_counts_total": {
            "commit_argmax": 0,
            "commit_pf": 0,
            "commit_fallback_max_prob": 0,
            "freeze_delay": 0,
            "freeze_cooldown": 0,
            "freeze_budget": 0,
        },
        "step_logs": [],
        "risk_trace_lines": [],
        "parser_feedback_counts": {"bracket": 0, "indent": 0},
        "parser_hotspot_counts": {"bracket": 0, "indent": 0},
        "rdd_rollback_count": 0,
        "rdd_remasked_tokens_total": 0,
        "rdd_rollback_events": [],
        "rdd_rollback_cleared_pf_count": 0,
        "repair_route_counts": {"normal": 0, "pf": 0, "rollback": 0},
        "pf_rb_route_counts": {"normal": 0, "pf": 0, "rollback": 0},
        "eb_step_count": 0,
        "eb_candidate_count": 0,
        "eb_allowed_count": 0,
        "eb_blocked_count": 0,
        "eb_min_fallback_count": 0,
        "eb_structure_blocked_count": 0,
        "eb_signature_blocked_count": 0,
        "eb_syntax_near_blocked_count": 0,
        "local_beam_branch_events": 0,
        "local_beam_accepted_alternatives": 0,
        "local_beam_delay_count": 0,
        "local_beam_extra_forwards": 0,
        "local_beam_beam_sizes": [],
        "local_beam_event_logs": [],
    }
    original_forward = getattr(model, "forward", None)
    forward_state: Dict[str, Any] = {"forward_fn": original_forward if callable(original_forward) else None}

    def _forward_logits_from_ids(full_ids_t: "torch.Tensor") -> "torch.Tensor":
        """
        Run one additional model forward on a perturbed full sequence and return
        logits in shape [seq_len, vocab] for 1D input or [batch, seq_len, vocab]
        for 2D batched input.
        """
        squeeze_batch = False
        if int(getattr(full_ids_t, "ndim", 0)) == 1:
            in_ids = full_ids_t.unsqueeze(0).to(device=device, dtype=torch.long)
            squeeze_batch = True
        elif int(getattr(full_ids_t, "ndim", 0)) == 2:
            in_ids = full_ids_t.to(device=device, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported ids ndim for extra forward: {getattr(full_ids_t, 'ndim', None)}")
        attn_mask_2d = torch.ones_like(in_ids, dtype=torch.bool, device=in_ids.device)
        forward_fn = forward_state.get("forward_fn")
        out_local = None
        try:
            if callable(forward_fn):
                out_local = forward_fn(input_ids=in_ids, attention_mask=attn_mask_2d)
            else:
                out_local = model(input_ids=in_ids, attention_mask=attn_mask_2d)
        except TypeError:
            if callable(forward_fn):
                out_local = forward_fn(input_ids=in_ids)
            else:
                out_local = model(input_ids=in_ids)
        logits_local = out_local.logits if hasattr(out_local, "logits") else out_local[0]
        if squeeze_batch and int(getattr(logits_local, "ndim", 0)) == 3:
            logits_local = logits_local[0]
        return logits_local.float()

    def _generation_logits_hook(step, x, logits):
        nonlocal pf_cooldown_remaining, branch_bank, next_branch_id, rdd_rollback_cooldown_remaining, prev_logits_comp_t
        if logits is None:
            return logits
        try:
            if x is None:
                return logits

            # Normalize x to [seq_len] for batch=1 hooks.
            if getattr(x, "ndim", 0) == 2:
                if int(x.shape[0]) != 1:
                    return logits
                x_seq_t = x[0]
            elif getattr(x, "ndim", 0) == 1:
                x_seq_t = x
            else:
                return logits

            # Normalize logits to [seq_len, vocab_size]. Dream hooks may deliver
            # either [1, seq_len, vocab_size] or [seq_len, vocab_size].
            if getattr(logits, "ndim", 0) == 3:
                if int(logits.shape[0]) != 1:
                    return logits
                logits_seq_t = logits[0]
            elif getattr(logits, "ndim", 0) == 2:
                logits_seq_t = logits
            else:
                return logits
            if int(getattr(logits_seq_t, "ndim", 0)) != 2:
                return logits

            seq_limit = int(min(int(x_seq_t.shape[0]), int(logits_seq_t.shape[0])))
            if seq_limit <= 0:
                return logits

            comp_start = min(prompt_len, seq_limit)
            comp_end = min(prompt_len + effective_seq_len, seq_limit)
            if comp_end <= comp_start:
                return logits

            x_comp = x_seq_t[comp_start:comp_end]
            logits_comp_t = logits_seq_t[comp_start:comp_end]

            x_comp_ids_t = x_comp.to(dtype=torch.long)
            persistent_mask_t = pf_persistent_ttl_t > 0
            persistent_lim = min(int(x_comp_ids_t.numel()), int(persistent_mask_t.numel()))
            if persistent_lim > 0 and bool(torch.any(persistent_mask_t[:persistent_lim]).item()):
                x_comp_ids_t = x_comp_ids_t.clone()
                x_comp_ids_t[:persistent_lim] = torch.where(
                    persistent_mask_t[:persistent_lim],
                    pf_persistent_tokens_t[:persistent_lim],
                    x_comp_ids_t[:persistent_lim],
                )
            committed_mask_t = x_comp_ids_t.ne(tok_mask_id)
            step_pf_forwards = 0
            branch_count_before = max(int(len(branch_bank)), 1)
            branch_pruned_count = 0
            branch_forked_this_step = False
            branch_trial_active = False
            branch_trial_merge_due = False
            branch_merge_selected = False
            active_branch_id: Optional[int] = None
            active_branch_score = 0.0
            active_branch_score_gap: Optional[float] = None
            active_branch_second_id: Optional[int] = None
            active_branch_second_score: Optional[float] = None
            branch_commit_allowed = False
            branch_parent_id: Optional[int] = None
            branch_parent_score = 0.0
            branch_parent_owned_mask_t = torch.zeros_like(committed_mask_t, dtype=torch.bool)
            branch_parent_tokens_t = torch.zeros_like(x_comp_ids_t, dtype=torch.long)
            active_branch_owned_mask_t = torch.zeros_like(committed_mask_t, dtype=torch.bool)
            active_branch_tokens_t = torch.zeros_like(x_comp_ids_t, dtype=torch.long)
            if branch_bank:
                branch_bank, branch_trial_merge_due = _advance_dream_branch_trials(branch_bank)
                branch_trial_active = bool(branch_bank) and not bool(branch_trial_merge_due)
                if branch_trial_merge_due:
                    branch_inputs: List["torch.Tensor"] = []
                    branch_states: List[Tuple[_DreamBranchState, "torch.Tensor", "torch.Tensor"]] = []
                    for branch in branch_bank:
                        branch_comp_ids_t, branch_committed_mask_t = _compose_branch_completion_state(
                            base_tokens=x_comp_ids_t,
                            base_committed_mask=committed_mask_t,
                            branch=branch,
                        )
                        full_ids_branch_t = x_seq_t.to(dtype=torch.long).clone()
                        full_ids_branch_t[comp_start:comp_end] = branch_comp_ids_t
                        branch_inputs.append(full_ids_branch_t)
                        branch_states.append((branch, branch_comp_ids_t, branch_committed_mask_t))
                    if branch_inputs:
                        branch_logits_batches: List["torch.Tensor"] = []
                        try:
                            branch_batch_t = torch.stack(branch_inputs, dim=0)
                            branch_logits_full_bt = _forward_logits_from_ids(branch_batch_t)
                            step_pf_forwards += 1
                            stats_state["extra_forwards_pf"] += 1
                            if int(getattr(branch_logits_full_bt, "ndim", 0)) == 2:
                                branch_logits_full_bt = branch_logits_full_bt.unsqueeze(0)
                            branch_logits_batches = [branch_logits_full_bt[idx] for idx in range(int(branch_logits_full_bt.shape[0]))]
                        except Exception:
                            branch_logits_batches = []
                            for branch_input_t in branch_inputs:
                                branch_logits_single_t = _forward_logits_from_ids(branch_input_t)
                                step_pf_forwards += 1
                                stats_state["extra_forwards_pf"] += 1
                                if int(getattr(branch_logits_single_t, "ndim", 0)) == 2:
                                    branch_logits_batches.append(branch_logits_single_t)
                        if branch_logits_batches:
                            evaluated_branches: List[Tuple[_DreamBranchState, "torch.Tensor", "torch.Tensor", "torch.Tensor"]] = []
                            for idx, (branch, branch_comp_ids_t, branch_committed_mask_t) in enumerate(branch_states):
                                if idx >= len(branch_logits_batches):
                                    continue
                                branch_logits_comp_t = branch_logits_batches[idx][comp_start:comp_end]
                                if int(getattr(branch_logits_comp_t, "ndim", 0)) != 2:
                                    continue
                                eval_score, eval_feedback, future_entropy = _score_dream_branch_state(
                                    branch=branch,
                                    branch_tokens_t=branch_comp_ids_t,
                                    logits_comp_t=branch_logits_comp_t,
                                    committed_mask_t=branch_committed_mask_t,
                                    token_ids_to_code=pf_token_ids_to_code,
                                    cfg=cfg,
                                    source_prefix=prompt,
                                )
                                branch.score = float(eval_score)
                                branch.eval_score = float(eval_score)
                                branch.parse_quality = float(eval_feedback.get("quality_score", 1.0))
                                branch.severity = float(eval_feedback.get("severity_score", 0.0))
                                branch.future_entropy = float(future_entropy)
                                evaluated_branches.append((branch, branch_comp_ids_t, branch_committed_mask_t, branch_logits_comp_t))
                            pruned_bank, branch_pruned_count = _prune_dream_branch_bank(
                                [entry[0] for entry in evaluated_branches],
                                beam_width=int(getattr(cfg, "beam_width", 1)),
                            )
                            entry_by_id = {int(entry[0].branch_id): entry for entry in evaluated_branches}
                            ranked_entries = [
                                entry_by_id[int(branch.branch_id)]
                                for branch in pruned_bank
                                if int(branch.branch_id) in entry_by_id
                            ]
                            best_entry = _select_best_dream_branch_entry(ranked_entries)
                            if len(ranked_entries) > 1:
                                best_score_val = float(max(ranked_entries[0][0].eval_score, ranked_entries[0][0].score))
                                second_score_val = float(max(ranked_entries[1][0].eval_score, ranked_entries[1][0].score))
                                active_branch_score_gap = float(best_score_val - second_score_val)
                                active_branch_second_id = int(ranked_entries[1][0].branch_id)
                                active_branch_second_score = float(second_score_val)
                            branch_bank = []
                            if best_entry is not None:
                                best_branch, best_comp_ids_t, best_committed_mask_t, best_logits_comp_t = best_entry
                                branch_parent_id = int(best_branch.branch_id)
                                branch_parent_score = float(max(best_branch.eval_score, best_branch.score))
                                branch_parent_owned_mask_t = best_branch.owned_mask.to(device=committed_mask_t.device, dtype=torch.bool).clone()
                                branch_parent_tokens_t = best_branch.owned_tokens.to(device=x_comp_ids_t.device, dtype=torch.long).clone()
                                x_comp_ids_t = best_comp_ids_t
                                committed_mask_t = best_committed_mask_t
                                logits_comp_t = best_logits_comp_t
                                active_branch_id = int(best_branch.branch_id)
                                active_branch_score = float(branch_parent_score)
                                branch_merge_selected = True
                                branch_commit_allowed = True
                                active_branch_owned_mask_t = branch_parent_owned_mask_t.clone()
                                active_branch_tokens_t = branch_parent_tokens_t.clone()
                                branch_lim = min(
                                    int(active_branch_owned_mask_t.numel()),
                                    int(active_branch_tokens_t.numel()),
                                    int(pf_persistent_ttl_t.numel()),
                                    int(pf_persistent_tokens_t.numel()),
                                )
                                persistence_steps = _resolve_pf_persistence_steps(
                                    t_remaining=t_remaining if 't_remaining' in locals() else total_steps - int(step),
                                    t_start=pf_t_start,
                                    t_end=pf_t_end,
                                    base_steps=pf_cooldown_steps,
                                )
                                if branch_lim > 0 and bool(torch.any(active_branch_owned_mask_t[:branch_lim]).item()):
                                    refresh_ttl = torch.full_like(
                                        pf_persistent_ttl_t[:branch_lim],
                                        fill_value=max(int(persistence_steps), 1),
                                    )
                                    pf_persistent_tokens_t[:branch_lim] = torch.where(
                                        active_branch_owned_mask_t[:branch_lim],
                                        active_branch_tokens_t[:branch_lim],
                                        pf_persistent_tokens_t[:branch_lim],
                                    )
                                    pf_persistent_ttl_t[:branch_lim] = torch.where(
                                        active_branch_owned_mask_t[:branch_lim],
                                        torch.maximum(pf_persistent_ttl_t[:branch_lim], refresh_ttl),
                                        pf_persistent_ttl_t[:branch_lim],
                                    )
                else:
                    branch_bank = [branch for branch in branch_bank if int(branch.ttl) > 0]
            masked_positions_t = torch.nonzero(~committed_mask_t, as_tuple=False).flatten()

            masked_count = int(masked_positions_t.numel())
            if masked_count == 0:
                return logits

            total_mask_slots = max(int(x_comp_ids_t.numel()), 1)
            mask_ratio = float(masked_count) / float(total_mask_slots)
            t_remaining = total_steps - int(step)
            base_pf_phase = bool(pf_t_start <= int(t_remaining) <= pf_t_end)
            committed_completion_source = ""
            committed_completion_token_count = 0
            committed_feedback = _default_parser_feedback()
            if pf_token_ids_to_code is not None:
                try:
                    committed_completion_source, committed_completion_token_count = _decode_committed_completion_prefix(
                        token_ids=x_comp_ids_t,
                        committed_mask=committed_mask_t,
                        token_ids_to_code=pf_token_ids_to_code,
                    )
                    committed_feedback = parser_feedback_from_source(
                        source=prompt + committed_completion_source,
                        min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
                    )
                except Exception:
                    committed_feedback = _default_parser_feedback()

            parser_feedback: Dict[str, Any] = committed_feedback
            if bool(getattr(cfg, "parser_feedback_enabled", True)) and pf_token_ids_to_code is not None:
                if bool(parser_feedback.get("observed")):
                    if bool(parser_feedback.get("bracket_issue")):
                        _bump_parser_issue_hist(parser_issue_histograms["bracket"], t_remaining)
                        stats_state["parser_feedback_counts"]["bracket"] += 1
                    if bool(parser_feedback.get("indent_issue")):
                        _bump_parser_issue_hist(parser_issue_histograms["indent"], t_remaining)
                        stats_state["parser_feedback_counts"]["indent"] += 1
            parser_gradient = _parser_gradient_metrics(
                parser_feedback=parser_feedback,
                prev_severity=float(parser_feedback_state.get("prev_severity", 0.0)),
                prev_delta=float(parser_feedback_state.get("prev_delta", 0.0)),
            )
            current_gradient_hotspot = bool(
                parser_feedback.get("observed")
                and parser_feedback.get("primary_issue", "none") != "none"
                and float(parser_gradient["score"]) >= float(parser_feedback_threshold)
            )
            if current_gradient_hotspot:
                issue_types = parser_feedback.get("issue_types", [])
                if not issue_types:
                    issue_types = [str(parser_feedback.get("primary_issue", "none"))]
                for issue_type in issue_types:
                    issue_key = str(issue_type)
                    if issue_key not in parser_hotspot_histograms:
                        continue
                    _bump_parser_score_hist(
                        parser_hotspot_histograms[issue_key],
                        t_remaining=t_remaining,
                        score=float(parser_gradient["score"]),
                    )
                    stats_state["parser_hotspot_counts"][issue_key] += 1
            parser_hotspot_score = _parser_hotspot_score(
                issue_histograms=parser_hotspot_histograms,
                t_remaining=t_remaining,
                radius=parser_feedback_radius,
            )
            parser_hotspot_active = bool(current_gradient_hotspot or parser_hotspot_score >= parser_feedback_threshold)
            parser_feedback_state["prev_severity"] = float(parser_gradient["severity"])
            parser_feedback_state["prev_delta"] = float(parser_gradient["delta"])
            rdd_rollback_triggered = False
            rdd_rollback_positions: List[int] = []
            rdd_rollback_reason = ""
            rdd_rollback_severity = 0.0
            rdd_rollback_cleared_pf_positions: List[int] = []
            rdd_cooldown_start = int(rdd_rollback_cooldown_remaining)
            rdd_cooldown_active = bool(rdd_rollback_cooldown_remaining > 0)
            allow_pf_phase = bool((base_pf_phase or parser_hotspot_active) and not branch_trial_active and not branch_trial_merge_due)
            cooldown_active = bool(pf_cooldown_remaining > 0)
            allow_pf_step = bool(cfg.pf_enabled and allow_pf_phase and not cooldown_active)
            if pf_rb_policy_enabled and rdd_cooldown_active:
                allow_pf_step = False

            rollback_available = bool(
                rollback_policy_enabled
                and rdd_rollback_enabled
                and pf_token_ids_to_code is not None
                and not bool(branch_trial_active)
                and not bool(branch_trial_merge_due)
                and not bool(torch.any(rdd_pending_remask_t).item())
                and not rdd_cooldown_active
                and int(stats_state["rdd_rollback_count"]) < int(rdd_rollback_max_events)
                and int(t_remaining) > 2
            )
            if _repair_policy_uses_baseline_sampling(repair_policy):
                repair_route = _route_rollback_only_action(
                    committed_feedback=committed_feedback,
                    prompt=prompt,
                    committed_completion=committed_completion_source,
                    committed_token_count=int(committed_completion_token_count),
                    rollback_available=bool(rollback_available),
                    rollback_min_severity=float(rdd_rollback_min_severity),
                    parser_gradient_hotspot=bool(current_gradient_hotspot),
                    repair_cooldown_active=bool(rdd_cooldown_active),
                )
                repair_action = str(repair_route.get("action", "normal"))
                if repair_action not in stats_state["repair_route_counts"]:
                    stats_state["repair_route_counts"][repair_action] = 0
                stats_state["repair_route_counts"][repair_action] += 1

                if rollback_policy_enabled and repair_action == "rollback":
                    rdd_rollback_positions = _select_rdd_rollback_positions(
                        committed_mask_t=committed_mask_t,
                        syntax_error_progress=float(repair_route.get("committed_error_completion_progress", 1.0)),
                        rollback_window=int(rdd_rollback_window),
                    )
                    if rdd_rollback_positions:
                        idx_t = torch.as_tensor(rdd_rollback_positions, dtype=torch.long, device=rdd_pending_remask_t.device)
                        valid_idx_t = idx_t[(idx_t >= 0) & (idx_t < int(rdd_pending_remask_t.numel()))]
                        if int(valid_idx_t.numel()) > 0:
                            rdd_rollback_cleared_pf_positions = [
                                int(v.item())
                                for v in valid_idx_t
                                if int(pf_persistent_ttl_t[int(v.item())].item()) > 0
                            ]
                            rdd_pending_remask_t[valid_idx_t] = True
                            pf_persistent_ttl_t[valid_idx_t] = 0
                            branch_bank = []
                            rdd_rollback_triggered = True
                            rdd_rollback_positions = [int(v.item()) for v in valid_idx_t]
                            rdd_rollback_reason = str(committed_feedback.get("primary_issue", "syntax"))
                            rdd_rollback_severity = float(committed_feedback.get("severity_score", 0.0))
                            stats_state["rdd_rollback_count"] += 1
                            stats_state["rdd_remasked_tokens_total"] += int(len(rdd_rollback_positions))
                            stats_state["rdd_rollback_cleared_pf_count"] += int(len(rdd_rollback_cleared_pf_positions))
                            stats_state["rdd_rollback_events"].append(
                                {
                                    "step": int(step),
                                    "t": int(t_remaining),
                                    "positions": [int(p) for p in rdd_rollback_positions],
                                    "cleared_pf_positions": [int(p) for p in rdd_rollback_cleared_pf_positions],
                                    "reason": str(rdd_rollback_reason),
                                    "severity": float(rdd_rollback_severity),
                                    "repair_policy": str(repair_policy),
                                    "route_reason": str(repair_route.get("reason", "")),
                                }
                            )

                cooldown_start = int(pf_cooldown_remaining)
                cooldown_end = int(max(pf_cooldown_remaining - 1, 0)) if pf_cooldown_remaining > 0 else int(pf_cooldown_remaining)
                if rdd_rollback_triggered and rdd_rollback_cooldown_steps > 0:
                    rdd_rollback_cooldown_remaining = int(rdd_rollback_cooldown_steps)
                elif rdd_rollback_cooldown_remaining > 0:
                    rdd_rollback_cooldown_remaining = int(rdd_rollback_cooldown_remaining - 1)
                rdd_cooldown_end = int(rdd_rollback_cooldown_remaining)

                risk_trace_line = (
                    f"risk_trace step={int(step)} t={int(t_remaining)} "
                    f"max_risk=0.000000 tau_high=0.000000 tau_pf=0.000000 tau_grad=0.000000 "
                    f"tau_pf_final=0.000000 allow_pf=False "
                    f"in_window={bool(base_pf_phase)} base_window={bool(base_pf_phase)} window=[{int(pf_t_start)},{int(pf_t_end)}] "
                    f"signal_mode=none parser_issue={str(parser_feedback.get('primary_issue', 'none'))} "
                    f"parser_hotspot={bool(parser_hotspot_active)} parser_grad={float(parser_gradient['delta']):.3f} "
                    f"parser_accel={float(parser_gradient['accel']):.3f} parser_score={float(parser_hotspot_score):.3f} "
                    f"cooldown={int(pf_cooldown_remaining)} particles=0 masked={int(masked_count)} "
                    f"repair_policy={str(repair_policy)} repair_action={str(repair_action)} "
                    f"pf_budget_mode=legacy budget_remaining=-1 kl_raw_max=0.000000 kl_raw_span=0.000000"
                )
                stats_state["risk_trace_lines"].append(risk_trace_line)
                logger.info(risk_trace_line)

                action_counts = {
                    "commit_argmax": 0,
                    "commit_pf": 0,
                    "commit_fallback_max_prob": 0,
                    "freeze_delay": int(masked_count),
                    "freeze_cooldown": 0,
                    "freeze_budget": 0,
                }
                for action_name, val in action_counts.items():
                    stats_state["action_counts_total"][action_name] += int(val)

                stats_state["step_logs"].append(
                    {
                        "step": int(step),
                        "t": int(t_remaining),
                        "secondary_signal_mode": "none",
                        "allow_pf_phase": False,
                        "allow_pf_phase_base_window": bool(base_pf_phase),
                        "allow_pf_step": False,
                        "allow_kl_step": False,
                        "cooldown_active": bool(cooldown_active),
                        "cooldown_remaining_start": int(cooldown_start),
                        "cooldown_remaining_end": int(cooldown_end),
                        "rdd_rollback_enabled": bool(rdd_rollback_enabled),
                        "rdd_rollback_triggered": bool(rdd_rollback_triggered),
                        "rdd_rollback_positions": [int(p) for p in rdd_rollback_positions],
                        "rdd_rollback_reason": str(rdd_rollback_reason),
                        "rdd_rollback_severity": float(rdd_rollback_severity),
                        "rdd_rollback_cleared_pf_positions": [int(p) for p in rdd_rollback_cleared_pf_positions],
                        "rdd_cooldown_active": bool(rdd_cooldown_active),
                        "rdd_cooldown_remaining_start": int(rdd_cooldown_start),
                        "rdd_cooldown_remaining_end": int(rdd_cooldown_end),
                        "repair_policy": str(repair_policy),
                        "repair_action": str(repair_action),
                        "repair_state": str(repair_route.get("state", "")),
                        "repair_reason": str(repair_route.get("reason", "")),
                        "repair_route": repair_route,
                        "pf_rb_policy_enabled": False,
                        "pf_rb_action": "normal",
                        "pf_rb_state": "",
                        "pf_rb_reason": "",
                        "pf_rb_route": _default_repair_route(),
                        "committed_feedback": committed_feedback,
                        "projected_feedback": _default_parser_feedback(),
                        "pf_triggered_this_step": False,
                        "phase_ratio_threshold": float(pf_phase_ratio),
                        "pf_phase_ratio_active": bool(pf_phase_ratio_active),
                        "mask_ratio": float(mask_ratio),
                        "pf_time_window_mode": str(cfg.pf_time_window_mode),
                        "pf_time_window_start_t": int(pf_t_start),
                        "pf_time_window_end_t": int(pf_t_end),
                        "in_pf_time_window": bool(base_pf_phase),
                        "pf_bandwidth": 0,
                        "pf_budget_mode": "legacy",
                        "pf_budget_active": False,
                        "pf_budget_blocked": False,
                        "pf_syntax_repair_mode": False,
                        "pf_extra_forward_budget": 0,
                        "pf_trigger_limit": int(pf_trigger_limit),
                        "pf_trigger_cap_reached": False,
                        "pf_remaining_extra_forward_budget": -1,
                        "pf_estimated_forward_cost": 0,
                        "pf_candidate_budget": 0,
                        "pf_candidate_positions": [],
                        "pf_candidate_meta": {},
                        "tau_low": 0.0,
                        "tau_high": 0.0,
                        "tau_high_effective": 0.0,
                        "tau_pf_trigger": 0.0,
                        "risk_mean": 0.0,
                        "risk_std": 0.0,
                        "risk_gradient_sigma": float(pf_risk_gradient_sigma),
                        "risk_gradient_gate": 0.0,
                        "risk_gradient_gate_effective": 0.0,
                        "pf_trigger_gate": 0.0,
                        "pf_trigger_gate_effective": 0.0,
                        "joint_gate_quantile": float(joint_gate_quantile),
                        "pf_trigger_quantile": float(pf_trigger_quantile),
                        "pf_particles_step": 0,
                        "pf_persistent_mask_size": int((pf_persistent_ttl_t > 0).sum().item()),
                        "pf_persistent_ttl_max": int(pf_persistent_ttl_t.max().item()) if int(pf_persistent_ttl_t.numel()) > 0 else 0,
                        "pf_persistent_positions": [
                            int(v.item()) for v in torch.nonzero(pf_persistent_ttl_t > 0, as_tuple=False).flatten()
                        ],
                        "joint_entropy_gate": 0.0,
                        "joint_influence_gate": 0.0,
                        "max_risk": 0.0,
                        "attention_proxy_raw_max": 0.0,
                        "attention_proxy_raw_span": 0.0,
                        "attention_proxy_raw_nonzero": 0,
                        "influence_kl_raw_max": 0.0,
                        "influence_kl_raw_span": 0.0,
                        "influence_kl_raw_nonzero": 0,
                        "entropy_min": 0.0,
                        "entropy_mean": 0.0,
                        "entropy_max": 0.0,
                        "parser_feedback": parser_feedback,
                        "parser_hotspot_score": float(parser_hotspot_score),
                        "parser_hotspot_active": bool(parser_hotspot_active),
                        "parser_gradient_score": float(parser_gradient["score"]),
                        "parser_gradient_delta": float(parser_gradient["delta"]),
                        "parser_gradient_accel": float(parser_gradient["accel"]),
                        "parser_gradient_hotspot": bool(current_gradient_hotspot),
                        "parser_quality_score": float(parser_gradient["quality"]),
                        "parser_severity_score": float(parser_gradient["severity"]),
                        "parser_gate_scale": 1.0,
                        "parser_feedback_step_counts": {
                            "bracket": int(parser_issue_histograms["bracket"].get(int(t_remaining), 0)),
                            "indent": int(parser_issue_histograms["indent"].get(int(t_remaining), 0)),
                        },
                        "parser_hotspot_step_scores": {
                            "bracket": float(parser_hotspot_histograms["bracket"].get(int(t_remaining), 0.0)),
                            "indent": float(parser_hotspot_histograms["indent"].get(int(t_remaining), 0.0)),
                        },
                        "entropy_histogram": {},
                        "uncommitted_mask_size": int(masked_count),
                        "valid_mask_size": int(masked_count),
                        "invalid_mask_size": 0,
                        "influence_target_mask_size": 0,
                        "high_risk_mask_size": 0,
                        "mid_risk_mask_size": 0,
                        "low_risk_mask_size": 0,
                        "influence_compute_count": 0,
                        "influence_targets": [],
                        "influence_target_tokens": [],
                        "influence_top_positions": [],
                        "action_counts": action_counts,
                        "token_replacement_count": 0,
                        "resample_count": 0,
                        "extra_forwards_influence_step": 0,
                        "extra_forwards_pf_step": int(step_pf_forwards),
                        "extra_forwards_step_total": int(step_pf_forwards),
                        "branch_count_before": int(branch_count_before),
                        "branch_count_after": int(max(len(branch_bank), 1)),
                        "branch_pruned_count": int(branch_pruned_count),
                        "branch_forked_this_step": bool(branch_forked_this_step),
                        "branch_trial_active": bool(branch_trial_active),
                        "branch_trial_merge_due": bool(branch_trial_merge_due),
                        "branch_trial_steps": int(getattr(cfg, "branch_trial_steps", 3)),
                        "branch_merge_selected": bool(branch_merge_selected),
                        "branch_parent_id": int(branch_parent_id) if branch_parent_id is not None else None,
                        "active_branch_id": int(active_branch_id) if active_branch_id is not None else None,
                        "branch_commit_allowed": bool(branch_commit_allowed),
                        "branch_commit_margin": float(getattr(cfg, "branch_commit_margin", 0.2)),
                        "branch_best_score": float(branch_parent_score) if branch_parent_id is not None else None,
                        "branch_second_id": int(active_branch_second_id) if active_branch_second_id is not None else None,
                        "branch_second_score": float(active_branch_second_score) if active_branch_second_score is not None else None,
                        "branch_score_gap": float(active_branch_score_gap) if active_branch_score_gap is not None else None,
                        "risk_band_histogram": {"low": 0, "mid": 0, "high": 0},
                        "risk_triggers": [],
                        "pf_decisions": {},
                    }
                )
                prev_logits_comp_t = logits_comp_t.float().detach().clone()
                branch_width_trace.append(int(max(len(branch_bank), 1)))
                return logits

            logits_comp_t_fp32 = logits_comp_t.float()
            probs_t = torch.softmax(logits_comp_t_fp32, dim=-1)
            entropy_all_t = -(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum(dim=-1)
            argmax_all_t = torch.argmax(logits_comp_t_fp32, dim=-1)
            need_projected_source = bool(
                (cfg.pf_enabled and pf_token_ids_to_code is not None)
                or pf_rb_policy_enabled
                or secondary_signal_mode == "constraint_identifier"
            )
            projected_source = ""
            if need_projected_source:
                projected_source = _project_completion_source(
                    current_tokens_t=x_comp_ids_t,
                    committed_mask_t=committed_mask_t,
                    predicted_tokens_t=argmax_all_t,
                    token_ids_to_code=pf_token_ids_to_code,
                )
            projected_feedback = _default_parser_feedback()
            if cfg.pf_enabled and pf_token_ids_to_code is not None:
                try:
                    projected_feedback = parser_feedback_from_source(
                        source=prompt + projected_source,
                        min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
                    )
                except Exception:
                    projected_feedback = _default_parser_feedback()
            strict_parse_repair_pf = bool(cfg.pf_enabled and pf_token_ids_to_code is not None)
            projected_pf_near = {
                "near": False,
                "error_pos": -1,
                "nearest_distance": None,
                "nearest_entropy": 0.0,
                "entropy_gate": 0.0,
                "completion_progress": 0.0,
                "in_completion": False,
            }
            if strict_parse_repair_pf and not bool(projected_feedback.get("parse_ok", True)):
                projected_pf_near = _projected_error_near_masked_high_entropy(
                    projected_feedback=projected_feedback,
                    prompt=prompt,
                    projected_completion=projected_source,
                    projected_token_count=int(x_comp_ids_t.numel()),
                    masked_positions_t=masked_positions_t,
                    entropy_all_t=entropy_all_t,
                    near_radius=int(parser_feedback_radius),
                )
            projected_pf_trigger_eligible = True
            if strict_parse_repair_pf:
                projected_pf_trigger_eligible = bool(
                    not bool(projected_feedback.get("parse_ok", True))
                    and (
                        _feedback_tokenize_bad(projected_feedback)
                        or (
                            _feedback_obvious_syntax_risk(projected_feedback, min_severity=0.0)
                            and bool(projected_pf_near["near"])
                        )
                    )
                )
            projected_identifier_counts = _extract_identifier_counts(projected_source)

            # Keep every still-masked position in the active risk set.
            valid_positions_t = masked_positions_t
            valid_count = int(valid_positions_t.numel())
            candidate_budget_base = min(max(int(getattr(cfg, "influence_top_k", 1)), 1), 2)
            if parser_hotspot_active or current_gradient_hotspot:
                candidate_budget = min(candidate_budget_base + 1, 2)
            else:
                candidate_budget = candidate_budget_base
            pf_candidate_positions: List[int] = []
            pf_candidate_meta: Dict[int, Dict[str, float]] = {}
            if cfg.pf_enabled and allow_pf_phase and valid_count > 0:
                pf_candidate_positions, pf_candidate_meta = _select_pf_candidate_positions(
                    masked_positions_t=valid_positions_t,
                    entropy_all_t=entropy_all_t,
                    argmax_all_t=argmax_all_t,
                    token_ids_to_code=pf_token_ids_to_code,
                    budget=int(candidate_budget),
                    parser_hotspot_active=bool(parser_hotspot_active),
                )
            if pf_candidate_positions:
                pf_candidate_positions_t = torch.as_tensor(
                    pf_candidate_positions,
                    device=masked_positions_t.device,
                    dtype=torch.long,
                )
            else:
                pf_candidate_positions_t = masked_positions_t[:0]
            pf_candidate_set = {int(pos) for pos in pf_candidate_positions}

            use_secondary_signal = bool(cfg.influence_enabled or secondary_signal_mode != "none")
            allow_secondary_signal_step = bool(
                use_secondary_signal
                and allow_pf_phase
                and int(pf_candidate_positions_t.numel()) > 0
                and not branch_trial_active
                and not branch_trial_merge_due
            )

            influences_all_t = torch.zeros_like(entropy_all_t)

            influence_top_positions: List[int] = []
            step_influence_forwards = 0
            influence_targets: List[int] = []
            influence_kl_raw_max = 0.0
            influence_kl_raw_span = 0.0
            influence_kl_raw_nonzero = 0

            def _sampler_step_for_influence(
                _latents_t: "torch.Tensor",
                committed_mask_local: "torch.Tensor",
                forced_tokens_local: Optional["torch.Tensor"],
            ) -> "torch.Tensor":
                nonlocal step_influence_forwards
                comp_ids_pert_t = x_comp_ids_t.clone()
                cm_t = (
                    committed_mask_local
                    if torch.is_tensor(committed_mask_local)
                    else torch.as_tensor(committed_mask_local, device=comp_ids_pert_t.device, dtype=torch.bool)
                )
                cm_t = cm_t.to(device=comp_ids_pert_t.device, dtype=torch.bool).reshape(-1)
                if forced_tokens_local is None:
                    ft_t = torch.zeros_like(cm_t, dtype=torch.long, device=comp_ids_pert_t.device)
                else:
                    ft_t = (
                        forced_tokens_local
                        if torch.is_tensor(forced_tokens_local)
                        else torch.as_tensor(forced_tokens_local, device=comp_ids_pert_t.device, dtype=torch.long)
                    )
                    ft_t = ft_t.to(device=comp_ids_pert_t.device, dtype=torch.long).reshape(-1)

                lim = min(int(comp_ids_pert_t.numel()), int(cm_t.numel()), int(ft_t.numel()))
                if lim > 0:
                    comp_ids_pert_t[:lim] = torch.where(cm_t[:lim], ft_t[:lim], comp_ids_pert_t[:lim])

                full_ids_pert_t = x_seq_t.to(dtype=torch.long).clone()
                full_ids_pert_t[comp_start:comp_end] = comp_ids_pert_t
                logits_full_pert_t = _forward_logits_from_ids(full_ids_pert_t)
                step_influence_forwards += 1
                stats_state["extra_forwards_influence"] += 1

                if int(getattr(logits_full_pert_t, "ndim", 0)) != 2:
                    return logits_comp_t_fp32

                seq_out = int(logits_full_pert_t.shape[0])
                start = min(comp_start, seq_out)
                stop = min(comp_end, seq_out)
                comp_logits_pert_t = logits_full_pert_t[start:stop]

                expected_len = int(logits_comp_t_fp32.shape[0])
                expected_vocab = int(logits_comp_t_fp32.shape[1]) if int(logits_comp_t_fp32.ndim) == 2 else 0
                if (
                    int(getattr(comp_logits_pert_t, "ndim", 0)) != 2
                    or int(comp_logits_pert_t.shape[0]) != expected_len
                    or int(comp_logits_pert_t.shape[1]) != expected_vocab
                ):
                    merged_t = logits_comp_t_fp32.clone()
                    if int(getattr(comp_logits_pert_t, "ndim", 0)) == 2 and expected_vocab > 0:
                        copy_len = min(expected_len, int(comp_logits_pert_t.shape[0]))
                        copy_vocab = min(expected_vocab, int(comp_logits_pert_t.shape[1]))
                        if copy_len > 0 and copy_vocab > 0:
                            merged_t[:copy_len, :copy_vocab] = comp_logits_pert_t[:copy_len, :copy_vocab]
                    return merged_t
                return comp_logits_pert_t

            if allow_secondary_signal_step:
                topk_n = min(int(pf_candidate_positions_t.numel()), 2)
                if topk_n > 0:
                    influence_targets_t = pf_candidate_positions_t[:topk_n]
                    stats_state["influence_compute_count"] += int(influence_targets_t.numel())
                    influence_targets = [int(p.item()) for p in influence_targets_t]

                    influence_raw_vals: List["torch.Tensor"] = []
                    for pos_t in influence_targets_t:
                        pos = int(pos_t.item())
                        try:
                            if secondary_signal_mode == "counterfactual_gain":
                                influence_val, _ = _counterfactual_repair_gain(
                                    pos=pos,
                                    current_tokens_t=x_comp_ids_t,
                                    committed_mask_t=committed_mask_t,
                                    argmax_all_t=argmax_all_t,
                                    token_ids_to_code=pf_token_ids_to_code,
                                    min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
                                    sampler_step_fn=_sampler_step_for_influence,
                                    rollout_steps=int(counterfactual_rollout_steps),
                                )
                                influence_raw_vals.append(
                                    torch.tensor(
                                        max(float(influence_val), 0.0),
                                        device=logits_comp_t_fp32.device,
                                        dtype=logits_comp_t_fp32.dtype,
                                    )
                                )
                            elif secondary_signal_mode == "constraint_identifier":
                                token_text = _decode_single_token_text(int(argmax_all_t[pos].item()), pf_token_ids_to_code)
                                influence_val, _ = _constraint_identifier_signal(
                                    pos=pos,
                                    token_text=token_text,
                                    seq_len=int(logits_comp_t_fp32.shape[0]),
                                    parser_feedback=parser_feedback,
                                    parser_hotspot_active=bool(parser_hotspot_active),
                                    prompt_identifier_counts=prompt_identifier_counts,
                                    source_identifier_counts=projected_identifier_counts,
                                )
                                influence_raw_vals.append(
                                    torch.tensor(
                                        max(float(influence_val), 0.0),
                                        device=logits_comp_t_fp32.device,
                                        dtype=logits_comp_t_fp32.dtype,
                                    )
                                )
                            else:
                                influence_val, _ = compute_influence(
                                    logits=logits_comp_t_fp32,
                                    pos=pos,
                                    eps=float(cfg.influence_eps),
                                    sampler_step_fn=_sampler_step_for_influence,
                                    committed_mask=committed_mask_t,
                                    forced_tokens=x_comp_ids_t,
                                    cfg=cfg,
                                    return_top_affected=False,
                                )
                                if torch.is_tensor(influence_val):
                                    influence_raw_vals.append(
                                        influence_val.to(device=logits_comp_t_fp32.device, dtype=logits_comp_t_fp32.dtype).clamp_min(0.0)
                                    )
                                else:
                                    influence_raw_vals.append(
                                        torch.tensor(
                                            max(float(influence_val), 0.0),
                                            device=logits_comp_t_fp32.device,
                                            dtype=logits_comp_t_fp32.dtype,
                                        )
                                    )
                        except Exception:
                            influence_raw_vals.append(
                                torch.tensor(0.0, device=logits_comp_t_fp32.device, dtype=logits_comp_t_fp32.dtype)
                            )

                    if influence_raw_vals:
                        raw_vals_t = torch.stack(influence_raw_vals)
                        raw_vals_t = torch.nan_to_num(raw_vals_t, nan=0.0, neginf=0.0, posinf=0.0)
                        raw_min_t = raw_vals_t.min()
                        raw_max_t = raw_vals_t.max()
                        influence_kl_raw_max = float(raw_max_t.item())
                        influence_kl_raw_span = float((raw_max_t - raw_min_t).item())
                        influence_kl_raw_nonzero = int((raw_vals_t > 1e-12).sum().item())

                        if secondary_signal_mode in {"counterfactual_gain", "constraint_identifier"}:
                            norm_vals_t = raw_vals_t.clamp(0.0, 1.0)
                        elif influence_kl_raw_span > 1e-12:
                            norm_vals_t = (raw_vals_t - raw_min_t) / (raw_max_t - raw_min_t + 1e-12)
                        elif influence_kl_raw_max > 0.0:
                            norm_vals_t = torch.ones_like(raw_vals_t)
                        else:
                            norm_vals_t = torch.zeros_like(raw_vals_t)
                        influences_all_t[influence_targets_t] = norm_vals_t.to(dtype=influences_all_t.dtype)

                        topk = min(16, int(raw_vals_t.numel()))
                        if topk > 0:
                            top_rel_idx_t = torch.topk(raw_vals_t, k=topk, largest=True).indices
                            influence_top_positions = [int(v.item()) for v in influence_targets_t[top_rel_idx_t]]

            if use_secondary_signal:
                risk_all_t = cfg.w_entropy * entropy_all_t + cfg.w_influence * influences_all_t
            else:
                risk_all_t = cfg.w_entropy * entropy_all_t

            masked_risk_t = risk_all_t[masked_positions_t]
            masked_entropy_t = entropy_all_t[masked_positions_t]
            masked_influence_t = influences_all_t[masked_positions_t]
            risk_mean = 0.0
            risk_std = 0.0
            risk_gradient_gate = 0.0
            entropy_high_joint = 0.0
            influence_high_joint = 0.0
            if int(masked_risk_t.numel()) > 0:
                tau_low_t = torch.quantile(masked_risk_t, float(cfg.risk_low_quantile))
                tau_high_t = torch.quantile(masked_risk_t, float(cfg.risk_high_quantile))
                risk_min_t = masked_risk_t.min()
                risk_max_t = masked_risk_t.max()
                tau_low = float(tau_low_t.item())
                tau_high_raw = float(tau_high_t.item())
                risk_span = float((risk_max_t - risk_min_t).clamp_min(0.0).item())
                risk_mean_t = masked_risk_t.mean()
                risk_std_t = masked_risk_t.std(unbiased=False)
                risk_mean = float(risk_mean_t.item())
                risk_std = float(risk_std_t.item())
                risk_gradient_gate = float((risk_mean_t + float(pf_risk_gradient_sigma) * risk_std_t).item())
                min_gap = float(cfg.risk_threshold_min_gap)
                if cfg.risk_fusion_mode == "entropy_only":
                    # Entropy-only mode has a narrower dynamic range; keep the gap adaptive.
                    min_gap = float(min(min_gap, max(risk_span * 0.1, 0.0)))
                tau_high = float(max(tau_high_raw, tau_low + min_gap))
                tau_high = float(min(tau_high, float(risk_max_t.item())))
                entropy_high_joint = float(
                    torch.quantile(masked_entropy_t, joint_gate_quantile).item()
                )
                influence_gate_source_t = influences_all_t[pf_candidate_positions_t]
                if use_secondary_signal and int(influence_gate_source_t.numel()) > 0:
                    influence_high_joint = float(
                        torch.quantile(influence_gate_source_t, joint_gate_quantile).item()
                    )
                tau_pf_trigger_raw = float(torch.quantile(masked_risk_t, pf_trigger_quantile).item())
                tau_pf_trigger = float(max(tau_high, tau_pf_trigger_raw))
            else:
                tau_low, tau_high = 0.0, 0.0
                tau_pf_trigger = 0.0
                risk_gradient_gate = 0.0

            joint_entropy_gate = float(entropy_high_joint)
            joint_influence_gate = float(influence_high_joint)
            if use_secondary_signal:
                # Dynamic-quantile aligned joint gates: keep quantile gates as primary,
                # and clip optional absolute floors to the observed step maxima.
                if int(masked_entropy_t.numel()) > 0:
                    entropy_floor = float(max(float(cfg.entropy_threshold), 0.0))
                    if entropy_floor > 0.0:
                        entropy_floor = float(min(entropy_floor, float(masked_entropy_t.max().item())))
                        joint_entropy_gate = float(max(joint_entropy_gate, entropy_floor))
                if int(masked_influence_t.numel()) > 0:
                    influence_floor = float(max(float(cfg.influence_trigger_floor), 0.0))
                    if influence_floor > 0.0:
                        influence_floor = float(min(influence_floor, float(masked_influence_t.max().item())))
                        joint_influence_gate = float(max(joint_influence_gate, influence_floor))
            pf_trigger_gate = float(max(float(tau_pf_trigger), float(risk_gradient_gate)))
            parser_gate_scale = float(parser_feedback_gate_scale_default if parser_hotspot_active else 1.0)
            tau_high_effective = float(tau_high * parser_gate_scale) if tau_high > 0.0 else float(tau_high)
            risk_gradient_gate_effective = (
                float(risk_gradient_gate * parser_gate_scale) if risk_gradient_gate > 0.0 else float(risk_gradient_gate)
            )
            pf_trigger_gate_effective = (
                float(pf_trigger_gate * parser_gate_scale) if pf_trigger_gate > 0.0 else float(pf_trigger_gate)
            )

            ranked_idx_t = torch.argsort(masked_risk_t, descending=True)
            ranked_positions_t = masked_positions_t[ranked_idx_t]
            local_beam_risks = []
            local_beam_trigger = None
            if (
                bool(getattr(cfg, "local_beam_enabled", False))
                and prev_logits_comp_t is not None
                and allow_pf_phase
                and not branch_trial_active
                and not branch_trial_merge_due
                and int(valid_positions_t.numel()) > 0
            ):
                try:
                    local_beam_risks = compute_local_beam_risks(
                        logits=logits_comp_t_fp32,
                        prev_logits=prev_logits_comp_t,
                        candidate_positions=[int(v.item()) for v in valid_positions_t],
                        committed_mask=committed_mask_t,
                        cfg=cfg,
                        token_ids_to_code=pf_token_ids_to_code,
                        forced_tokens=x_comp_ids_t,
                        sampler=pf_sampler,
                    )
                    local_beam_trigger = select_local_beam_trigger(
                        risks=local_beam_risks,
                        cfg=cfg,
                        branch_events=int(stats_state["local_beam_branch_events"]),
                    )
                    if local_beam_trigger is not None:
                        trigger_pos_t = torch.as_tensor(
                            [int(local_beam_trigger.pos)],
                            device=ranked_positions_t.device,
                            dtype=ranked_positions_t.dtype,
                        )
                        ranked_positions_t = torch.cat(
                            [
                                trigger_pos_t,
                                ranked_positions_t[ranked_positions_t != int(local_beam_trigger.pos)],
                            ],
                            dim=0,
                        )
                except Exception as lb_exc:
                    stats_state["local_beam_event_logs"].append(
                        {
                            "step": int(step),
                            "t": int(t_remaining),
                            "event": "risk_compute_error",
                            "error": f"{lb_exc}",
                        }
                    )
                    local_beam_risks = []
                    local_beam_trigger = None
            pf_particles_step = resolve_pf_particles_for_t(
                t_remaining=t_remaining,
                total_steps=total_steps,
                cfg=cfg,
                t_start=pf_t_start,
                t_end=pf_t_end,
            )
            pf_budget_controls = _resolve_budgeted_entropy_pf_controls(
                mode=pf_budget_mode,
                base_allow_pf_step=bool(allow_pf_step),
                pf_positions_cap=int(pf_positions_cap),
                pf_particles_step=int(pf_particles_step),
                pf_horizon_steps=int(getattr(pf_scoring_cfg, "pf_horizon_steps", 0)),
                extra_forwards_used=int(stats_state["extra_forwards_influence"] + stats_state["extra_forwards_pf"]),
                extra_forward_budget=int(pf_extra_forward_budget),
                parser_hotspot_active=bool(parser_hotspot_active),
                current_gradient_hotspot=bool(current_gradient_hotspot),
            )
            allow_pf_step = bool(pf_budget_controls["allow_pf_step"])
            pf_positions_cap_step = int(pf_budget_controls["pf_positions_cap"])
            pf_particles_step = int(pf_budget_controls["pf_particles_step"])
            pf_budget_blocked = bool(pf_budget_controls["budget_blocked"])

            max_risk_step = float(masked_risk_t.max().item()) if int(masked_risk_t.numel()) > 0 else 0.0
            pf_trigger_cap_reached = bool(
                pf_trigger_limit > 0
                and int(stats_state["pf_trigger_count"]) >= int(pf_trigger_limit)
            )
            high_risk_masked_position = bool(
                allow_pf_step
                and cfg.pf_enabled
                and (not strict_parse_repair_pf or projected_pf_trigger_eligible)
                and not pf_trigger_cap_reached
                and int(masked_risk_t.numel()) > 0
                and max_risk_step >= float(pf_trigger_gate_effective)
            )
            repair_route = _default_repair_route()
            if repair_policy == _REPAIR_POLICY_PF_RB:
                repair_route = _route_pf_rb_action(
                    committed_feedback=committed_feedback,
                    projected_feedback=projected_feedback,
                    prompt=prompt,
                    committed_completion=committed_completion_source,
                    projected_completion=projected_source,
                    committed_token_count=int(committed_completion_token_count),
                    projected_token_count=int(x_comp_ids_t.numel()),
                    masked_positions_t=masked_positions_t,
                    entropy_all_t=entropy_all_t,
                    rollback_available=bool(rollback_available),
                    pf_available=bool(allow_pf_step and cfg.pf_enabled and not pf_trigger_cap_reached),
                    high_risk_masked_position=bool(high_risk_masked_position),
                    rollback_min_severity=float(rdd_rollback_min_severity),
                    near_radius=int(parser_feedback_radius),
                    parser_gradient_hotspot=bool(current_gradient_hotspot),
                    repair_cooldown_active=bool(rdd_cooldown_active),
                )
            elif repair_policy == _REPAIR_POLICY_ROLLBACK_ONLY:
                repair_route = _route_rollback_only_action(
                    committed_feedback=committed_feedback,
                    prompt=prompt,
                    committed_completion=committed_completion_source,
                    committed_token_count=int(committed_completion_token_count),
                    rollback_available=bool(rollback_available),
                    rollback_min_severity=float(rdd_rollback_min_severity),
                    parser_gradient_hotspot=bool(current_gradient_hotspot),
                    repair_cooldown_active=bool(rdd_cooldown_active),
                )
            repair_action = str(repair_route.get("action", "normal"))
            if repair_action not in stats_state["repair_route_counts"]:
                stats_state["repair_route_counts"][repair_action] = 0
            stats_state["repair_route_counts"][repair_action] += 1
            if repair_policy == _REPAIR_POLICY_PF_RB:
                if repair_action not in stats_state["pf_rb_route_counts"]:
                    stats_state["pf_rb_route_counts"][repair_action] = 0
                stats_state["pf_rb_route_counts"][repair_action] += 1

            if rollback_policy_enabled and repair_action == "rollback":
                rdd_rollback_positions = _select_rdd_rollback_positions(
                    committed_mask_t=committed_mask_t,
                    syntax_error_progress=float(repair_route.get("committed_error_completion_progress", 1.0)),
                    rollback_window=int(rdd_rollback_window),
                )
                if rdd_rollback_positions:
                    idx_t = torch.as_tensor(rdd_rollback_positions, dtype=torch.long, device=rdd_pending_remask_t.device)
                    valid_idx_t = idx_t[(idx_t >= 0) & (idx_t < int(rdd_pending_remask_t.numel()))]
                    if int(valid_idx_t.numel()) > 0:
                        rdd_rollback_cleared_pf_positions = [
                            int(v.item())
                            for v in valid_idx_t
                            if int(pf_persistent_ttl_t[int(v.item())].item()) > 0
                        ]
                        rdd_pending_remask_t[valid_idx_t] = True
                        pf_persistent_ttl_t[valid_idx_t] = 0
                        branch_bank = []
                        rdd_rollback_triggered = True
                        rdd_rollback_positions = [int(v.item()) for v in valid_idx_t]
                        rdd_rollback_reason = str(committed_feedback.get("primary_issue", "syntax"))
                        rdd_rollback_severity = float(committed_feedback.get("severity_score", 0.0))
                        stats_state["rdd_rollback_count"] += 1
                        stats_state["rdd_remasked_tokens_total"] += int(len(rdd_rollback_positions))
                        stats_state["rdd_rollback_cleared_pf_count"] += int(len(rdd_rollback_cleared_pf_positions))
                        stats_state["rdd_rollback_events"].append(
                            {
                                "step": int(step),
                                "t": int(t_remaining),
                                "positions": [int(p) for p in rdd_rollback_positions],
                                "cleared_pf_positions": [int(p) for p in rdd_rollback_cleared_pf_positions],
                                "reason": str(rdd_rollback_reason),
                                "severity": float(rdd_rollback_severity),
                                "repair_policy": str(repair_policy),
                                "route_reason": str(repair_route.get("reason", "")),
                            }
                        )
                allow_pf_step = False
            elif pf_rb_policy_enabled and repair_action != "pf":
                allow_pf_step = False

            risk_trace_line = (
                f"risk_trace step={int(step)} t={int(t_remaining)} "
                f"max_risk={max_risk_step:.6f} tau_high={float(tau_high):.6f} "
                f"tau_pf={float(tau_pf_trigger):.6f} tau_grad={float(risk_gradient_gate):.6f} "
                f"tau_pf_final={float(pf_trigger_gate_effective):.6f} allow_pf={bool(allow_pf_step)} "
                f"in_window={bool(allow_pf_phase)} base_window={bool(base_pf_phase)} window=[{int(pf_t_start)},{int(pf_t_end)}] "
                f"signal_mode={str(secondary_signal_mode)} "
                f"parser_issue={str(parser_feedback.get('primary_issue', 'none'))} "
                f"parser_hotspot={bool(parser_hotspot_active)} parser_grad={float(parser_gradient['delta']):.3f} "
                f"parser_accel={float(parser_gradient['accel']):.3f} "
                f"parser_score={float(parser_hotspot_score):.3f} "
                f"cooldown={int(pf_cooldown_remaining)} particles={int(pf_particles_step)} masked={int(masked_count)} "
                f"repair_policy={str(repair_policy)} repair_action={str(repair_action)} "
                f"pf_budget_mode={str(pf_budget_mode)} budget_remaining={int(pf_budget_controls['remaining_extra_forward_budget'])} "
                f"kl_raw_max={float(influence_kl_raw_max):.6f} "
                f"kl_raw_span={float(influence_kl_raw_span):.6f}"
            )
            stats_state["risk_trace_lines"].append(risk_trace_line)
            logger.info(risk_trace_line)

            decisions: List[Dict[str, Any]] = []
            pf_positions_used = 0
            pf_triggered_this_step = False
            action_counts = {
                "commit_argmax": 0,
                "commit_pf": 0,
                "commit_fallback_max_prob": 0,
                "freeze_delay": 0,
                "freeze_cooldown": 0,
                "freeze_budget": 0,
            }
            high_risk_positions: List[int] = []

            def _apply_token_bias(pos: int, token: int) -> None:
                abs_pos = comp_start + int(pos)
                if abs_pos < 0 or abs_pos >= int(logits_seq_t.shape[0]):
                    return
                row = logits_seq_t[abs_pos]
                row_min = torch.min(row)
                row[:] = row_min - float(pf_logit_bias)
                row[int(token)] = row_min + float(pf_logit_bias)

            pf_context_committed_t = committed_mask_t.clone()
            pf_forced_tokens_t = torch.where(pf_context_committed_t, x_comp_ids_t, torch.zeros_like(x_comp_ids_t))

            for pos_t in ranked_positions_t:
                pos = int(pos_t.item())
                argmax_token = int(argmax_all_t[pos].item())
                entropy_val = float(entropy_all_t[pos].item())
                influence_val = float(influences_all_t[pos].item())
                risk_val = float(risk_all_t[pos].item())
                decision = {
                    "pos": pos,
                    "risk": risk_val,
                    "entropy": entropy_val,
                    "influence": influence_val,
                    "argmax_token": argmax_token,
                    "chosen_token": None,
                    "pf_forward_calls": 0,
                    "applied": False,
                }

                if rollback_policy_enabled and repair_action == "rollback":
                    action_counts["freeze_delay"] += 1
                    decision["reason"] = (
                        "remask_pf_rb_rollback"
                        if repair_policy == _REPAIR_POLICY_PF_RB
                        else "remask_rdd_rollback"
                    )
                    decisions.append(decision)
                    continue

                if local_beam_trigger is not None and pos == int(local_beam_trigger.pos):
                    if cooldown_active:
                        action_counts["freeze_cooldown"] += 1
                        decision["reason"] = "remask_local_beam_cooldown"
                        decisions.append(decision)
                        continue
                    try:
                        lb_result = run_commit_timing_local_beam(
                            logits=logits_comp_t_fp32,
                            prev_logits=prev_logits_comp_t,
                            risk=local_beam_trigger,
                            committed_mask=pf_context_committed_t,
                            forced_tokens=pf_forced_tokens_t,
                            cfg=cfg,
                            sampler=pf_sampler,
                            latents=logits_comp_t_fp32,
                            token_ids_to_code=pf_token_ids_to_code,
                            source_prefix=prompt,
                        )
                    except Exception as lb_exc:
                        action_counts["freeze_delay"] += 1
                        decision["reason"] = "local_beam_error_remask"
                        decision["local_beam_error"] = f"{lb_exc}"
                        decisions.append(decision)
                        continue

                    step_forward_calls = int(lb_result.extra_forwards)
                    stats_state["extra_forwards_pf"] += step_forward_calls
                    stats_state["local_beam_extra_forwards"] += step_forward_calls
                    step_pf_forwards += step_forward_calls
                    stats_state["local_beam_branch_events"] += 1
                    stats_state["risk_trigger_count"] += 1
                    stats_state["local_beam_beam_sizes"].append(int(len(lb_result.particle_logs)))
                    high_risk_positions.append(int(pos))
                    pf_triggered_this_step = True
                    if bool(lb_result.accepted_alternative):
                        stats_state["local_beam_accepted_alternatives"] += 1
                    if bool(lb_result.delay_selected):
                        stats_state["local_beam_delay_count"] += 1

                    selected_token = lb_result.selected_token
                    event_log = {
                        "step": int(step),
                        "t": int(t_remaining),
                        "pos": int(pos),
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
                    stats_state["local_beam_event_logs"].append(event_log)
                    decision["reason"] = f"local_beam_{lb_result.reason}"
                    decision["chosen_token"] = int(selected_token) if selected_token is not None else None
                    decision["token_replaced"] = bool(selected_token is not None and int(selected_token) != int(argmax_token))
                    decision["local_beam"] = event_log
                    decision["pf_forward_calls"] = int(step_forward_calls)
                    decision["pf_particles"] = int(len(lb_result.particle_logs))
                    if selected_token is None:
                        local_beam_delay_mask_t[int(pos)] = True
                        action_counts["freeze_delay"] += 1
                        decision["applied"] = False
                    else:
                        _apply_token_bias(pos, int(selected_token))
                        decision["applied"] = True
                        if bool(lb_result.accepted_alternative):
                            action_counts["commit_pf"] += 1
                            persistence_steps = _resolve_pf_persistence_steps(
                                t_remaining=t_remaining,
                                t_start=pf_t_start,
                                t_end=pf_t_end,
                                base_steps=pf_cooldown_steps,
                            )
                            decision["pf_persistence_steps"] = int(persistence_steps)
                            if persistence_steps > 0 and 0 <= pos < int(pf_persistent_ttl_t.numel()):
                                pf_persistent_tokens_t[pos] = int(selected_token)
                                pf_persistent_ttl_t[pos] = max(
                                    int(pf_persistent_ttl_t[pos].item()),
                                    int(persistence_steps),
                                )
                                pf_context_committed_t[pos] = True
                                pf_forced_tokens_t[pos] = int(selected_token)
                        else:
                            action_counts["commit_argmax"] += 1
                    decisions.append(decision)
                    continue

                if risk_val < tau_low:
                    _apply_token_bias(pos, argmax_token)
                    action_counts["commit_argmax"] += 1
                    decision["reason"] = "freeze_low_risk"
                    decision["chosen_token"] = int(argmax_token)
                    decision["applied"] = True
                    decisions.append(decision)
                    continue

                if branch_trial_active:
                    action_counts["freeze_delay"] += 1
                    decision["reason"] = "remask_branch_trial_active"
                    decisions.append(decision)
                    continue

                if branch_trial_merge_due:
                    action_counts["freeze_delay"] += 1
                    decision["reason"] = "remask_branch_trial_merge"
                    decisions.append(decision)
                    continue

                if not allow_pf_phase:
                    action_counts["freeze_delay"] += 1
                    decision["reason"] = "remask_outside_pf_time_window"
                    decisions.append(decision)
                    continue

                if cooldown_active:
                    action_counts["freeze_cooldown"] += 1
                    decision["reason"] = "remask_pf_cooldown"
                    decisions.append(decision)
                    continue

                if pf_budget_blocked:
                    action_counts["freeze_budget"] += 1
                    decision["reason"] = "remask_pf_extra_forward_budget"
                    decision["remaining_extra_forward_budget"] = int(pf_budget_controls["remaining_extra_forward_budget"])
                    decisions.append(decision)
                    continue

                if not allow_pf_step:
                    action_counts["freeze_delay"] += 1
                    decision["reason"] = "remask_pf_rb_not_routed_to_pf" if pf_rb_policy_enabled else "remask_pf_step_blocked"
                    decisions.append(decision)
                    continue

                if strict_parse_repair_pf and not projected_pf_trigger_eligible:
                    action_counts["freeze_delay"] += 1
                    decision["reason"] = (
                        "remask_projected_parse_ok_no_pf"
                        if bool(projected_feedback.get("parse_ok", True))
                        else "remask_projected_not_obvious_syntax_risk"
                    )
                    decisions.append(decision)
                    continue

                if use_secondary_signal:
                    if entropy_val < joint_entropy_gate:
                        action_counts["freeze_delay"] += 1
                        decision["reason"] = "remask_entropy_below_joint_high"
                        decisions.append(decision)
                        continue
                    if pos in pf_candidate_set and influence_val < joint_influence_gate:
                        action_counts["freeze_delay"] += 1
                        decision["reason"] = "remask_influence_below_joint_high"
                        decisions.append(decision)
                        continue

                if risk_val >= tau_high_effective and cfg.pf_enabled:
                    if risk_val < float(risk_gradient_gate_effective):
                        action_counts["freeze_delay"] += 1
                        decision["reason"] = "remask_below_risk_gradient_gate"
                        decisions.append(decision)
                        continue
                    if risk_val < float(pf_trigger_gate_effective):
                        action_counts["freeze_delay"] += 1
                        decision["reason"] = "remask_below_pf_quantile"
                        decisions.append(decision)
                        continue
                    if pf_positions_used >= pf_positions_cap_step:
                        action_counts["freeze_budget"] += 1
                        decision["reason"] = "freeze_budget_cap"
                        decisions.append(decision)
                        continue
                    if pf_trigger_limit > 0 and int(stats_state["pf_trigger_count"]) >= int(pf_trigger_limit):
                        action_counts["freeze_budget"] += 1
                        decision["reason"] = "remask_pf_trigger_cap"
                        decision["pf_trigger_limit"] = int(pf_trigger_limit)
                        decisions.append(decision)
                        continue
                    try:
                        chosen, pf_logs, pf_particles = run_local_pf(
                            logits=logits_comp_t_fp32,
                            pos=pos,
                            committed_mask=pf_context_committed_t,
                            forced_tokens=pf_forced_tokens_t,
                            cfg=pf_scoring_cfg,
                            sampler=pf_sampler,
                            latents=logits_comp_t_fp32,
                            token_ids_to_code=pf_token_ids_to_code,
                            force_fallback=False,
                            pf_particles_override=int(pf_particles_step),
                            current_t=t_remaining,
                            pf_window=(pf_t_start, pf_t_end),
                            source_prefix=prompt,
                            return_particles=True,
                        )
                    except Exception as pf_exc:
                        action_counts["freeze_delay"] += 1
                        decision["reason"] = "pf_error"
                        decision["pf_error"] = f"{pf_exc}"
                        decisions.append(decision)
                        continue
                    step_forward_calls = int(
                        sum(int(log.get("lookahead_forward_calls_total", 0)) for log in pf_logs)
                    )
                    stats_state["extra_forwards_pf"] += step_forward_calls
                    step_pf_forwards += step_forward_calls
                    stats_state["pf_trigger_count"] += 1
                    stats_state["risk_trigger_count"] += 1
                    pf_positions_used += 1
                    pf_triggered_this_step = True
                    decision["pf_forward_calls"] = step_forward_calls
                    decision["pf_particles"] = int(pf_particles_step)
                    decision["reason"] = "trigger_pf"
                    decision["pf_budget_mode"] = str(pf_budget_mode)
                    decision["pf_budget_remaining_before"] = int(pf_budget_controls["remaining_extra_forward_budget"])
                    if chosen is None:
                        do_no_harm_reason = next(
                            (
                                str(log.get("pf_do_no_harm_reason", ""))
                                for log in pf_logs
                                if bool(log.get("pf_do_no_harm_rejected", False))
                            ),
                            "",
                        )
                        if pf_rb_policy_enabled:
                            action_counts["freeze_delay"] += 1
                            decision["applied"] = False
                            decision["reason"] = (
                                "pf_rejected_do_no_harm_remask"
                                if do_no_harm_reason
                                else "pf_no_candidate_remask"
                            )
                            if do_no_harm_reason:
                                decision["pf_do_no_harm_reason"] = do_no_harm_reason
                            decisions.append(decision)
                            continue
                        _apply_token_bias(pos, argmax_token)
                        action_counts["commit_argmax"] += 1
                        decision["chosen_token"] = int(argmax_token)
                        decision["applied"] = True
                        decision["reason"] = (
                            "pf_rejected_do_no_harm_commit_argmax"
                            if do_no_harm_reason
                            else "pf_no_candidate_commit_argmax"
                        )
                        if do_no_harm_reason:
                            decision["pf_do_no_harm_reason"] = do_no_harm_reason
                        decisions.append(decision)
                        continue
                    pf_candidate_feedback, _pf_candidate_source = _parser_feedback_after_forced_token(
                        current_tokens_t=x_comp_ids_t,
                        committed_mask_t=pf_context_committed_t,
                        predicted_tokens_t=argmax_all_t,
                        pos=int(pos),
                        token=int(chosen),
                        token_ids_to_code=pf_token_ids_to_code,
                        source_prefix=prompt,
                    )
                    decision["pf_candidate_ast_parse_ok"] = bool(pf_candidate_feedback.get("parse_ok", False))
                    decision["pf_candidate_quality_score"] = float(pf_candidate_feedback.get("quality_score", 0.0))
                    decision["pf_base_quality_score"] = float(projected_feedback.get("quality_score", 1.0))
                    if (
                        pf_rb_policy_enabled
                        and (
                            not bool(pf_candidate_feedback.get("parse_ok", False))
                            or float(pf_candidate_feedback.get("quality_score", 0.0)) + float(pf_acceptance_tolerance)
                            < float(projected_feedback.get("quality_score", 1.0))
                        )
                    ):
                        action_counts["freeze_delay"] += 1
                        decision["applied"] = False
                        decision["reason"] = (
                            "pf_rejected_ast_parse_failed"
                            if not bool(pf_candidate_feedback.get("parse_ok", False))
                            else "pf_rejected_ast_parse_regression"
                        )
                        decision["pf_candidate_primary_issue"] = str(pf_candidate_feedback.get("primary_issue", "none"))
                        decisions.append(decision)
                        continue
                    high_risk_positions.append(pos)
                    fork_started = False
                    if (
                        bool(current_gradient_hotspot)
                        and not pf_rb_policy_enabled
                        and int(getattr(cfg, "beam_width", 1)) > 1
                        and pf_particles
                        and not branch_trial_active
                        and not branch_trial_merge_due
                        and not branch_bank
                    ):
                        particle_groups: List[Tuple[int, List[Any]]] = []
                        fork_ttl = max(int(getattr(cfg, "branch_trial_steps", 3)), 1)
                        primary_particles = list(pf_particles[: max(int(getattr(cfg, "beam_width", 1)), 1)])
                        if primary_particles:
                            particle_groups.append((int(pos), primary_particles))
                        max_joint_positions = min(
                            max(int(getattr(cfg, "joint_fork_positions", 1)), 1),
                            max(int(pf_positions_cap_step - pf_positions_used + 1), 1),
                        )
                        if bool(getattr(cfg, "joint_fork_enabled", False)) and max_joint_positions > 1:
                            for extra_pos_t in ranked_positions_t:
                                extra_pos = int(extra_pos_t.item())
                                if extra_pos == int(pos):
                                    continue
                                if pf_trigger_limit > 0 and int(stats_state["pf_trigger_count"]) >= int(pf_trigger_limit):
                                    break
                                if len(particle_groups) >= int(max_joint_positions):
                                    break
                                extra_entropy = float(entropy_all_t[extra_pos].item())
                                extra_influence = float(influences_all_t[extra_pos].item())
                                extra_risk = float(risk_all_t[extra_pos].item())
                                if use_secondary_signal and (
                                    extra_entropy < joint_entropy_gate or extra_influence < joint_influence_gate
                                ):
                                    continue
                                if extra_risk < float(tau_high_effective):
                                    continue
                                if extra_risk < float(risk_gradient_gate_effective):
                                    continue
                                if extra_risk < float(pf_trigger_gate_effective):
                                    continue
                                if pf_positions_used >= pf_positions_cap_step:
                                    break
                                try:
                                    extra_chosen, extra_pf_logs, extra_particles = run_local_pf(
                                        logits=logits_comp_t_fp32,
                                        pos=extra_pos,
                                        committed_mask=pf_context_committed_t,
                                        forced_tokens=pf_forced_tokens_t,
                                        cfg=pf_scoring_cfg,
                                        sampler=pf_sampler,
                                        latents=logits_comp_t_fp32,
                                        token_ids_to_code=pf_token_ids_to_code,
                                        force_fallback=False,
                                        pf_particles_override=int(pf_particles_step),
                                        current_t=t_remaining,
                                        pf_window=(pf_t_start, pf_t_end),
                                        source_prefix=prompt,
                                        return_particles=True,
                                    )
                                except Exception:
                                    continue
                                extra_forward_calls = int(
                                    sum(int(log.get("lookahead_forward_calls_total", 0)) for log in extra_pf_logs)
                                )
                                stats_state["extra_forwards_pf"] += extra_forward_calls
                                step_pf_forwards += extra_forward_calls
                                stats_state["pf_trigger_count"] += 1
                                stats_state["risk_trigger_count"] += 1
                                pf_positions_used += 1
                                if extra_chosen is not None and extra_particles:
                                    particle_groups.append(
                                        (
                                            int(extra_pos),
                                            list(extra_particles[: max(int(getattr(cfg, "beam_width", 1)), 1)]),
                                        )
                                    )
                                    high_risk_positions.append(int(extra_pos))
                        branch_candidates = _build_joint_branch_candidates(
                            particle_groups=particle_groups,
                            base_owned_mask_t=branch_parent_owned_mask_t,
                            base_owned_tokens_t=branch_parent_tokens_t,
                            base_tokens_t=x_comp_ids_t,
                            base_committed_t=committed_mask_t,
                            parent_score=float(branch_parent_score),
                            next_branch_id=int(next_branch_id),
                            t_remaining=int(t_remaining),
                            beam_width=int(getattr(cfg, "beam_width", 1)),
                            trial_steps=int(fork_ttl),
                        )
                        if branch_candidates:
                            branch_bank, fork_pruned = _prune_dream_branch_bank(
                                branch_bank + branch_candidates,
                                beam_width=int(getattr(cfg, "beam_width", 1)),
                            )
                            next_branch_id += int(len(branch_candidates))
                            branch_pruned_count += int(fork_pruned)
                            branch_forked_this_step = True
                            branch_trial_active = True
                            fork_started = True
                            decision["forked_branch_count"] = int(len(branch_bank))
                            decision["fork_source"] = "parser_gradient_hotspot"
                            decision["joint_fork_positions"] = [int(group[0]) for group in particle_groups]
                            decision["branch_trial_steps"] = int(fork_ttl)
                            decision["reason"] = "fork_pf_trial"

                    if chosen is not None and (budgeted_entropy_pf or pf_rb_policy_enabled):
                        parser_delta = _selected_pf_particle_parser_delta(pf_logs, chosen_token=int(chosen))
                        decision["pf_parser_delta"] = parser_delta
                        if bool(parser_delta.get("observed", False)) and not _should_accept_budgeted_pf_choice(
                            base_quality=float(parser_delta.get("base_quality", 0.0)),
                            candidate_quality=float(parser_delta.get("candidate_quality", 0.0)),
                            candidate_parse_ok=bool(parser_delta.get("candidate_parse_ok", True)),
                            tolerance=float(pf_acceptance_tolerance),
                        ):
                            chosen = None
                            decision["reason"] = "pf_rejected_parser_regression"

                    if chosen is not None and not fork_started:
                        _apply_token_bias(pos, int(chosen))
                        action_counts["commit_pf"] += 1
                        decision["chosen_token"] = int(chosen)
                        decision["applied"] = True
                        persistence_steps = _resolve_pf_persistence_steps(
                            t_remaining=t_remaining,
                            t_start=pf_t_start,
                            t_end=pf_t_end,
                            base_steps=pf_cooldown_steps,
                        )
                        decision["pf_persistence_steps"] = int(persistence_steps)
                        if persistence_steps > 0 and 0 <= pos < int(pf_persistent_ttl_t.numel()):
                            pf_persistent_tokens_t[pos] = int(chosen)
                            pf_persistent_ttl_t[pos] = max(int(pf_persistent_ttl_t[pos].item()), int(persistence_steps))
                            pf_context_committed_t[pos] = True
                            pf_forced_tokens_t[pos] = int(chosen)
                    elif fork_started:
                        action_counts["freeze_delay"] += 1
                        decision["applied"] = False
                    else:
                        action_counts["freeze_delay"] += 1
                    decisions.append(decision)
                    continue

                action_counts["freeze_delay"] += 1
                decision["reason"] = "remask_late_mid_risk"
                decisions.append(decision)

            cooldown_start = int(pf_cooldown_remaining)
            if pf_triggered_this_step and pf_cooldown_steps > 0:
                pf_cooldown_remaining = int(pf_cooldown_steps)
            elif pf_cooldown_remaining > 0:
                pf_cooldown_remaining = int(pf_cooldown_remaining - 1)
            cooldown_end = int(pf_cooldown_remaining)
            if rdd_rollback_triggered and rdd_rollback_cooldown_steps > 0:
                rdd_rollback_cooldown_remaining = int(rdd_rollback_cooldown_steps)
            elif rdd_rollback_cooldown_remaining > 0:
                rdd_rollback_cooldown_remaining = int(rdd_rollback_cooldown_remaining - 1)
            rdd_cooldown_end = int(rdd_rollback_cooldown_remaining)

            low_count = int((masked_risk_t < float(tau_low)).sum().item())
            high_count = int((masked_risk_t >= float(tau_high)).sum().item())
            mid_count = int(int(masked_risk_t.numel()) - low_count - high_count)

            for action_name, val in action_counts.items():
                stats_state["action_counts_total"][action_name] += int(val)
            stats_state["n_commits"] += int(action_counts["commit_argmax"] + action_counts["commit_pf"])

            if int(masked_entropy_t.numel()) > 0:
                entropy_min = float(masked_entropy_t.min().item())
                entropy_mean = float(masked_entropy_t.mean().item())
                entropy_max = float(masked_entropy_t.max().item())
            else:
                entropy_min = 0.0
                entropy_mean = 0.0
                entropy_max = 0.0
            stats_state["step_logs"].append(
                {
                    "step": int(step),
                    "t": int(t_remaining),
                    "secondary_signal_mode": str(secondary_signal_mode),
                    "allow_pf_phase": bool(allow_pf_phase),
                    "allow_pf_phase_base_window": bool(base_pf_phase),
                    "allow_pf_step": bool(allow_pf_step),
                    "allow_kl_step": bool(allow_secondary_signal_step),
                    "cooldown_active": bool(cooldown_active),
                    "cooldown_remaining_start": int(cooldown_start),
                    "cooldown_remaining_end": int(cooldown_end),
                    "rdd_rollback_enabled": bool(rdd_rollback_enabled),
                    "rdd_rollback_triggered": bool(rdd_rollback_triggered),
                    "rdd_rollback_positions": [int(p) for p in rdd_rollback_positions],
                    "rdd_rollback_reason": str(rdd_rollback_reason),
                    "rdd_rollback_severity": float(rdd_rollback_severity),
                    "rdd_rollback_cleared_pf_positions": [int(p) for p in rdd_rollback_cleared_pf_positions],
                    "rdd_cooldown_active": bool(rdd_cooldown_active),
                    "rdd_cooldown_remaining_start": int(rdd_cooldown_start),
                    "rdd_cooldown_remaining_end": int(rdd_cooldown_end),
                    "repair_policy": str(repair_policy),
                    "repair_action": str(repair_action),
                    "repair_state": str(repair_route.get("state", "")),
                    "repair_reason": str(repair_route.get("reason", "")),
                    "repair_route": repair_route,
                    "pf_rb_policy_enabled": bool(pf_rb_policy_enabled),
                    "pf_rb_action": str(repair_action if repair_policy == _REPAIR_POLICY_PF_RB else "normal"),
                    "pf_rb_state": str(repair_route.get("state", "") if repair_policy == _REPAIR_POLICY_PF_RB else ""),
                    "pf_rb_reason": str(repair_route.get("reason", "") if repair_policy == _REPAIR_POLICY_PF_RB else ""),
                    "pf_rb_route": repair_route if repair_policy == _REPAIR_POLICY_PF_RB else _default_repair_route(),
                    "committed_feedback": committed_feedback,
                    "projected_feedback": projected_feedback,
                    "pf_triggered_this_step": bool(pf_triggered_this_step),
                    "phase_ratio_threshold": float(pf_phase_ratio),
                    "pf_phase_ratio_active": bool(pf_phase_ratio_active),
                    "mask_ratio": float(mask_ratio),
                    "pf_time_window_mode": str(cfg.pf_time_window_mode),
                    "pf_time_window_start_t": int(pf_t_start),
                    "pf_time_window_end_t": int(pf_t_end),
                    "in_pf_time_window": bool(allow_pf_phase),
                    "pf_bandwidth": int(pf_positions_cap_step),
                    "pf_budget_mode": str(pf_budget_mode),
                    "pf_budget_active": bool(pf_budget_controls["budget_active"]),
                    "pf_budget_blocked": bool(pf_budget_blocked),
                    "pf_syntax_repair_mode": bool(pf_budget_controls["syntax_repair_mode"]),
                    "pf_extra_forward_budget": int(pf_extra_forward_budget),
                    "pf_trigger_limit": int(pf_trigger_limit),
                    "pf_trigger_cap_reached": bool(pf_trigger_cap_reached),
                    "pf_remaining_extra_forward_budget": int(pf_budget_controls["remaining_extra_forward_budget"]),
                    "pf_estimated_forward_cost": int(pf_budget_controls["estimated_pf_forward_cost"]),
                    "pf_candidate_budget": int(candidate_budget),
                    "pf_candidate_positions": [int(p) for p in pf_candidate_positions],
                    "pf_candidate_meta": {
                        str(int(pos)): {
                            "entropy": float(meta.get("entropy", 0.0)),
                            "structure_score": float(meta.get("structure_score", 0.0)),
                            "priority_score": float(meta.get("priority_score", 0.0)),
                        }
                        for pos, meta in pf_candidate_meta.items()
                    },
                    "tau_low": float(tau_low),
                    "tau_high": float(tau_high),
                    "tau_high_effective": float(tau_high_effective),
                    "tau_pf_trigger": float(tau_pf_trigger),
                    "risk_mean": float(risk_mean),
                    "risk_std": float(risk_std),
                    "risk_gradient_sigma": float(pf_risk_gradient_sigma),
                    "risk_gradient_gate": float(risk_gradient_gate),
                    "risk_gradient_gate_effective": float(risk_gradient_gate_effective),
                    "pf_trigger_gate": float(pf_trigger_gate),
                    "pf_trigger_gate_effective": float(pf_trigger_gate_effective),
                    "joint_gate_quantile": float(joint_gate_quantile),
                    "pf_trigger_quantile": float(pf_trigger_quantile),
                    "pf_particles_step": int(pf_particles_step),
                    "pf_persistent_mask_size": int((pf_persistent_ttl_t > 0).sum().item()),
                    "pf_persistent_ttl_max": int(pf_persistent_ttl_t.max().item()) if int(pf_persistent_ttl_t.numel()) > 0 else 0,
                    "pf_persistent_positions": [
                        int(v.item()) for v in torch.nonzero(pf_persistent_ttl_t > 0, as_tuple=False).flatten()
                    ],
                    "joint_entropy_gate": float(joint_entropy_gate),
                    "joint_influence_gate": float(joint_influence_gate),
                    "max_risk": float(max_risk_step),
                    "attention_proxy_raw_max": float(influence_kl_raw_max),
                    "attention_proxy_raw_span": float(influence_kl_raw_span),
                    "attention_proxy_raw_nonzero": int(influence_kl_raw_nonzero),
                    "influence_kl_raw_max": float(influence_kl_raw_max),
                    "influence_kl_raw_span": float(influence_kl_raw_span),
                    "influence_kl_raw_nonzero": int(influence_kl_raw_nonzero),
                    "entropy_min": float(entropy_min),
                    "entropy_mean": float(entropy_mean),
                    "entropy_max": float(entropy_max),
                    "parser_feedback": parser_feedback,
                    "parser_hotspot_score": float(parser_hotspot_score),
                    "parser_hotspot_active": bool(parser_hotspot_active),
                    "parser_gradient_score": float(parser_gradient["score"]),
                    "parser_gradient_delta": float(parser_gradient["delta"]),
                    "parser_gradient_accel": float(parser_gradient["accel"]),
                    "parser_gradient_hotspot": bool(current_gradient_hotspot),
                    "parser_quality_score": float(parser_gradient["quality"]),
                    "parser_severity_score": float(parser_gradient["severity"]),
                    "parser_gate_scale": float(parser_gate_scale),
                    "parser_feedback_step_counts": {
                        "bracket": int(parser_issue_histograms["bracket"].get(int(t_remaining), 0)),
                        "indent": int(parser_issue_histograms["indent"].get(int(t_remaining), 0)),
                    },
                    "parser_hotspot_step_scores": {
                        "bracket": float(parser_hotspot_histograms["bracket"].get(int(t_remaining), 0.0)),
                        "indent": float(parser_hotspot_histograms["indent"].get(int(t_remaining), 0.0)),
                    },
                    "entropy_histogram": {},
                    "uncommitted_mask_size": int(masked_count),
                    "valid_mask_size": int(valid_count),
                    "invalid_mask_size": int(max(masked_count - valid_count, 0)),
                    "influence_target_mask_size": int(len(influence_targets)),
                    "high_risk_mask_size": int(high_count),
                    "mid_risk_mask_size": int(mid_count),
                    "low_risk_mask_size": int(low_count),
                    "influence_compute_count": int(len(influence_targets)),
                    "influence_targets": [int(p) for p in influence_targets],
                    "influence_target_tokens": [],
                    "influence_top_positions": [int(p) for p in influence_top_positions],
                    "local_beam_enabled": bool(getattr(cfg, "local_beam_enabled", False)),
                    "local_beam_triggered_this_step": bool(
                        local_beam_trigger is not None
                        and any(int(p) == int(local_beam_trigger.pos) for p in high_risk_positions)
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
                    "action_counts": action_counts,
                    "token_replacement_count": int(
                        sum(
                            1
                            for d in decisions
                            if d.get("chosen_token") is not None
                            and int(d.get("chosen_token")) != int(d.get("argmax_token"))
                        )
                    ),
                    "resample_count": int(sum(1 for d in decisions if d.get("chosen_token") is not None)),
                    "extra_forwards_influence_step": int(step_influence_forwards),
                    "extra_forwards_pf_step": int(step_pf_forwards),
                    "extra_forwards_step_total": int(step_pf_forwards + step_influence_forwards),
                    "branch_count_before": int(branch_count_before),
                    "branch_count_after": int(max(len(branch_bank), 1)),
                    "branch_pruned_count": int(branch_pruned_count),
                    "branch_forked_this_step": bool(branch_forked_this_step),
                    "branch_trial_active": bool(branch_trial_active),
                    "branch_trial_merge_due": bool(branch_trial_merge_due),
                    "branch_trial_steps": int(getattr(cfg, "branch_trial_steps", 3)),
                    "branch_merge_selected": bool(branch_merge_selected),
                    "branch_parent_id": int(branch_parent_id) if branch_parent_id is not None else None,
                    "active_branch_id": int(active_branch_id) if active_branch_id is not None else None,
                    "branch_commit_allowed": bool(branch_commit_allowed),
                    "branch_commit_margin": float(getattr(cfg, "branch_commit_margin", 0.2)),
                    "branch_best_score": float(branch_parent_score) if branch_parent_id is not None else None,
                    "branch_second_id": int(active_branch_second_id) if active_branch_second_id is not None else None,
                    "branch_second_score": float(active_branch_second_score) if active_branch_second_score is not None else None,
                    "branch_score_gap": float(active_branch_score_gap) if active_branch_score_gap is not None else None,
                    "risk_band_histogram": {
                        "low": int(low_count),
                        "mid": int(mid_count),
                        "high": int(high_count),
                    },
                    "risk_triggers": [int(p) for p in high_risk_positions],
                    "pf_decisions": {int(d["pos"]): d for d in decisions},
                }
            )
            prev_logits_comp_t = logits_comp_t_fp32.detach().clone()
            branch_width_trace.append(int(max(len(branch_bank), 1)))
            return logits
        except Exception as exc:
            x_shape = tuple(x.shape) if hasattr(x, "shape") else None
            logits_shape = tuple(logits.shape) if hasattr(logits, "shape") else None
            hook_traceback = traceback.format_exc(limit=6)
            stats_state["step_logs"].append(
                {
                    "step": int(step),
                    "hook_error": f"{exc}",
                    "x_shape": x_shape,
                    "logits_shape": logits_shape,
                    "hook_traceback": hook_traceback,
                }
            )
            return logits

    def _generation_tokens_hook(step, x, logits):
        del step, logits
        try:
            persistent_mask_t = pf_persistent_ttl_t > 0
            pending_rdd_mask_t = rdd_pending_remask_t.clone()
            has_persistent = bool(torch.any(persistent_mask_t).item())
            has_rdd_remask = bool(torch.any(pending_rdd_mask_t).item())
            if not has_persistent and not has_rdd_remask:
                return x
            x_out = x
            if has_persistent:
                x_out = _apply_completion_token_overrides(
                    x=x_out,
                    prompt_len=prompt_len,
                    completion_len=effective_seq_len,
                    override_mask=persistent_mask_t,
                    override_tokens=pf_persistent_tokens_t,
                )
                pf_persistent_ttl_t.sub_(persistent_mask_t.to(dtype=pf_persistent_ttl_t.dtype))
                pf_persistent_ttl_t.clamp_(min=0)
            if has_rdd_remask:
                positions = [int(v.item()) for v in torch.nonzero(pending_rdd_mask_t, as_tuple=False).flatten()]
                x_out = _apply_completion_remask(
                    x=x_out,
                    prompt_len=prompt_len,
                    completion_len=effective_seq_len,
                    remask_positions=positions,
                    mask_token_id=int(tok_mask_id),
                )
                pf_persistent_ttl_t[pending_rdd_mask_t] = 0
                rdd_pending_remask_t.zero_()
            return x_out
        except Exception:
            return x

    dream_top_k = getattr(getattr(model, "generation_config", None), "top_k", None)
    dream_eps = float(getattr(getattr(model, "generation_config", None), "eps", 1e-3) or 1e-3)
    x_loop_t, loop_attention_mask, loop_tok_idx = _prepare_dream_diffusion_state(
        input_ids_t=input_ids,
        attention_mask_t=attention_mask,
        max_new_tokens=int(max_new_tokens),
        mask_token_id=int(tok_mask_id),
    )
    timesteps_t = torch.linspace(1.0, float(dream_eps), int(total_steps) + 1, device=x_loop_t.device)
    branch_batch_x_t: Optional["torch.Tensor"] = None
    x_loop_t = _generation_tokens_hook(None, x_loop_t, None)

    with torch.no_grad():
        for step_idx in range(int(total_steps)):
            t_remaining = int(total_steps - int(step_idx))
            t_value = float(timesteps_t[step_idx].item())
            s_value = float(timesteps_t[step_idx + 1].item())

            if branch_batch_x_t is not None and branch_bank:
                branch_count_before = int(len(branch_bank))
                batch_attention_mask, batch_tok_idx = _expand_dream_loop_context(
                    loop_attention_mask,
                    loop_tok_idx,
                    batch_size=branch_count_before,
                )
                branch_logits_t = _dream_forward_step_logits(
                    model=model,
                    x_t=branch_batch_x_t,
                    attention_mask_t=batch_attention_mask,
                    tok_idx_t=batch_tok_idx,
                )
                branch_next_x_t = _dream_apply_sampling_step(
                    x_t=branch_batch_x_t,
                    logits_t=branch_logits_t,
                    mask_token_id=int(tok_mask_id),
                    t_value=t_value,
                    s_value=s_value,
                    alg=str(alg),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=dream_top_k,
                    alg_temp=float(alg_temp),
                    final_step=bool(step_idx == int(total_steps) - 1),
                )
                if pf_cooldown_remaining > 0:
                    pf_cooldown_remaining = int(pf_cooldown_remaining - 1)
                for branch in branch_bank:
                    branch.ttl = max(int(branch.ttl) - 1, 0)

                branch_trial_merge_due = bool(any(int(branch.ttl) <= 0 for branch in branch_bank))
                branch_merge_selected = False
                branch_extended_this_step = False
                branch_count_after = branch_count_before
                branch_pruned_count = 0
                active_branch_id: Optional[int] = None
                branch_parent_id: Optional[int] = None
                branch_best_score: Optional[float] = None
                branch_second_id: Optional[int] = None
                branch_second_score: Optional[float] = None
                branch_score_gap: Optional[float] = None
                parser_feedback_step: Dict[str, Any] = {
                    "observed": False,
                    "parse_ok": True,
                    "primary_issue": "none",
                    "issue_types": [],
                    "quality_score": 1.0,
                    "severity_score": 0.0,
                }
                parser_gradient_step = {
                    "quality": float(parser_feedback_step["quality_score"]),
                    "severity": float(parser_feedback_step["severity_score"]),
                    "delta": 0.0,
                    "accel": 0.0,
                    "score": 0.0,
                }

                if branch_trial_merge_due:
                    evaluated_branches: List[Tuple[_DreamBranchState, "torch.Tensor", "torch.Tensor", "torch.Tensor"]] = []
                    for branch_idx, branch in enumerate(branch_bank):
                        branch_comp_ids_t = branch_next_x_t[branch_idx, comp_start:comp_end].to(dtype=torch.long)
                        branch_committed_mask_t = branch_comp_ids_t.ne(tok_mask_id)
                        branch_logits_comp_t = branch_logits_t[branch_idx, comp_start:comp_end]
                        eval_score, eval_feedback, future_entropy = _score_dream_branch_state(
                            branch=branch,
                            branch_tokens_t=branch_comp_ids_t,
                            logits_comp_t=branch_logits_comp_t,
                            committed_mask_t=branch_committed_mask_t,
                            token_ids_to_code=pf_token_ids_to_code,
                            cfg=cfg,
                            source_prefix=prompt,
                        )
                        branch.score = float(eval_score)
                        branch.eval_score = float(eval_score)
                        branch.parse_quality = float(eval_feedback.get("quality_score", 1.0))
                        branch.severity = float(eval_feedback.get("severity_score", 0.0))
                        branch.future_entropy = float(future_entropy)
                        evaluated_branches.append((branch, branch_comp_ids_t, branch_committed_mask_t, branch_logits_comp_t))

                    pruned_bank, branch_pruned_count = _prune_dream_branch_bank(
                        [entry[0] for entry in evaluated_branches],
                        beam_width=int(getattr(cfg, "beam_width", 1)),
                    )
                    entry_by_id = {int(entry[0].branch_id): entry for entry in evaluated_branches}
                    ranked_entries = [
                        entry_by_id[int(branch.branch_id)]
                        for branch in pruned_bank
                        if int(branch.branch_id) in entry_by_id
                    ]
                    if ranked_entries:
                        branch_best_score = float(max(ranked_entries[0][0].eval_score, ranked_entries[0][0].score))
                        branch_parent_id = int(ranked_entries[0][0].branch_id)
                        active_branch_id = int(ranked_entries[0][0].branch_id)
                    if len(ranked_entries) > 1:
                        branch_second_id = int(ranked_entries[1][0].branch_id)
                        branch_second_score = float(max(ranked_entries[1][0].eval_score, ranked_entries[1][0].score))
                        branch_score_gap = float(branch_best_score - branch_second_score) if branch_best_score is not None else None
                    if _should_extend_dream_branches(ranked_entries, cfg):
                        extended_branches: List[_DreamBranchState] = []
                        extended_rows: List["torch.Tensor"] = []
                        extension_steps = max(int(getattr(cfg, "branch_extension_steps", 2)), 1)
                        entry_by_id = {int(entry[0].branch_id): entry for entry in ranked_entries}
                        for entry in ranked_entries[: max(int(getattr(cfg, "beam_width", 1)), 2)]:
                            branch = entry[0]
                            branch.ttl = int(extension_steps)
                            branch.extensions_used = int(branch.extensions_used) + 1
                            extended_branches.append(branch)
                            branch_idx = next(
                                idx for idx, existing in enumerate(branch_bank)
                                if int(existing.branch_id) == int(branch.branch_id)
                            )
                            extended_rows.append(branch_next_x_t[branch_idx].clone())
                        branch_bank = extended_branches
                        branch_batch_x_t = torch.stack(extended_rows, dim=0)
                        branch_count_after = int(len(branch_bank))
                        branch_extended_this_step = True
                    else:
                        best_entry = _select_best_dream_branch_entry(ranked_entries)
                        if best_entry is not None:
                            best_branch = best_entry[0]
                            best_index = next(
                                idx for idx, branch in enumerate(branch_bank)
                                if int(branch.branch_id) == int(best_branch.branch_id)
                            )
                            x_loop_t = branch_next_x_t[best_index : best_index + 1].clone()
                            parser_feedback_step = parser_feedback_from_tokens(
                                forced_tokens=x_loop_t[0, comp_start:comp_end].to(dtype=torch.long),
                                committed_mask=x_loop_t[0, comp_start:comp_end].ne(tok_mask_id),
                                token_ids_to_code=pf_token_ids_to_code,
                                min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
                            )
                            parser_gradient_step = _parser_gradient_metrics(
                                parser_feedback=parser_feedback_step,
                                prev_severity=float(parser_feedback_state.get("prev_severity", 0.0)),
                                prev_delta=float(parser_feedback_state.get("prev_delta", 0.0)),
                            )
                        else:
                            x_loop_t = branch_next_x_t[:1].clone()
                        branch_bank = []
                        branch_batch_x_t = None
                        branch_merge_selected = True
                        branch_count_after = 1
                        parser_feedback_state["prev_severity"] = float(parser_gradient_step["severity"])
                        parser_feedback_state["prev_delta"] = float(parser_gradient_step["delta"])
                else:
                    branch_batch_x_t = branch_next_x_t

                masked_count_step = int((branch_next_x_t[0, comp_start:comp_end] == tok_mask_id).sum().item())
                stats_state["step_logs"].append(
                    {
                        "step": int(step_idx),
                        "t": int(t_remaining),
                        "allow_pf_phase": False,
                        "allow_pf_phase_base_window": False,
                        "allow_pf_step": False,
                        "allow_kl_step": False,
                        "cooldown_active": bool(pf_cooldown_remaining > 0),
                        "cooldown_remaining_start": int(max(pf_cooldown_remaining, 0)),
                        "cooldown_remaining_end": int(max(pf_cooldown_remaining, 0)),
                        "pf_triggered_this_step": False,
                        "phase_ratio_threshold": float(pf_phase_ratio),
                        "pf_phase_ratio_active": bool(pf_phase_ratio_active),
                        "mask_ratio": float(masked_count_step / float(max(effective_seq_len, 1))),
                        "pf_time_window_mode": str(cfg.pf_time_window_mode),
                        "pf_time_window_start_t": int(pf_t_start),
                        "pf_time_window_end_t": int(pf_t_end),
                        "in_pf_time_window": bool(pf_t_start <= int(t_remaining) <= pf_t_end),
                        "pf_bandwidth": int(pf_positions_cap),
                        "tau_low": 0.0,
                        "tau_high": 0.0,
                        "tau_high_effective": 0.0,
                        "tau_pf_trigger": 0.0,
                        "risk_mean": 0.0,
                        "risk_std": 0.0,
                        "risk_gradient_sigma": float(pf_risk_gradient_sigma),
                        "risk_gradient_gate": 0.0,
                        "risk_gradient_gate_effective": 0.0,
                        "pf_trigger_gate": 0.0,
                        "pf_trigger_gate_effective": 0.0,
                        "joint_gate_quantile": float(joint_gate_quantile),
                        "pf_trigger_quantile": float(pf_trigger_quantile),
                        "pf_particles_step": 0,
                        "pf_persistent_mask_size": int((pf_persistent_ttl_t > 0).sum().item()),
                        "pf_persistent_ttl_max": int(pf_persistent_ttl_t.max().item()) if int(pf_persistent_ttl_t.numel()) > 0 else 0,
                        "pf_persistent_positions": [
                            int(v.item()) for v in torch.nonzero(pf_persistent_ttl_t > 0, as_tuple=False).flatten()
                        ],
                        "joint_entropy_gate": 0.0,
                        "joint_influence_gate": 0.0,
                        "max_risk": 0.0,
                        "attention_proxy_raw_max": 0.0,
                        "attention_proxy_raw_span": 0.0,
                        "attention_proxy_raw_nonzero": 0,
                        "influence_kl_raw_max": 0.0,
                        "influence_kl_raw_span": 0.0,
                        "influence_kl_raw_nonzero": 0,
                        "entropy_min": 0.0,
                        "entropy_mean": 0.0,
                        "entropy_max": 0.0,
                        "parser_feedback": parser_feedback_step,
                        "parser_hotspot_score": 0.0,
                        "parser_hotspot_active": False,
                        "parser_gradient_score": float(parser_gradient_step["score"]),
                        "parser_gradient_delta": float(parser_gradient_step["delta"]),
                        "parser_gradient_accel": float(parser_gradient_step["accel"]),
                        "parser_gradient_hotspot": False,
                        "parser_quality_score": float(parser_gradient_step["quality"]),
                        "parser_severity_score": float(parser_gradient_step["severity"]),
                        "parser_gate_scale": 1.0,
                        "parser_feedback_step_counts": {"bracket": 0, "indent": 0},
                        "parser_hotspot_step_scores": {"bracket": 0.0, "indent": 0.0},
                        "entropy_histogram": {},
                        "uncommitted_mask_size": int(masked_count_step),
                        "valid_mask_size": int(masked_count_step),
                        "invalid_mask_size": 0,
                        "influence_target_mask_size": 0,
                        "high_risk_mask_size": 0,
                        "mid_risk_mask_size": 0,
                        "low_risk_mask_size": 0,
                        "influence_compute_count": 0,
                        "influence_targets": [],
                        "influence_target_tokens": [],
                        "influence_top_positions": [],
                        "action_counts": {
                            "commit_argmax": 0,
                            "commit_pf": 0,
                            "commit_fallback_max_prob": 0,
                            "freeze_delay": int(masked_count_step),
                            "freeze_cooldown": 0,
                            "freeze_budget": 0,
                        },
                        "token_replacement_count": 0,
                        "resample_count": 0,
                        "extra_forwards_influence_step": 0,
                        "extra_forwards_pf_step": 0,
                        "extra_forwards_step_total": 0,
                        "branch_count_before": int(branch_count_before),
                        "branch_count_after": int(branch_count_after),
                        "branch_pruned_count": int(branch_pruned_count),
                        "branch_forked_this_step": False,
                        "branch_extended_this_step": bool(branch_extended_this_step),
                        "branch_trial_active": bool(not branch_trial_merge_due),
                        "branch_trial_merge_due": bool(branch_trial_merge_due),
                        "branch_trial_steps": int(getattr(cfg, "branch_trial_steps", 3)),
                        "branch_merge_selected": bool(branch_merge_selected),
                        "branch_parent_id": int(branch_parent_id) if branch_parent_id is not None else None,
                        "active_branch_id": int(active_branch_id) if active_branch_id is not None else None,
                        "branch_commit_allowed": bool(branch_merge_selected),
                        "branch_commit_margin": float(getattr(cfg, "branch_commit_margin", 0.2)),
                        "branch_best_score": float(branch_best_score) if branch_best_score is not None else None,
                        "branch_second_id": int(branch_second_id) if branch_second_id is not None else None,
                        "branch_second_score": float(branch_second_score) if branch_second_score is not None else None,
                        "branch_score_gap": float(branch_score_gap) if branch_score_gap is not None else None,
                        "risk_band_histogram": {"low": 0, "mid": 0, "high": 0},
                        "risk_triggers": [],
                        "pf_decisions": {},
                    }
                )
                branch_width_trace.append(int(max(branch_count_after, 1)))
                continue

            main_attention_mask, main_tok_idx = _expand_dream_loop_context(
                loop_attention_mask,
                loop_tok_idx,
                batch_size=int(x_loop_t.shape[0]),
            )
            logits_t = _dream_forward_step_logits(
                model=model,
                x_t=x_loop_t,
                attention_mask_t=main_attention_mask,
                tok_idx_t=main_tok_idx,
            )
            logits_t = _generation_logits_hook(step_idx, x_loop_t, logits_t)

            if branch_bank:
                x_comp_current_t = x_loop_t[0, comp_start:comp_end].to(dtype=torch.long)
                committed_current_t = x_comp_current_t.ne(tok_mask_id)
                branch_rows: List["torch.Tensor"] = []
                for branch in branch_bank:
                    branch_comp_ids_t, _ = _compose_branch_completion_state(
                        base_tokens=x_comp_current_t,
                        base_committed_mask=committed_current_t,
                        branch=branch,
                    )
                    branch_row_t = x_loop_t[0].to(dtype=torch.long).clone()
                    branch_row_t[comp_start:comp_end] = branch_comp_ids_t
                    branch_rows.append(branch_row_t)
                if branch_rows:
                    branch_batch_x_t = torch.stack(branch_rows, dim=0)
                    continue

            eb_transfer_allowed_t = None
            if bool(getattr(cfg, "eb_sampler_enabled", False)):
                projected_feedback_for_eb = _default_parser_feedback()
                try:
                    logits_seq_for_eb_t = logits_t[0] if int(getattr(logits_t, "ndim", 0)) == 3 else logits_t
                    x_comp_for_eb_t = x_loop_t[0, comp_start:comp_end].to(dtype=torch.long)
                    committed_for_eb_t = x_comp_for_eb_t.ne(tok_mask_id)
                    argmax_for_eb_t = torch.argmax(logits_seq_for_eb_t[comp_start:comp_end].float(), dim=-1)
                    projected_source_for_eb = _project_completion_source(
                        current_tokens_t=x_comp_for_eb_t,
                        committed_mask_t=committed_for_eb_t,
                        predicted_tokens_t=argmax_for_eb_t,
                        token_ids_to_code=pf_token_ids_to_code,
                    )
                    projected_feedback_for_eb = parser_feedback_from_source(
                        source=prompt + projected_source_for_eb,
                        min_prefix_chars=int(getattr(cfg, "parser_feedback_min_prefix_chars", 24)),
                    )
                except Exception:
                    projected_feedback_for_eb = _default_parser_feedback()
                try:
                    eb_transfer_allowed_t, eb_step_meta = _build_eb_transfer_allowed_mask(
                        x_t=x_loop_t,
                        logits_t=logits_t,
                        comp_start=comp_start,
                        comp_end=comp_end,
                        mask_token_id=int(tok_mask_id),
                        cfg=cfg,
                        token_ids_to_code=pf_token_ids_to_code,
                        source_prefix=prompt,
                        parser_feedback=projected_feedback_for_eb,
                    )
                except Exception as eb_exc:
                    eb_transfer_allowed_t = None
                    eb_step_meta = {"enabled": True, "error": f"{eb_exc}", "candidate_count": 0, "allowed_count": 0, "blocked_count": 0}
                stats_state["eb_step_count"] += 1
                stats_state["eb_candidate_count"] += int(eb_step_meta.get("candidate_count", 0))
                stats_state["eb_allowed_count"] += int(eb_step_meta.get("allowed_count", 0))
                stats_state["eb_blocked_count"] += int(eb_step_meta.get("blocked_count", 0))
                stats_state["eb_min_fallback_count"] += int(eb_step_meta.get("min_fallback_count", 0))
                stats_state["eb_structure_blocked_count"] += int(eb_step_meta.get("structure_blocked_count", 0))
                stats_state["eb_signature_blocked_count"] += int(eb_step_meta.get("signature_blocked_count", 0))
                stats_state["eb_syntax_near_blocked_count"] += int(eb_step_meta.get("syntax_near_blocked_count", 0))
                if stats_state["step_logs"] and int(stats_state["step_logs"][-1].get("step", -1)) == int(step_idx):
                    compact_meta = {
                        key: eb_step_meta.get(key)
                        for key in (
                            "enabled",
                            "candidate_count",
                            "allowed_count",
                            "blocked_count",
                            "structure_blocked_count",
                            "signature_blocked_count",
                            "syntax_near_blocked_count",
                            "min_fallback_count",
                            "entropy_quantile",
                            "error",
                        )
                        if key in eb_step_meta
                    }
                    stats_state["step_logs"][-1]["eb_sampler"] = compact_meta

            transfer_allowed_t = _merge_completion_transfer_block_mask(
                allowed_t=eb_transfer_allowed_t,
                x_t=x_loop_t,
                prompt_len=prompt_len,
                completion_len=effective_seq_len,
                blocked_completion_mask_t=local_beam_delay_mask_t,
            )
            x_loop_t = _dream_apply_sampling_step(
                x_t=x_loop_t,
                logits_t=logits_t,
                mask_token_id=int(tok_mask_id),
                t_value=t_value,
                s_value=s_value,
                alg=str(alg),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=dream_top_k,
                alg_temp=float(alg_temp),
                final_step=bool(step_idx == int(total_steps) - 1),
                transfer_allowed_mask_t=transfer_allowed_t,
            )
            local_beam_delay_mask_t.zero_()
            x_loop_t = _generation_tokens_hook(step_idx, x_loop_t, logits_t)

    sequences = x_loop_t
    generated_ids = sequences[0, input_ids.shape[1] :]
    if hasattr(generated_ids, "detach"):
        generated_ids = generated_ids.detach().cpu()
    generated = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and eos_token in generated:
        generated = generated.split(eos_token)[0]

    baseline_forwards = int(max(diffusion_steps, 1))
    extra_forwards_influence = int(stats_state["extra_forwards_influence"])
    extra_forwards_pf = int(stats_state["extra_forwards_pf"])
    extra_forwards_total = int(extra_forwards_influence + extra_forwards_pf)
    avg_branch_width = float(sum(branch_width_trace) / len(branch_width_trace)) if branch_width_trace else 1.0
    max_branch_width = int(max(branch_width_trace)) if branch_width_trace else 1
    stats = {
        "pf_trigger_count": int(stats_state["pf_trigger_count"]),
        "extra_forwards": extra_forwards_total,
        "extra_forwards_influence": extra_forwards_influence,
        "extra_forwards_pf": extra_forwards_pf,
        "extra_forwards_local_beam": int(stats_state["local_beam_extra_forwards"]),
        "risk_trigger_count": int(stats_state["risk_trigger_count"]),
        "influence_compute_count": int(stats_state["influence_compute_count"]),
        "n_commits": int(stats_state["n_commits"]),
        "action_counts_total": stats_state["action_counts_total"],
        "parser_feedback_counts": stats_state["parser_feedback_counts"],
        "parser_hotspot_counts": stats_state["parser_hotspot_counts"],
        "parser_feedback_histograms": {
            issue_type: {int(t): int(count) for t, count in hist.items()}
            for issue_type, hist in parser_issue_histograms.items()
        },
        "parser_hotspot_histograms": {
            issue_type: {int(t): float(score) for t, score in hist.items()}
            for issue_type, hist in parser_hotspot_histograms.items()
        },
        "parser_feedback_top_timesteps": _top_parser_issue_timesteps(parser_issue_histograms),
        "parser_hotspot_top_timesteps": _top_parser_score_timesteps(parser_hotspot_histograms),
        "rdd_rollback_enabled": bool(rdd_rollback_enabled),
        "rdd_rollback_count": int(stats_state["rdd_rollback_count"]),
        "rdd_remasked_tokens_total": int(stats_state["rdd_remasked_tokens_total"]),
        "rdd_rollback_cleared_pf_count": int(stats_state["rdd_rollback_cleared_pf_count"]),
        "rdd_rollback_events": stats_state["rdd_rollback_events"],
        "repair_policy": str(repair_policy),
        "repair_route_counts": {
            str(key): int(value)
            for key, value in stats_state["repair_route_counts"].items()
        },
        "pf_rb_route_counts": {
            str(key): int(value)
            for key, value in stats_state["pf_rb_route_counts"].items()
        },
        "step_logs": stats_state["step_logs"],
        "risk_trace_lines": stats_state["risk_trace_lines"],
        "commit_events": [],
        "avg_branch_width": float(avg_branch_width),
        "max_branch_width": int(max_branch_width),
        "baseline_forwards": baseline_forwards,
        "latency_ratio_vs_baseline": float(extra_forwards_total / baseline_forwards + 1.0),
        "branch_width_trace": [int(v) for v in branch_width_trace] if branch_width_trace else [1 for _ in range(int(diffusion_steps))],
        "pf_budget_mode": str(pf_budget_mode),
        "pf_extra_forward_budget": int(pf_extra_forward_budget),
        "pf_trigger_limit": int(pf_trigger_limit),
        "eb_sampler_enabled": bool(getattr(cfg, "eb_sampler_enabled", False)),
        "eb_step_count": int(stats_state["eb_step_count"]),
        "eb_candidate_count": int(stats_state["eb_candidate_count"]),
        "eb_allowed_count": int(stats_state["eb_allowed_count"]),
        "eb_blocked_count": int(stats_state["eb_blocked_count"]),
        "eb_min_fallback_count": int(stats_state["eb_min_fallback_count"]),
        "eb_structure_blocked_count": int(stats_state["eb_structure_blocked_count"]),
        "eb_signature_blocked_count": int(stats_state["eb_signature_blocked_count"]),
        "eb_syntax_near_blocked_count": int(stats_state["eb_syntax_near_blocked_count"]),
        "eb_allowed_per_step": float(stats_state["eb_allowed_count"] / max(int(stats_state["eb_step_count"]), 1)),
        "eb_blocked_per_step": float(stats_state["eb_blocked_count"] / max(int(stats_state["eb_step_count"]), 1)),
        "local_beam_enabled": bool(getattr(cfg, "local_beam_enabled", False)),
        "local_beam_branch_events": int(stats_state["local_beam_branch_events"]),
        "local_beam_accepted_alternatives": int(stats_state["local_beam_accepted_alternatives"]),
        "local_beam_delay_count": int(stats_state["local_beam_delay_count"]),
        "local_beam_avg_beam_size": float(
            sum(stats_state["local_beam_beam_sizes"]) / max(len(stats_state["local_beam_beam_sizes"]), 1)
        ),
        "local_beam_event_logs": stats_state["local_beam_event_logs"],
    }
    return generated, stats


def _build_step_trace(step_log: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": int(step_log.get("step", -1)),
        "entropy": {
            "min": float(step_log.get("entropy_min", 0.0)),
            "mean": float(step_log.get("entropy_mean", 0.0)),
            "max": float(step_log.get("entropy_max", 0.0)),
            "histogram": step_log.get("entropy_histogram", {}),
        },
        "mask_size": {
            "uncommitted": int(step_log.get("uncommitted_mask_size", 0)),
            "valid": int(step_log.get("valid_mask_size", 0)),
            "invalid": int(step_log.get("invalid_mask_size", 0)),
            "influence_targets": int(step_log.get("influence_target_mask_size", 0)),
            "high_risk": int(step_log.get("high_risk_mask_size", 0)),
            "mid_risk": int(step_log.get("mid_risk_mask_size", 0)),
            "low_risk": int(step_log.get("low_risk_mask_size", 0)),
        },
        "influence_compute_count": int(step_log.get("influence_compute_count", 0)),
        "influence_targets": step_log.get("influence_targets", []),
        "influence_target_tokens": step_log.get("influence_target_tokens", []),
        "influence_top_positions": step_log.get("influence_top_positions", []),
        "local_beam": {
            "enabled": bool(step_log.get("local_beam_enabled", False)),
            "triggered": bool(step_log.get("local_beam_triggered_this_step", False)),
            "risk_top": step_log.get("local_beam_risk_top", []),
        },
        "actions": {
            "counts": step_log.get("action_counts", {}),
            "token_replacements": int(step_log.get("token_replacement_count", 0)),
            "resamples": int(step_log.get("resample_count", 0)),
        },
        "extra_forwards_step": {
            "total": int(step_log.get("extra_forwards_step_total", 0)),
            "influence": int(step_log.get("extra_forwards_influence_step", 0)),
            "pf": int(step_log.get("extra_forwards_pf_step", 0)),
        },
        "parser_feedback": {
            "observed": bool((step_log.get("parser_feedback") or {}).get("observed", False)),
            "primary_issue": str((step_log.get("parser_feedback") or {}).get("primary_issue", "none")),
            "issue_types": (step_log.get("parser_feedback") or {}).get("issue_types", []),
            "hotspot_score": float(step_log.get("parser_hotspot_score", 0.0)),
            "hotspot_active": bool(step_log.get("parser_hotspot_active", False)),
            "gradient_score": float(step_log.get("parser_gradient_score", 0.0)),
            "gradient_delta": float(step_log.get("parser_gradient_delta", 0.0)),
            "gradient_accel": float(step_log.get("parser_gradient_accel", 0.0)),
            "gradient_hotspot": bool(step_log.get("parser_gradient_hotspot", False)),
            "quality_score": float(step_log.get("parser_quality_score", 1.0)),
            "severity_score": float(step_log.get("parser_severity_score", 0.0)),
            "gate_scale": float(step_log.get("parser_gate_scale", 1.0)),
        },
        "repair": {
            "policy": str(step_log.get("repair_policy", _REPAIR_POLICY_NONE)),
            "action": str(step_log.get("repair_action", "normal")),
            "state": str(step_log.get("repair_state", "")),
            "reason": str(step_log.get("repair_reason", "")),
            "rollback_triggered": bool(step_log.get("rdd_rollback_triggered", False)),
            "rollback_positions": step_log.get("rdd_rollback_positions", []),
            "cleared_pf_positions": step_log.get("rdd_rollback_cleared_pf_positions", []),
        },
        "pf_rb": {
            "enabled": bool(step_log.get("pf_rb_policy_enabled", False)),
            "action": str(step_log.get("pf_rb_action", "normal")),
            "state": str(step_log.get("pf_rb_state", "")),
            "reason": str(step_log.get("pf_rb_reason", "")),
            "rollback_triggered": bool(step_log.get("rdd_rollback_triggered", False)),
            "rollback_positions": step_log.get("rdd_rollback_positions", []),
            "cleared_pf_positions": step_log.get("rdd_rollback_cleared_pf_positions", []),
        },
        "branch": {
            "before": int(step_log.get("branch_count_before", 1)),
            "after": int(step_log.get("branch_count_after", 1)),
            "pruned": int(step_log.get("branch_pruned_count", 0)),
            "trial_active": bool(step_log.get("branch_trial_active", False)),
            "trial_merge_due": bool(step_log.get("branch_trial_merge_due", False)),
            "trial_steps": int(step_log.get("branch_trial_steps", 0)),
            "merge_selected": bool(step_log.get("branch_merge_selected", False)),
            "parent_id": step_log.get("branch_parent_id"),
            "active_id": step_log.get("active_branch_id"),
            "commit_allowed": bool(step_log.get("branch_commit_allowed", False)),
            "commit_margin": float(step_log.get("branch_commit_margin", 0.0)),
            "best_score": step_log.get("branch_best_score"),
            "second_id": step_log.get("branch_second_id"),
            "second_score": step_log.get("branch_second_score"),
            "score_gap": step_log.get("branch_score_gap"),
            "band_histogram": step_log.get("risk_band_histogram", {}),
        },
    }


def _emit_single_sample_trace(sample_index: int, stats: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "sample_index": sample_index,
        "pf_trigger_count": int(stats.get("pf_trigger_count", 0)),
        "risk_trigger_count": int(stats.get("risk_trigger_count", 0)),
        "extra_forwards": {
            "total": int(stats.get("extra_forwards", 0)),
            "influence": int(stats.get("extra_forwards_influence", 0)),
            "pf": int(stats.get("extra_forwards_pf", 0)),
        },
        "influence_compute_count": int(stats.get("influence_compute_count", 0)),
        "avg_branch_width": float(stats.get("avg_branch_width", 1.0)),
        "latency_ratio_vs_baseline": float(stats.get("latency_ratio_vs_baseline", 1.0)),
        "action_counts_total": stats.get("action_counts_total", {}),
        "parser_feedback_counts": stats.get("parser_feedback_counts", {}),
        "parser_feedback_top_timesteps": stats.get("parser_feedback_top_timesteps", {}),
        "parser_hotspot_counts": stats.get("parser_hotspot_counts", {}),
        "parser_hotspot_top_timesteps": stats.get("parser_hotspot_top_timesteps", {}),
        "rdd_rollback_count": int(stats.get("rdd_rollback_count", 0)),
        "rdd_rollback_cleared_pf_count": int(stats.get("rdd_rollback_cleared_pf_count", 0)),
        "repair_policy": str(stats.get("repair_policy", _REPAIR_POLICY_NONE)),
        "repair_route_counts": stats.get("repair_route_counts", {}),
        "pf_rb_route_counts": stats.get("pf_rb_route_counts", {}),
        "local_beam_enabled": bool(stats.get("local_beam_enabled", False)),
        "local_beam_branch_events": int(stats.get("local_beam_branch_events", 0)),
        "local_beam_accepted_alternatives": int(stats.get("local_beam_accepted_alternatives", 0)),
        "local_beam_delay_count": int(stats.get("local_beam_delay_count", 0)),
        "local_beam_avg_beam_size": float(stats.get("local_beam_avg_beam_size", 0.0)),
        "steps": [_build_step_trace(step_log) for step_log in stats.get("step_logs", [])],
    }
    print(json.dumps({"sample_trace": payload}, ensure_ascii=False))
    return payload


def _collect_risk_trace_lines(stats: Dict[str, Any]) -> List[str]:
    trace_lines = stats.get("risk_trace_lines")
    if isinstance(trace_lines, list) and trace_lines:
        return [str(line) for line in trace_lines]

    # Fallback: reconstruct from step logs if direct trace lines are unavailable.
    lines: List[str] = []
    for step_log in stats.get("step_logs", []):
        if not isinstance(step_log, dict):
            continue
        if "hook_error" in step_log:
            continue
        step = step_log.get("step")
        t = step_log.get("t")
        tau_high = step_log.get("tau_high")
        allow_pf = step_log.get("allow_pf_phase")
        if step is None or t is None or tau_high is None or allow_pf is None:
            continue
        masked = int(step_log.get("uncommitted_mask_size", 0))
        pf_decisions = step_log.get("pf_decisions", {})
        max_risk = float(step_log.get("max_risk", 0.0) or 0.0)
        if max_risk <= 0.0 and isinstance(pf_decisions, dict) and pf_decisions:
            risks = []
            for decision in pf_decisions.values():
                if isinstance(decision, dict) and decision.get("risk") is not None:
                    risks.append(float(decision["risk"]))
            if risks:
                max_risk = float(max(risks))
        lines.append(
            (
                f"risk_trace step={int(step)} t={int(t)} "
                f"max_risk={max_risk:.6f} tau_high={float(tau_high):.6f} "
                f"allow_pf={bool(allow_pf)} masked={masked}"
            )
        )
    return lines


def _prompt_from_item(item: Dict[str, Any], task_mode: str) -> str:
    if task_mode == "infill" and item.get("prefix") and item.get("suffix"):
        return item["prefix"] + "\n# [FILL]\n" + item["suffix"]
    prompt = item.get("prompt", item.get("entry_point", "def f(): pass"))
    if isinstance(prompt, dict):
        return prompt.get("prompt", "def f(): pass")
    if "test_list" in item and "entry_point" not in item:
        tests = item.get("test_list", [])
        if not isinstance(tests, list):
            tests = str(item.get("test", "")).splitlines()
        lines = ["# " + line for line in str(prompt).splitlines()]
        lines.append("#")
        lines.append("# Your code should satisfy these tests:")
        lines.extend("# " + str(test) for test in tests if str(test).strip())
        lines.append("#")
        lines.append("# Return only Python code.")
        return "\n".join(lines) + "\n"
    return prompt


def _collect_warmup_statistics(items: list, make_sampler_and_decode, cfg: DecoderConfig, max_steps: int = 1) -> Dict[str, list]:
    entropy_values: list = []
    influence_values: list = []
    risk_values: list = []
    del max_steps

    for item in items:
        prompt = _prompt_from_item(item, task_mode="completion")
        sampler, _ = make_sampler_and_decode(prompt)
        eot_token_id = getattr(sampler, "eot_token_id", None)
        if eot_token_id is None:
            eot_token_id = getattr(sampler, "eos_token_id", None)
        seq_len = getattr(sampler, "seq_len", 64)
        committed_mask = np.zeros(seq_len, dtype=bool)
        forced_tokens = np.zeros(seq_len, dtype=np.int64)
        latents = None

        if hasattr(sampler, "step_with_aux"):
            out = sampler.step_with_aux(latents, committed_mask, forced_tokens)
            if isinstance(out, tuple) and len(out) == 2:
                logits, aux = out
            else:
                logits, aux = out, {}
        else:
            logits = sampler.step(latents, committed_mask, forced_tokens)
            aux = {}
        uncommitted = np.where(~committed_mask)[0]
        if cfg.dynamic_valid_mask_enabled:
            valid_positions, _ = dynamic_valid_positions(
                logits=logits,
                candidate_positions=(int(idx) for idx in uncommitted),
                committed_mask=committed_mask,
                max_prob_threshold=cfg.valid_mask_max_prob,
                eot_token_id=eot_token_id,
                exclude_eot=cfg.valid_mask_exclude_eot,
            )
        else:
            valid_positions = [int(idx) for idx in uncommitted]

        if not valid_positions:
            continue

        entropies = {idx: entropy_at_position(logits, idx) for idx in valid_positions}
        entropy_values.extend(float(v) for v in entropies.values())

        if cfg.influence_enabled and cfg.use_attention_influence_proxy:
            influences_raw = attention_influence_proxy(
                attention=aux.get("last_attention") if isinstance(aux, dict) else None,
                committed_mask=committed_mask,
                candidate_positions=valid_positions,
            )
        else:
            influences_raw = {int(idx): 0.0 for idx in valid_positions}
        influences = normalize_influence_scores(influences_raw)
        influence_values.extend(float(influences_raw.get(int(idx), 0.0)) for idx in valid_positions)

        for idx in valid_positions:
            ent = entropies[idx]
            inf = influences.get(idx, 0.0)
            risk = risk_score(
                ent,
                inf,
                cfg,
                None,
                use_entropy=True,
                use_influence=cfg.influence_enabled,
            )
            risk_values.append(float(risk))

    return {
        "entropy": entropy_values,
        "influence": influence_values,
        "risk": risk_values,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Risk-aware PF decoder evaluation")
    p.add_argument("--decoder", choices=["risk_pf", "baseline"], default="risk_pf")
    p.add_argument("--dataset", choices=["humaneval", "mbpp", "apps", "codecontests"], default="humaneval")
    p.add_argument("--data_path", default="", help="Path to dataset JSON")
    p.add_argument("--max_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default="", help="Deprecated alias of --result_root")
    p.add_argument("--result_root", default="", help="Result root dir. A timestamp folder is created per run.")
    p.add_argument(
        "--output_rawdata",
        dest="output_rawdata",
        action="store_true",
        default=True,
        help="Write raw prompts/completions and per-step influence/entropy stats into rawdata/.",
    )
    p.add_argument(
        "--no_output_rawdata",
        dest="output_rawdata",
        action="store_false",
        help="Disable rawdata payload dump; only keep result analysis JSON.",
    )
    p.add_argument(
        "--preset",
        default="",
        choices=[
            "",
            "fast_baseline",
            "no_pf",
            "entropy_only",
            "influence_only",
            "pf_no_delay",
            "delay_no_pf",
            "lb_entropy_only",
            "lb_kl_only",
            "lb_entropy_kl",
            "lb_entropy_kl_struct",
            "lb_delay_only",
            "local_beam",
        ],
    )
    p.add_argument("--task", default="completion", choices=["completion", "infill"])

    # Risk / influence / PF knobs
    p.add_argument("--entropy_threshold", type=float, default=0.05)
    p.add_argument("--entropy_low", type=float, default=0.2)
    p.add_argument("--entropy_high", type=float, default=2.5)
    p.add_argument("--commit_entropy_floor", type=float, default=0.3)
    p.add_argument("--risk_threshold", type=float, default=0.6)
    p.add_argument("--risk_low_q", type=float, default=0.25)
    p.add_argument("--risk_high_q", type=float, default=0.75)
    p.add_argument("--joint_gate_quantile", type=float, default=0.75)
    p.add_argument("--pf_trigger_quantile", type=float, default=0.75)
    p.add_argument("--risk_beta", type=float, default=1.0)
    p.add_argument(
        "--risk_fusion_mode",
        choices=["weighted_sum", "independent_triggers", "entropy_only"],
        default="entropy_only",
    )
    p.add_argument("--influence_trigger_floor", type=float, default=0.0)
    p.add_argument("--valid_mask_max_prob", type=float, default=1.0)
    p.add_argument("--w_entropy", type=float, default=1.0)
    p.add_argument("--w_influence", type=float, default=0.0)
    p.add_argument("--influence_enabled", action="store_true", default=False)
    p.add_argument("--influence_eps", type=float, default=0.1)
    p.add_argument("--influence_positions_sample_ratio", type=float, default=0.3)
    p.add_argument("--influence_top_k", type=int, default=1)
    p.add_argument(
        "--correctness_signal_mode",
        choices=["none", "counterfactual_gain", "constraint_identifier"],
        default="none",
        help="Experimental secondary signal used in place of KL influence for Dream PF routing.",
    )
    p.add_argument(
        "--counterfactual_rollout_steps",
        type=int,
        default=1,
        help="Short rollout steps for counterfactual_gain signal.",
    )
    p.add_argument("--pf_enabled", action="store_true", default=False)
    p.add_argument("--pf_top_k", type=int, default=5)
    p.add_argument("--pf_particles", type=int, default=4)
    p.add_argument("--pf_particles_min", type=int, default=1)
    p.add_argument(
        "--pf_particles_schedule",
        choices=["fixed", "time_linear"],
        default="time_linear",
        help="PF particles schedule inside time window.",
    )
    p.add_argument("--pf_horizon_steps", type=int, default=3)
    p.add_argument("--pf_rep_lambda", type=float, default=1.0)
    p.add_argument("--pf_syntax_reward", type=float, default=2.0)
    p.add_argument("--pf_stability_weight", type=float, default=0.5)
    p.add_argument("--pf_repetition_ngram", type=int, default=3)
    p.add_argument("--pf_win_margin", type=float, default=0.2)
    p.add_argument(
        "--pf_budget_mode",
        choices=["legacy", "budgeted_entropy"],
        default="legacy",
        help="PF trigger/budget controller. budgeted_entropy uses entropy-only routing with a small extra-forward budget.",
    )
    p.add_argument(
        "--pf_extra_forward_budget",
        type=int,
        default=_DEFAULT_PF_EXTRA_FORWARD_BUDGET,
        help="Per-sample extra forward budget used by --pf_budget_mode budgeted_entropy. 0 disables the hard cap.",
    )
    p.add_argument(
        "--pf_acceptance_tolerance",
        type=float,
        default=0.02,
        help="Allowed parser-quality drop before budgeted PF rejects a candidate.",
    )
    p.add_argument(
        "--pf_max_triggers_per_sample",
        type=int,
        default=1,
        help="Hard cap for PF interventions per sample. Use 0 for unlimited.",
    )
    p.add_argument(
        "--pf_parse_fail_penalty",
        type=float,
        default=8.0,
        help="Score penalty applied to PF particles that fail python parse checks.",
    )
    p.add_argument(
        "--no_pf_do_no_harm",
        dest="pf_do_no_harm_enabled",
        action="store_false",
        default=True,
        help="Disable the PF parser do-no-harm acceptance gate.",
    )
    p.add_argument(
        "--pf_do_no_harm_margin",
        type=float,
        default=0.2,
        help="Minimum PF score margin over the original argmax path when syntax quality does not improve.",
    )
    p.add_argument(
        "--pf_do_no_harm_min_quality_gain",
        type=float,
        default=0.05,
        help="Minimum parser-quality gain required before PF may replace the original argmax token.",
    )
    p.add_argument(
        "--eb_sampler_enabled",
        action="store_true",
        default=False,
        help="Enable train-free entropy-bounded unmasking: low-entropy tokens may commit, high-entropy tokens stay masked.",
    )
    p.add_argument("--eb_entropy_quantile", type=float, default=0.35)
    p.add_argument("--eb_min_commit_per_step", type=int, default=1)
    p.add_argument("--eb_structure_entropy_scale", type=float, default=0.65)
    p.add_argument("--eb_signature_entropy_scale", type=float, default=0.5)
    p.add_argument("--eb_syntax_near_entropy_scale", type=float, default=0.5)
    p.add_argument("--eb_near_radius", type=int, default=2)
    p.add_argument("--local_beam_enabled", action="store_true", default=False)
    p.add_argument(
        "--local_beam_mode",
        choices=["entropy_only", "kl_only", "entropy_kl", "entropy_kl_struct", "beam", "delay_only"],
        default="entropy_kl_struct",
    )
    p.add_argument("--local_beam_size", type=int, default=4)
    p.add_argument("--local_beam_top_k", type=int, default=5)
    p.add_argument("--local_beam_horizon", type=int, default=2)
    p.add_argument("--local_beam_max_events", type=int, default=1)
    p.add_argument("--local_beam_top_m", type=int, default=1)
    p.add_argument("--local_beam_tau_entropy", type=float, default=0.45)
    p.add_argument("--local_beam_tau_kl", type=float, default=0.8)
    p.add_argument("--local_beam_tau_risk", type=float, default=0.45)
    p.add_argument("--local_beam_lambda_kl", type=float, default=1.0)
    p.add_argument("--local_beam_struct_weight", type=float, default=0.75)
    p.add_argument("--local_beam_margin_base", type=float, default=0.15)
    p.add_argument("--local_beam_margin_branch", type=float, default=0.05)
    p.add_argument("--local_beam_entropy_top_k", type=int, default=64)
    p.add_argument("--local_beam_kl_top_k", type=int, default=32)
    p.add_argument("--local_beam_use_visible_tests", action="store_true", default=False)
    p.add_argument("--local_beam_no_preserve_baseline", dest="local_beam_preserve_baseline", action="store_false", default=True)
    p.add_argument("--entropy_kl_logging", dest="entropy_kl_logging_enabled", action="store_true", default=False)
    p.add_argument("--shadow_mode", dest="shadow_mode_enabled", action="store_true", default=False)
    p.add_argument("--shadow_top_m", type=int, default=3)
    p.add_argument("--shadow_max_events", type=int, default=8)
    p.add_argument("--shadow_risk_threshold", type=float, default=0.45)
    p.add_argument("--shadow_commit_lag_window", type=int, default=3)
    p.add_argument("--shadow_entropy_top_k", type=int, default=16)
    p.add_argument("--shadow_kl_top_k", type=int, default=8)
    p.add_argument("--shadow_token_top_k", type=int, default=5)
    p.add_argument(
        "--branch_observe_mode",
        dest="branch_observe_enabled",
        action="store_true",
        default=False,
        help="Run Priority-3 branch-observe rollouts from official Dream trajectory while forcing final output to baseline.",
    )
    p.add_argument("--branch_observe_beam_size", type=int, default=3)
    p.add_argument("--branch_observe_top_k", type=int, default=3)
    p.add_argument("--branch_observe_max_events", type=int, default=1)
    p.add_argument("--branch_observe_horizon", type=int, default=2)
    p.add_argument(
        "--branch_observe_trigger_mode",
        choices=[
            "auto",
            "legacy_entropy_kl_struct",
            "entropy_only",
            "kl_only",
            "entropy_kl",
            "entropy_kl_struct",
            "entropy_struct",
            "kl_struct",
        ],
        default="auto",
    )
    p.add_argument(
        "--branch_observe_event_policy",
        choices=["top_risk", "random_masked", "random_structural", "highest_conf_structural"],
        default="top_risk",
    )
    p.add_argument("--branch_observe_no_delay", dest="branch_observe_include_delay", action="store_false", default=True)
    p.add_argument(
        "--branch_observe_no_token_fallback",
        action='store_true',
        default=False,
        help='If set, do not synthesize a duplicate top token when unique branch candidates are exhausted.',
    )
    p.add_argument(
        "--branch_observe_allow_output_replace",
        dest="branch_observe_force_baseline_output",
        action="store_false",
        default=True,
        help="Diagnostic only; by default branch-observe always returns official baseline output.",
    )
    p.add_argument(
        "--branch_select_enabled",
        action="store_true",
        default=False,
        help="Enable Priority-4 conservative baseline-preserving selection over branch-observe particles.",
    )
    p.add_argument(
        "--branch_select_verifier",
        choices=["level0", "level1", "oracle"],
        default="level0",
        help="Verifier used for branch selection. level1 uses visible/public tests; oracle is analysis-only.",
    )
    p.add_argument(
        "--branch_select_allow_baseline_level0_pass",
        dest="branch_select_require_baseline_failure",
        action="store_false",
        default=True,
        help="Allow replacement even when baseline already passes Level0 checks.",
    )
    p.add_argument("--branch_select_min_score_gain", type=float, default=1.0)
    p.add_argument("--branch_select_visible_min_pass_gain", type=int, default=1)
    p.add_argument(
        "--branch_select_visible_allow_level0_regression",
        dest="branch_select_visible_require_level0",
        action="store_false",
        default=True,
        help="Allow visible-test selector to ignore Level0 parse/format guard.",
    )
    p.add_argument(
        "--eval_compare_baseline",
        action="store_true",
        default=False,
        help="Run the matching baseline decoder per sample and report recovery/damage statistics. This is evaluation-only.",
    )
    p.add_argument("--diffusion_mid_step", type=int, default=3)
    p.add_argument(
        "--pf_phase_ratio",
        type=float,
        default=0.7,
        help="Enable PF phase only when remaining MASK ratio < pf_phase_ratio.",
    )
    p.add_argument(
        "--pf_time_window_mode",
        choices=["proportional", "absolute"],
        default="proportional",
        help="PF trigger window mode: proportional mapping or absolute t bounds.",
    )
    p.add_argument("--pf_time_window_start", type=int, default=30)
    p.add_argument("--pf_time_window_end", type=int, default=70)
    p.add_argument("--pf_time_window_ref_steps", type=int, default=96)
    p.add_argument("--parser_feedback_enabled", dest="parser_feedback_enabled", action="store_true", default=True)
    p.add_argument("--no_parser_feedback", dest="parser_feedback_enabled", action="store_false")
    p.add_argument("--parser_feedback_min_prefix_chars", type=int, default=24)
    p.add_argument("--parser_feedback_window_radius", type=int, default=2)
    p.add_argument("--parser_feedback_hotspot_threshold", type=float, default=0.1)
    p.add_argument("--parser_feedback_gate_scale", type=float, default=0.85)
    p.add_argument(
        "--rdd_rollback_enabled",
        action="store_true",
        default=False,
        help="Enable train-free RDD-style rollback by remasking a small syntax-error neighborhood.",
    )
    p.add_argument("--rdd_rollback_window", type=int, default=8)
    p.add_argument("--rdd_rollback_max_events", type=int, default=2)
    p.add_argument("--rdd_rollback_min_severity", type=float, default=0.25)
    p.add_argument("--rdd_rollback_cooldown_steps", type=int, default=2)
    p.add_argument("--pf_cooldown_steps", type=int, default=3)
    p.add_argument(
        "--pf_risk_gradient_sigma",
        type=float,
        default=1.5,
        help="Additional PF trigger gate: risk >= mean + sigma * std on masked positions.",
    )
    p.add_argument("--max_delay_steps", type=int, default=5)
    p.add_argument("--fallback_policy", choices=["best_particle", "max_prob"], default="best_particle")

    # Branching and adaptive budget
    p.add_argument("--beam_width", type=int, default=3)
    p.add_argument(
        "--branch_commit_margin",
        type=float,
        default=0.2,
        help="Deprecated compatibility knob; Dream official PF now merges branches after a fixed trial period.",
    )
    p.add_argument(
        "--branch_trial_steps",
        type=int,
        default=3,
        help="Forked branches stay alive for this many diffusion steps before one batched merge evaluation selects a winner.",
    )
    p.add_argument("--branch_extension_enabled", action="store_true", default=False)
    p.add_argument("--branch_extension_steps", type=int, default=2)
    p.add_argument("--branch_extension_margin", type=float, default=0.4)
    p.add_argument("--branch_max_extensions", type=int, default=1)
    p.add_argument("--joint_fork_enabled", action="store_true", default=False)
    p.add_argument("--joint_fork_positions", type=int, default=2)
    p.add_argument("--max_branch_positions_per_step", type=int, default=8)
    p.add_argument("--adaptive_budget_enabled", action="store_true", default=False)
    p.add_argument("--overhead_cap_ratio", type=float, default=3.0)

    # Calibration
    p.add_argument("--calibrate_warmup_samples", type=int, default=0)
    p.add_argument("--thresholds_path", default="", help="Optional JSON path to save calibrated thresholds")

    # Metrics / tracing
    p.add_argument("--unit_timeout", type=float, default=3.0)
    p.add_argument("--logging_level", default="INFO")
    p.add_argument("--trace_sample", action="store_true", default=False)
    p.add_argument("--trace_sample_index", type=int, default=0)

    p.add_argument("--backend", choices=["placeholder", "llada", "dream"], default="placeholder")

    p.add_argument("--llada_model_path", default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--llada_device", default="cuda")
    p.add_argument("--llada_seq_len", type=int, default=256)
    p.add_argument("--llada_max_prompt_tokens", type=int, default=512)
    p.add_argument("--llada_mask_id", type=int, default=-1, help="Use tokenizer mask_token_id when < 0")
    p.add_argument("--llada_use_chat_template", action="store_true", default=False)

    p.add_argument("--dream_model_path", default="Dream-org/Dream-Coder-v0-Instruct-7B")
    p.add_argument("--dream_device", default="cuda")
    p.add_argument("--dream_seq_len", type=int, default=256)
    p.add_argument("--dream_max_prompt_tokens", type=int, default=512)
    p.add_argument("--dream_mask_id", type=int, default=-1, help="Use tokenizer mask_token_id when < 0")
    p.add_argument("--dream_use_chat_template", action="store_true", default=False)
    p.add_argument(
        "--dream_baseline_strategy",
        choices=["official_diffusion", "legacy_sampler"],
        default="official_diffusion",
        help="For --backend dream and --decoder baseline: official diffusion_generate (recommended) or legacy sampler+argmax.",
    )
    p.add_argument("--dream_diffusion_steps", type=int, default=768)
    p.add_argument("--dream_max_new_tokens", type=int, default=768)
    p.add_argument("--dream_temperature", type=float, default=0.1)
    p.add_argument("--dream_top_p", type=float, default=0.95)
    p.add_argument("--dream_alg", default="entropy")
    p.add_argument("--dream_alg_temp", type=float, default=0.0)
    p.add_argument("--dream_eos_penalty", type=float, default=3.0)
    p.add_argument(
        "--dream_risk_pf_strategy",
        choices=["official_diffusion", "step_sampler"],
        default="official_diffusion",
        help="For --backend dream and --decoder risk_pf: official diffusion loop with PF hooks (recommended) or legacy step-sampler decoder.",
    )
    p.add_argument("--dream_pf_positions_per_step", type=int, default=1)
    p.add_argument(
        "--dream_pf_bandwidth",
        type=int,
        default=None,
        help="Alias of --dream_pf_positions_per_step for concurrent PF intervention slots per diffusion step.",
    )
    p.add_argument("--dream_pf_logit_bias", type=float, default=100.0)
    p.add_argument("--dream_pf_parse_checks", action="store_true", default=False)

    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.logging_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    _set_global_seed(args.seed)

    if args.backend == "dream" and args.decoder == "risk_pf":
        logger.info(
            "Dream PF mode=%s. Reference baseline: --decoder baseline --dream_baseline_strategy official_diffusion.",
            args.dream_risk_pf_strategy,
        )

    local_beam_preset_modes = {
        "lb_entropy_only": "entropy_only",
        "lb_kl_only": "kl_only",
        "lb_entropy_kl": "entropy_kl",
        "lb_entropy_kl_struct": "entropy_kl_struct",
        "lb_delay_only": "delay_only",
    }

    if args.preset == "fast_baseline":
        cfg = fast_baseline()
    elif args.preset == "no_pf":
        cfg = ablation_no_pf()
    elif args.preset == "entropy_only":
        cfg = ablation_entropy_only()
    elif args.preset == "influence_only":
        cfg = ablation_influence_only()
    elif args.preset == "pf_no_delay":
        cfg = ablation_pf_no_delay()
    elif args.preset == "delay_no_pf":
        cfg = ablation_delay_no_pf()
    elif args.preset in local_beam_preset_modes:
        cfg = ablation_local_beam(local_beam_preset_modes[args.preset])
    elif args.preset == "local_beam":
        cfg = ablation_local_beam(args.local_beam_mode)
    else:
        cfg = DecoderConfig()

    cfg.random_seed = args.seed
    cfg.entropy_threshold = args.entropy_threshold
    cfg.entropy_windowing = (args.entropy_low, args.entropy_high)
    cfg.entropy_low = args.entropy_low
    cfg.entropy_high = args.entropy_high
    cfg.commit_entropy_floor = args.commit_entropy_floor
    cfg.risk_threshold = args.risk_threshold
    cfg.risk_low_quantile = args.risk_low_q
    cfg.risk_high_quantile = args.risk_high_q
    cfg.joint_gate_quantile = args.joint_gate_quantile
    cfg.pf_trigger_quantile = args.pf_trigger_quantile
    cfg.risk_beta = args.risk_beta
    cfg.risk_fusion_mode = args.risk_fusion_mode
    cfg.influence_trigger_floor = args.influence_trigger_floor
    cfg.valid_mask_max_prob = args.valid_mask_max_prob
    cfg.risk_weights = (args.w_entropy, args.w_influence)
    cfg.w_entropy = args.w_entropy
    cfg.w_influence = args.w_influence
    cfg.influence_enabled = args.influence_enabled or cfg.influence_enabled
    cfg.influence_eps = args.influence_eps
    cfg.influence_positions_sample_ratio = args.influence_positions_sample_ratio
    cfg.influence_top_k = args.influence_top_k
    cfg.correctness_signal_mode = args.correctness_signal_mode
    cfg.counterfactual_rollout_steps = args.counterfactual_rollout_steps
    cfg.pf_enabled = args.pf_enabled or cfg.pf_enabled
    cfg.pf_top_k = args.pf_top_k
    cfg.pf_particles = args.pf_particles
    cfg.pf_particles_min = args.pf_particles_min
    cfg.pf_particles_schedule = args.pf_particles_schedule
    cfg.pf_horizon_steps = args.pf_horizon_steps
    cfg.pf_rep_lambda = args.pf_rep_lambda
    cfg.pf_badness_beta = args.pf_rep_lambda
    cfg.pf_syntax_reward = args.pf_syntax_reward
    cfg.pf_stability_weight = args.pf_stability_weight
    cfg.pf_repetition_ngram = args.pf_repetition_ngram
    cfg.pf_win_margin = args.pf_win_margin
    cfg.pf_budget_mode = args.pf_budget_mode
    cfg.pf_extra_forward_budget = args.pf_extra_forward_budget
    cfg.pf_acceptance_tolerance = args.pf_acceptance_tolerance
    cfg.pf_max_triggers_per_sample = args.pf_max_triggers_per_sample
    cfg.pf_parse_fail_penalty = args.pf_parse_fail_penalty
    cfg.pf_do_no_harm_enabled = args.pf_do_no_harm_enabled
    cfg.pf_do_no_harm_margin = args.pf_do_no_harm_margin
    cfg.pf_do_no_harm_min_quality_gain = args.pf_do_no_harm_min_quality_gain
    cfg.eb_sampler_enabled = args.eb_sampler_enabled
    cfg.eb_entropy_quantile = args.eb_entropy_quantile
    cfg.eb_min_commit_per_step = args.eb_min_commit_per_step
    cfg.eb_structure_entropy_scale = args.eb_structure_entropy_scale
    cfg.eb_signature_entropy_scale = args.eb_signature_entropy_scale
    cfg.eb_syntax_near_entropy_scale = args.eb_syntax_near_entropy_scale
    cfg.eb_near_radius = args.eb_near_radius
    cfg.local_beam_enabled = args.local_beam_enabled or cfg.local_beam_enabled
    if args.preset not in local_beam_preset_modes:
        cfg.local_beam_mode = args.local_beam_mode
    cfg.local_beam_size = args.local_beam_size
    cfg.local_beam_top_k = args.local_beam_top_k
    cfg.local_beam_horizon = args.local_beam_horizon
    cfg.local_beam_max_events = args.local_beam_max_events
    cfg.local_beam_top_m = args.local_beam_top_m
    cfg.local_beam_tau_entropy = args.local_beam_tau_entropy
    cfg.local_beam_tau_kl = args.local_beam_tau_kl
    cfg.local_beam_tau_risk = args.local_beam_tau_risk
    cfg.local_beam_lambda_kl = args.local_beam_lambda_kl
    cfg.local_beam_struct_weight = args.local_beam_struct_weight
    cfg.local_beam_margin_base = args.local_beam_margin_base
    cfg.local_beam_margin_branch = args.local_beam_margin_branch
    cfg.local_beam_entropy_top_k = args.local_beam_entropy_top_k
    cfg.local_beam_kl_top_k = args.local_beam_kl_top_k
    cfg.local_beam_use_visible_tests = args.local_beam_use_visible_tests
    cfg.local_beam_preserve_baseline = args.local_beam_preserve_baseline
    cfg.entropy_kl_logging_enabled = args.entropy_kl_logging_enabled
    cfg.shadow_mode_enabled = args.shadow_mode_enabled
    cfg.shadow_top_m = args.shadow_top_m
    cfg.shadow_max_events = args.shadow_max_events
    cfg.shadow_risk_threshold = args.shadow_risk_threshold
    cfg.shadow_commit_lag_window = args.shadow_commit_lag_window
    cfg.shadow_entropy_top_k = args.shadow_entropy_top_k
    cfg.shadow_kl_top_k = args.shadow_kl_top_k
    cfg.shadow_token_top_k = args.shadow_token_top_k
    cfg.branch_observe_enabled = bool(args.branch_observe_enabled or args.branch_select_enabled)
    cfg.branch_observe_beam_size = args.branch_observe_beam_size
    cfg.branch_observe_top_k = args.branch_observe_top_k
    cfg.branch_observe_max_events = args.branch_observe_max_events
    cfg.branch_observe_horizon = args.branch_observe_horizon
    cfg.branch_observe_trigger_mode = args.branch_observe_trigger_mode
    cfg.branch_observe_event_policy = args.branch_observe_event_policy
    cfg.branch_observe_include_delay = args.branch_observe_include_delay
    cfg.branch_observe_token_fallback = not bool(args.branch_observe_no_token_fallback)
    cfg.branch_observe_force_baseline_output = args.branch_observe_force_baseline_output
    cfg.branch_select_enabled = args.branch_select_enabled
    cfg.branch_select_verifier = args.branch_select_verifier
    cfg.branch_select_require_baseline_failure = args.branch_select_require_baseline_failure
    cfg.branch_select_min_score_gain = args.branch_select_min_score_gain
    cfg.branch_select_visible_min_pass_gain = args.branch_select_visible_min_pass_gain
    cfg.branch_select_visible_require_level0 = args.branch_select_visible_require_level0
    cfg.diffusion_mid_step = args.diffusion_mid_step
    cfg.pf_phase_ratio = args.pf_phase_ratio
    cfg.pf_time_window_mode = args.pf_time_window_mode
    cfg.pf_time_window_start = args.pf_time_window_start
    cfg.pf_time_window_end = args.pf_time_window_end
    cfg.pf_time_window_ref_steps = args.pf_time_window_ref_steps
    cfg.parser_feedback_enabled = args.parser_feedback_enabled
    cfg.parser_feedback_min_prefix_chars = args.parser_feedback_min_prefix_chars
    cfg.parser_feedback_window_radius = args.parser_feedback_window_radius
    cfg.parser_feedback_hotspot_threshold = args.parser_feedback_hotspot_threshold
    cfg.parser_feedback_gate_scale = args.parser_feedback_gate_scale
    cfg.rdd_rollback_enabled = args.rdd_rollback_enabled
    cfg.rdd_rollback_window = args.rdd_rollback_window
    cfg.rdd_rollback_max_events = args.rdd_rollback_max_events
    cfg.rdd_rollback_min_severity = args.rdd_rollback_min_severity
    cfg.rdd_rollback_cooldown_steps = args.rdd_rollback_cooldown_steps
    cfg.pf_cooldown_steps = args.pf_cooldown_steps
    cfg.pf_risk_gradient_sigma = args.pf_risk_gradient_sigma
    cfg.max_delay_steps = args.max_delay_steps
    cfg.fallback_policy = args.fallback_policy
    cfg.logging_level = args.logging_level.upper()

    cfg.beam_width = args.beam_width
    cfg.branch_commit_margin = args.branch_commit_margin
    cfg.branch_trial_steps = args.branch_trial_steps
    cfg.branch_extension_enabled = args.branch_extension_enabled
    cfg.branch_extension_steps = args.branch_extension_steps
    cfg.branch_extension_margin = args.branch_extension_margin
    cfg.branch_max_extensions = args.branch_max_extensions
    cfg.joint_fork_enabled = args.joint_fork_enabled
    cfg.joint_fork_positions = args.joint_fork_positions
    cfg.max_branch_positions_per_step = args.max_branch_positions_per_step
    cfg.adaptive_budget_enabled = args.adaptive_budget_enabled
    cfg.overhead_cap_ratio = args.overhead_cap_ratio
    cfg.__post_init__()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    result_root = _resolve_result_root(project_root=root, result_root=args.result_root, log_dir=args.log_dir)
    result_paths = _prepare_result_dirs(result_root=result_root)
    logger.info("Run result dir: %s", result_paths["run_dir"])
    risk_trace_log_path = Path(result_paths["rawdata_dir"]) / "risk_trace.log"
    risk_trace_handler = _attach_risk_trace_log_handler(risk_trace_log_path)

    data_path = args.data_path or os.path.join(root, "data", f"{args.dataset}.json")
    total_needed = int(args.max_samples + max(args.calibrate_warmup_samples, 0))
    items_all = load_json_items(data_path, total_needed)
    if not items_all:
        logger.warning("Dataset empty or missing at %s; running with 2 synthetic prompts", data_path)
        items_all = [{"prompt": "def f(): pass", "task_id": "0"}, {"prompt": "def g(): return 1", "task_id": "1"}]

    warmup_items = items_all[: args.calibrate_warmup_samples] if args.calibrate_warmup_samples > 0 else []
    items = items_all[args.calibrate_warmup_samples : args.calibrate_warmup_samples + args.max_samples]
    if not items:
        items = items_all[: args.max_samples]

    seq_len = 64
    llada_model, llada_tokenizer = None, None
    dream_model, dream_tokenizer = None, None

    if args.backend == "llada":
        llada_model, llada_tokenizer = load_llada_model(args.llada_model_path, device=args.llada_device)
        seq_len = args.llada_seq_len
        logger.info("Loaded LLaDA model from %s", args.llada_model_path)
    elif args.backend == "dream":
        dream_model, dream_tokenizer = load_dream_model(args.dream_model_path, device=args.dream_device)
        seq_len = args.dream_seq_len
        logger.info("Loaded Dream model from %s", args.dream_model_path)

    def make_sampler_and_decode(prompt: str):
        if args.backend == "llada":
            sampler = LLaDASamplerAdapter(
                prompt=prompt,
                model=llada_model,
                tokenizer=llada_tokenizer,
                device=args.llada_device,
                seq_len=seq_len,
                max_prompt_tokens=args.llada_max_prompt_tokens,
                mask_id=None if args.llada_mask_id < 0 else args.llada_mask_id,
                use_chat_template=args.llada_use_chat_template,
            )

            def token_ids_to_code(ids):
                return llada_tokenizer.decode(ids.tolist(), skip_special_tokens=True)

            return sampler, token_ids_to_code

        if args.backend == "dream":
            sampler = DreamSamplerAdapter(
                prompt=prompt,
                model=dream_model,
                tokenizer=dream_tokenizer,
                device=args.dream_device,
                seq_len=seq_len,
                max_prompt_tokens=args.dream_max_prompt_tokens,
                mask_id=None if args.dream_mask_id < 0 else args.dream_mask_id,
                use_chat_template=args.dream_use_chat_template,
            )

            def token_ids_to_code(ids):
                return dream_tokenizer.decode(ids.tolist(), skip_special_tokens=True)

            return sampler, token_ids_to_code

        sampler = PlaceholderDiffusionSampler(seq_len=64, vocab_size=32000, random_seed=args.seed, deterministic=True)
        return sampler, None

    warmup_stats_output = None
    if args.decoder == "risk_pf" and warmup_items:
        warm_stats = _collect_warmup_statistics(warmup_items, make_sampler_and_decode, cfg)
        warmup_stats_output = warm_stats
        payload = apply_warmup_calibration(
            cfg=cfg,
            risk_values=warm_stats["risk"],
            entropy_values=warm_stats["entropy"],
            influence_values=warm_stats["influence"],
            low_q=args.risk_low_q,
            high_q=args.risk_high_q,
        )
        logger.info(
            "Calibrated thresholds: low=%.4f high=%.4f",
            cfg.risk_low_threshold,
            cfg.risk_high_threshold,
        )
        if args.thresholds_path:
            save_calibration(args.thresholds_path, payload)
            logger.info("Saved calibration to %s", args.thresholds_path)

    format_ok = 0
    parse_ok = 0
    pass_ok = 0
    raw_format_ok = 0
    raw_parse_ok = 0
    total = 0

    pf_triggers_list = []
    rollback_counts_list = []
    rollback_cleared_pf_list = []
    overhead_list = []
    branch_width_list = []
    latency_ratio_list = []
    eb_allowed_per_step_list = []
    eb_blocked_per_step_list = []
    eb_min_fallback_list = []
    local_beam_branch_events_list = []
    local_beam_avg_beam_size_list = []
    local_beam_accepted_list = []
    local_beam_delay_list = []
    dream_noop_fast_path_list = []
    shadow_enabled_list = []
    shadow_max_risk_list = []
    shadow_mean_risk_list = []
    shadow_num_risk_events_list = []
    shadow_early_commit_list = []
    sample_latency = []
    sample_analysis_records: List[Dict[str, Any]] = []
    sample_raw_records: List[Dict[str, Any]] = []
    sample_trace_records: List[Dict[str, Any]] = []
    baseline_compare_records: List[Dict[str, Any]] = []
    recovered_baseline_failures = 0
    damaged_baseline_successes = 0
    parse_recovered_baseline_failures = 0
    parse_damaged_baseline_successes = 0
    baseline_compare_latency: List[float] = []
    dream_pf_positions_per_step = (
        int(args.dream_pf_bandwidth)
        if args.dream_pf_bandwidth is not None
        else int(args.dream_pf_positions_per_step)
    )

    def run_reference_baseline(prompt_text: str) -> str:
        if args.backend == "dream" and args.dream_baseline_strategy == "official_diffusion":
            return run_dream_official_baseline(
                prompt=prompt_text,
                model=dream_model,
                tokenizer=dream_tokenizer,
                device=args.dream_device,
                max_new_tokens=args.dream_max_new_tokens,
                diffusion_steps=args.dream_diffusion_steps,
                temperature=args.dream_temperature,
                top_p=args.dream_top_p,
                alg=args.dream_alg,
                alg_temp=args.dream_alg_temp,
                eos_penalty=args.dream_eos_penalty,
                use_chat_template=args.dream_use_chat_template,
                max_prompt_tokens=args.dream_max_prompt_tokens,
            )
        baseline_sampler, baseline_token_ids_to_code = make_sampler_and_decode(prompt_text)
        baseline_slen = getattr(baseline_sampler, "seq_len", 64)
        return run_baseline(
            prompt_text,
            baseline_sampler,
            max_steps=20,
            seq_len=baseline_slen,
            token_ids_to_code=baseline_token_ids_to_code,
        )

    for item in items:
        prompt = _prompt_from_item(item, task_mode=args.task)

        total += 1
        sample_index = total - 1
        sample_seed = int(args.seed) + int(sample_index)
        _set_global_seed(sample_seed)
        sampler, token_ids_to_code = make_sampler_and_decode(prompt)
        slen = getattr(sampler, "seq_len", 64)
        trace_payload = None

        t0 = time.perf_counter()
        if args.decoder == "risk_pf":
            if args.backend == "dream" and args.dream_risk_pf_strategy == "official_diffusion":
                out, stats = run_dream_official_pf(
                    prompt=prompt,
                    model=dream_model,
                    tokenizer=dream_tokenizer,
                    cfg=cfg,
                    device=args.dream_device,
                    max_new_tokens=args.dream_max_new_tokens,
                    diffusion_steps=args.dream_diffusion_steps,
                    temperature=args.dream_temperature,
                    top_p=args.dream_top_p,
                    alg=args.dream_alg,
                    alg_temp=args.dream_alg_temp,
                    eos_penalty=args.dream_eos_penalty,
                    use_chat_template=args.dream_use_chat_template,
                    max_prompt_tokens=args.dream_max_prompt_tokens,
                    mask_id=None if args.dream_mask_id < 0 else args.dream_mask_id,
                    pf_positions_per_step=dream_pf_positions_per_step,
                    pf_logit_bias=args.dream_pf_logit_bias,
                    pf_parse_checks=args.dream_pf_parse_checks,
                )
            else:
                decoder = RiskAwarePFDecoder(cfg, sampler=sampler)
                out, stats = decoder.generate_with_stats(
                    prompt=prompt,
                    cfg=cfg,
                    sampler=sampler,
                    max_steps=20,
                    token_ids_to_code=token_ids_to_code,
                )
            if args.trace_sample and sample_index == args.trace_sample_index:
                trace_payload = _emit_single_sample_trace(sample_index=sample_index, stats=stats)
            pf_triggers_list.append(stats.get("pf_trigger_count", 0))
            rollback_counts_list.append(stats.get("rdd_rollback_count", 0))
            rollback_cleared_pf_list.append(stats.get("rdd_rollback_cleared_pf_count", 0))
            overhead_list.append(stats.get("extra_forwards", 0))
            branch_width_list.append(stats.get("avg_branch_width", 1.0))
            latency_ratio_list.append(stats.get("latency_ratio_vs_baseline", 1.0))
            eb_allowed_per_step_list.append(stats.get("eb_allowed_per_step", 0.0))
            eb_blocked_per_step_list.append(stats.get("eb_blocked_per_step", 0.0))
            eb_min_fallback_list.append(stats.get("eb_min_fallback_count", 0))
            local_beam_branch_events_list.append(stats.get("local_beam_branch_events", 0))
            local_beam_avg_beam_size_list.append(stats.get("local_beam_avg_beam_size", 0.0))
            local_beam_accepted_list.append(stats.get("local_beam_accepted_alternatives", 0))
            local_beam_delay_list.append(stats.get("local_beam_delay_count", 0))
            dream_noop_fast_path_list.append(bool(stats.get("dream_noop_fast_path", False)))
            shadow_enabled_list.append(bool(stats.get("shadow_mode_enabled", False)))
            shadow_max_risk_list.append(float(stats.get("shadow_max_risk", 0.0)))
            shadow_mean_risk_list.append(float(stats.get("shadow_mean_risk", 0.0)))
            shadow_num_risk_events_list.append(int(stats.get("shadow_num_risk_events", 0)))
            shadow_early_commit_list.append(int(stats.get("shadow_high_risk_early_commit_count", 0)))
        else:
            if args.backend == "dream" and args.dream_baseline_strategy == "official_diffusion":
                out = run_dream_official_baseline(
                    prompt=prompt,
                    model=dream_model,
                    tokenizer=dream_tokenizer,
                    device=args.dream_device,
                    max_new_tokens=args.dream_max_new_tokens,
                    diffusion_steps=args.dream_diffusion_steps,
                    temperature=args.dream_temperature,
                    top_p=args.dream_top_p,
                    alg=args.dream_alg,
                    alg_temp=args.dream_alg_temp,
                    eos_penalty=args.dream_eos_penalty,
                    use_chat_template=args.dream_use_chat_template,
                    max_prompt_tokens=args.dream_max_prompt_tokens,
                )
            else:
                out = run_baseline(prompt, sampler, max_steps=20, seq_len=slen, token_ids_to_code=token_ids_to_code)
            stats = {}
        elapsed = time.perf_counter() - t0
        sample_latency.append(elapsed)

        raw_out = out
        eval_out, postprocess_meta = postprocess_completion_for_eval(
            item=item,
            completion=raw_out,
            dataset=args.dataset,
        )

        raw_fmt_ok = format_success(raw_out)
        raw_prs_ok = parse_success_for_eval(item, prompt, raw_out, args.dataset)
        fmt_ok = format_success(eval_out)
        prs_ok = parse_success_for_eval(item, prompt, eval_out, args.dataset)
        tst_ok = unit_test_pass(item, eval_out, dataset=args.dataset, timeout=args.unit_timeout)
        selection_verifier = str(getattr(cfg, "branch_select_verifier", "level0") or "level0").lower()
        visible_selection_enabled = bool(getattr(cfg, "branch_select_enabled", False) and selection_verifier == "level1")
        baseline_visible_score = (
            _visible_test_score(
                item=item,
                prompt=prompt,
                completion=eval_out,
                dataset=args.dataset,
                timeout=args.unit_timeout,
            )
            if visible_selection_enabled
            else {
                "enabled": False,
                "passed": 0,
                "total": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "all_passed": False,
            }
        )
        branch_observe_eval: Dict[str, Any] = {
            "enabled": bool(stats.get("branch_observe_enabled", False)),
            "branch_events": int(stats.get("branch_observe_branch_events", 0)),
            "rollout_count": int(stats.get("branch_observe_rollout_count", 0)),
            "extra_forwards": int(stats.get("branch_observe_extra_forwards", 0)),
            "avg_beam_size": float(stats.get("branch_observe_avg_beam_size", 0.0)),
            "force_baseline_output": bool(stats.get("branch_observe_force_baseline_output", True)),
            "event_logs": [],
            "any_particle_unit_pass": bool(tst_ok),
            "any_nonbaseline_unit_pass": False,
            "any_particle_parse_ok": bool(prs_ok),
            "any_nonbaseline_parse_ok": False,
            "score_selected_unit_pass": bool(tst_ok),
            "score_selected_parse_ok": bool(prs_ok),
            "score_selected_kind": "baseline",
            "score_selected_nonbaseline": False,
            "potential_recovery": False,
            "potential_parse_recovery": False,
            "score_selected_recovery": False,
            "score_selected_damage": False,
            "errors": stats.get("branch_observe_errors", []),
        }
        if bool(branch_observe_eval["enabled"]):
            evaluated_events: List[Dict[str, Any]] = []
            score_selected_particles: List[Dict[str, Any]] = []
            any_nonbaseline_pass = False
            any_nonbaseline_parse = False
            for event in stats.get("branch_observe_event_logs", []) or []:
                event_eval = dict(event)
                particle_evals: List[Dict[str, Any]] = []
                selected_idx = int(event.get("selected_particle", 0))
                for pidx, particle in enumerate(event.get("particle_logs", []) or []):
                    particle_eval = dict(particle)
                    cand_raw = str(particle.get("raw_output", ""))
                    if str(particle.get("kind", "")) == "baseline" and cand_raw == str(raw_out):
                        cand_eval = eval_out
                        cand_post = postprocess_meta
                        cand_fmt = bool(fmt_ok)
                        cand_parse = bool(prs_ok)
                        cand_pass = bool(tst_ok)
                    else:
                        cand_eval, cand_post = postprocess_completion_for_eval(
                            item=item,
                            completion=cand_raw,
                            dataset=args.dataset,
                        )
                        cand_fmt = format_success(cand_eval)
                        cand_parse = parse_success_for_eval(item, prompt, cand_eval, args.dataset)
                        cand_pass = unit_test_pass(item, cand_eval, dataset=args.dataset, timeout=args.unit_timeout)
                    cand_visible_score = (
                        _visible_test_score(
                            item=item,
                            prompt=prompt,
                            completion=cand_eval,
                            dataset=args.dataset,
                            timeout=args.unit_timeout,
                        )
                        if visible_selection_enabled
                        else {
                            "enabled": False,
                            "passed": 0,
                            "total": 0,
                            "failed": 0,
                            "pass_rate": 0.0,
                            "all_passed": False,
                        }
                    )
                    particle_eval.update(
                        {
                            "completion": cand_eval,
                            "postprocess": cand_post,
                            "format_ok_eval": bool(cand_fmt),
                            "parse_ok_eval": bool(cand_parse),
                            "unit_pass": bool(cand_pass),
                            "visible_test": cand_visible_score,
                            "selected_by_score": bool(pidx == selected_idx or particle.get("selected_by_score", False)),
                        }
                    )
                    if str(particle_eval.get("kind", "")) != "baseline":
                        any_nonbaseline_pass = bool(any_nonbaseline_pass or cand_pass)
                        any_nonbaseline_parse = bool(any_nonbaseline_parse or cand_parse)
                    if bool(particle_eval.get("selected_by_score", False)):
                        score_selected_particles.append(particle_eval)
                    particle_evals.append(particle_eval)
                event_eval["particle_logs"] = particle_evals
                evaluated_events.append(event_eval)
            selected_particle = score_selected_particles[0] if score_selected_particles else None
            if selected_particle is not None:
                branch_observe_eval["score_selected_unit_pass"] = bool(selected_particle.get("unit_pass", False))
                branch_observe_eval["score_selected_parse_ok"] = bool(selected_particle.get("parse_ok_eval", False))
                branch_observe_eval["score_selected_kind"] = str(selected_particle.get("kind", ""))
                branch_observe_eval["score_selected_nonbaseline"] = bool(
                    str(selected_particle.get("kind", "")) != "baseline"
                )
            branch_observe_eval["event_logs"] = evaluated_events
            branch_observe_eval["any_nonbaseline_unit_pass"] = bool(any_nonbaseline_pass)
            branch_observe_eval["any_nonbaseline_parse_ok"] = bool(any_nonbaseline_parse)
            branch_observe_eval["any_particle_unit_pass"] = bool(tst_ok or any_nonbaseline_pass)
            branch_observe_eval["any_particle_parse_ok"] = bool(prs_ok or any_nonbaseline_parse)
            branch_observe_eval["potential_recovery"] = bool((not tst_ok) and any_nonbaseline_pass)
            branch_observe_eval["potential_parse_recovery"] = bool((not prs_ok) and any_nonbaseline_parse)
            branch_observe_eval["score_selected_recovery"] = bool(
                (not tst_ok) and bool(branch_observe_eval["score_selected_unit_pass"])
            )
            branch_observe_eval["score_selected_damage"] = bool(
                tst_ok
                and bool(branch_observe_eval["score_selected_nonbaseline"])
                and not bool(branch_observe_eval["score_selected_unit_pass"])
            )

        baseline_before_select = {
            "raw_completion": raw_out,
            "completion": eval_out,
            "postprocess": postprocess_meta,
            "raw_format_ok": bool(raw_fmt_ok),
            "raw_parse_ok": bool(raw_prs_ok),
            "format_ok": bool(fmt_ok),
            "parse_ok": bool(prs_ok),
            "unit_pass": bool(tst_ok),
            "obvious_truncation": bool(_obvious_truncation_for_eval(eval_out)),
            "target_function_present": bool(
                _target_function_present_for_eval(
                    item=item,
                    prompt=prompt,
                    completion=eval_out,
                    dataset=args.dataset,
                )
            ),
            "level0_score": float(_level0_selection_score(fmt_ok, prs_ok, eval_out)),
            "level0_pass": bool(fmt_ok and prs_ok and not _obvious_truncation_for_eval(eval_out)),
            "visible_test": baseline_visible_score,
        }
        branch_select_meta: Dict[str, Any] = {
            "enabled": bool(getattr(cfg, "branch_select_enabled", False)),
            "verifier": str(getattr(cfg, "branch_select_verifier", "level0") or "level0"),
            "selected": False,
            "reason": "disabled",
            "baseline": {
                key: value
                for key, value in baseline_before_select.items()
                if key not in {"raw_completion", "completion"}
            },
            "selected_kind": "baseline",
            "selected_event_index": -1,
            "selected_particle_index": -1,
            "selected_token_id": None,
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "min_score_gain": float(getattr(cfg, "branch_select_min_score_gain", 1.0)),
            "require_baseline_failure": bool(getattr(cfg, "branch_select_require_baseline_failure", True)),
            "visible_min_pass_gain": int(getattr(cfg, "branch_select_visible_min_pass_gain", 1)),
            "visible_require_level0": bool(getattr(cfg, "branch_select_visible_require_level0", True)),
        }
        if bool(branch_select_meta["enabled"]):
            verifier = str(branch_select_meta["verifier"]).lower()
            require_baseline_failure = bool(branch_select_meta["require_baseline_failure"])
            baseline_level0_pass = bool(baseline_before_select["level0_pass"])
            baseline_visible = baseline_before_select.get("visible_test", {})
            baseline_visible_enabled = bool(isinstance(baseline_visible, dict) and baseline_visible.get("enabled", False))
            baseline_visible_passed = int(baseline_visible.get("passed", 0)) if baseline_visible_enabled else 0
            baseline_visible_total = int(baseline_visible.get("total", 0)) if baseline_visible_enabled else 0
            baseline_level1_pass = bool(
                baseline_level0_pass
                and baseline_visible_enabled
                and baseline_visible_total > 0
                and bool(baseline_visible.get("all_passed", False))
            )
            if require_baseline_failure and baseline_level0_pass and verifier == "level0":
                branch_select_meta["reason"] = "keep_baseline_level0_pass"
            elif verifier == "level1" and baseline_level0_pass and not baseline_visible_enabled:
                branch_select_meta["reason"] = "keep_baseline_no_visible_tests"
            elif require_baseline_failure and verifier == "level1" and baseline_level1_pass:
                branch_select_meta["reason"] = "keep_baseline_level1_pass"
            elif require_baseline_failure and bool(tst_ok) and verifier == "oracle":
                branch_select_meta["reason"] = "keep_baseline_oracle_pass"
            else:
                candidates: List[Dict[str, Any]] = []
                for event_idx, event in enumerate(branch_observe_eval.get("event_logs", []) or []):
                    for particle_idx, particle in enumerate(event.get("particle_logs", []) or []):
                        if str(particle.get("kind", "")) == "baseline":
                            continue
                        cand_completion = str(particle.get("completion", ""))
                        cand_fmt = bool(particle.get("format_ok_eval", False))
                        cand_parse = bool(particle.get("parse_ok_eval", False))
                        cand_pass = bool(particle.get("unit_pass", False))
                        cand_level0 = float(_level0_selection_score(cand_fmt, cand_parse, cand_completion))
                        cand_truncated = bool(_obvious_truncation_for_eval(cand_completion))
                        cand_target_function = bool(
                            _target_function_present_for_eval(
                                item=item,
                                prompt=prompt,
                                completion=cand_completion,
                                dataset=args.dataset,
                            )
                        )
                        cand_visible = particle.get("visible_test", {})
                        cand_visible_enabled = bool(isinstance(cand_visible, dict) and cand_visible.get("enabled", False))
                        cand_visible_passed = int(cand_visible.get("passed", 0)) if cand_visible_enabled else 0
                        cand_visible_total = int(cand_visible.get("total", 0)) if cand_visible_enabled else 0
                        cand_visible_gain = int(cand_visible_passed - baseline_visible_passed)
                        model_score = float(particle.get("score", 0.0))
                        if verifier == "oracle":
                            eligible = bool(cand_pass and not bool(tst_ok))
                            select_score = float(100.0 + cand_level0 + 0.01 * model_score)
                        elif verifier == "level1":
                            decision = _level1_branch_candidate_decision(
                                baseline_level0_pass=baseline_level0_pass,
                                baseline_level0_score=float(baseline_before_select["level0_score"]),
                                baseline_visible=baseline_visible if isinstance(baseline_visible, dict) else {},
                                cand_format_ok=cand_fmt,
                                cand_parse_ok=cand_parse,
                                cand_obvious_truncation=cand_truncated,
                                cand_level0_score=cand_level0,
                                cand_visible=cand_visible if isinstance(cand_visible, dict) else {},
                                model_score=model_score,
                                min_score_gain=float(branch_select_meta["min_score_gain"]),
                                visible_min_pass_gain=int(branch_select_meta["visible_min_pass_gain"]),
                                visible_require_level0=bool(branch_select_meta["visible_require_level0"]),
                            )
                            eligible = bool(decision["eligible"])
                            select_score = float(decision["select_score"])
                        else:
                            eligible = bool(
                                cand_fmt
                                and cand_parse
                                and not cand_truncated
                                and cand_level0
                                >= float(baseline_before_select["level0_score"])
                                + float(branch_select_meta["min_score_gain"])
                            )
                            select_score = float(cand_level0 + 0.01 * model_score)
                        candidates.append(
                            {
                                "event_index": int(event_idx),
                                "particle_index": int(particle_idx),
                                "particle": particle,
                                "eligible": bool(eligible),
                                "select_score": float(select_score),
                                "level0_score": float(cand_level0),
                                "format_ok": bool(cand_fmt),
                                "parse_ok": bool(cand_parse),
                                "unit_pass": bool(cand_pass),
                                "obvious_truncation": bool(cand_truncated),
                                "target_function_present": bool(cand_target_function),
                                "visible_passed": int(cand_visible_passed),
                                "visible_total": int(cand_visible_total),
                                "visible_gain": int(cand_visible_gain),
                                "level0_gain_ok": bool(decision["level0_gain_ok"]) if verifier == "level1" else False,
                                "visible_gain_ok": bool(decision["visible_gain_ok"]) if verifier == "level1" else False,
                                "repair_mode": str(decision["repair_mode"]) if verifier == "level1" else "",
                            }
                        )
                branch_select_meta["candidate_count"] = int(len(candidates))
                eligible_candidates = [cand for cand in candidates if bool(cand["eligible"])]
                branch_select_meta["eligible_candidate_count"] = int(len(eligible_candidates))
                if eligible_candidates:
                    best = max(
                        eligible_candidates,
                        key=lambda cand: (
                            float(cand["select_score"]),
                            float(cand["level0_score"]),
                            bool(cand["unit_pass"]),
                        ),
                    )
                    particle = dict(best["particle"])
                    raw_out = str(particle.get("raw_output", ""))
                    eval_out = str(particle.get("completion", ""))
                    postprocess_meta = particle.get("postprocess", postprocess_meta)
                    raw_fmt_ok = format_success(raw_out)
                    raw_prs_ok = parse_success_for_eval(item, prompt, raw_out, args.dataset)
                    fmt_ok = format_success(eval_out)
                    prs_ok = parse_success_for_eval(item, prompt, eval_out, args.dataset)
                    tst_ok = unit_test_pass(item, eval_out, dataset=args.dataset, timeout=args.unit_timeout)
                    selected_visible_score = (
                        _visible_test_score(
                            item=item,
                            prompt=prompt,
                            completion=eval_out,
                            dataset=args.dataset,
                            timeout=args.unit_timeout,
                        )
                        if verifier == "level1"
                        else {
                            "enabled": False,
                            "passed": 0,
                            "total": 0,
                            "failed": 0,
                            "pass_rate": 0.0,
                            "all_passed": False,
                        }
                    )
                    branch_select_meta.update(
                        {
                            "selected": True,
                            "reason": (
                                f"selected_{verifier}_{str(best.get('repair_mode', 'repair'))}_repair"
                                if verifier == "level1"
                                else f"selected_{verifier}_strict_gain"
                            ),
                            "selected_kind": str(particle.get("kind", "")),
                            "selected_event_index": int(best["event_index"]),
                            "selected_particle_index": int(best["particle_index"]),
                            "selected_token_id": particle.get("token_id"),
                            "selected_level0_score": float(best["level0_score"]),
                            "selected_select_score": float(best["select_score"]),
                            "selected_format_ok": bool(fmt_ok),
                            "selected_parse_ok": bool(prs_ok),
                            "selected_unit_pass": bool(tst_ok),
                            "selected_obvious_truncation": bool(best.get("obvious_truncation", False)),
                            "selected_target_function_present": bool(best.get("target_function_present", False)),
                            "selected_visible_test": selected_visible_score,
                            "parse_recovered_baseline": bool((not baseline_before_select["parse_ok"]) and prs_ok),
                            "parse_damaged_baseline": bool(baseline_before_select["parse_ok"] and not prs_ok),
                            "unit_recovered_baseline": bool((not baseline_before_select["unit_pass"]) and tst_ok),
                            "unit_damaged_baseline": bool(baseline_before_select["unit_pass"] and not tst_ok),
                            "first_raw_text_diff_from_baseline": _first_text_diff(
                                str(baseline_before_select["raw_completion"]),
                                raw_out,
                            ),
                            "first_eval_text_diff_from_baseline": _first_text_diff(
                                str(baseline_before_select["completion"]),
                                eval_out,
                            ),
                        }
                    )
                else:
                    branch_select_meta["reason"] = "keep_baseline_no_eligible_candidate"
        branch_observe_eval["branch_select"] = branch_select_meta
        baseline_compare_meta: Optional[Dict[str, Any]] = None
        if bool(args.eval_compare_baseline) and args.decoder == "risk_pf":
            baseline_t0 = time.perf_counter()
            try:
                _set_global_seed(sample_seed)
                baseline_raw_out = run_reference_baseline(prompt)
                baseline_elapsed = time.perf_counter() - baseline_t0
                baseline_compare_latency.append(float(baseline_elapsed))
                baseline_eval_out, baseline_postprocess_meta = postprocess_completion_for_eval(
                    item=item,
                    completion=baseline_raw_out,
                    dataset=args.dataset,
                )
                baseline_fmt_ok = format_success(baseline_eval_out)
                baseline_prs_ok = parse_success_for_eval(item, prompt, baseline_eval_out, args.dataset)
                baseline_tst_ok = unit_test_pass(
                    item,
                    baseline_eval_out,
                    dataset=args.dataset,
                    timeout=args.unit_timeout,
                )
                raw_exact_output_match = bool(str(baseline_raw_out) == str(raw_out))
                exact_output_match = bool(str(baseline_eval_out) == str(eval_out))
                recovered = bool((not baseline_tst_ok) and tst_ok)
                damaged = bool(baseline_tst_ok and not tst_ok)
                parse_recovered = bool((not baseline_prs_ok) and prs_ok)
                parse_damaged = bool(baseline_prs_ok and not prs_ok)
                recovered_baseline_failures += int(recovered)
                damaged_baseline_successes += int(damaged)
                parse_recovered_baseline_failures += int(parse_recovered)
                parse_damaged_baseline_successes += int(parse_damaged)
                baseline_compare_meta = {
                    "enabled": True,
                    "baseline_completion": baseline_eval_out,
                    "baseline_raw_completion": baseline_raw_out,
                    "baseline_postprocess": baseline_postprocess_meta,
                    "raw_exact_output_match": bool(raw_exact_output_match),
                    "exact_output_match": bool(exact_output_match),
                    "first_raw_text_diff": _first_text_diff(baseline_raw_out, raw_out),
                    "first_eval_text_diff": _first_text_diff(baseline_eval_out, eval_out),
                    "first_raw_token_diff": _first_token_diff(dream_tokenizer, baseline_raw_out, raw_out),
                    "first_eval_token_diff": _first_token_diff(dream_tokenizer, baseline_eval_out, eval_out),
                    "baseline_format_ok": bool(baseline_fmt_ok),
                    "baseline_parse_ok": bool(baseline_prs_ok),
                    "baseline_unit_pass": bool(baseline_tst_ok),
                    "method_format_ok": bool(fmt_ok),
                    "method_parse_ok": bool(prs_ok),
                    "method_unit_pass": bool(tst_ok),
                    "recovered_baseline_failure": bool(recovered),
                    "damaged_baseline_success": bool(damaged),
                    "parse_recovered_baseline_failure": bool(parse_recovered),
                    "parse_damaged_baseline_success": bool(parse_damaged),
                    "baseline_latency_sec": float(baseline_elapsed),
                }
            except Exception as base_exc:
                baseline_compare_meta = {
                    "enabled": True,
                    "error": f"{type(base_exc).__name__}: {base_exc}",
                }
            if baseline_compare_meta is not None:
                baseline_compare_records.append(
                    {
                        "sample_index": int(sample_index),
                        "task_id": item.get("task_id", str(sample_index)),
                        **baseline_compare_meta,
                    }
                )

        format_ok += int(fmt_ok)
        parse_ok += int(prs_ok)
        pass_ok += int(tst_ok)
        raw_format_ok += int(raw_fmt_ok)
        raw_parse_ok += int(raw_prs_ok)

        logger.info(
            "sample %s format_ok=%s parse_ok=%s unit_pass=%s len=%s raw_parse_ok=%s latency=%.3fs",
            total,
            fmt_ok,
            prs_ok,
            tst_ok,
            len(eval_out),
            raw_prs_ok,
            elapsed,
        )

        sample_task_id = item.get("task_id", str(sample_index))
        sample_risk_trace_lines = _collect_risk_trace_lines(stats) if args.decoder == "risk_pf" else []
        sample_analysis = {
            "sample_index": int(sample_index),
            "task_id": sample_task_id,
            "entry_point": item.get("entry_point", ""),
            "format_ok": bool(fmt_ok),
            "parse_ok": bool(prs_ok),
            "unit_pass": bool(tst_ok),
            "obvious_truncation": bool(_obvious_truncation_for_eval(eval_out)),
            "target_function_present": bool(
                _target_function_present_for_eval(
                    item=item,
                    prompt=prompt,
                    completion=eval_out,
                    dataset=args.dataset,
                )
            ),
            "raw_format_ok": bool(raw_fmt_ok),
            "raw_parse_ok": bool(raw_prs_ok),
            "completion_len": int(len(eval_out)),
            "raw_completion_len": int(len(raw_out)),
            "completion": eval_out,
            "raw_completion": raw_out,
            "postprocess": postprocess_meta,
            "latency_sec": float(elapsed),
            "dream_noop_fast_path": bool(stats.get("dream_noop_fast_path", False)),
            "shadow_mode_enabled": bool(stats.get("shadow_mode_enabled", False)),
            "shadow_num_steps": int(stats.get("shadow_num_steps", 0)),
            "shadow_num_risk_events": int(stats.get("shadow_num_risk_events", 0)),
            "shadow_max_risk": float(stats.get("shadow_max_risk", 0.0)),
            "shadow_mean_risk": float(stats.get("shadow_mean_risk", 0.0)),
            "shadow_high_risk_early_commit_count": int(stats.get("shadow_high_risk_early_commit_count", 0)),
            "shadow_top_risk_events": stats.get("shadow_top_risk_events", []),
            "shadow_step_commit_stats": stats.get("shadow_step_commit_stats", []),
            "shadow_errors": stats.get("shadow_errors", []),
            "branch_observe": branch_observe_eval,
            "branch_select": branch_select_meta,
            "pf_trigger_count": int(stats.get("pf_trigger_count", 0)),
            "risk_trigger_count": int(stats.get("risk_trigger_count", 0)),
            "extra_forwards": int(stats.get("extra_forwards", 0)),
            "influence_compute_count": int(stats.get("influence_compute_count", 0)),
            "parser_feedback_counts": stats.get("parser_feedback_counts", {}),
            "parser_feedback_top_timesteps": stats.get("parser_feedback_top_timesteps", {}),
            "parser_hotspot_counts": stats.get("parser_hotspot_counts", {}),
            "parser_hotspot_top_timesteps": stats.get("parser_hotspot_top_timesteps", {}),
            "rdd_rollback_count": int(stats.get("rdd_rollback_count", 0)),
            "rdd_remasked_tokens_total": int(stats.get("rdd_remasked_tokens_total", 0)),
            "rdd_rollback_cleared_pf_count": int(stats.get("rdd_rollback_cleared_pf_count", 0)),
            "repair_policy": str(stats.get("repair_policy", _REPAIR_POLICY_NONE)),
            "repair_route_counts": stats.get("repair_route_counts", {}),
            "pf_rb_route_counts": stats.get("pf_rb_route_counts", {}),
            "eb_sampler_enabled": bool(stats.get("eb_sampler_enabled", False)),
            "eb_step_count": int(stats.get("eb_step_count", 0)),
            "eb_candidate_count": int(stats.get("eb_candidate_count", 0)),
            "eb_allowed_count": int(stats.get("eb_allowed_count", 0)),
            "eb_blocked_count": int(stats.get("eb_blocked_count", 0)),
            "eb_min_fallback_count": int(stats.get("eb_min_fallback_count", 0)),
            "eb_allowed_per_step": float(stats.get("eb_allowed_per_step", 0.0)),
            "eb_blocked_per_step": float(stats.get("eb_blocked_per_step", 0.0)),
            "local_beam_enabled": bool(stats.get("local_beam_enabled", False)),
            "local_beam_branch_events": int(stats.get("local_beam_branch_events", 0)),
            "local_beam_accepted_alternatives": int(stats.get("local_beam_accepted_alternatives", 0)),
            "local_beam_delay_count": int(stats.get("local_beam_delay_count", 0)),
            "local_beam_avg_beam_size": float(stats.get("local_beam_avg_beam_size", 0.0)),
            "baseline_compare": baseline_compare_meta or {"enabled": False},
        }
        sample_analysis_records.append(sample_analysis)

        if trace_payload is not None:
            sample_trace_records.append(trace_payload)

        if args.output_rawdata:
            sample_raw_records.append(
                {
                    "sample_index": int(sample_index),
                    "task_id": sample_task_id,
                    "item": item,
                    "prompt": prompt,
                    "completion": eval_out,
                    "raw_completion": raw_out,
                    "postprocess": postprocess_meta,
                    "format_ok": bool(fmt_ok),
                    "parse_ok": bool(prs_ok),
                    "unit_pass": bool(tst_ok),
                    "raw_format_ok": bool(raw_fmt_ok),
                    "raw_parse_ok": bool(raw_prs_ok),
                    "completion_len": int(len(eval_out)),
                    "raw_completion_len": int(len(raw_out)),
                    "latency_sec": float(elapsed),
                    "baseline_compare": baseline_compare_meta or {"enabled": False},
                    "risk_trace_lines": sample_risk_trace_lines,
                    "decoder_stats": stats,
                }
            )

    format_success_rate = format_ok / total if total else 0.0
    parse_success_rate = parse_ok / total if total else 0.0
    unit_test_pass_rate = pass_ok / total if total else 0.0
    raw_format_success_rate = raw_format_ok / total if total else 0.0
    raw_parse_success_rate = raw_parse_ok / total if total else 0.0
    format_to_parse_rate = parse_ok / format_ok if format_ok else 0.0
    parse_to_pass_rate = pass_ok / parse_ok if parse_ok else 0.0

    syntax_error_rate = 1.0 - parse_success_rate
    avg_pf_triggers = (sum(pf_triggers_list) / len(pf_triggers_list)) if pf_triggers_list else 0.0
    avg_rollback_count = (sum(rollback_counts_list) / len(rollback_counts_list)) if rollback_counts_list else 0.0
    total_rollback_cleared_pf = int(sum(rollback_cleared_pf_list)) if rollback_cleared_pf_list else 0
    avg_overhead = (sum(overhead_list) / len(overhead_list)) if overhead_list else 0.0
    avg_branch_width = (sum(branch_width_list) / len(branch_width_list)) if branch_width_list else 1.0
    latency_ratio_vs_baseline = (sum(latency_ratio_list) / len(latency_ratio_list)) if latency_ratio_list else 1.0
    avg_latency_sec = (sum(sample_latency) / len(sample_latency)) if sample_latency else 0.0
    avg_eb_allowed_per_step = (sum(eb_allowed_per_step_list) / len(eb_allowed_per_step_list)) if eb_allowed_per_step_list else 0.0
    avg_eb_blocked_per_step = (sum(eb_blocked_per_step_list) / len(eb_blocked_per_step_list)) if eb_blocked_per_step_list else 0.0
    avg_eb_min_fallback = (sum(eb_min_fallback_list) / len(eb_min_fallback_list)) if eb_min_fallback_list else 0.0
    avg_local_beam_branch_events = (
        sum(local_beam_branch_events_list) / len(local_beam_branch_events_list)
        if local_beam_branch_events_list
        else 0.0
    )
    avg_local_beam_beam_size = (
        sum(local_beam_avg_beam_size_list) / len(local_beam_avg_beam_size_list)
        if local_beam_avg_beam_size_list
        else 0.0
    )
    total_local_beam_accepted = int(sum(local_beam_accepted_list)) if local_beam_accepted_list else 0
    total_local_beam_delay = int(sum(local_beam_delay_list)) if local_beam_delay_list else 0
    total_dream_noop_fast_path = int(sum(1 for value in dream_noop_fast_path_list if value))
    avg_shadow_max_risk = (sum(shadow_max_risk_list) / len(shadow_max_risk_list)) if shadow_max_risk_list else 0.0
    avg_shadow_mean_risk = (sum(shadow_mean_risk_list) / len(shadow_mean_risk_list)) if shadow_mean_risk_list else 0.0
    avg_shadow_num_risk_events = (
        sum(shadow_num_risk_events_list) / len(shadow_num_risk_events_list) if shadow_num_risk_events_list else 0.0
    )
    total_shadow_early_commit_events = int(sum(shadow_early_commit_list)) if shadow_early_commit_list else 0
    shadow_pass_records = [row for row in sample_analysis_records if bool(row.get("unit_pass", False))]
    shadow_fail_records = [row for row in sample_analysis_records if not bool(row.get("unit_pass", False))]
    shadow_pass_max_risk_mean = (
        sum(float(row.get("shadow_max_risk", 0.0)) for row in shadow_pass_records) / len(shadow_pass_records)
        if shadow_pass_records
        else 0.0
    )
    shadow_fail_max_risk_mean = (
        sum(float(row.get("shadow_max_risk", 0.0)) for row in shadow_fail_records) / len(shadow_fail_records)
        if shadow_fail_records
        else 0.0
    )
    shadow_pass_risk_event_mean = (
        sum(float(row.get("shadow_num_risk_events", 0.0)) for row in shadow_pass_records) / len(shadow_pass_records)
        if shadow_pass_records
        else 0.0
    )
    shadow_fail_risk_event_mean = (
        sum(float(row.get("shadow_num_risk_events", 0.0)) for row in shadow_fail_records) / len(shadow_fail_records)
        if shadow_fail_records
        else 0.0
    )
    branch_observe_records = [
        row.get("branch_observe", {})
        for row in sample_analysis_records
        if isinstance(row.get("branch_observe", {}), dict) and bool(row.get("branch_observe", {}).get("enabled", False))
    ]
    branch_observe_enabled_count = int(len(branch_observe_records))
    branch_observe_total_events = int(sum(int(row.get("branch_events", 0)) for row in branch_observe_records))
    branch_observe_total_rollouts = int(sum(int(row.get("rollout_count", 0)) for row in branch_observe_records))
    branch_observe_total_extra_forwards = int(sum(int(row.get("extra_forwards", 0)) for row in branch_observe_records))
    branch_observe_avg_beam_size = (
        sum(float(row.get("avg_beam_size", 0.0)) for row in branch_observe_records) / len(branch_observe_records)
        if branch_observe_records
        else 0.0
    )
    branch_observe_oracle_union_pass_count = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("any_particle_unit_pass", row.get("unit_pass", False))))
    )
    branch_observe_oracle_nonbaseline_pass_count = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("any_nonbaseline_unit_pass", False)))
    )
    branch_observe_oracle_union_parse_count = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("any_particle_parse_ok", row.get("parse_ok", False))))
    )
    branch_observe_score_selected_pass_count = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("score_selected_unit_pass", row.get("unit_pass", False))))
    )
    branch_observe_score_selected_parse_count = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("score_selected_parse_ok", row.get("parse_ok", False))))
    )
    branch_observe_score_selected_nonbaseline_count = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("score_selected_nonbaseline", False)))
    )
    branch_observe_potential_recoveries = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("potential_recovery", False)))
    )
    branch_observe_potential_parse_recoveries = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("potential_parse_recovery", False)))
    )
    branch_observe_score_selected_recoveries = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("score_selected_recovery", False)))
    )
    branch_observe_score_selected_damages = int(
        sum(1 for row in sample_analysis_records if bool(row.get("branch_observe", {}).get("score_selected_damage", False)))
    )
    branch_select_records = [
        row.get("branch_select", {})
        for row in sample_analysis_records
        if isinstance(row.get("branch_select", {}), dict) and bool(row.get("branch_select", {}).get("enabled", False))
    ]
    branch_select_enabled_count = int(len(branch_select_records))
    branch_select_selected_count = int(sum(1 for row in branch_select_records if bool(row.get("selected", False))))
    branch_select_parse_recoveries = int(
        sum(1 for row in branch_select_records if bool(row.get("parse_recovered_baseline", False)))
    )
    branch_select_parse_damages = int(
        sum(1 for row in branch_select_records if bool(row.get("parse_damaged_baseline", False)))
    )
    branch_select_unit_recoveries = int(
        sum(1 for row in branch_select_records if bool(row.get("unit_recovered_baseline", False)))
    )
    branch_select_unit_damages = int(
        sum(1 for row in branch_select_records if bool(row.get("unit_damaged_baseline", False)))
    )
    branch_select_eligible_count = int(sum(int(row.get("eligible_candidate_count", 0)) for row in branch_select_records))
    branch_select_visible_selected_count = int(
        sum(
            1
            for row in branch_select_records
            if bool(row.get("selected", False)) and str(row.get("verifier", "")).lower() == "level1"
        )
    )
    branch_select_visible_pass_gain_total = int(
        sum(
            max(
                int((row.get("selected_visible_test") or {}).get("passed", 0))
                - int(((row.get("baseline") or {}).get("visible_test") or {}).get("passed", 0)),
                0,
            )
            for row in branch_select_records
            if bool(row.get("selected", False)) and str(row.get("verifier", "")).lower() == "level1"
        )
    )
    branch_select_breakdown = _summarize_branch_select_records(branch_select_records)
    baseline_compare_count = int(len(baseline_compare_records))
    exact_output_match_count = int(sum(1 for rec in baseline_compare_records if rec.get("exact_output_match") is True))
    raw_exact_output_match_count = int(
        sum(1 for rec in baseline_compare_records if rec.get("raw_exact_output_match") is True)
    )
    net_change = int(recovered_baseline_failures - damaged_baseline_successes)
    parse_net_change = int(parse_recovered_baseline_failures - parse_damaged_baseline_successes)
    avg_baseline_compare_latency_sec = (
        sum(baseline_compare_latency) / len(baseline_compare_latency)
        if baseline_compare_latency
        else 0.0
    )

    logger.info(
        (
            "format_success_rate=%.2f parse_success_rate=%.2f unit_test_pass_rate=%.2f "
            "format_to_parse=%.2f parse_to_pass=%.2f avg_pf_triggers=%.1f "
            "avg_extra_forwards=%.1f avg_branch_width=%.2f latency_ratio_vs_baseline=%.2f"
        ),
        format_success_rate,
        parse_success_rate,
        unit_test_pass_rate,
        format_to_parse_rate,
        parse_to_pass_rate,
        avg_pf_triggers,
        avg_overhead,
        avg_branch_width,
        latency_ratio_vs_baseline,
    )

    summary_payload = {
        "format_success_rate": format_success_rate,
        "parse_success_rate": parse_success_rate,
        "unit_test_pass_rate": unit_test_pass_rate,
        "postprocess_enabled": True,
        "postprocess_mode": "humaneval_function_body" if args.dataset == "humaneval" else "strip_markdown_fence",
        "raw_format_success_rate": raw_format_success_rate,
        "raw_parse_success_rate": raw_parse_success_rate,
        "format_to_parse_rate": format_to_parse_rate,
        "parse_to_pass_rate": parse_to_pass_rate,
        "syntax_error_rate": syntax_error_rate,
        "n_samples": total,
        "avg_pf_triggers": avg_pf_triggers,
        "avg_rollback_count": avg_rollback_count,
        "total_rollback_cleared_pf": total_rollback_cleared_pf,
        "avg_extra_forwards": avg_overhead,
        "avg_branch_width": avg_branch_width,
        "latency_ratio_vs_baseline": latency_ratio_vs_baseline,
        "avg_latency_sec": avg_latency_sec,
        "risk_low_threshold": cfg.risk_low_threshold,
        "risk_high_threshold": cfg.risk_high_threshold,
        "risk_trigger_mode": "dynamic_quantile",
        "risk_low_quantile": cfg.risk_low_quantile,
        "risk_high_quantile": cfg.risk_high_quantile,
        "risk_threshold_min_gap": cfg.risk_threshold_min_gap,
        "backend": args.backend,
        "decoder": args.decoder,
        "correctness_signal_mode": args.correctness_signal_mode,
        "pf_budget_mode": args.pf_budget_mode,
        "pf_extra_forward_budget": int(args.pf_extra_forward_budget),
        "pf_max_triggers_per_sample": int(args.pf_max_triggers_per_sample),
        "pf_parse_fail_penalty": float(args.pf_parse_fail_penalty),
        "pf_do_no_harm_enabled": bool(args.pf_do_no_harm_enabled),
        "pf_do_no_harm_margin": float(args.pf_do_no_harm_margin),
        "pf_do_no_harm_min_quality_gain": float(args.pf_do_no_harm_min_quality_gain),
        "eb_sampler_enabled": bool(args.eb_sampler_enabled),
        "eb_entropy_quantile": float(args.eb_entropy_quantile),
        "eb_min_commit_per_step": int(args.eb_min_commit_per_step),
        "eb_structure_entropy_scale": float(args.eb_structure_entropy_scale),
        "eb_signature_entropy_scale": float(args.eb_signature_entropy_scale),
        "eb_syntax_near_entropy_scale": float(args.eb_syntax_near_entropy_scale),
        "eb_near_radius": int(args.eb_near_radius),
        "avg_eb_allowed_per_step": float(avg_eb_allowed_per_step),
        "avg_eb_blocked_per_step": float(avg_eb_blocked_per_step),
        "avg_eb_min_fallback_count": float(avg_eb_min_fallback),
        "local_beam_enabled": bool(cfg.local_beam_enabled),
        "local_beam_mode": str(cfg.local_beam_mode),
        "local_beam_size": int(cfg.local_beam_size),
        "local_beam_horizon": int(cfg.local_beam_horizon),
        "local_beam_max_events": int(cfg.local_beam_max_events),
        "local_beam_tau_entropy": float(cfg.local_beam_tau_entropy),
        "local_beam_tau_kl": float(cfg.local_beam_tau_kl),
        "local_beam_tau_risk": float(cfg.local_beam_tau_risk),
        "local_beam_struct_weight": float(cfg.local_beam_struct_weight),
        "local_beam_preserve_baseline": bool(cfg.local_beam_preserve_baseline),
        "local_beam_use_visible_tests": bool(cfg.local_beam_use_visible_tests),
        "avg_local_beam_branch_events": float(avg_local_beam_branch_events),
        "avg_local_beam_beam_size": float(avg_local_beam_beam_size),
        "total_local_beam_accepted_alternatives": int(total_local_beam_accepted),
        "total_local_beam_delay_count": int(total_local_beam_delay),
        "entropy_kl_logging_enabled": bool(getattr(cfg, "entropy_kl_logging_enabled", False)),
        "dream_noop_fast_path_count": int(total_dream_noop_fast_path),
        "shadow_mode_enabled": bool(getattr(cfg, "shadow_mode_enabled", False)),
        "shadow_enabled_count": int(sum(1 for value in shadow_enabled_list if value)),
        "shadow_entropy_top_k": int(getattr(cfg, "shadow_entropy_top_k", 16)),
        "shadow_kl_top_k": int(getattr(cfg, "shadow_kl_top_k", 8)),
        "shadow_token_top_k": int(getattr(cfg, "shadow_token_top_k", 5)),
        "avg_shadow_max_risk": float(avg_shadow_max_risk),
        "avg_shadow_mean_risk": float(avg_shadow_mean_risk),
        "avg_shadow_num_risk_events": float(avg_shadow_num_risk_events),
        "total_shadow_high_risk_early_commit_count": int(total_shadow_early_commit_events),
        "shadow_pass_max_risk_mean": float(shadow_pass_max_risk_mean),
        "shadow_fail_max_risk_mean": float(shadow_fail_max_risk_mean),
        "shadow_pass_risk_event_mean": float(shadow_pass_risk_event_mean),
        "shadow_fail_risk_event_mean": float(shadow_fail_risk_event_mean),
        "branch_observe_enabled": bool(getattr(cfg, "branch_observe_enabled", False)),
        "branch_observe_enabled_count": int(branch_observe_enabled_count),
        "branch_observe_beam_size": int(getattr(cfg, "branch_observe_beam_size", 3)),
        "branch_observe_top_k": int(getattr(cfg, "branch_observe_top_k", 3)),
        "branch_observe_max_events": int(getattr(cfg, "branch_observe_max_events", 1)),
        "branch_observe_horizon": int(getattr(cfg, "branch_observe_horizon", 2)),
        "branch_observe_trigger_mode": str(getattr(cfg, "branch_observe_trigger_mode", "auto") or "auto"),
        "branch_observe_event_policy": str(getattr(cfg, "branch_observe_event_policy", "top_risk") or "top_risk"),
        "branch_observe_total_events": int(branch_observe_total_events),
        "branch_observe_total_rollouts": int(branch_observe_total_rollouts),
        "branch_observe_total_extra_forwards": int(branch_observe_total_extra_forwards),
        "branch_observe_avg_beam_size": float(branch_observe_avg_beam_size),
        "branch_observe_oracle_union_pass_count": int(branch_observe_oracle_union_pass_count),
        "branch_observe_oracle_nonbaseline_pass_count": int(branch_observe_oracle_nonbaseline_pass_count),
        "branch_observe_oracle_union_parse_count": int(branch_observe_oracle_union_parse_count),
        "branch_observe_score_selected_pass_count": int(branch_observe_score_selected_pass_count),
        "branch_observe_score_selected_parse_count": int(branch_observe_score_selected_parse_count),
        "branch_observe_score_selected_nonbaseline_count": int(branch_observe_score_selected_nonbaseline_count),
        "branch_observe_potential_recoveries": int(branch_observe_potential_recoveries),
        "branch_observe_potential_parse_recoveries": int(branch_observe_potential_parse_recoveries),
        "branch_observe_score_selected_recoveries": int(branch_observe_score_selected_recoveries),
        "branch_observe_score_selected_damages": int(branch_observe_score_selected_damages),
        "branch_select_enabled": bool(getattr(cfg, "branch_select_enabled", False)),
        "branch_select_verifier": str(getattr(cfg, "branch_select_verifier", "level0") or "level0"),
        "branch_select_min_score_gain": float(getattr(cfg, "branch_select_min_score_gain", 1.0)),
        "branch_select_visible_min_pass_gain": int(getattr(cfg, "branch_select_visible_min_pass_gain", 1)),
        "branch_select_visible_require_level0": bool(
            getattr(cfg, "branch_select_visible_require_level0", True)
        ),
        "branch_select_require_baseline_failure": bool(
            getattr(cfg, "branch_select_require_baseline_failure", True)
        ),
        "branch_select_enabled_count": int(branch_select_enabled_count),
        "branch_select_selected_count": int(branch_select_selected_count),
        "branch_select_eligible_candidate_count": int(branch_select_eligible_count),
        "branch_select_parse_recoveries": int(branch_select_parse_recoveries),
        "branch_select_parse_damages": int(branch_select_parse_damages),
        "branch_select_unit_recoveries": int(branch_select_unit_recoveries),
        "branch_select_unit_damages": int(branch_select_unit_damages),
        "branch_select_visible_selected_count": int(branch_select_visible_selected_count),
        "branch_select_visible_pass_gain_total": int(branch_select_visible_pass_gain_total),
        "branch_select_reason_counts": branch_select_breakdown["reason_counts"],
        "branch_select_selected_kind_counts": branch_select_breakdown["selected_kind_counts"],
        "branch_select_selected_candidate_count": int(branch_select_breakdown["selected_candidate_count"]),
        "branch_select_selected_delay_count": int(branch_select_breakdown["selected_delay_count"]),
        "branch_select_selected_level0_parse_repair_count": int(
            branch_select_breakdown["selected_level0_parse_repair_count"]
        ),
        "branch_select_selected_level0_format_repair_count": int(
            branch_select_breakdown["selected_level0_format_repair_count"]
        ),
        "branch_select_selected_level0_truncation_repair_count": int(
            branch_select_breakdown["selected_level0_truncation_repair_count"]
        ),
        "branch_select_selected_level0_target_func_repair_count": int(
            branch_select_breakdown["selected_level0_target_func_repair_count"]
        ),
        "branch_select_net_unit_change": int(branch_select_unit_recoveries - branch_select_unit_damages),
        "branch_select_net_parse_change": int(branch_select_parse_recoveries - branch_select_parse_damages),
        "baseline_compare_enabled": bool(args.eval_compare_baseline),
        "baseline_compare_count": int(baseline_compare_count),
        "exact_output_match_count": int(exact_output_match_count),
        "raw_exact_output_match_count": int(raw_exact_output_match_count),
        "recovered_baseline_failures": int(recovered_baseline_failures),
        "damaged_baseline_successes": int(damaged_baseline_successes),
        "net_change": int(net_change),
        "parse_recovered_baseline_failures": int(parse_recovered_baseline_failures),
        "parse_damaged_baseline_successes": int(parse_damaged_baseline_successes),
        "parse_net_change": int(parse_net_change),
        "avg_baseline_compare_latency_sec": float(avg_baseline_compare_latency_sec),
        "rdd_rollback_enabled": bool(args.rdd_rollback_enabled),
        "repair_policy": str(_resolve_dream_repair_policy(cfg) if args.decoder == "risk_pf" else _REPAIR_POLICY_NONE),
        "rdd_rollback_window": int(args.rdd_rollback_window),
        "rdd_rollback_max_events": int(args.rdd_rollback_max_events),
        "rdd_rollback_min_severity": float(args.rdd_rollback_min_severity),
        "dream_baseline_strategy": args.dream_baseline_strategy if args.backend == "dream" else "",
        "dream_risk_pf_strategy": args.dream_risk_pf_strategy if args.backend == "dream" else "",
        "result_dir": result_paths["run_dir"],
        "json_dir": result_paths["json_dir"],
        "rawdata_dir": result_paths["rawdata_dir"],
        "risk_trace_log_path": str(risk_trace_log_path),
        "output_rawdata": bool(args.output_rawdata),
    }

    _write_json(Path(result_paths["json_dir"]) / "summary.json", summary_payload)
    _write_json(Path(result_paths["json_dir"]) / "sample_metrics.json", sample_analysis_records)
    _write_json(Path(result_paths["json_dir"]) / "baseline_compare.json", baseline_compare_records)
    _write_json(
        Path(result_paths["json_dir"]) / "run_metadata.json",
        {
            "timestamp": result_paths["timestamp"],
            "result_root": result_paths["result_root"],
            "run_dir": result_paths["run_dir"],
            "dataset": args.dataset,
            "data_path": data_path,
            "total_dataset_items_loaded": int(len(items_all)),
            "warmup_samples": int(len(warmup_items)),
            "evaluated_samples": int(len(items)),
            "risk_trace_log_path": summary_payload["risk_trace_log_path"],
            "cli_args": vars(args),
            "decoder_config": dict(cfg.__dict__),
        },
    )

    risk_trace_handler.flush()
    logging.getLogger().removeHandler(risk_trace_handler)
    risk_trace_handler.close()

    if args.output_rawdata:
        _write_json(Path(result_paths["rawdata_dir"]) / "input_items.json", items)
        _write_json(Path(result_paths["rawdata_dir"]) / "warmup_items.json", warmup_items)
        _write_json(Path(result_paths["rawdata_dir"]) / "sample_details.json", sample_raw_records)
        _write_json(Path(result_paths["rawdata_dir"]) / "sample_traces.json", sample_trace_records)
        if warmup_stats_output is not None:
            _write_json(Path(result_paths["rawdata_dir"]) / "warmup_statistics.json", warmup_stats_output)
    else:
        _write_json(
            Path(result_paths["rawdata_dir"]) / "rawdata_disabled.json",
            {
                "output_rawdata": False,
                "message": "Raw data output was disabled via --no_output_rawdata.",
            },
        )

    print(json.dumps(summary_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
