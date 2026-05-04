"""Commit-timing-aware local beam utilities.

This module is deliberately train-free: it only consumes logits from the
ordinary denoising trajectory plus optional parser/tokenizer callbacks.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import DecoderConfig
from .sampler_adapter import SamplerAdapter


_STRUCTURE_KEYWORDS = {
    "def",
    "class",
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
    "lambda",
    "and",
    "or",
    "not",
    "in",
    "is",
    "range",
    "len",
    "enumerate",
    "zip",
}
_STRUCTURE_CHARS = set("()[]{}:'\"=<>.,")


def _ensure_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - torch-less envs are unsupported here
        raise ImportError("local beam decoding requires torch.") from exc


@dataclass
class LocalBeamRiskCandidate:
    pos: int
    entropy_norm: float
    kl_raw: float
    kl_norm: float
    structure_score: float
    structure_weight: float
    risk: float
    top_tokens: List[int] = field(default_factory=list)
    top_token_texts: List[str] = field(default_factory=list)
    triggered: bool = False
    trigger_reason: str = ""


@dataclass
class LocalBeamParticle:
    kind: str
    token_id: Optional[int]
    forced_tokens: Any
    committed_mask: Any
    score: float = 0.0
    model_score: float = 0.0
    stability_score: float = 0.0
    structure_score: float = 0.0
    badness_penalty: float = 0.0
    parse_ok: bool = True
    format_ok: bool = True
    lookahead_forward_calls: int = 0


@dataclass
class LocalBeamResult:
    selected_kind: str
    selected_token: Optional[int]
    selected_particle_index: int
    branch_event: bool
    accepted_alternative: bool
    delay_selected: bool
    baseline_preserved: bool
    extra_forwards: int
    risk: LocalBeamRiskCandidate
    particle_logs: List[Dict[str, Any]]
    reason: str


def _as_torch_2d(value: Any, device: Optional[Any] = None, dtype: Optional[Any] = None) -> Any:
    torch = _ensure_torch()
    if torch.is_tensor(value):
        out = value
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    if int(getattr(out, "ndim", 0)) != 2:
        raise ValueError("expected a 2D [seq_len, vocab] tensor")
    return out


def _as_1d_bool(value: Any, length: int, device: Any) -> Any:
    torch = _ensure_torch()
    if value is None:
        out = torch.zeros(int(length), dtype=torch.bool, device=device)
    elif torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.bool).reshape(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if int(out.numel()) < int(length):
        pad = torch.zeros(int(length) - int(out.numel()), dtype=torch.bool, device=device)
        out = torch.cat([out, pad], dim=0)
    return out[: int(length)].clone()


def _as_1d_long(value: Any, length: int, device: Any) -> Any:
    torch = _ensure_torch()
    if value is None:
        out = torch.zeros(int(length), dtype=torch.long, device=device)
    elif torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long).reshape(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.long).reshape(-1)
    if int(out.numel()) < int(length):
        pad = torch.zeros(int(length) - int(out.numel()), dtype=torch.long, device=device)
        out = torch.cat([out, pad], dim=0)
    return out[: int(length)].clone()


def _decode_token_text(token_id: int, token_ids_to_code: Optional[Callable[[Any], str]]) -> str:
    if token_ids_to_code is None:
        return ""
    token_arr = np.asarray([int(token_id)], dtype=np.int64)
    try:
        out = token_ids_to_code(token_arr)
    except Exception:
        try:
            out = token_ids_to_code(token_arr.tolist())
        except Exception:
            return ""
    return out if isinstance(out, str) else str(out)


def _decode_token_sequence(token_ids: Any, token_ids_to_code: Optional[Callable[[Any], str]]) -> str:
    if token_ids_to_code is None:
        return ""
    torch = _ensure_torch()
    try:
        return str(token_ids_to_code(token_ids))
    except Exception:
        pass
    if torch.is_tensor(token_ids):
        cpu_tokens = token_ids.detach()
        if cpu_tokens.device.type != "cpu":
            cpu_tokens = cpu_tokens.to(device="cpu")
        try:
            return str(token_ids_to_code(cpu_tokens))
        except Exception:
            try:
                return str(token_ids_to_code(cpu_tokens.tolist()))
            except Exception:
                return ""
    if hasattr(token_ids, "tolist"):
        try:
            return str(token_ids_to_code(token_ids.tolist()))
        except Exception:
            return ""
    return ""


def structure_token_score(token_text: str) -> float:
    """Return a clipped [0, 1] structure score for one decoded token string."""
    text = (
        str(token_text or "")
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\n")
        .replace("▁", " ")
    )
    normalized = text.strip().lower()
    if not normalized:
        return 0.0
    score = 0.0
    if any(ch in text for ch in "()[]{}"):
        score += 0.45
    if any(ch in text for ch in "'\""):
        score += 0.25
    if ":" in text:
        score += 0.35
    if "\n" in text or "\t" in text:
        score += 0.35
    if any(ch in text for ch in "<>="):
        score += 0.20
    if normalized in _STRUCTURE_KEYWORDS:
        score += 0.50
    if any(word in normalized.split() for word in _STRUCTURE_KEYWORDS):
        score += 0.25
    if normalized in {",", ".", "):", "]:", "},", "):\n"}:
        score += 0.20
    return float(np.clip(score, 0.0, 1.0))


def _structure_context_score(
    forced_tokens: Optional[Any],
    committed_mask: Optional[Any],
    pos: int,
    token_ids_to_code: Optional[Callable[[Any], str]],
    radius: int = 3,
) -> float:
    if token_ids_to_code is None or forced_tokens is None or committed_mask is None:
        return 0.0
    torch = _ensure_torch()
    ft = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
    cm = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    ft = ft.to(dtype=torch.long).reshape(-1)
    cm = cm.to(device=ft.device, dtype=torch.bool).reshape(-1)
    lim = min(int(ft.numel()), int(cm.numel()))
    if lim <= 0:
        return 0.0
    start = max(int(pos) - max(int(radius), 0), 0)
    end = min(int(pos) + max(int(radius), 0) + 1, lim)
    if end <= start:
        return 0.0
    neighbor_tokens = ft[start:end][cm[start:end]]
    if int(neighbor_tokens.numel()) <= 0:
        return 0.0
    text = _decode_token_sequence(neighbor_tokens, token_ids_to_code)
    lowered = text.lower()
    if any(ch in lowered for ch in _STRUCTURE_CHARS) or any(word in lowered for word in _STRUCTURE_KEYWORDS):
        return 0.5
    return 0.0


def _forbidden_token_ids(sampler: Optional[SamplerAdapter], vocab_size: int) -> List[int]:
    ids: set[int] = set()
    for source in (sampler, getattr(sampler, "tokenizer", None) if sampler is not None else None):
        if source is None:
            continue
        for attr in (
            "eot_token_id",
            "eos_token_id",
            "mask_id",
            "mask_token_id",
            "pad_token_id",
            "bos_token_id",
            "unk_token_id",
            "sep_token_id",
            "cls_token_id",
            "additional_special_tokens_ids",
            "all_special_ids",
        ):
            value = getattr(source, attr, None)
            if value is None:
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                try:
                    token_id = int(item)
                except Exception:
                    continue
                if 0 <= token_id < int(vocab_size):
                    ids.add(token_id)
    return sorted(ids)


def topk_entropy_norm(logits: Any, positions: Sequence[int], top_k: int = 64) -> Dict[int, float]:
    """Top-k normalized entropy for selected positions."""
    torch = _ensure_torch()
    logits_t = _as_torch_2d(logits, dtype=torch.float32)
    seq_len, vocab_size = int(logits_t.shape[0]), int(logits_t.shape[1])
    k = min(max(int(top_k), 2), max(int(vocab_size), 2))
    out: Dict[int, float] = {}
    for pos in positions:
        idx = int(pos)
        if idx < 0 or idx >= seq_len:
            continue
        vals_t = torch.topk(logits_t[idx], k=k, dim=-1).values
        probs_t = torch.softmax(vals_t, dim=-1)
        ent_t = -(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum()
        out[idx] = float((ent_t / np.log(float(k))).clamp(0.0, 1.0).item())
    return out


def temporal_kl_topk(
    logits: Any,
    prev_logits: Any,
    positions: Sequence[int],
    top_k: int = 32,
    symmetric: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Approximate temporal KL over the union of current/previous top-k ids."""
    torch = _ensure_torch()
    logits_t = _as_torch_2d(logits, dtype=torch.float32)
    prev_t = _as_torch_2d(prev_logits, device=logits_t.device, dtype=torch.float32)
    seq_len = min(int(logits_t.shape[0]), int(prev_t.shape[0]))
    vocab_size = min(int(logits_t.shape[1]), int(prev_t.shape[1]))
    if seq_len <= 0 or vocab_size <= 0:
        return {}, {}
    k = min(max(int(top_k), 1), vocab_size)
    raw: Dict[int, float] = {}
    for pos in positions:
        idx = int(pos)
        if idx < 0 or idx >= seq_len:
            continue
        cur_row = logits_t[idx, :vocab_size]
        prev_row = prev_t[idx, :vocab_size]
        cur_top = torch.topk(cur_row, k=k, dim=-1).indices
        prev_top = torch.topk(prev_row, k=k, dim=-1).indices
        union_idx = torch.unique(torch.cat([cur_top, prev_top], dim=0))
        cur_log = torch.log_softmax(cur_row, dim=-1)[union_idx]
        prev_log = torch.log_softmax(prev_row, dim=-1)[union_idx]
        cur_p = torch.exp(cur_log)
        prev_p = torch.exp(prev_log)
        cur_p = cur_p / cur_p.sum().clamp_min(1e-12)
        prev_p = prev_p / prev_p.sum().clamp_min(1e-12)
        cur_log_sel = torch.log(cur_p.clamp_min(1e-12))
        prev_log_sel = torch.log(prev_p.clamp_min(1e-12))
        kl_val = (cur_p * (cur_log_sel - prev_log_sel)).sum()
        if symmetric:
            rev = (prev_p * (prev_log_sel - cur_log_sel)).sum()
            kl_val = 0.5 * (kl_val + rev)
        raw[idx] = float(torch.nan_to_num(kl_val, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0).item())

    if not raw:
        return raw, {}
    items = sorted(raw.items(), key=lambda item: item[1])
    norm: Dict[int, float] = {}
    if len(items) == 1:
        only_pos, only_val = items[0]
        norm[int(only_pos)] = 1.0 if float(only_val) > 1e-12 else 0.0
    else:
        denom = float(len(items) - 1)
        for rank, (idx, _) in enumerate(items):
            norm[int(idx)] = float(rank / denom)
    return raw, norm


def _top_token_meta(
    logits_t: Any,
    pos: int,
    token_ids_to_code: Optional[Callable[[Any], str]],
    top_k: int,
    forbidden_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[str], float]:
    torch = _ensure_torch()
    row = logits_t[int(pos)].float().clone()
    if forbidden_ids:
        idx_t = torch.as_tensor(list(forbidden_ids), dtype=torch.long, device=row.device)
        idx_t = idx_t[(idx_t >= 0) & (idx_t < int(row.numel()))]
        if int(idx_t.numel()) > 0:
            row[idx_t] = torch.finfo(row.dtype).min
    k = min(max(int(top_k), 1), int(row.numel()))
    token_ids = [int(v.item()) for v in torch.topk(row, k=k, dim=-1).indices]
    texts = [_decode_token_text(token_id, token_ids_to_code) for token_id in token_ids]
    structure_score = max((structure_token_score(text) for text in texts), default=0.0)
    return token_ids, texts, float(structure_score)


def compute_local_beam_risks(
    logits: Any,
    prev_logits: Any,
    candidate_positions: Iterable[int],
    committed_mask: Optional[Any],
    cfg: DecoderConfig,
    token_ids_to_code: Optional[Callable[[Any], str]] = None,
    forced_tokens: Optional[Any] = None,
    sampler: Optional[SamplerAdapter] = None,
) -> List[LocalBeamRiskCandidate]:
    """Compute entropy/KL/structure risk for eligible positions."""
    torch = _ensure_torch()
    logits_t = _as_torch_2d(logits, dtype=torch.float32)
    prev_t = _as_torch_2d(prev_logits, device=logits_t.device, dtype=torch.float32)
    seq_len = min(int(logits_t.shape[0]), int(prev_t.shape[0]))
    cm = _as_1d_bool(committed_mask, seq_len, logits_t.device) if committed_mask is not None else None
    positions: List[int] = []
    for raw_pos in candidate_positions:
        pos = int(raw_pos)
        if pos < 0 or pos >= seq_len:
            continue
        if cm is not None and bool(cm[pos].item()):
            continue
        positions.append(pos)
    if not positions:
        return []

    entropy_top_k = int(getattr(cfg, "local_beam_entropy_top_k", 64))
    kl_top_k = int(getattr(cfg, "local_beam_kl_top_k", 32))
    top_k = int(getattr(cfg, "local_beam_top_k", 5))
    ent = topk_entropy_norm(logits_t, positions, top_k=entropy_top_k)
    kl_raw, kl_norm = temporal_kl_topk(logits_t, prev_t, positions, top_k=kl_top_k)
    forbidden = _forbidden_token_ids(sampler=sampler, vocab_size=int(logits_t.shape[1]))

    alpha = max(float(getattr(cfg, "local_beam_struct_weight", 0.75)), 0.0)
    mode = str(getattr(cfg, "local_beam_mode", "entropy_kl_struct") or "entropy_kl_struct")
    use_structure = mode in {"entropy_kl_struct", "beam", "delay_only"}
    risks: List[LocalBeamRiskCandidate] = []
    for pos in positions:
        top_tokens, top_texts, token_struct = _top_token_meta(
            logits_t=logits_t,
            pos=pos,
            token_ids_to_code=token_ids_to_code,
            top_k=top_k,
            forbidden_ids=forbidden,
        )
        ctx_struct = _structure_context_score(
            forced_tokens=forced_tokens,
            committed_mask=committed_mask,
            pos=pos,
            token_ids_to_code=token_ids_to_code,
        )
        structure_score = float(np.clip(max(token_struct, ctx_struct), 0.0, 1.0))
        structure_weight = float(1.0 + (alpha * structure_score if use_structure else 0.0))
        h = float(ent.get(pos, 0.0))
        kln = float(kl_norm.get(pos, 0.0))
        if mode == "entropy_only":
            risk = h * structure_weight
        elif mode == "kl_only":
            risk = kln * structure_weight
        else:
            risk = h * kln * structure_weight
        risks.append(
            LocalBeamRiskCandidate(
                pos=int(pos),
                entropy_norm=float(h),
                kl_raw=float(kl_raw.get(pos, 0.0)),
                kl_norm=float(kln),
                structure_score=float(structure_score),
                structure_weight=float(structure_weight),
                risk=float(risk),
                top_tokens=[int(v) for v in top_tokens],
                top_token_texts=top_texts,
            )
        )
    return sorted(risks, key=lambda cand: cand.risk, reverse=True)


def select_local_beam_trigger(
    risks: Sequence[LocalBeamRiskCandidate],
    cfg: DecoderConfig,
    branch_events: int = 0,
) -> Optional[LocalBeamRiskCandidate]:
    if not risks:
        return None
    if int(branch_events) >= max(int(getattr(cfg, "local_beam_max_events", 1)), 0):
        return None
    top_m = max(int(getattr(cfg, "local_beam_top_m", 1)), 1)
    tau_h = float(getattr(cfg, "local_beam_tau_entropy", 0.45))
    tau_kl = float(getattr(cfg, "local_beam_tau_kl", 0.8))
    tau_r = float(getattr(cfg, "local_beam_tau_risk", 0.45))
    mode = str(getattr(cfg, "local_beam_mode", "entropy_kl_struct") or "entropy_kl_struct")
    for cand in risks[:top_m]:
        reasons = []
        if mode not in {"kl_only"} and cand.entropy_norm < tau_h:
            reasons.append("entropy_below_tau")
        if mode not in {"entropy_only"} and cand.kl_norm < tau_kl:
            reasons.append("kl_below_tau")
        if cand.risk < tau_r:
            reasons.append("risk_below_tau")
        if reasons:
            cand.triggered = False
            cand.trigger_reason = ",".join(reasons)
            continue
        cand.triggered = True
        cand.trigger_reason = "trigger_local_beam"
        return cand
    return None


def _bracket_balance_ok(source: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in str(source or ""):
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _format_ok(source: str) -> bool:
    text = str(source or "").strip()
    if not text:
        return False
    if "```" in text:
        return False
    return not text.lower().startswith(("here", "sure", "explanation"))


def _particle_source(
    forced_tokens: Any,
    committed_mask: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
    source_prefix: str,
) -> str:
    torch = _ensure_torch()
    if token_ids_to_code is None:
        return ""
    ft = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
    cm = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    ft = ft.to(dtype=torch.long).reshape(-1)
    cm = cm.to(device=ft.device, dtype=torch.bool).reshape(-1)
    lim = min(int(ft.numel()), int(cm.numel()))
    if lim <= 0:
        completion = ""
    else:
        committed = ft[:lim][cm[:lim]]
        completion = _decode_token_sequence(committed, token_ids_to_code) if int(committed.numel()) > 0 else ""
    prefix = str(source_prefix or "")
    if prefix and completion.startswith(prefix):
        return completion
    return prefix + completion


def minimal_structure_check(
    forced_tokens: Any,
    committed_mask: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
    source_prefix: str = "",
) -> Dict[str, Any]:
    source = _particle_source(
        forced_tokens=forced_tokens,
        committed_mask=committed_mask,
        token_ids_to_code=token_ids_to_code,
        source_prefix=source_prefix,
    )
    format_ok = _format_ok(source) if token_ids_to_code is not None else True
    bracket_ok = _bracket_balance_ok(source) if token_ids_to_code is not None else True
    parse_ok = True
    parse_quality = 1.0
    syntax_error = ""
    if token_ids_to_code is not None and source.strip():
        try:
            ast.parse(source.rstrip() + "\n")
        except SyntaxError as exc:
            parse_ok = False
            syntax_error = str(getattr(exc, "msg", "") or "")
            parse_quality = 0.0 if not bracket_ok else 0.4
        except Exception as exc:
            parse_ok = False
            syntax_error = f"{type(exc).__name__}: {exc}"
            parse_quality = 0.0
    return {
        "source": source,
        "format_ok": bool(format_ok),
        "bracket_ok": bool(bracket_ok),
        "parse_ok": bool(parse_ok),
        "parse_quality": float(parse_quality),
        "syntax_error": syntax_error,
    }


def _lookahead_entropy_norm(lookahead_logits: Any, pos: int, top_k: int) -> float:
    values = topk_entropy_norm(lookahead_logits, [int(pos)], top_k=top_k)
    return float(values.get(int(pos), 0.0))


def _make_particle(
    kind: str,
    token_id: Optional[int],
    base_forced_t: Any,
    base_committed_t: Any,
    pos: int,
) -> Tuple[Any, Any]:
    ft = base_forced_t.clone()
    cm = base_committed_t.clone()
    if kind != "delay" and token_id is not None:
        ft[int(pos)] = int(token_id)
        cm[int(pos)] = True
    elif kind == "delay":
        cm[int(pos)] = False
    return ft, cm


def build_local_beam_token_candidates(
    logits: Any,
    pos: int,
    cfg: DecoderConfig,
    sampler: Optional[SamplerAdapter] = None,
) -> List[Dict[str, Any]]:
    """Return baseline/top-k/delay token proposals without running lookahead."""
    torch = _ensure_torch()
    logits_t = _as_torch_2d(logits, dtype=torch.float32)
    row = logits_t[int(pos)].clone()
    forbidden = _forbidden_token_ids(sampler=sampler, vocab_size=int(row.numel()))
    if forbidden:
        idx_t = torch.as_tensor(forbidden, dtype=torch.long, device=row.device)
        idx_t = idx_t[(idx_t >= 0) & (idx_t < int(row.numel()))]
        if int(idx_t.numel()) > 0:
            row[idx_t] = torch.finfo(row.dtype).min
    probs = torch.softmax(row, dim=-1)
    beam_size = max(int(getattr(cfg, "local_beam_size", 4)), 1)
    top_k = min(max(int(getattr(cfg, "local_beam_top_k", 5)), 1), int(row.numel()))
    top_tokens = [int(v.item()) for v in torch.topk(probs, k=top_k, dim=-1).indices]
    proposals: List[Dict[str, Any]] = []
    baseline_token = int(top_tokens[0]) if top_tokens else int(torch.argmax(probs).item())
    proposals.append({"kind": "baseline", "token_id": int(baseline_token), "prob": float(probs[baseline_token].item())})
    for token_id in top_tokens:
        if len(proposals) >= max(beam_size - 1, 1):
            break
        if int(token_id) == baseline_token and bool(getattr(cfg, "local_beam_preserve_baseline", True)):
            continue
        proposals.append({"kind": "candidate", "token_id": int(token_id), "prob": float(probs[token_id].item())})
    if len(proposals) < beam_size:
        proposals.append({"kind": "delay", "token_id": None, "prob": 0.0})
    return proposals[:beam_size]


def run_commit_timing_local_beam(
    logits: Any,
    prev_logits: Any,
    risk: LocalBeamRiskCandidate,
    committed_mask: Any,
    forced_tokens: Optional[Any],
    cfg: DecoderConfig,
    sampler: SamplerAdapter,
    latents: Any,
    token_ids_to_code: Optional[Callable[[Any], str]] = None,
    source_prefix: str = "",
) -> LocalBeamResult:
    """Run a conservative short-horizon local beam around one risky token."""
    torch = _ensure_torch()
    logits_t = _as_torch_2d(logits, dtype=torch.float32)
    seq_len = int(logits_t.shape[0])
    device = logits_t.device
    pos = int(risk.pos)
    base_cm_t = _as_1d_bool(committed_mask, seq_len, device)
    base_ft_t = _as_1d_long(forced_tokens, seq_len, device)
    proposals = build_local_beam_token_candidates(logits_t, pos=pos, cfg=cfg, sampler=sampler)
    probs_t = torch.softmax(logits_t[pos], dim=-1)
    horizon = max(int(getattr(cfg, "local_beam_horizon", 2)), 0)
    entropy_top_k = int(getattr(cfg, "local_beam_entropy_top_k", 64))

    particles: List[LocalBeamParticle] = []
    for proposal in proposals:
        kind = str(proposal["kind"])
        token_id = proposal.get("token_id")
        ft, cm = _make_particle(kind=kind, token_id=token_id, base_forced_t=base_ft_t, base_committed_t=base_cm_t, pos=pos)
        lookahead_latents = latents
        lookahead_logits = logits_t
        calls = 0
        for _ in range(horizon):
            try:
                lookahead_logits = sampler.step(lookahead_latents, cm, ft)
                lookahead_logits = _as_torch_2d(lookahead_logits, device=device, dtype=torch.float32)
                lookahead_latents = lookahead_logits
                calls += 1
            except Exception:
                break

        if token_id is None:
            model_score = -0.05
        else:
            model_score = float(torch.log(probs_t[int(token_id)].clamp_min(1e-12)).item())
        end_entropy = _lookahead_entropy_norm(lookahead_logits, pos=pos, top_k=entropy_top_k) if calls > 0 else risk.entropy_norm
        stability_score = float(risk.entropy_norm - end_entropy - float(getattr(cfg, "local_beam_lambda_kl", 1.0)) * risk.kl_norm)
        structure = minimal_structure_check(
            forced_tokens=ft,
            committed_mask=cm,
            token_ids_to_code=token_ids_to_code,
            source_prefix=source_prefix,
        )
        parse_quality = float(structure.get("parse_quality", 1.0))
        struct_score = float((0.5 if structure.get("bracket_ok", True) else 0.0) + 0.5 * parse_quality)
        badness = 0.0
        if not bool(structure.get("format_ok", True)):
            badness += 0.5
        if not bool(structure.get("bracket_ok", True)):
            badness += 0.5
        if not bool(structure.get("parse_ok", True)):
            badness += 0.75
        score = float(model_score + 0.5 * stability_score + 0.5 * struct_score - badness)
        particles.append(
            LocalBeamParticle(
                kind=kind,
                token_id=int(token_id) if token_id is not None else None,
                forced_tokens=ft,
                committed_mask=cm,
                score=score,
                model_score=float(model_score),
                stability_score=float(stability_score),
                structure_score=float(struct_score),
                badness_penalty=float(badness),
                parse_ok=bool(structure.get("parse_ok", True)),
                format_ok=bool(structure.get("format_ok", True)),
                lookahead_forward_calls=int(calls),
            )
        )

    if not particles:
        raise RuntimeError("local beam constructed no particles")

    baseline_idx = next((idx for idx, part in enumerate(particles) if part.kind == "baseline"), 0)
    baseline_particle = particles[baseline_idx]
    ranked = sorted(enumerate(particles), key=lambda item: item[1].score, reverse=True)
    best_idx, best = ranked[0]
    second_non_baseline = next((part for _, part in ranked if part.kind != "baseline" and part is not best), None)
    margin_base = float(getattr(cfg, "local_beam_margin_base", 0.15))
    margin_branch = float(getattr(cfg, "local_beam_margin_branch", 0.05))
    reason = "keep_baseline"

    selected_idx = baseline_idx
    selected = baseline_particle
    if best.kind == "delay" and str(getattr(cfg, "local_beam_mode", "")) == "delay_only":
        selected_idx = best_idx
        selected = best
        reason = "delay_only"
    elif best.kind != "baseline":
        base_gap = float(best.score - baseline_particle.score)
        branch_gap = margin_branch
        if second_non_baseline is not None and second_non_baseline is not best:
            branch_gap = float(best.score - second_non_baseline.score)
        clear_base = base_gap >= margin_base
        clear_branch = branch_gap >= margin_branch
        minimal_ok = bool(best.format_ok and (best.parse_ok or not baseline_particle.parse_ok))
        no_parse_regression = bool(best.parse_ok or not baseline_particle.parse_ok)
        if clear_base and clear_branch and minimal_ok and no_parse_regression:
            selected_idx = best_idx
            selected = best
            reason = "accepted_clear_margin"
        else:
            reason = "rejected_conservative_margin"

    particle_logs = [
        {
            "kind": part.kind,
            "token_id": part.token_id,
            "score": float(part.score),
            "model_score": float(part.model_score),
            "stability_score": float(part.stability_score),
            "structure_score": float(part.structure_score),
            "badness_penalty": float(part.badness_penalty),
            "parse_ok": bool(part.parse_ok),
            "format_ok": bool(part.format_ok),
            "lookahead_forward_calls": int(part.lookahead_forward_calls),
        }
        for part in particles
    ]
    return LocalBeamResult(
        selected_kind=str(selected.kind),
        selected_token=int(selected.token_id) if selected.token_id is not None else None,
        selected_particle_index=int(selected_idx),
        branch_event=True,
        accepted_alternative=bool(selected.kind == "candidate"),
        delay_selected=bool(selected.kind == "delay"),
        baseline_preserved=bool(any(part.kind == "baseline" for part in particles)),
        extra_forwards=int(sum(part.lookahead_forward_calls for part in particles)),
        risk=risk,
        particle_logs=particle_logs,
        reason=reason,
    )
