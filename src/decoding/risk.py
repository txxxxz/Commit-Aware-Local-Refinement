"""Risk utilities: entropy, attention-based influence proxy, and joint risk scoring."""
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import DecoderConfig


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def entropy_at_position(logits: np.ndarray, pos: int) -> float:
    """Token entropy (base-e) at one position."""
    p = np.clip(softmax(logits[pos]), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def dynamic_valid_positions(
    logits: np.ndarray,
    candidate_positions: Iterable[int],
    committed_mask: Optional[np.ndarray],
    max_prob_threshold: float = 0.99,
    eot_token_id: Optional[int] = None,
    exclude_eot: bool = True,
) -> Tuple[List[int], Dict[int, Dict[str, float]]]:
    """
    Collect active positions for risk statistics.

    Rules:
    - remove committed positions;
    - keep every remaining MASK position in valid_set.

    Notes:
    - `max_prob_threshold` and `exclude_eot` are kept in the signature for
      backward compatibility but are intentionally ignored.
    """
    valid: List[int] = []
    meta: Dict[int, Dict[str, float]] = {}
    del max_prob_threshold, eot_token_id, exclude_eot
    candidates = np.fromiter((int(idx) for idx in candidate_positions), dtype=np.int64)
    if candidates.size == 0:
        return valid, meta

    if committed_mask is not None:
        cm = np.asarray(committed_mask, dtype=bool)
        candidates = candidates[~cm[candidates]]
    if candidates.size == 0:
        return valid, meta

    valid = candidates.astype(np.int64).tolist()
    # Metadata is intentionally omitted on this hot path to reduce Python overhead.
    meta = {}
    return valid, meta


def _mean_attention_matrix(attention: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalize attention tensor to [seq_len, seq_len].

    Supported shapes:
    - [seq_len, seq_len]
    - [heads, seq_len, seq_len]
    - [batch, heads, seq_len, seq_len]
    """
    if attention is None:
        return None
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a.mean(axis=0)
    if a.ndim == 4:
        return a.mean(axis=(0, 1))
    return None


def attention_influence_proxy(
    attention: Optional[np.ndarray],
    committed_mask: np.ndarray,
    candidate_positions: Iterable[int],
) -> Dict[int, float]:
    """
    Compute forward-only influence proxy:
      I_i = sum_{j != i} A_{i,j} * (1 - M_j)

    Here committed_mask=True means committed -> (1 - M_j)=1.
    """
    scores: Dict[int, float] = {}
    a = _mean_attention_matrix(attention)
    if a is None:
        for idx in candidate_positions:
            scores[int(idx)] = 0.0
        return scores

    seq_len = int(min(a.shape[0], a.shape[1], committed_mask.shape[0]))
    committed = committed_mask[:seq_len].astype(np.float64)
    for idx_raw in candidate_positions:
        idx = int(idx_raw)
        if idx < 0 or idx >= seq_len:
            scores[idx] = 0.0
            continue
        row = np.array(a[idx, :seq_len], copy=True)
        row[idx] = 0.0
        scores[idx] = float(np.sum(row * committed))
    return scores


def attention_influence_from_attentions(
    attentions: Optional[Any],
    valid_mask: Any,
    batch_idx: int = 0,
    eps: float = 1e-9,
    return_debug: bool = False,
):
    """
    Zero-cost attention influence proxy from the model's main forward pass.

    Args:
    - attentions: tuple/list of attention tensors; last layer expected in shape
      [batch, num_heads, seq_len, seq_len] (or compatible 2D/3D variants).
    - valid_mask: boolean tensor/array of shape [seq_len], True for current [MASK] tokens.
    - batch_idx: batch row to use (defaults to 0).
    - eps: small constant for min-max denominator stability.

    Returns:
    - Tensor of shape [seq_len], zero outside valid_mask, min-max normalized on valid_mask.
    - If `return_debug=True`, also returns a debug dict with raw pre-normalization stats.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch-less environments
        raise ImportError("attention_influence_from_attentions requires torch.") from exc

    valid_mask_t = valid_mask if torch.is_tensor(valid_mask) else torch.as_tensor(valid_mask)
    if valid_mask_t.ndim != 1:
        valid_mask_t = valid_mask_t.reshape(-1)
    valid_mask_t = valid_mask_t.to(dtype=torch.bool)

    seq_len = int(valid_mask_t.numel())
    out = torch.zeros(seq_len, dtype=torch.float32, device=valid_mask_t.device)
    debug = {"raw_min": 0.0, "raw_max": 0.0, "raw_span": 0.0, "raw_nonzero": 0}
    if seq_len == 0:
        return (out, debug) if return_debug else out

    if not attentions:
        return (out, debug) if return_debug else out

    last_layer_attn = attentions[-1]
    if last_layer_attn is None:
        return (out, debug) if return_debug else out

    last = last_layer_attn if torch.is_tensor(last_layer_attn) else torch.as_tensor(last_layer_attn)

    # Normalize to [batch, heads, seq, seq] defensively.
    if last.ndim == 2:
        last = last.unsqueeze(0).unsqueeze(0)
    elif last.ndim == 3:
        last = last.unsqueeze(0)
    elif last.ndim != 4:
        return (out, debug) if return_debug else out

    if int(last.shape[0]) <= 0:
        return (out, debug) if return_debug else out
    if batch_idx < 0 or batch_idx >= int(last.shape[0]):
        batch_idx = 0

    # [heads, seq, seq]
    last = last[batch_idx]
    if last.ndim == 2:
        last = last.unsqueeze(0)
    if last.ndim != 3:
        return (out, debug) if return_debug else out

    # Always compute and return on the same device as model attentions.
    compute_device = last.device
    valid_mask_t = valid_mask_t.to(device=compute_device)
    out = torch.zeros(seq_len, dtype=torch.float32, device=compute_device)

    seq_lim = int(min(seq_len, int(last.shape[-2]), int(last.shape[-1])))
    if seq_lim <= 0:
        return (out, debug) if return_debug else out

    # Mask and attention now live on the same device.
    valid_mask_lim = valid_mask_t[:seq_lim]
    if int(valid_mask_lim.sum().item()) == 0:
        return (out, debug) if return_debug else out

    # 1) Average heads -> [seq, seq]
    mean_attn = last[:, :seq_lim, :seq_lim].float().mean(dim=0)
    # 2) Select masked rows i where valid_mask[i] == True
    masked_rows = mean_attn[valid_mask_lim]
    if masked_rows.ndim == 1:
        masked_rows = masked_rows.unsqueeze(0)
    # 3) Column sum: I_j = sum_{i in Masked} A_{i,j}
    influence = masked_rows.sum(dim=0)
    if int(influence.numel()) > 0:
        raw_min = float(influence.min().item())
        raw_max = float(influence.max().item())
        debug["raw_min"] = raw_min
        debug["raw_max"] = raw_max
        debug["raw_span"] = float(raw_max - raw_min)
        debug["raw_nonzero"] = int((influence.abs() > float(eps)).sum().item())

    # 4) Min-max normalize only on valid positions.
    norm = torch.zeros_like(influence)
    valid_scores = influence[valid_mask_lim]
    if int(valid_scores.numel()) > 0:
        v_min = valid_scores.min()
        v_max = valid_scores.max()
        denom = (v_max - v_min).clamp_min(float(eps))
        norm_valid = (valid_scores - v_min) / denom
        norm[valid_mask_lim] = norm_valid

    out[:seq_lim] = norm.to(device=out.device, dtype=out.dtype)
    return (out, debug) if return_debug else out


def normalize_influence_scores(influences: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize influence values into [0, 1]."""
    if not influences:
        return {}
    vals = np.asarray(list(influences.values()), dtype=np.float64)
    vals = np.nan_to_num(vals, nan=0.0, neginf=0.0, posinf=0.0)
    low = float(vals.min())
    high = float(vals.max())
    if high <= low + 1e-12:
        return {int(k): 0.0 for k in influences.keys()}
    return {
        int(k): float(np.clip((float(v) - low) / (high - low + 1e-12), 0.0, 1.0))
        for k, v in influences.items()
    }


def dynamic_risk_thresholds(
    risk_values: Iterable[float],
    low_q: float = 0.25,
    high_q: float = 0.75,
    min_gap: float = 1e-6,
    entropy_only: bool = False,
) -> Tuple[float, float]:
    """Return (tau_low, tau_high) from quantiles over current uncommitted risks."""
    arr = np.asarray(list(risk_values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    low_q = float(np.clip(low_q, 0.0, 1.0))
    high_q = float(np.clip(high_q, 0.0, 1.0))
    if low_q > high_q:
        low_q, high_q = high_q, low_q
    tau_low = float(np.quantile(arr, low_q))
    tau_high = float(np.quantile(arr, high_q))
    risk_span = float(max(float(arr.max() - arr.min()), 0.0))
    min_gap = max(float(min_gap), 0.0)
    if entropy_only:
        min_gap = min(min_gap, max(risk_span * 0.1, 0.0))
    if tau_high < tau_low + min_gap:
        tau_high = tau_low + min_gap
    tau_high = float(min(tau_high, float(arr.max())))
    return tau_low, tau_high


def risk_score(
    entropy: float,
    influence: float,
    cfg: DecoderConfig,
    running_stats: Optional[Dict[str, float]] = None,
    use_entropy: bool = True,
    use_influence: bool = True,
) -> float:
    """
    Joint risk score with soft gating:
      R_i = H_i * (1 + beta * norm(I_i))

    `influence` is expected to be normalized (or near-normalized) before calling.
    """
    del running_stats
    if not use_entropy:
        return 0.0
    h = max(float(entropy), 0.0)
    if cfg.risk_fusion_mode == "entropy_only":
        return h
    ni = max(float(influence), 0.0) if use_influence else 0.0
    return float(h * (1.0 + float(cfg.risk_beta) * ni))


def risk_score_with_components(
    entropy: float,
    influence: float,
    cfg: DecoderConfig,
    running_stats: Optional[Dict[str, float]] = None,
    use_entropy: bool = True,
    use_influence: bool = True,
) -> Tuple[float, float, float]:
    """Return (risk, entropy, normalized_influence)."""
    del running_stats
    ni = float(np.clip(influence, 0.0, 1.0))
    score = risk_score(
        entropy=entropy,
        influence=ni,
        cfg=cfg,
        running_stats=None,
        use_entropy=use_entropy,
        use_influence=use_influence,
    )
    return score, max(float(entropy), 0.0), ni


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), base-e (NumPy fallback path)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _torch_kl_divergence(p_t, q_t, eps: float = 1e-12):
    p_t = p_t.clamp_min(float(eps))
    q_t = q_t.clamp_min(float(eps))
    return (p_t * (torch.log(p_t) - torch.log(q_t))).sum(dim=-1)


def _perturb_distribution(p: np.ndarray, eps: float, mode: str) -> Tuple[np.ndarray, int]:
    """Return (perturbed_distribution, chosen_token_id)."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        return p, 0
    p = np.clip(p, 1e-12, 1.0)
    p = p / (p.sum() + 1e-12)
    top = np.argsort(-p)
    best = int(top[0])
    eps = float(np.clip(eps, 0.0, 1.0))
    if mode == "top2_mass" and len(top) > 1:
        second = int(top[1])
        p_pert = (1.0 - eps) * p
        p_pert[second] += eps
        p_pert = p_pert / (p_pert.sum() + 1e-12)
        return p_pert, second
    p_pert = (1.0 - eps) * p
    p_pert[best] += eps
    p_pert = p_pert / (p_pert.sum() + 1e-12)
    return p_pert, best


def _choose_perturb_token_torch(p_t, mode: str) -> int:
    top_idx_t = torch.argsort(p_t, descending=True)
    if mode == "top2_mass" and int(top_idx_t.numel()) > 1:
        return int(top_idx_t[1].item())
    return int(top_idx_t[0].item())


def _compute_influence_numpy(
    logits: np.ndarray,
    pos: int,
    eps: float,
    sampler_step_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    committed_mask: np.ndarray,
    forced_tokens: Optional[np.ndarray],
    cfg: DecoderConfig,
    return_top_affected: bool = False,
):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] <= 0:
        return (0.0, []) if return_top_affected else (0.0, None)
    seq_len = int(logits.shape[0])
    pos = int(pos)
    if pos < 0 or pos >= seq_len:
        return (0.0, []) if return_top_affected else (0.0, None)

    p = softmax(logits[pos])
    _, tok_pert = _perturb_distribution(p, eps, cfg.influence_perturb_mode)

    forced_pert = (
        np.array(forced_tokens, copy=True)
        if forced_tokens is not None
        else np.zeros(seq_len, dtype=np.int64)
    )
    if forced_pert.shape[0] != seq_len:
        forced_pert = np.resize(forced_pert, (seq_len,)).astype(np.int64)
    forced_pert[pos] = int(tok_pert)

    mask_pert = np.array(committed_mask, copy=True).astype(bool)
    if mask_pert.shape[0] != seq_len:
        mask_pert = np.resize(mask_pert, (seq_len,)).astype(bool)
    mask_pert[pos] = True

    logits_new = np.asarray(sampler_step_fn(logits, mask_pert, forced_pert), dtype=np.float64)
    if logits_new.ndim != 2 or logits_new.shape[0] != seq_len:
        return (0.0, []) if return_top_affected else (0.0, None)

    all_positions = [j for j in range(seq_len) if j != pos]
    if not all_positions:
        return (0.0, []) if return_top_affected else (0.0, None)

    if cfg.influence_approx_mode == "sample_positions":
        n_sample = max(1, int(len(all_positions) * cfg.influence_positions_sample_ratio))
        rng = np.random.default_rng(cfg.random_seed + pos)
        positions = list(rng.choice(all_positions, size=min(n_sample, len(all_positions)), replace=False))
    elif cfg.influence_approx_mode == "single_forward":
        n_sample = min(8, len(all_positions))
        positions = all_positions[:n_sample]
    else:
        positions = all_positions

    total_kl = 0.0
    kls: List[Tuple[int, float]] = []
    for j in positions:
        p_old = softmax(logits[j])
        p_new = softmax(logits_new[j])
        k = max(0.0, kl_divergence(p_old, p_new))
        total_kl += k
        kls.append((int(j), float(k)))

    if return_top_affected:
        kls.sort(key=lambda item: -item[1])
        return float(total_kl), [idx for idx, _ in kls[:5]]
    return float(total_kl), None


def compute_influence(
    logits: Any,
    pos: int,
    eps: float,
    sampler_step_fn: Callable[[Any, Any, Optional[Any]], Any],
    committed_mask: Any,
    forced_tokens: Optional[Any],
    cfg: DecoderConfig,
    return_top_affected: bool = False,
):
    """
    Compute KL-based local influence for one position.

    Procedure:
    1) perturb token distribution at `pos` and force a concrete token at `pos`;
    2) run one additional forward step via `sampler_step_fn`;
    3) aggregate KL differences over selected other positions.
    """
    use_torch = False
    try:
        global torch  # noqa: PLW0603 - lazy import cache
        import torch  # type: ignore
        use_torch = bool(
            torch.is_tensor(logits)
            or torch.is_tensor(committed_mask)
            or (forced_tokens is not None and torch.is_tensor(forced_tokens))
        )
    except Exception:
        use_torch = False

    if not use_torch:
        return _compute_influence_numpy(
            logits=np.asarray(logits),
            pos=pos,
            eps=eps,
            sampler_step_fn=sampler_step_fn,
            committed_mask=np.asarray(committed_mask),
            forced_tokens=None if forced_tokens is None else np.asarray(forced_tokens),
            cfg=cfg,
            return_top_affected=return_top_affected,
        )

    logits_t = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    if int(getattr(logits_t, "ndim", 0)) != 2 or int(logits_t.shape[0]) <= 0:
        out = torch.tensor(0.0, device=logits_t.device if torch.is_tensor(logits_t) else "cpu", dtype=torch.float32)
        return (out, []) if return_top_affected else (out, None)
    logits_t = logits_t.float()
    device = logits_t.device
    seq_len = int(logits_t.shape[0])

    pos = int(pos)
    if pos < 0 or pos >= seq_len:
        out = torch.zeros((), device=device, dtype=logits_t.dtype)
        return (out, []) if return_top_affected else (out, None)

    p_t = torch.softmax(logits_t[pos], dim=-1).clamp_min(1e-12)
    tok_pert = _choose_perturb_token_torch(p_t, cfg.influence_perturb_mode)

    if forced_tokens is None:
        forced_pert_t = torch.zeros(seq_len, dtype=torch.long, device=device)
    else:
        forced_pert_t = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
        forced_pert_t = forced_pert_t.to(device=device, dtype=torch.long).reshape(-1)
        if int(forced_pert_t.numel()) < seq_len:
            pad_n = seq_len - int(forced_pert_t.numel())
            forced_pert_t = torch.cat([forced_pert_t, torch.zeros(pad_n, dtype=torch.long, device=device)], dim=0)
        if int(forced_pert_t.numel()) > seq_len:
            forced_pert_t = forced_pert_t[:seq_len]
    forced_pert_t = forced_pert_t.clone()
    forced_pert_t[pos] = int(tok_pert)

    mask_pert_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    mask_pert_t = mask_pert_t.to(device=device, dtype=torch.bool).reshape(-1)
    if int(mask_pert_t.numel()) < seq_len:
        pad_n = seq_len - int(mask_pert_t.numel())
        mask_pert_t = torch.cat([mask_pert_t, torch.zeros(pad_n, dtype=torch.bool, device=device)], dim=0)
    if int(mask_pert_t.numel()) > seq_len:
        mask_pert_t = mask_pert_t[:seq_len]
    mask_pert_t = mask_pert_t.clone()
    mask_pert_t[pos] = True

    logits_new_t = sampler_step_fn(logits_t, mask_pert_t, forced_pert_t)
    if not torch.is_tensor(logits_new_t):
        logits_new_t = torch.as_tensor(logits_new_t, device=device)
    logits_new_t = logits_new_t.to(device=device, dtype=logits_t.dtype)
    if int(getattr(logits_new_t, "ndim", 0)) != 2 or int(logits_new_t.shape[0]) != seq_len:
        out = torch.zeros((), device=device, dtype=logits_t.dtype)
        return (out, []) if return_top_affected else (out, None)

    all_positions_t = torch.arange(seq_len, device=device, dtype=torch.long)
    all_positions_t = all_positions_t[all_positions_t != int(pos)]
    if int(all_positions_t.numel()) == 0:
        out = torch.zeros((), device=device, dtype=logits_t.dtype)
        return (out, []) if return_top_affected else (out, None)

    if cfg.influence_approx_mode == "sample_positions":
        n_sample = max(1, int(int(all_positions_t.numel()) * float(cfg.influence_positions_sample_ratio)))
        rng = np.random.default_rng(int(cfg.random_seed) + int(pos))
        sampled_rel_idx_np = rng.choice(
            int(all_positions_t.numel()),
            size=min(n_sample, int(all_positions_t.numel())),
            replace=False,
        )
        sampled_rel_idx_t = torch.as_tensor(sampled_rel_idx_np, device=device, dtype=torch.long)
        positions_t = all_positions_t[sampled_rel_idx_t]
    elif cfg.influence_approx_mode == "single_forward":
        positions_t = all_positions_t[: min(8, int(all_positions_t.numel()))]
    else:
        positions_t = all_positions_t

    p_old_t = torch.softmax(logits_t[positions_t], dim=-1)
    p_new_t = torch.softmax(logits_new_t[positions_t], dim=-1)
    kls_t = _torch_kl_divergence(p_old_t, p_new_t).clamp_min(0.0)
    total_kl_t = kls_t.sum()

    if return_top_affected:
        if int(kls_t.numel()) > 0:
            top_n = min(5, int(kls_t.numel()))
            top_rel_idx_t = torch.topk(kls_t, k=top_n, largest=True).indices
            top_positions = [int(v.item()) for v in positions_t[top_rel_idx_t]]
        else:
            top_positions = []
        return total_kl_t, top_positions
    return total_kl_t, None
