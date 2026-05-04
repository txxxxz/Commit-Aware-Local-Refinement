"""Local particle-filter utilities for high-risk token decisions."""
import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import DecoderConfig, resolve_pf_time_window
from .sampler_adapter import SamplerAdapter


def _ensure_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError("PF torch path requires torch. Install with: pip install torch")


def _decode_tokens(
    token_ids: Any,
    token_ids_to_code: Callable[[Any], str],
) -> str:
    """
    Invoke decoder callback without requiring NumPy.
    """
    torch = _ensure_torch()
    try:
        return token_ids_to_code(token_ids)
    except Exception:
        pass

    if torch.is_tensor(token_ids):
        token_cpu = token_ids.detach()
        if token_cpu.device.type != "cpu":
            token_cpu = token_cpu.to(device="cpu")
        try:
            return token_ids_to_code(token_cpu)
        except Exception:
            return token_ids_to_code(token_cpu.tolist())

    if hasattr(token_ids, "tolist"):
        return token_ids_to_code(token_ids.tolist())
    return token_ids_to_code(token_ids)


@dataclass
class ParticleState:
    token_at_pos: int
    forced_tokens: Any
    committed_mask: Any
    source_pos: int = -1
    score: float = 0.0
    log_prob: float = 0.0
    rep_penalty: float = 0.0
    syntax_reward: float = 0.0
    parse_ok: bool = True
    lookahead_forward_calls: int = 0
    lookahead_bonus: float = 0.0
    lookahead_confidence: float = 0.0
    syntax_base_quality: float = 0.0
    syntax_candidate_quality: float = 0.0


def _parse_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _balanced_brackets(source: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in source:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _indentation_sanity(source: str) -> bool:
    level = 0
    for line in source.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            level += 1
        if stripped and stripped[0] == "\t":
            return False
    return level >= 0


def _committed_prefix_tokens(
    forced_tokens: Any,
    committed_mask: Any,
) -> Any:
    """Decode only the contiguous committed prefix to avoid fabricating holes."""
    torch = _ensure_torch()
    ft_t = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
    cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    ft_t = ft_t.to(dtype=torch.long).reshape(-1)
    cm_t = cm_t.to(dtype=torch.bool).reshape(-1)
    lim = min(int(ft_t.numel()), int(cm_t.numel()))
    if lim <= 0:
        return ft_t[:0]

    ft_t = ft_t[:lim]
    cm_t = cm_t[:lim]
    first_uncommitted = torch.nonzero(~cm_t, as_tuple=False)
    prefix_len = int(first_uncommitted[0].item()) if int(first_uncommitted.numel()) > 0 else lim
    if prefix_len <= 0:
        return ft_t[:0]
    return ft_t[:prefix_len].clone()


def _projected_committed_tokens(
    forced_tokens: Any,
    committed_mask: Any,
    upto_pos: Optional[int] = None,
) -> Any:
    """Collect committed tokens in order up to a position, dropping masked holes."""
    torch = _ensure_torch()
    ft_t = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
    cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    ft_t = ft_t.to(dtype=torch.long).reshape(-1)
    cm_t = cm_t.to(dtype=torch.bool).reshape(-1)
    lim = min(int(ft_t.numel()), int(cm_t.numel()))
    if upto_pos is None:
        end = lim
    else:
        end = min(max(int(upto_pos) + 1, 0), lim)
    if end <= 0:
        return ft_t[:0]
    proj_t = ft_t[:end][cm_t[:end]]
    return proj_t.clone()


def _bracket_health(source: str) -> float:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    bad_closers = 0
    for ch in source:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                bad_closers += 1
            else:
                stack.pop()
    if bad_closers > 0:
        return float(max(0.0, 1.0 - 0.5 * float(bad_closers)))
    return float(max(0.0, 1.0 - 0.08 * float(len(stack))))


def _bracket_issue_stats(source: str) -> Dict[str, int]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    bad_closers = 0
    for ch in source:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                bad_closers += 1
            else:
                stack.pop()
    unclosed_openers = len(stack)
    return {
        "bad_closers": int(bad_closers),
        "unclosed_openers": int(unclosed_openers),
        "imbalance": int(bad_closers + unclosed_openers),
    }


def _syntax_error_progress(source: str, exc: SyntaxError) -> float:
    lines = source.splitlines() or [source]
    total_chars = max(sum(len(line) + 1 for line in lines), 1)
    line_no = int(getattr(exc, "lineno", 1) or 1)
    line_no = min(max(line_no, 1), len(lines))
    line_text = lines[line_no - 1]
    offset = getattr(exc, "offset", None)
    if offset is None:
        offset = len(line_text) + 1
    offset = min(max(int(offset) - 1, 0), len(line_text))
    prefix_chars = sum(len(line) + 1 for line in lines[: line_no - 1]) + offset
    return float(min(max(prefix_chars / float(total_chars), 0.0), 1.0))


def _trailing_operator_penalty(source: str) -> float:
    stripped = source.rstrip()
    if not stripped:
        return 0.0

    punctuation_suffixes = (
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        "(",
        "[",
        "{",
        ",",
        ".",
        ":",
        "\\",
        "==",
        "!=",
        "<=",
        ">=",
        "<",
        ">",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
    )
    if stripped.endswith(punctuation_suffixes):
        return 1.0

    word_suffixes = {"and", "or", "not", "in", "is", "lambda", "return", "raise", "from"}
    last_token = stripped.split()[-1]
    return 1.0 if last_token in word_suffixes else 0.0


def _syntax_quality_from_exception(
    text: str,
    exc: SyntaxError,
    bracket_health: Optional[float] = None,
    indent_ok: Optional[bool] = None,
) -> float:
    progress = _syntax_error_progress(text, exc)
    if bracket_health is None:
        bracket_health = _bracket_health(text)
    if indent_ok is None:
        indent_ok = _indentation_sanity(text)
    indent_health = 1.0 if bool(indent_ok) else 0.0
    trailing_penalty = _trailing_operator_penalty(text)
    exc_msg = str(getattr(exc, "msg", "") or "").lower()
    severe_penalty = 0.0
    if "unmatched" in exc_msg or "never closed" in exc_msg:
        severe_penalty += 0.25
    if "unexpected indent" in exc_msg:
        severe_penalty += 0.15
    quality = (
        0.55 * progress
        + 0.25 * float(bracket_health)
        + 0.20 * indent_health
        - 0.25 * trailing_penalty
        - severe_penalty
    )
    return float(min(max(quality, -1.0), 0.95))


def _python_syntax_quality(source: str) -> Tuple[float, bool]:
    text = source.rstrip()
    if not text.strip():
        return 0.0, True

    try:
        ast.parse(text)
        return 1.0, True
    except SyntaxError as exc:
        quality = _syntax_quality_from_exception(text=text, exc=exc)
        return float(min(max(quality, -1.0), 0.95)), False


def _compose_parse_source(source_prefix: str, completion: str) -> str:
    prefix = str(source_prefix or "")
    text = str(completion or "")
    if prefix and text.startswith(prefix):
        return text
    return prefix + text


def parser_feedback_from_source(
    source: str,
    min_prefix_chars: int = 24,
) -> Dict[str, Any]:
    text = str(source or "").rstrip()
    stripped = text.strip()
    bracket_stats = _bracket_issue_stats(text)
    bracket_health = _bracket_health(text)
    indent_ok = _indentation_sanity(text)

    feedback: Dict[str, Any] = {
        "observed": False,
        "parse_ok": True,
        "primary_issue": "none",
        "issue_types": [],
        "bracket_issue": False,
        "indent_issue": False,
        "syntax_error_message": "",
        "syntax_error_progress": 0.0,
        "code_chars": int(len(text)),
        "code_lines": int(len(text.splitlines())) if text else 0,
        "bracket_health": float(bracket_health),
        "indent_ok": bool(indent_ok),
        "bad_closers": int(bracket_stats["bad_closers"]),
        "unclosed_openers": int(bracket_stats["unclosed_openers"]),
        "bracket_imbalance": int(bracket_stats["imbalance"]),
        "quality_raw_score": 1.0,
        "quality_score": 1.0,
        "severity_score": 0.0,
    }
    if not stripped or len(stripped) < int(min_prefix_chars):
        return feedback

    feedback["observed"] = True
    try:
        ast.parse(text)
        return feedback
    except SyntaxError as exc:
        exc_msg = str(getattr(exc, "msg", "") or "")
        exc_msg_lower = exc_msg.lower()
        bracket_issue = (
            "unmatched" in exc_msg_lower
            or "never closed" in exc_msg_lower
            or "closing parenthesis" in exc_msg_lower
            or int(bracket_stats["imbalance"]) > 0
        )
        indent_issue = ("indent" in exc_msg_lower) or (not indent_ok)
        issue_types: List[str] = []
        if bracket_issue:
            issue_types.append("bracket")
        if indent_issue:
            issue_types.append("indent")
        if not issue_types:
            issue_types.append("syntax")
        quality_raw = _syntax_quality_from_exception(
            text=text,
            exc=exc,
            bracket_health=bracket_health,
            indent_ok=indent_ok,
        )
        quality_score = float(min(max((quality_raw + 1.0) * 0.5, 0.0), 1.0))
        severity_score = float(min(max(1.0 - quality_score, 0.0), 1.0))

        feedback.update(
            {
                "parse_ok": False,
                "primary_issue": issue_types[0],
                "issue_types": issue_types,
                "bracket_issue": bool(bracket_issue),
                "indent_issue": bool(indent_issue),
                "syntax_error_message": exc_msg,
                "syntax_error_progress": float(_syntax_error_progress(text, exc)),
                "quality_raw_score": float(quality_raw),
                "quality_score": float(quality_score),
                "severity_score": float(severity_score),
            }
        )
        return feedback


def parser_feedback_from_tokens(
    forced_tokens: Any,
    committed_mask: Any,
    token_ids_to_code: Optional[Callable[[Any], str]],
    min_prefix_chars: int = 24,
) -> Dict[str, Any]:
    if token_ids_to_code is None:
        return {
            "observed": False,
            "parse_ok": True,
            "primary_issue": "none",
            "issue_types": [],
            "quality_raw_score": 1.0,
            "quality_score": 1.0,
            "severity_score": 0.0,
        }

    try:
        prefix_tokens = _committed_prefix_tokens(forced_tokens=forced_tokens, committed_mask=committed_mask)
        if int(getattr(prefix_tokens, "numel", lambda: 0)()) <= 0:
            return {
                "observed": False,
                "parse_ok": True,
                "primary_issue": "none",
                "issue_types": [],
                "quality_raw_score": 1.0,
                "quality_score": 1.0,
                "severity_score": 0.0,
            }
        source = _decode_tokens(prefix_tokens, token_ids_to_code)
    except Exception:
        return {
            "observed": False,
            "parse_ok": False,
            "primary_issue": "decode_error",
            "issue_types": ["decode_error"],
            "bracket_issue": False,
            "indent_issue": False,
            "syntax_error_message": "decode_error",
            "syntax_error_progress": 0.0,
            "code_chars": 0,
            "code_lines": 0,
            "bracket_health": 0.0,
            "indent_ok": True,
            "bad_closers": 0,
            "unclosed_openers": 0,
            "bracket_imbalance": 0,
            "quality_raw_score": -1.0,
            "quality_score": 0.0,
            "severity_score": 1.0,
        }
    return parser_feedback_from_source(source=source, min_prefix_chars=min_prefix_chars)


def _extend_token_id_set(target: set[int], value: Any) -> None:
    torch = _ensure_torch()
    if value is None:
        return
    if torch.is_tensor(value):
        values = value.detach().reshape(-1).tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    for item in values:
        try:
            target.add(int(item))
        except Exception:
            continue


def _collect_forbidden_pf_token_ids(
    sampler: Optional[SamplerAdapter],
    vocab_size: int,
) -> List[int]:
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
            _extend_token_id_set(ids, getattr(source, attr, None))
    return sorted(token_id for token_id in ids if 0 <= int(token_id) < int(vocab_size))


def badness_particle(
    forced_tokens: Any,
    cfg: DecoderConfig,
    token_ids_to_code: Optional[Callable[[Any], str]] = None,
) -> Tuple[float, bool]:
    """
    Backward-compatible structure badness signal.

    New PF scoring does not consume this value directly.
    """
    if not cfg.parsing_checks_enabled or cfg.language != "python" or token_ids_to_code is None:
        return 0.0, True

    try:
        code = _decode_tokens(forced_tokens, token_ids_to_code)
    except Exception:
        return 10.0, False

    parse_ok = _parse_python(code)
    bracket_ok = _balanced_brackets(code)
    indent_ok = _indentation_sanity(code)
    if parse_ok and bracket_ok and indent_ok:
        return 0.0, True
    badness = 10.0 if not parse_ok else 2.0
    if not bracket_ok:
        badness += 1.0
    if not indent_ok:
        badness += 1.0
    return badness, False


def repetition_penalty_particle(
    forced_tokens: Any,
    committed_mask: Any,
    ngram_size: int = 3,
) -> float:
    """Simple repetition penalty over committed tokens only."""
    torch = _ensure_torch()
    ft_t = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
    cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    ft_t = ft_t.to(dtype=torch.long).reshape(-1)
    cm_t = cm_t.to(dtype=torch.bool).reshape(-1)
    lim = min(int(ft_t.numel()), int(cm_t.numel()))
    if lim <= 1:
        return 0.0
    ft_t = ft_t[:lim]
    cm_t = cm_t[:lim]
    toks_t = ft_t[cm_t]
    if int(toks_t.numel()) < 2:
        return 0.0

    adjacent_repeats = int((toks_t[1:] == toks_t[:-1]).sum().item())
    ngram_size = max(int(ngram_size), 2)
    if int(toks_t.numel()) < ngram_size:
        return float(adjacent_repeats)

    windows = toks_t.unfold(0, ngram_size, 1)
    _, counts = torch.unique(windows, dim=0, return_counts=True)
    ngram_repeats = int(torch.clamp(counts - 1, min=0).sum().item())
    return float(adjacent_repeats + ngram_repeats)


def syntax_reward_particle(
    forced_tokens: Any,
    committed_mask: Any,
    cfg: DecoderConfig,
    token_ids_to_code: Optional[Callable[[Any], str]],
    current_t: Optional[int],
    pf_window: Optional[Tuple[int, int]] = None,
    candidate_pos: Optional[int] = None,
    source_prefix: str = "",
) -> Tuple[float, bool, float, float]:
    """
    Candidate-sensitive syntax signal:
    - enabled only inside the PF trigger window,
    - compares projected committed code with and without the current candidate,
    - enabled only for python + parse checks + decoder callback.
    """
    torch = _ensure_torch()
    if pf_window is None:
        t_start, t_end = resolve_pf_time_window(total_steps=max(int(getattr(cfg, "pf_time_window_ref_steps", 96)), 1), cfg=cfg)
    else:
        t_start, t_end = int(pf_window[0]), int(pf_window[1])
    in_pf_window = current_t is not None and int(t_start) <= int(current_t) <= int(t_end)

    if (
        not in_pf_window
        or not cfg.parsing_checks_enabled
        or cfg.language != "python"
        or token_ids_to_code is None
    ):
        return 0.0, True, 0.0, 0.0

    cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    cm_t = cm_t.to(dtype=torch.bool).reshape(-1)
    base_cm_t = cm_t.clone()
    if candidate_pos is not None and 0 <= int(candidate_pos) < int(base_cm_t.numel()):
        base_cm_t[int(candidate_pos)] = False

    if candidate_pos is not None:
        candidate_tokens = _projected_committed_tokens(
            forced_tokens=forced_tokens,
            committed_mask=cm_t,
            upto_pos=int(candidate_pos),
        )
        base_tokens = _projected_committed_tokens(
            forced_tokens=forced_tokens,
            committed_mask=base_cm_t,
            upto_pos=int(candidate_pos),
        )
    else:
        candidate_tokens = _committed_prefix_tokens(forced_tokens=forced_tokens, committed_mask=cm_t)
        base_tokens = candidate_tokens[:0]
    if int(getattr(candidate_tokens, "numel", lambda: 0)()) <= 0:
        return 0.0, True, 0.0, 0.0

    try:
        candidate_code = _decode_tokens(candidate_tokens, token_ids_to_code)
    except Exception:
        return 0.0, False, 0.0, -1.0

    if not candidate_code.strip():
        return 0.0, True, 0.0, 0.0

    base_quality = 0.0
    if int(getattr(base_tokens, "numel", lambda: 0)()) > 0:
        try:
            base_code = _decode_tokens(base_tokens, token_ids_to_code)
            base_quality, _ = _python_syntax_quality(_compose_parse_source(source_prefix, base_code))
        except Exception:
            base_quality = 0.0

    candidate_quality, parse_ok = _python_syntax_quality(_compose_parse_source(source_prefix, candidate_code))
    syntax_signal = float(cfg.pf_syntax_reward) * float(candidate_quality - base_quality)
    return float(syntax_signal), parse_ok, float(base_quality), float(candidate_quality)


def lookahead_bonus_particle(
    lookahead_logits: Any,
    committed_mask: Any,
    cfg: DecoderConfig,
) -> Tuple[float, float]:
    """Reward branches that make remaining masked positions easier to decode."""
    torch = _ensure_torch()
    if lookahead_logits is None:
        return 0.0, 0.0

    logits_t = lookahead_logits if torch.is_tensor(lookahead_logits) else torch.as_tensor(lookahead_logits)
    logits_t = logits_t.to(dtype=torch.float32)
    if int(getattr(logits_t, "ndim", 0)) != 2:
        return 0.0, 0.0

    cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    cm_t = cm_t.to(device=logits_t.device, dtype=torch.bool).reshape(-1)
    lim = min(int(logits_t.shape[0]), int(cm_t.numel()))
    if lim <= 0:
        return 0.0, 0.0

    future_mask_t = ~cm_t[:lim]
    if int(future_mask_t.sum().item()) <= 0:
        return 0.0, 0.0

    future_logits_t = logits_t[:lim][future_mask_t]
    future_probs_t = torch.softmax(future_logits_t, dim=-1)
    mean_top1_conf_t = future_probs_t.max(dim=-1).values.mean()
    mean_top1_conf = float(mean_top1_conf_t.item())
    bonus = float(max(float(cfg.pf_stability_weight), 0.0) * mean_top1_conf)
    return bonus, mean_top1_conf


def run_local_pf(
    logits: Any,
    pos: int,
    committed_mask: Any,
    forced_tokens: Optional[Any],
    cfg: DecoderConfig,
    sampler: SamplerAdapter,
    latents: Any,
    token_ids_to_code: Optional[Callable[[Any], str]] = None,
    force_fallback: bool = False,
    pf_top_k_override: Optional[int] = None,
    pf_particles_override: Optional[int] = None,
    pf_horizon_steps_override: Optional[int] = None,
    current_t: Optional[int] = None,
    pf_window: Optional[Tuple[int, int]] = None,
    source_prefix: str = "",
    return_particles: bool = False,
) -> Any:
    """
    Local PF with Top-B branch cloning and K-step lookahead.

    Scoring:
      w_k = log P(x_k) - lambda_rep * Penalty_rep(x_k)
            + Reward_syntax(x_k) + Reward_lookahead(x_k)
    """
    torch = _ensure_torch()

    if torch.is_tensor(logits):
        logits_t = logits.to(dtype=torch.float32)
    else:
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
    if int(getattr(logits_t, "ndim", 0)) != 2:
        return None, []
    device = logits_t.device
    seq_len, vocab_size = int(logits_t.shape[0]), int(logits_t.shape[1])

    if forced_tokens is None:
        base_forced_t = torch.zeros(seq_len, dtype=torch.long, device=device)
    else:
        base_forced_t = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
        base_forced_t = base_forced_t.to(device=device, dtype=torch.long).reshape(-1)
        if int(base_forced_t.numel()) < seq_len:
            base_forced_t = torch.cat(
                [base_forced_t, torch.zeros(seq_len - int(base_forced_t.numel()), dtype=torch.long, device=device)],
                dim=0,
            )
        base_forced_t = base_forced_t[:seq_len]
    base_forced_t = base_forced_t.clone()

    base_committed_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
    base_committed_t = base_committed_t.to(device=device, dtype=torch.bool).reshape(-1)
    if int(base_committed_t.numel()) < seq_len:
        base_committed_t = torch.cat(
            [base_committed_t, torch.zeros(seq_len - int(base_committed_t.numel()), dtype=torch.bool, device=device)],
            dim=0,
        )
    base_committed_t = base_committed_t[:seq_len].clone()

    if latents is None:
        latents_t = None
    elif torch.is_tensor(latents):
        latents_t = latents.to(device=device, dtype=torch.float32)
    else:
        latents_t = torch.as_tensor(latents, device=device, dtype=torch.float32)

    pos = int(pos)
    if pos < 0 or pos >= seq_len:
        return None, []

    probs_t = torch.softmax(logits_t[pos], dim=-1)

    pf_top_k = max(int(cfg.pf_top_k if pf_top_k_override is None else pf_top_k_override), 1)
    pf_particles = max(int(cfg.pf_particles if pf_particles_override is None else pf_particles_override), 1)
    pf_horizon_steps = max(int(cfg.pf_horizon_steps if pf_horizon_steps_override is None else pf_horizon_steps_override), 0)

    forbidden_token_ids = _collect_forbidden_pf_token_ids(sampler=sampler, vocab_size=vocab_size)
    candidate_probs_t = probs_t.clone()
    if forbidden_token_ids:
        forbidden_idx_t = torch.as_tensor(forbidden_token_ids, device=device, dtype=torch.long)
        candidate_probs_t[forbidden_idx_t] = -1.0

    allowed_count = int((candidate_probs_t >= 0.0).sum().item())
    if allowed_count <= 0:
        return None, []

    top_k = min(pf_top_k, allowed_count)
    top_indices_t = torch.topk(candidate_probs_t, k=top_k, largest=True).indices
    n_particles = min(pf_particles, int(top_indices_t.numel()))

    particles: List[ParticleState] = []
    for token_t in top_indices_t[:n_particles]:
        token = int(token_t.item())
        ft = base_forced_t.clone()
        cm = base_committed_t.clone()
        ft[pos] = token
        cm[pos] = True

        lookahead_latents = latents_t
        lookahead_logits = logits_t
        lookahead_calls = 0
        lookahead_failed = False
        for _ in range(pf_horizon_steps):
            try:
                lookahead_logits = sampler.step(lookahead_latents, cm, ft)
                if not torch.is_tensor(lookahead_logits):
                    lookahead_logits = torch.as_tensor(lookahead_logits, device=device, dtype=torch.float32)
                else:
                    lookahead_logits = lookahead_logits.to(device=device, dtype=torch.float32)
                lookahead_latents = lookahead_logits
                lookahead_calls += 1
            except Exception:
                lookahead_failed = True
                break

        log_p = float(torch.log(probs_t[token].clamp_min(1e-12)).item())
        rep_pen = repetition_penalty_particle(
            forced_tokens=ft,
            committed_mask=cm,
            ngram_size=cfg.pf_repetition_ngram,
        )
        syn_reward, parse_ok, syntax_base_quality, syntax_candidate_quality = syntax_reward_particle(
            forced_tokens=ft,
            committed_mask=cm,
            cfg=cfg,
            token_ids_to_code=token_ids_to_code,
            current_t=current_t,
            pf_window=pf_window,
            candidate_pos=pos,
            source_prefix=source_prefix,
        )
        lookahead_bonus = 0.0
        lookahead_confidence = 0.0
        if lookahead_calls > 0:
            lookahead_bonus, lookahead_confidence = lookahead_bonus_particle(
                lookahead_logits=lookahead_logits,
                committed_mask=cm,
                cfg=cfg,
            )

        parse_penalty = 0.0
        if (
            cfg.parsing_checks_enabled
            and cfg.language == "python"
            and token_ids_to_code is not None
            and not bool(parse_ok)
        ):
            parse_penalty = float(getattr(cfg, "pf_parse_fail_penalty", 0.0))

        score = float(log_p - float(cfg.pf_rep_lambda) * rep_pen + syn_reward + lookahead_bonus - parse_penalty)

        particles.append(
            ParticleState(
                token_at_pos=token,
                forced_tokens=ft,
                committed_mask=cm,
                source_pos=int(pos),
                score=score,
                log_prob=log_p,
                rep_penalty=rep_pen,
                syntax_reward=syn_reward,
                parse_ok=parse_ok,
                lookahead_forward_calls=lookahead_calls,
                lookahead_bonus=lookahead_bonus,
                lookahead_confidence=lookahead_confidence,
                syntax_base_quality=syntax_base_quality,
                syntax_candidate_quality=syntax_candidate_quality,
            )
        )
        if lookahead_failed:
            # Degrade gracefully instead of aborting the whole decode step.
            continue

    if not particles:
        if return_particles:
            return None, [], []
        return None, []

    particles.sort(key=lambda p: p.score, reverse=True)
    chosen_particle = particles[0]
    chosen = chosen_particle.token_at_pos
    argmax_token = int(top_indices_t[0].item())
    argmax_particle = next((particle for particle in particles if int(particle.token_at_pos) == argmax_token), None)
    gate_rejected = False
    gate_reason = "accepted"
    gate_quality_gain: Optional[float] = None
    gate_score_gain: Optional[float] = None
    if (
        bool(getattr(cfg, "pf_do_no_harm_enabled", True))
        and cfg.parsing_checks_enabled
        and cfg.language == "python"
        and token_ids_to_code is not None
        and int(chosen) != int(argmax_token)
    ):
        min_quality_gain = max(float(getattr(cfg, "pf_do_no_harm_min_quality_gain", 0.05)), 0.0)
        if not bool(chosen_particle.parse_ok):
            gate_rejected = True
            gate_reason = "candidate_parse_failed"
        elif argmax_particle is None:
            gate_rejected = True
            gate_reason = "argmax_particle_missing"
        else:
            gate_quality_gain = float(chosen_particle.syntax_candidate_quality - argmax_particle.syntax_candidate_quality)
            gate_score_gain = float(chosen_particle.score - argmax_particle.score)
            if gate_quality_gain < min_quality_gain:
                gate_rejected = True
                gate_reason = "candidate_syntax_not_improved"
    if gate_rejected:
        chosen = None

    logs = [
        {
            "pf_step": 0,
            "lookahead_steps": int(pf_horizon_steps),
            "lookahead_forward_calls_total": int(sum(p.lookahead_forward_calls for p in particles)),
            "particle_tokens": [int(p.token_at_pos) for p in particles],
            "particle_scores": [float(p.score) for p in particles],
            "particle_log_probs": [float(p.log_prob) for p in particles],
            "particle_rep_penalty": [float(p.rep_penalty) for p in particles],
            "particle_syntax_reward": [float(p.syntax_reward) for p in particles],
            "particle_parse_ok": [bool(p.parse_ok) for p in particles],
            "particle_lookahead_bonus": [float(p.lookahead_bonus) for p in particles],
            "particle_lookahead_confidence": [float(p.lookahead_confidence) for p in particles],
            "particle_syntax_base_quality": [float(p.syntax_base_quality) for p in particles],
            "particle_syntax_candidate_quality": [float(p.syntax_candidate_quality) for p in particles],
            "forbidden_token_ids": [int(v) for v in forbidden_token_ids],
            "argmax_token": int(argmax_token),
            "chosen_token_before_gate": int(chosen_particle.token_at_pos),
            "pf_do_no_harm_enabled": bool(getattr(cfg, "pf_do_no_harm_enabled", True)),
            "pf_do_no_harm_rejected": bool(gate_rejected),
            "pf_do_no_harm_reason": str(gate_reason),
            "pf_do_no_harm_quality_gain": gate_quality_gain,
            "pf_do_no_harm_score_gain": gate_score_gain,
            "pf_do_no_harm_min_quality_gain": float(getattr(cfg, "pf_do_no_harm_min_quality_gain", 0.05)),
        }
    ]

    if chosen is None and force_fallback and not gate_rejected:
        fallback = int(top_indices_t[0].item())
        if return_particles:
            return fallback, logs, particles
        return fallback, logs
    if chosen is None:
        if return_particles:
            return None, logs, particles
        return None, logs
    if return_particles:
        return int(chosen), logs, particles
    return int(chosen), logs
