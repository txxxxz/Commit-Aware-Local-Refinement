"""
Sampler adapter: abstract interface and placeholder implementation.
Replace PlaceholderDiffusionSampler with Open-dLLM step(...) wrapper for real deployment.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional


def _ensure_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError("SamplerAdapter torch path requires torch. Install with: pip install torch")


class SamplerAdapter(ABC):
    """Abstract interface for one diffusion step."""

    @abstractmethod
    def step(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
    ) -> Any:
        """
        One diffusion step.
        Args:
            latents: current state (model-specific)
            committed_mask: bool tensor/array [seq_len], True = already committed
            forced_tokens: int tensor/array [seq_len] or None; token id to force at committed positions
        Returns:
            logits: float tensor [seq_len, vocab_size]
        """
        pass

    def step_with_aux(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
    ) -> tuple[Any, dict]:
        """
        One diffusion step with optional auxiliary tensors.

        Returns:
            (logits, aux_dict) where aux_dict may contain model-specific
            extras (for example: {"last_attention": torch.Tensor}).
        """
        return self.step(latents, committed_mask, forced_tokens), {}

    def step_with_forced_position(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
        pos: int,
        token_id: int,
    ) -> Any:
        """
        Convenience wrapper for perturbation-based influence computation.
        """
        torch = _ensure_torch()
        cm = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
        cm = cm.to(dtype=torch.bool).reshape(-1).clone()
        if forced_tokens is None:
            ft = torch.zeros(cm.shape[0], dtype=torch.long, device=cm.device)
        else:
            ft = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
            ft = ft.to(device=cm.device, dtype=torch.long).reshape(-1).clone()
            if int(ft.numel()) < int(cm.numel()):
                pad_n = int(cm.numel()) - int(ft.numel())
                ft = torch.cat([ft, torch.zeros(pad_n, dtype=torch.long, device=cm.device)], dim=0)
            if int(ft.numel()) > int(cm.numel()):
                ft = ft[: int(cm.numel())]
        cm[pos] = True
        ft[pos] = int(token_id)
        return self.step(latents, cm, ft)


class PlaceholderDiffusionSampler(SamplerAdapter):
    """
    Placeholder sampler for testing and local runs.
    Returns deterministic or pseudo-random logits; respects committed_mask and forced_tokens.
    """

    def __init__(
        self,
        seq_len: int = 64,
        vocab_size: int = 32000,
        random_seed: int = 42,
        deterministic: bool = False,
        device: str = "cpu",
    ) -> None:
        torch = _ensure_torch()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.eot_token_id = None
        try:
            self._rng = torch.Generator(device=self.device)
            self._rng_device = self.device
        except Exception:
            self._rng = torch.Generator(device="cpu")
            self._rng_device = torch.device("cpu")
        self._rng.manual_seed(int(random_seed))
        self._random_seed = int(random_seed)
        self._deterministic = deterministic
        self._step_count = 0

    def step(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
    ) -> Any:
        logits, _ = self.step_with_aux(latents, committed_mask, forced_tokens)
        return logits

    def step_with_aux(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
    ) -> tuple[Any, dict]:
        torch = _ensure_torch()
        self._step_count += 1
        if self._deterministic:
            self._rng.manual_seed(int(self._random_seed + self._step_count))
        logits = torch.randn(
            (self.seq_len, self.vocab_size),
            generator=self._rng,
            dtype=torch.float32,
            device=self._rng_device,
        )
        if logits.device != self.device:
            logits = logits.to(device=self.device)
        logits = logits * 0.5

        cm_t = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
        cm_t = cm_t.to(device=self.device, dtype=torch.bool).reshape(-1)[: self.seq_len]
        if int(cm_t.numel()) < self.seq_len:
            cm_t = torch.cat(
                [cm_t, torch.zeros(self.seq_len - int(cm_t.numel()), dtype=torch.bool, device=self.device)], dim=0
            )

        if forced_tokens is None:
            ft_t = torch.zeros(self.seq_len, dtype=torch.long, device=self.device)
        else:
            ft_t = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
            ft_t = ft_t.to(device=self.device, dtype=torch.long).reshape(-1)[: self.seq_len]
            if int(ft_t.numel()) < self.seq_len:
                ft_t = torch.cat(
                    [ft_t, torch.zeros(self.seq_len - int(ft_t.numel()), dtype=torch.long, device=self.device)], dim=0
                )

        if int(cm_t.sum().item()) > 0:
            idx = torch.nonzero(cm_t, as_tuple=False).flatten()
            logits[idx] = -10.0
            logits[idx, ft_t[idx]] = 10.0

        # Synthetic row-normalized attention proxy for local tests.
        base = torch.linspace(1.0, 2.0, steps=self.seq_len, device=self.device, dtype=torch.float32)
        attn = base.unsqueeze(0).repeat(self.seq_len, 1)
        committed_idx = torch.nonzero(cm_t, as_tuple=False).flatten()
        if int(committed_idx.numel()) > 0:
            attn[:, committed_idx] = attn[:, committed_idx] * 2.0
        diag = torch.arange(self.seq_len, device=self.device)
        attn[diag, diag] = attn[diag, diag] * 0.5
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)
        return logits, {"last_attention": attn}
