"""
Dream sampler adapter: wraps Dream-style diffusion model forward into
step(latents, committed_mask, forced_tokens) -> logits.
"""
from typing import Any, Optional

import numpy as np

from .sampler_adapter import SamplerAdapter


def _ensure_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError("Dream backend requires torch. Install with: pip install torch")


def _ensure_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer

        return AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("Dream backend requires transformers. Install with: pip install transformers")


class DreamSamplerAdapter(SamplerAdapter):
    """
    SamplerAdapter for Dream mask-prediction forward.

    For each step:
      - completion positions are initialized as [MASK];
      - committed positions are forced with chosen token ids;
      - model returns per-position logits for completion slice.
    """

    def __init__(
        self,
        prompt: str,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        seq_len: int = 256,
        max_prompt_tokens: int = 512,
        mask_id: Optional[int] = None,
        use_chat_template: bool = False,
        fill_uncommitted_with_latent_argmax: bool = False,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.max_prompt_tokens = max_prompt_tokens
        self.use_chat_template = use_chat_template
        self.fill_uncommitted_with_latent_argmax = fill_uncommitted_with_latent_argmax

        tok_mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is not None:
            self.mask_id = int(mask_id)
        elif tok_mask_id is not None and int(tok_mask_id) >= 0:
            self.mask_id = int(tok_mask_id)
        else:
            raise ValueError(
                "Dream tokenizer has no mask_token_id; provide --dream_mask_id for step decoding."
            )

        self.vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.eot_token_id = int(eos_token_id) if eos_token_id is not None and int(eos_token_id) >= 0 else None
        self._prompt_ids: Optional[np.ndarray] = None
        self._prompt_len: int = 0

    def _get_prompt_ids(self) -> np.ndarray:
        if self._prompt_ids is not None:
            return self._prompt_ids

        if self.use_chat_template:
            try:
                messages = [{"role": "user", "content": self.prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
                ids = inputs["input_ids"].squeeze(0).numpy()
            except Exception:
                ids = self.tokenizer.encode(self.prompt, add_special_tokens=True)
                if hasattr(ids, "tolist"):
                    ids = np.array(ids)
        else:
            ids = self.tokenizer.encode(self.prompt, add_special_tokens=True)
            if hasattr(ids, "tolist"):
                ids = np.array(ids)

        if len(ids) > self.max_prompt_tokens:
            ids = ids[-self.max_prompt_tokens:]
        self._prompt_ids = np.asarray(ids, dtype=np.int64)
        self._prompt_len = len(self._prompt_ids)
        return self._prompt_ids

    def _build_completion_ids(
        self,
        seq_len: int,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Any,
    ) -> Any:
        torch = _ensure_torch()
        comp_ids = torch.full((seq_len,), int(self.mask_id), dtype=torch.long, device=self.device)
        committed = committed_mask if torch.is_tensor(committed_mask) else torch.as_tensor(committed_mask)
        committed = committed.to(device=self.device, dtype=torch.bool).reshape(-1)
        forced = forced_tokens if torch.is_tensor(forced_tokens) else torch.as_tensor(forced_tokens)
        forced = forced.to(device=self.device, dtype=torch.long).reshape(-1)

        if int(committed.numel()) < seq_len:
            committed = torch.cat(
                [committed, torch.zeros(seq_len - int(committed.numel()), dtype=torch.bool, device=self.device)], dim=0
            )
        committed = committed[:seq_len]

        if int(forced.numel()) < seq_len:
            forced = torch.cat(
                [forced, torch.zeros(seq_len - int(forced.numel()), dtype=torch.long, device=self.device)], dim=0
            )
        forced = forced[:seq_len]

        if self.fill_uncommitted_with_latent_argmax and latents is not None and hasattr(latents, "shape"):
            lat_t = latents if torch.is_tensor(latents) else torch.as_tensor(latents, device=self.device)
            if int(lat_t.ndim) >= 2 and int(lat_t.shape[0]) >= seq_len:
                latent_argmax = lat_t[:seq_len].argmax(dim=-1).to(dtype=torch.long, device=self.device)
                comp_ids[~committed] = latent_argmax[~committed]
        comp_ids[committed] = forced[committed]
        return comp_ids

    def step(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
    ) -> Any:
        logits, _ = self.step_with_aux(
            latents=latents,
            committed_mask=committed_mask,
            forced_tokens=forced_tokens,
            need_attention=False,
        )
        return logits

    def step_with_aux(
        self,
        latents: Any,
        committed_mask: Any,
        forced_tokens: Optional[Any],
        need_attention: bool = False,
    ) -> tuple[Any, dict]:
        torch = _ensure_torch()
        prompt_ids = self._get_prompt_ids()
        if committed_mask is not None and hasattr(committed_mask, "shape"):
            seq_len = min(self.seq_len, int(committed_mask.shape[0]))
        else:
            seq_len = self.seq_len
        if forced_tokens is None:
            forced_tokens = torch.zeros(seq_len, dtype=torch.long, device=self.device)

        comp_ids = self._build_completion_ids(
            seq_len=seq_len,
            latents=latents,
            committed_mask=committed_mask,
            forced_tokens=forced_tokens,
        )

        prompt_ids_t = torch.as_tensor(prompt_ids, dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_ids_t, comp_ids], dim=0)
        input_ids = full_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)

        self.model.eval()
        attn = None
        with torch.no_grad():
            out = None
            if need_attention:
                try:
                    total_len = int(input_ids.shape[-1])
                    # Dream forward with output_attentions=True indexes mask as [:, :, :, :L],
                    # so it requires a 4D mask tensor.
                    attention_mask_4d = torch.ones((1, 1, total_len, total_len), dtype=torch.bool, device=self.device)
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_4d,
                        output_attentions=True,
                    )
                except Exception:
                    out = None
            if out is None:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if int(getattr(logits, "ndim", 0)) == 3:
            logits = logits[0]
        logits = logits.float()
        attentions = getattr(out, "attentions", None) if need_attention else None

        prompt_len = self._prompt_len
        completion_logits = torch.zeros((seq_len, int(logits.shape[-1])), dtype=torch.float32, device=logits.device)
        start = prompt_len
        end = min(prompt_len + seq_len, int(logits.shape[0]))
        n = max(end - start, 0)
        if n > 0:
            completion_logits[:n] = logits[start:end]

        if attentions:
            try:
                last = attentions[-1]
                # [batch, heads, total_len, total_len] -> [heads, seq_len, seq_len]
                if int(getattr(last, "ndim", 0)) == 4:
                    last = last[0]
                last = last.float()
                attn = last[:, start:end, start:end]
            except Exception:
                attn = None

        aux = {"last_attention": attn} if attn is not None else {}
        return completion_logits, aux


def load_dream_model(
    model_path: str = "Dream-org/Dream-Coder-v0-Instruct-7B",
    device: str = "cuda",
    torch_dtype: Optional[Any] = None,
):
    """Load Dream model and tokenizer."""
    torch = _ensure_torch()
    AutoModel, AutoTokenizer = _ensure_transformers()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer
