"""Dream official diffusion baseline, shadow, and PF runtime entry points."""

from ._runtime import (
    _DreamBranchState,
    _advance_dream_branch_trials,
    _apply_completion_remask,
    _apply_completion_token_overrides,
    _build_dream_noop_stats,
    _build_eb_transfer_allowed_mask,
    _build_joint_branch_candidates,
    _compose_branch_completion_state,
    _dream_apply_sampling_step,
    _dream_forward_step_logits,
    _dream_sample_mask_logits,
    _dream_sample_mask_logits_chunked,
    _dream_top_k_logits,
    _dream_top_p_logits,
    _is_dream_noop_config,
    _merge_completion_transfer_block_mask,
    _prepare_dream_diffusion_state,
    _prune_dream_branch_bank,
    _restore_rng_state,
    _run_baseline_with_sampler,
    _select_best_dream_branch_entry,
    _set_global_seed,
    _should_extend_dream_branches,
    run_baseline,
    run_dream_official_baseline,
    run_dream_official_pf,
    run_dream_official_shadow,
)

__all__ = [name for name in globals() if not name.startswith("__")]
