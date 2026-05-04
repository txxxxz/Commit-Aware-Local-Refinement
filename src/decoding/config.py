"""Decoder configuration for risk-aware delayed commit + local PF."""
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

NormalizeMode = Literal["running_stats", "fixed_scale"]
InfluenceApproxMode = Literal["full", "sample_positions", "single_forward"]
InfluencePerturbMode = Literal["argmax_onehot", "top2_mass"]
FallbackPolicy = Literal["best_particle", "max_prob"]
RiskBand = Literal["low", "mid", "high"]
RiskFusionMode = Literal["weighted_sum", "independent_triggers", "entropy_only"]
PFTimeWindowMode = Literal["proportional", "absolute"]
PFParticleSchedule = Literal["fixed", "time_linear"]
LocalBeamMode = Literal[
    "entropy_only",
    "kl_only",
    "entropy_kl",
    "entropy_kl_struct",
    "beam",
    "delay_only",
]


@dataclass
class BudgetTier:
    """PF budget tier used by risk-band adaptive decoding."""

    pf_top_k: int = 1
    pf_particles: int = 1
    pf_horizon_steps: int = 0


@dataclass
class DecoderConfig:
    # --- Entropy ---
    entropy_threshold: float = 0.05
    entropy_windowing: Tuple[float, float] = (0.2, 2.5)
    commit_entropy_floor: float = 0.3
    entropy_scale: float = 2.0

    # Backward-compatible aliases
    entropy_low: Optional[float] = None
    entropy_high: Optional[float] = None
    entropy_commit_floor: Optional[float] = None

    # --- Influence ---
    influence_enabled: bool = False
    use_attention_influence_proxy: bool = True
    influence_eps: float = 0.1
    influence_positions_sample_ratio: float = 0.3
    influence_approx_mode: InfluenceApproxMode = "sample_positions"
    influence_perturb_mode: InfluencePerturbMode = "argmax_onehot"
    influence_scale: float = 1.0
    influence_top_k: int = 1

    # --- Risk ---
    risk_weights: Tuple[float, float] = (1.0, 0.0)  # (w_entropy, w_influence)
    risk_beta: float = 1.0
    risk_threshold: float = 0.6
    risk_low_quantile: float = 0.25
    risk_high_quantile: float = 0.75
    risk_low_threshold: Optional[float] = None
    risk_high_threshold: Optional[float] = None
    risk_normalize: NormalizeMode = "fixed_scale"
    risk_fusion_mode: RiskFusionMode = "entropy_only"
    risk_renormalize_available: bool = True
    entropy_norm_range: Optional[Tuple[float, float]] = None
    influence_norm_range: Optional[Tuple[float, float]] = None
    risk_threshold_min_gap: float = 0.02
    influence_trigger_floor: float = 0.0

    # --- Dynamic valid mask ---
    dynamic_valid_mask_enabled: bool = True
    valid_mask_max_prob: float = 1.0
    valid_mask_exclude_eot: bool = True

    # Backward-compatible aliases
    w_entropy: Optional[float] = 1.0
    w_influence: Optional[float] = 0.0

    # --- PF ---
    pf_enabled: bool = True
    pf_top_k: int = 5
    pf_particles: int = 4
    pf_particles_min: int = 1
    pf_particles_schedule: PFParticleSchedule = "time_linear"
    pf_horizon_steps: int = 3
    pf_rep_lambda: float = 1.0
    pf_syntax_reward: float = 2.0
    pf_repetition_ngram: int = 3
    pf_win_margin: float = 0.2
    pf_steps_before_commit: int = 2
    pf_badness_beta: float = 1.0
    pf_stability_weight: float = 0.5
    correctness_signal_mode: str = "none"
    counterfactual_rollout_steps: int = 1
    pf_budget_mode: str = "legacy"
    pf_extra_forward_budget: int = 0
    pf_acceptance_tolerance: float = 0.02
    pf_max_triggers_per_sample: int = 1
    pf_parse_fail_penalty: float = 8.0
    pf_do_no_harm_enabled: bool = True
    pf_do_no_harm_margin: float = 0.2
    pf_do_no_harm_min_quality_gain: float = 0.05

    # --- Time-aware scheduling ---
    diffusion_mid_step: int = 3
    pf_phase_ratio: float = 0.7
    pf_cooldown_steps: int = 3
    pf_risk_gradient_sigma: float = 1.5
    joint_gate_quantile: float = 0.75
    pf_trigger_quantile: float = 0.75
    pf_time_window_mode: PFTimeWindowMode = "proportional"
    pf_time_window_start: int = 30
    pf_time_window_end: int = 70
    pf_time_window_ref_steps: int = 96
    parser_feedback_enabled: bool = True
    parser_feedback_min_prefix_chars: int = 24
    parser_feedback_window_radius: int = 2
    parser_feedback_hotspot_threshold: float = 0.1
    parser_feedback_gate_scale: float = 0.85

    # --- RDD-style rollback/remasking ---
    rdd_rollback_enabled: bool = False
    rdd_rollback_window: int = 8
    rdd_rollback_max_events: int = 2
    rdd_rollback_min_severity: float = 0.25
    rdd_rollback_cooldown_steps: int = 2

    # --- Branching ---
    beam_width: int = 3
    max_branch_positions_per_step: int = 8
    branch_mid_candidates: int = 2
    branch_high_candidates: int = 3
    branch_parse_weight: float = 2.0
    branch_entropy_weight: float = 0.2
    branch_commit_margin: float = 0.2
    branch_trial_steps: int = 3
    branch_extension_enabled: bool = False
    branch_extension_steps: int = 2
    branch_extension_margin: float = 0.4
    branch_max_extensions: int = 1
    joint_fork_enabled: bool = False
    joint_fork_positions: int = 2

    # --- Adaptive budget ---
    adaptive_budget_enabled: bool = True
    overhead_cap_ratio: float = 3.0
    budget_tier_low: BudgetTier = field(default_factory=lambda: BudgetTier(1, 1, 0))
    budget_tier_mid: BudgetTier = field(default_factory=lambda: BudgetTier(2, 2, 1))
    budget_tier_high: BudgetTier = field(default_factory=lambda: BudgetTier(3, 3, 1))

    # --- Delayed commit ---
    max_delay_steps: int = 5
    fallback_policy: FallbackPolicy = "best_particle"
    delay_commit_enabled: bool = True
    step_commit_ratio: float = 0.35
    step_commit_min: int = 1
    argmax_commit_min_prob: float = 0.0
    argmax_commit_min_margin: float = 0.0

    # --- EB-style entropy-bounded unmasking ---
    eb_sampler_enabled: bool = False
    eb_entropy_quantile: float = 0.35
    eb_min_commit_per_step: int = 1
    eb_structure_entropy_scale: float = 0.65
    eb_signature_entropy_scale: float = 0.5
    eb_syntax_near_entropy_scale: float = 0.5
    eb_near_radius: int = 2

    # --- Commit-timing-aware local beam ---
    local_beam_enabled: bool = False
    local_beam_mode: LocalBeamMode = "entropy_kl_struct"
    local_beam_size: int = 4
    local_beam_top_k: int = 5
    local_beam_horizon: int = 2
    local_beam_max_events: int = 1
    local_beam_top_m: int = 1
    local_beam_tau_entropy: float = 0.45
    local_beam_tau_kl: float = 0.8
    local_beam_tau_risk: float = 0.45
    local_beam_lambda_kl: float = 1.0
    local_beam_struct_weight: float = 0.75
    local_beam_margin_base: float = 0.15
    local_beam_margin_branch: float = 0.05
    local_beam_entropy_top_k: int = 64
    local_beam_kl_top_k: int = 32
    local_beam_use_visible_tests: bool = False
    local_beam_preserve_baseline: bool = True
    entropy_kl_logging_enabled: bool = False
    shadow_mode_enabled: bool = False
    shadow_top_m: int = 3
    shadow_max_events: int = 8
    shadow_risk_threshold: float = 0.45
    shadow_commit_lag_window: int = 3
    shadow_entropy_top_k: int = 16
    shadow_kl_top_k: int = 8
    shadow_token_top_k: int = 5
    branch_observe_enabled: bool = False
    branch_observe_beam_size: int = 3
    branch_observe_top_k: int = 3
    branch_observe_max_events: int = 1
    branch_observe_horizon: int = 2
    branch_observe_trigger_mode: str = "auto"
    branch_observe_event_policy: str = "top_risk"
    branch_observe_include_delay: bool = True
    branch_observe_token_fallback: bool = True
    branch_observe_force_baseline_output: bool = True
    branch_select_enabled: bool = False
    branch_select_verifier: str = "level0"
    branch_select_require_baseline_failure: bool = True
    branch_select_min_score_gain: float = 1.0
    branch_select_visible_min_pass_gain: int = 1
    branch_select_visible_require_level0: bool = True

    # --- Structure checks ---
    parsing_checks_enabled: bool = True
    language: str = "python"

    # --- Reproducibility & logging ---
    random_seed: int = 42
    logging_level: str = "INFO"

    def __post_init__(self) -> None:
        # Canonicalize entropy aliases.
        low, high = self.entropy_windowing
        if self.entropy_low is not None:
            low = self.entropy_low
        if self.entropy_high is not None:
            high = self.entropy_high
        self.entropy_windowing = (float(low), float(high))
        self.entropy_low = self.entropy_windowing[0]
        self.entropy_high = self.entropy_windowing[1]

        if self.entropy_commit_floor is not None:
            self.commit_entropy_floor = float(self.entropy_commit_floor)
        self.entropy_commit_floor = self.commit_entropy_floor

        # Canonicalize risk-weight aliases.
        w_e, w_i = self.risk_weights
        if self.w_entropy is not None:
            w_e = self.w_entropy
        if self.w_influence is not None:
            w_i = self.w_influence
        self.risk_weights = (float(w_e), float(w_i))
        self.w_entropy = self.risk_weights[0]
        self.w_influence = self.risk_weights[1]
        self.risk_beta = max(float(self.risk_beta), 0.0)

        # Calibrate dual-threshold defaults from legacy single threshold.
        if self.risk_high_threshold is None:
            self.risk_high_threshold = float(self.risk_threshold)
        if self.risk_low_threshold is None:
            self.risk_low_threshold = float(self.risk_high_threshold * 0.67)
        if self.risk_low_threshold > self.risk_high_threshold:
            self.risk_low_threshold, self.risk_high_threshold = (
                self.risk_high_threshold,
                self.risk_low_threshold,
            )

        # Ensure numeric ranges are sane.
        self.risk_low_quantile = float(np.clip(self.risk_low_quantile, 0.0, 1.0))
        self.risk_high_quantile = float(np.clip(self.risk_high_quantile, 0.0, 1.0))
        if self.risk_low_quantile > self.risk_high_quantile:
            self.risk_low_quantile, self.risk_high_quantile = (
                self.risk_high_quantile,
                self.risk_low_quantile,
            )

        self.beam_width = max(int(self.beam_width), 1)
        self.max_branch_positions_per_step = max(int(self.max_branch_positions_per_step), 1)
        self.branch_mid_candidates = max(int(self.branch_mid_candidates), 1)
        self.branch_high_candidates = max(int(self.branch_high_candidates), 1)
        self.branch_commit_margin = max(float(self.branch_commit_margin), 0.0)
        self.branch_trial_steps = max(int(self.branch_trial_steps), 1)
        self.branch_extension_steps = max(int(self.branch_extension_steps), 1)
        self.branch_extension_margin = max(float(self.branch_extension_margin), 0.0)
        self.branch_max_extensions = max(int(self.branch_max_extensions), 0)
        self.joint_fork_positions = max(int(self.joint_fork_positions), 1)
        self.overhead_cap_ratio = max(float(self.overhead_cap_ratio), 1.0)
        self.risk_threshold_min_gap = max(float(self.risk_threshold_min_gap), 1e-6)
        self.valid_mask_max_prob = float(np.clip(self.valid_mask_max_prob, 0.0, 1.0))
        self.influence_trigger_floor = max(float(self.influence_trigger_floor), 0.0)
        self.influence_top_k = max(int(self.influence_top_k), 1)
        self.diffusion_mid_step = max(int(self.diffusion_mid_step), 1)
        self.pf_phase_ratio = float(np.clip(self.pf_phase_ratio, 0.0, 1.0))
        # Force cooldown to be at least 3 so PF decisions have time to settle.
        self.pf_cooldown_steps = max(int(self.pf_cooldown_steps), 3)
        self.pf_risk_gradient_sigma = max(float(self.pf_risk_gradient_sigma), 0.0)
        self.joint_gate_quantile = float(np.clip(self.joint_gate_quantile, 0.0, 1.0))
        self.pf_trigger_quantile = float(np.clip(self.pf_trigger_quantile, 0.0, 1.0))
        if self.pf_trigger_quantile < self.risk_high_quantile:
            self.pf_trigger_quantile = float(self.risk_high_quantile)
        mode = str(self.pf_time_window_mode).lower()
        if mode not in {"proportional", "absolute"}:
            mode = "proportional"
        self.pf_time_window_mode = mode
        self.pf_time_window_ref_steps = max(int(self.pf_time_window_ref_steps), 1)
        self.pf_time_window_start = max(int(self.pf_time_window_start), 1)
        self.pf_time_window_end = max(int(self.pf_time_window_end), 1)
        if self.pf_time_window_start > self.pf_time_window_end:
            self.pf_time_window_start, self.pf_time_window_end = (
                self.pf_time_window_end,
                self.pf_time_window_start,
            )
        self.parser_feedback_min_prefix_chars = max(int(self.parser_feedback_min_prefix_chars), 0)
        self.parser_feedback_window_radius = max(int(self.parser_feedback_window_radius), 0)
        self.parser_feedback_hotspot_threshold = max(float(self.parser_feedback_hotspot_threshold), 0.0)
        self.parser_feedback_gate_scale = float(np.clip(self.parser_feedback_gate_scale, 0.1, 1.0))
        self.rdd_rollback_window = max(int(self.rdd_rollback_window), 1)
        self.rdd_rollback_max_events = max(int(self.rdd_rollback_max_events), 0)
        self.rdd_rollback_min_severity = float(np.clip(self.rdd_rollback_min_severity, 0.0, 1.0))
        self.rdd_rollback_cooldown_steps = max(int(self.rdd_rollback_cooldown_steps), 0)

        self.pf_top_k = max(int(self.pf_top_k), 1)
        self.pf_particles = max(int(self.pf_particles), 1)
        self.pf_particles_min = max(int(self.pf_particles_min), 1)
        if self.pf_particles_min > self.pf_particles:
            self.pf_particles_min = int(self.pf_particles)
        schedule = str(self.pf_particles_schedule).lower()
        if schedule not in {"fixed", "time_linear"}:
            schedule = "time_linear"
        self.pf_particles_schedule = schedule
        self.pf_horizon_steps = max(int(self.pf_horizon_steps), 0)
        self.pf_rep_lambda = max(float(self.pf_rep_lambda), 0.0)
        self.pf_syntax_reward = float(self.pf_syntax_reward)
        self.pf_stability_weight = max(float(self.pf_stability_weight), 0.0)
        self.pf_repetition_ngram = max(int(self.pf_repetition_ngram), 2)
        mode_secondary = str(self.correctness_signal_mode or "none").lower()
        if mode_secondary not in {"none", "counterfactual_gain", "constraint_identifier"}:
            mode_secondary = "none"
        self.correctness_signal_mode = mode_secondary
        self.counterfactual_rollout_steps = max(int(self.counterfactual_rollout_steps), 1)
        mode_budget = str(self.pf_budget_mode or "legacy").lower()
        if mode_budget not in {"legacy", "budgeted_entropy"}:
            mode_budget = "legacy"
        self.pf_budget_mode = mode_budget
        self.pf_extra_forward_budget = max(int(self.pf_extra_forward_budget), 0)
        self.pf_acceptance_tolerance = max(float(self.pf_acceptance_tolerance), 0.0)
        self.pf_max_triggers_per_sample = max(int(self.pf_max_triggers_per_sample), 0)
        self.pf_parse_fail_penalty = max(float(self.pf_parse_fail_penalty), 0.0)
        self.pf_do_no_harm_margin = max(float(self.pf_do_no_harm_margin), 0.0)
        self.pf_do_no_harm_min_quality_gain = max(float(self.pf_do_no_harm_min_quality_gain), 0.0)
        # Backward-compatible alias: legacy pf_badness_beta now maps to pf_rep_lambda.
        if abs(self.pf_rep_lambda - 1.0) < 1e-12:
            self.pf_rep_lambda = max(float(self.pf_badness_beta), 0.0)
        self.pf_badness_beta = float(self.pf_rep_lambda)
        self.step_commit_min = max(int(self.step_commit_min), 0)
        self.argmax_commit_min_prob = float(np.clip(self.argmax_commit_min_prob, 0.0, 1.0))
        self.argmax_commit_min_margin = max(float(self.argmax_commit_min_margin), 0.0)
        self.eb_entropy_quantile = float(np.clip(self.eb_entropy_quantile, 0.0, 1.0))
        self.eb_min_commit_per_step = max(int(self.eb_min_commit_per_step), 0)
        self.eb_structure_entropy_scale = float(np.clip(self.eb_structure_entropy_scale, 0.05, 1.0))
        self.eb_signature_entropy_scale = float(np.clip(self.eb_signature_entropy_scale, 0.05, 1.0))
        self.eb_syntax_near_entropy_scale = float(np.clip(self.eb_syntax_near_entropy_scale, 0.05, 1.0))
        self.eb_near_radius = max(int(self.eb_near_radius), 0)
        local_mode = str(self.local_beam_mode or "entropy_kl_struct").lower()
        if local_mode not in {
            "entropy_only",
            "kl_only",
            "entropy_kl",
            "entropy_kl_struct",
            "beam",
            "delay_only",
        }:
            local_mode = "entropy_kl_struct"
        self.local_beam_mode = local_mode
        self.local_beam_size = max(int(self.local_beam_size), 1)
        self.local_beam_top_k = max(int(self.local_beam_top_k), 1)
        self.local_beam_horizon = max(int(self.local_beam_horizon), 0)
        self.local_beam_max_events = max(int(self.local_beam_max_events), 0)
        self.local_beam_top_m = max(int(self.local_beam_top_m), 1)
        self.local_beam_tau_entropy = float(np.clip(self.local_beam_tau_entropy, 0.0, 1.0))
        self.local_beam_tau_kl = float(np.clip(self.local_beam_tau_kl, 0.0, 1.0))
        self.local_beam_tau_risk = max(float(self.local_beam_tau_risk), 0.0)
        self.local_beam_lambda_kl = max(float(self.local_beam_lambda_kl), 0.0)
        self.local_beam_struct_weight = max(float(self.local_beam_struct_weight), 0.0)
        self.local_beam_margin_base = max(float(self.local_beam_margin_base), 0.0)
        self.local_beam_margin_branch = max(float(self.local_beam_margin_branch), 0.0)
        self.local_beam_entropy_top_k = max(int(self.local_beam_entropy_top_k), 2)
        self.local_beam_kl_top_k = max(int(self.local_beam_kl_top_k), 1)
        self.shadow_top_m = max(int(self.shadow_top_m), 1)
        self.shadow_max_events = max(int(self.shadow_max_events), 0)
        self.shadow_risk_threshold = max(float(self.shadow_risk_threshold), 0.0)
        self.shadow_commit_lag_window = max(int(self.shadow_commit_lag_window), 0)
        self.shadow_entropy_top_k = max(int(self.shadow_entropy_top_k), 2)
        self.shadow_kl_top_k = max(int(self.shadow_kl_top_k), 1)
        self.shadow_token_top_k = max(int(self.shadow_token_top_k), 1)
        self.branch_observe_beam_size = max(int(self.branch_observe_beam_size), 1)
        self.branch_observe_top_k = max(int(self.branch_observe_top_k), 0)
        self.branch_observe_max_events = max(int(self.branch_observe_max_events), 0)
        self.branch_observe_horizon = max(int(self.branch_observe_horizon), 1)
        trigger_mode = str(self.branch_observe_trigger_mode or "auto").lower()
        if trigger_mode not in {
            "auto",
            "legacy_entropy_kl_struct",
            "entropy_only",
            "kl_only",
            "entropy_kl",
            "entropy_kl_struct",
            "entropy_struct",
            "kl_struct",
        }:
            trigger_mode = "auto"
        self.branch_observe_trigger_mode = trigger_mode
        event_policy = str(self.branch_observe_event_policy or "top_risk").lower()
        if event_policy not in {"top_risk", "random_masked", "random_structural", "highest_conf_structural"}:
            event_policy = "top_risk"
        self.branch_observe_event_policy = event_policy
        verifier = str(self.branch_select_verifier or "level0").lower()
        if verifier not in {"level0", "level1", "oracle"}:
            verifier = "level0"
        self.branch_select_verifier = verifier
        self.branch_select_min_score_gain = max(float(self.branch_select_min_score_gain), 0.0)
        self.branch_select_visible_min_pass_gain = max(int(self.branch_select_visible_min_pass_gain), 1)

        if self.risk_normalize == "running_stats":
            self._entropy_mean = 0.0
            self._entropy_m2 = 0.0
            self._influence_mean = 0.0
            self._influence_m2 = 0.0
            self._n_stats = 0


def fast_baseline() -> DecoderConfig:
    """Preset: influence off, PF off, for quick baseline runs."""
    return DecoderConfig(
        influence_enabled=False,
        pf_enabled=False,
        w_influence=0.0,
        delay_commit_enabled=False,
        logging_level="WARNING",
    )


def ablation_no_pf() -> DecoderConfig:
    """(1) Risk only, no PF."""
    return DecoderConfig(pf_enabled=False)


def ablation_entropy_only() -> DecoderConfig:
    """(2) Entropy-only risk."""
    return DecoderConfig(influence_enabled=False, w_entropy=1.0, w_influence=0.0, risk_fusion_mode="entropy_only")


def ablation_influence_only() -> DecoderConfig:
    """(3) Influence-only risk."""
    return DecoderConfig(w_entropy=0.0)


def ablation_pf_no_delay() -> DecoderConfig:
    """(4) PF but no delayed commit (allow commit every step)."""
    return DecoderConfig(delay_commit_enabled=False, risk_threshold=2.0)


def ablation_delay_no_pf() -> DecoderConfig:
    """(5) Delayed commit but no PF."""
    return DecoderConfig(pf_enabled=False, delay_commit_enabled=True)


def ablation_local_beam(mode: LocalBeamMode = "entropy_kl_struct") -> DecoderConfig:
    """Commit-timing local beam ablation preset."""
    return DecoderConfig(
        pf_enabled=False,
        influence_enabled=False,
        delay_commit_enabled=True,
        local_beam_enabled=True,
        local_beam_mode=mode,
        local_beam_max_events=1,
        local_beam_size=4,
        local_beam_horizon=2,
    )


def resolve_pf_time_window(total_steps: int, cfg: DecoderConfig) -> Tuple[int, int]:
    """Resolve PF-allowed t-window [start, end] (inclusive) for this run."""
    total = max(int(total_steps), 1)
    mode = str(getattr(cfg, "pf_time_window_mode", "proportional")).lower()
    start = max(int(getattr(cfg, "pf_time_window_start", 30)), 1)
    end = max(int(getattr(cfg, "pf_time_window_end", 70)), 1)
    if start > end:
        start, end = end, start

    if mode == "absolute":
        t_start, t_end = int(start), int(end)
    else:
        ref_steps = max(int(getattr(cfg, "pf_time_window_ref_steps", 96)), 1)
        t_start = int(round(float(total) * (float(start) / float(ref_steps))))
        t_end = int(round(float(total) * (float(end) / float(ref_steps))))

    t_start = min(max(int(t_start), 1), total)
    t_end = min(max(int(t_end), 1), total)
    if t_start > t_end:
        t_start, t_end = t_end, t_start
    return t_start, t_end


def is_in_pf_time_window(t_remaining: int, total_steps: int, cfg: DecoderConfig) -> bool:
    """Return True when current t is inside PF-allowed window."""
    t_start, t_end = resolve_pf_time_window(total_steps=total_steps, cfg=cfg)
    t_now = int(t_remaining)
    return bool(t_start <= t_now <= t_end)


def resolve_pf_particles_for_t(
    t_remaining: int,
    total_steps: int,
    cfg: DecoderConfig,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> int:
    """
    Resolve PF particle count for current t.
    - fixed: always use cfg.pf_particles
    - time_linear: use more particles at higher t in the PF window, fewer near window start.
    """
    p_max = max(int(getattr(cfg, "pf_particles", 1)), 1)
    p_min = max(int(getattr(cfg, "pf_particles_min", 1)), 1)
    if p_min > p_max:
        p_min = p_max

    schedule = str(getattr(cfg, "pf_particles_schedule", "time_linear")).lower()
    if schedule != "time_linear":
        return int(p_max)

    if t_start is None or t_end is None:
        t_start, t_end = resolve_pf_time_window(total_steps=total_steps, cfg=cfg)
    lo = int(min(t_start, t_end))
    hi = int(max(t_start, t_end))
    if hi <= lo:
        return int(p_min)

    t_now = float(np.clip(float(t_remaining), float(lo), float(hi)))
    ratio = (t_now - float(lo)) / float(hi - lo)
    particles = int(round(float(p_min) + ratio * float(p_max - p_min)))
    return int(min(max(particles, p_min), p_max))
