"""Tests for PF time-window resolution and membership checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoding.config import (
    DecoderConfig,
    is_in_pf_time_window,
    resolve_pf_particles_for_t,
    resolve_pf_time_window,
)


def test_resolve_pf_time_window_proportional_96():
    cfg = DecoderConfig(
        pf_time_window_mode="proportional",
        pf_time_window_start=30,
        pf_time_window_end=70,
        pf_time_window_ref_steps=96,
    )
    assert resolve_pf_time_window(total_steps=96, cfg=cfg) == (30, 70)


def test_resolve_pf_time_window_proportional_128():
    cfg = DecoderConfig(
        pf_time_window_mode="proportional",
        pf_time_window_start=30,
        pf_time_window_end=70,
        pf_time_window_ref_steps=96,
    )
    assert resolve_pf_time_window(total_steps=128, cfg=cfg) == (40, 93)


def test_resolve_pf_time_window_absolute_with_swap_and_clamp():
    cfg = DecoderConfig(
        pf_time_window_mode="absolute",
        pf_time_window_start=90,
        pf_time_window_end=10,
    )
    assert resolve_pf_time_window(total_steps=50, cfg=cfg) == (10, 50)


def test_is_in_pf_time_window_closed_interval():
    cfg = DecoderConfig(
        pf_time_window_mode="proportional",
        pf_time_window_start=30,
        pf_time_window_end=70,
        pf_time_window_ref_steps=96,
    )
    assert is_in_pf_time_window(t_remaining=30, total_steps=96, cfg=cfg) is True
    assert is_in_pf_time_window(t_remaining=70, total_steps=96, cfg=cfg) is True
    assert is_in_pf_time_window(t_remaining=29, total_steps=96, cfg=cfg) is False
    assert is_in_pf_time_window(t_remaining=71, total_steps=96, cfg=cfg) is False


def test_pf_time_window_mode_falls_back_to_proportional():
    cfg = DecoderConfig(pf_time_window_mode="not-a-mode")
    assert cfg.pf_time_window_mode == "proportional"


def test_pf_cooldown_min_is_enforced():
    cfg = DecoderConfig(pf_cooldown_steps=0)
    assert cfg.pf_cooldown_steps >= 3


def test_resolve_pf_particles_for_t_time_linear_schedule():
    cfg = DecoderConfig(
        pf_particles=8,
        pf_particles_min=2,
        pf_particles_schedule="time_linear",
        pf_time_window_mode="proportional",
        pf_time_window_start=30,
        pf_time_window_end=70,
        pf_time_window_ref_steps=96,
    )
    assert resolve_pf_particles_for_t(t_remaining=70, total_steps=96, cfg=cfg) == 8
    assert resolve_pf_particles_for_t(t_remaining=30, total_steps=96, cfg=cfg) == 2
    mid = resolve_pf_particles_for_t(t_remaining=50, total_steps=96, cfg=cfg)
    assert 2 <= mid <= 8
