#!/usr/bin/env python3
"""Pulse-response plots for every device the sweep visits."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no DISPLAY needed
import matplotlib.pyplot as plt

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import PowStepDevice, ExpStepDevice
from aihwkit.simulator.presets.devices import (
    EcRamPresetDevice,
    EcRamMOPresetDevice,
    ReRamArrayHfO2PresetDevice,
    ReRamArrayOMPresetDevice,
)
from aihwkit.utils.visualization import plot_response_overview

OUTDIR = Path("~/analog_stress_test/results/device_curves").expanduser()
OUTDIR.mkdir(parents=True, exist_ok=True)

N_LOOPS  = 4    # number of up/down hyper-cycles plotted
N_TRACES = 3    # device-to-device variation lines (low because we set d2d=0 for Pow/Exp)
DPI      = 120

# Device builders — mirror NeurIPS-2025/S2-mnist-FCN.py:get_RPU_device exactly.
# dw_min = 2 * tau / num_of_states.

def pow_device(tau: float, res_gamma: float, num_of_states: int) -> PowStepDevice:
    dw_min = 2.0 * tau / num_of_states
    d = PowStepDevice(
        dw_min=dw_min,
        pow_gamma_dtod=0,
        w_max=tau, w_min=-tau,
        w_max_dtod=0, w_min_dtod=0,
    )
    if res_gamma > 0:
        d.pow_gamma = res_gamma
    return d

def exp_device(tau: float, res_gamma: float, num_of_states: int) -> ExpStepDevice:
    dw_min = 2.0 * tau / num_of_states
    d = ExpStepDevice(
        dw_min=dw_min,
        w_max=tau, w_min=-tau,
        w_max_dtod=0, w_min_dtod=0,
    )
    if res_gamma > 0:
        d.gamma_up = res_gamma
        d.gamma_down = res_gamma
    return d

def save_plot(device, name: str, title: str = "") -> None:
    rpu_config = SingleRPUConfig(device=device)
    plt.figure(figsize=(10, 8))
    try:
        plot_response_overview(rpu_config, n_loops=N_LOOPS, n_traces=N_TRACES)
    except Exception as exc:
        print(f"[WARN] {name}: plot_response_overview failed: {exc}")
        plt.close("all")
        return
    if title:
        plt.suptitle(title, y=1.02, fontsize=12)
    out = OUTDIR / f"{name}.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  -> {out}")

# Sweeps

print("=== Axis 1 — Pow granularity sweep (tau=0.7, rg=0.5) ===")
for rs in [14000, 1400, 280, 140, 28, 14]:
    d = pow_device(tau=0.7, res_gamma=0.5, num_of_states=rs)
    save_plot(d, f"axis1_pow_tau0.7_rg0.5_rs{rs}",
              f"Axis 1 — Pow  tau=0.7  res-gamma=0.5  res-state={rs}")

print("=== Axis 2 — Pow response curve sweep (tau=0.7, rs=140) ===")
for rg in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
    d = pow_device(tau=0.7, res_gamma=rg, num_of_states=140)
    save_plot(d, f"axis2_pow_tau0.7_rg{rg}_rs140",
              f"Axis 2 — Pow  tau=0.7  res-gamma={rg}  res-state=140")

print("=== Axis 3 — measurement-based device presets ===")
PRESETS = [
    ("HfO2",    ReRamArrayHfO2PresetDevice),
    ("EcRam",   EcRamPresetDevice),
    ("EcRamMO", EcRamMOPresetDevice),
    ("OM",      ReRamArrayOMPresetDevice),
]
for name, cls in PRESETS:
    save_plot(cls(), f"axis3_preset_{name}", f"Axis 3 — Preset: {name}")

print(f"\nAll plots saved under: {OUTDIR}")
