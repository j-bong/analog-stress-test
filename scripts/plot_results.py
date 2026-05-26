#!/usr/bin/env python3
"""Aggregate sweep logs and produce per-axis plots."""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_ROOT = Path("~/analog_stress_test/logs").expanduser()
OUT_ROOT = Path("~/analog_stress_test/results/plots").expanduser()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALGOS = ["Analog_SGD", "TT-v1", "TT-v2", "RL-v2"]
ALGO_COLOR = {
    "Analog_SGD": "#d62728",  # red
    "TT-v1":      "#1f77b4",  # blue
    "TT-v2":      "#2ca02c",  # green
    "RL-v2":      "#9467bd",  # purple
}
ALGO_MARKER = {
    "Analog_SGD": "o",
    "TT-v1":      "s",
    "TT-v2":      "^",
    "RL-v2":      "D",
}
ALGO_LABEL = {
    "Analog_SGD": "Analog SGD",
    "TT-v1":      "TT-v1",
    "TT-v2":      "TT-v2",
    "RL-v2":      "RL-v2",
}

# Axis parameter orderings
RS_VALUES  = [14000, 1400, 280, 140, 28, 14]
RG_VALUES  = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
PRESETS    = ["HfO2", "EcRam", "EcRamMO", "OM"]
IO_GRIDS   = ["io_default", "io_6b_mild", "io_4b_extreme"]

EPOCH_RE = re.compile(r"Test Accuracy:\s*([0-9.]+)")
DONE_MARKER = "Training Time"

def parse_final_accuracy(log_path: Path):
    if not log_path.exists():
        return None
    has_done = False
    last_acc = None
    with open(log_path, errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                try:
                    last_acc = float(m.group(1))
                except ValueError:
                    pass
            if DONE_MARKER in line:
                has_done = True
    return last_acc if has_done else None

def collect_axis1(model: str):
    """Returns {algo: {rs: [accuracies across seeds]}}"""
    out = defaultdict(lambda: defaultdict(list))
    base = LOG_ROOT / f"{model}_sweep" / "axis1_granularity"
    for algo in ALGOS:
        for rs in RS_VALUES:
            for s in [1, 2, 3]:
                # name pattern: pow_rs{rs}_tau0.7_rg0.5_seed{s}
                log = base / algo / f"pow_rs{rs}_tau0.7_rg0.5_seed{s}.log"
                acc = parse_final_accuracy(log)
                if acc is not None:
                    out[algo][rs].append(acc)
    return out

def collect_axis2(model: str):
    """Returns {algo: {rg: [accuracies]}}"""
    out = defaultdict(lambda: defaultdict(list))
    base = LOG_ROOT / f"{model}_sweep" / "axis2_pow_response"
    for algo in ALGOS:
        for rg in RG_VALUES:
            for s in [1, 2, 3]:
                log = base / algo / f"pow_rg{rg}_tau0.7_rs140_seed{s}.log"
                acc = parse_final_accuracy(log)
                if acc is not None:
                    out[algo][rg].append(acc)
    return out

def collect_axis3(model: str):
    """Returns {algo: {preset: [accuracies]}}"""
    out = defaultdict(lambda: defaultdict(list))
    base = LOG_ROOT / f"{model}_sweep" / "axis3_presets"
    for algo in ALGOS:
        for preset in PRESETS:
            for s in [1, 2, 3]:
                log = base / algo / f"{preset}_seed{s}.log"
                acc = parse_final_accuracy(log)
                if acc is not None:
                    out[algo][preset].append(acc)
    return out

def collect_axis4(model: str):
    """Returns {algo: {io_grid: [accuracies]}}"""
    out = defaultdict(lambda: defaultdict(list))
    base = LOG_ROOT / f"{model}_sweep" / "axis4_io"
    for algo in ALGOS:
        for grid in IO_GRIDS:
            for s in [1, 2, 3]:
                log = base / algo / f"pow_rs140_{grid}_seed{s}.log"
                acc = parse_final_accuracy(log)
                if acc is not None:
                    out[algo][grid].append(acc)
    return out

def plot_line(ax, data, x_values, x_label, title, log_x=True):
    """Line plot: x = parameter, y = mean accuracy ± std, lines per algo."""
    for algo in ALGOS:
        means, stds, xs = [], [], []
        for x in x_values:
            accs = data.get(algo, {}).get(x, [])
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
                xs.append(x)
        if xs:
            xs = np.array(xs)
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(xs, means, yerr=stds,
                        color=ALGO_COLOR[algo], marker=ALGO_MARKER[algo],
                        markersize=7, linewidth=1.8, capsize=4,
                        label=ALGO_LABEL[algo])
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

def plot_bar(ax, data, categories, x_label, title):
    """Grouped bar chart: x = categories, bars per algo."""
    n_algo = len(ALGOS)
    n_cat = len(categories)
    width = 0.8 / n_algo
    x_pos = np.arange(n_cat)
    for i, algo in enumerate(ALGOS):
        means, stds = [], []
        for cat in categories:
            accs = data.get(algo, {}).get(cat, [])
            means.append(np.mean(accs) if accs else 0.0)
            stds.append(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
        offset = (i - (n_algo - 1) / 2) * width
        ax.bar(x_pos + offset, means, width, yerr=stds,
               color=ALGO_COLOR[algo], capsize=3,
               label=ALGO_LABEL[algo], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)

def generate_plots_for(model: str):
    """Generate all 4 single-axis plots + 1 combined panel for the given model."""
    model_upper = model.upper()
    print(f"\n=== {model_upper} ===")

    a1 = collect_axis1(model)
    a2 = collect_axis2(model)
    a3 = collect_axis3(model)
    a4 = collect_axis4(model)

    for axis_name, data, count in [("axis1", a1, sum(len(v) for d in a1.values() for v in d.values())),
                                    ("axis2", a2, sum(len(v) for d in a2.values() for v in d.values())),
                                    ("axis3", a3, sum(len(v) for d in a3.values() for v in d.values())),
                                    ("axis4", a4, sum(len(v) for d in a4.values() for v in d.values()))]:
        print(f"  {axis_name}: {count} acc datapoints loaded")

    # --- Single plots ---
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, a1, RS_VALUES, "res-state (number of conductance levels)",
              f"{model_upper} MNIST — Axis 1: Pow granularity sweep (tau=0.7, rg=0.5)")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"axis1_{model}_granularity.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, a2, RG_VALUES, "res-gamma (Pow nonlinearity)",
              f"{model_upper} MNIST — Axis 2: Pow response curve sweep (tau=0.7, rs=140)")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"axis2_{model}_pow_response.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_bar(ax, a3, PRESETS, "Measurement-based device preset",
             f"{model_upper} MNIST — Axis 3: Real device presets")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"axis3_{model}_presets.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_bar(ax, a4, IO_GRIDS, "IO quantization (default → 4b extreme)",
             f"{model_upper} MNIST — Axis 4: IO quantization (Pow base, rs=140, rg=0.5)")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"axis4_{model}_io.png", dpi=140)
    plt.close()

    # --- Combined 2x2 panel ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_line(axes[0, 0], a1, RS_VALUES, "res-state",
              "Axis 1: Pow granularity")
    plot_line(axes[0, 1], a2, RG_VALUES, "res-gamma",
              "Axis 2: Pow response (nonlinearity)")
    plot_bar(axes[1, 0], a3, PRESETS, "Preset",
             "Axis 3: Measurement device presets")
    plot_bar(axes[1, 1], a4, IO_GRIDS, "IO grid",
             "Axis 4: IO quantization")
    fig.suptitle(f"{model_upper} @ MNIST — Paper-faithful sweep summary (α=0.05, β=0.01, 3 seeds)",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / f"summary_{model}_all.png", dpi=140)
    plt.close()

    print(f"  Saved 5 PNGs to {OUT_ROOT}")

def main():
    print(f"Reading logs from {LOG_ROOT}")
    print(f"Writing plots to  {OUT_ROOT}")
    for model in ["fcn", "cnn"]:
        generate_plots_for(model)
    print("\n=== Done ===")
    print(f"Total plots: 10  ({OUT_ROOT})")
    for p in sorted(OUT_ROOT.glob("*.png")):
        print(f"  {p.name}")

if __name__ == "__main__":
    main()
