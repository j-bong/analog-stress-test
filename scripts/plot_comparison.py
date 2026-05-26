#!/usr/bin/env python3
"""Side-by-side comparison plots across two configurations."""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIMARY_ROOT  = Path("~/analog_stress_test/logs").expanduser()
BASELINE_ROOT = Path("~/analog_stress_test/baseline_logs").expanduser()
OUT_ROOT    = Path("~/analog_stress_test/results/plots").expanduser()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALGOS = ["Analog_SGD", "TT-v1", "TT-v2", "RL-v2"]
ALGO_COLOR = {"Analog_SGD": "#d62728", "TT-v1": "#1f77b4", "TT-v2": "#2ca02c", "RL-v2": "#9467bd"}
ALGO_MARKER = {"Analog_SGD": "o", "TT-v1": "s", "TT-v2": "^", "RL-v2": "D"}
ALGO_LABEL = {"Analog_SGD": "Analog SGD", "TT-v1": "TT-v1", "TT-v2": "TT-v2", "RL-v2": "RL-v2"}

RS_VALUES = [14000, 1400, 280, 140, 28, 14]
RG_VALUES = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
PRESETS   = ["HfO2", "EcRam", "EcRamMO", "OM"]

EPOCH_RE = re.compile(r"Test Accuracy:\s*([0-9.]+)")
DONE = "Training Time"

def parse_acc(p: Path):
    if not p.exists(): return None
    has, last = False, None
    with open(p, errors="replace") as f:
        for ln in f:
            m = EPOCH_RE.search(ln)
            if m: last = float(m.group(1))
            if DONE in ln: has = True
    return last if has else None

def collect_axis_line(root: Path, model: str, axis_name: str, param_pattern, x_values, sub_root_name=None):
    """Collect line-axis data (axis 1 or 2). param_pattern: function(p_val) -> log filename middle part."""
    base = root / (sub_root_name or f"{model}_sweep") / axis_name
    out = defaultdict(lambda: defaultdict(list))
    for algo in ALGOS:
        for x in x_values:
            for s in [1, 2, 3]:
                log = base / algo / f"{param_pattern(x)}_seed{s}.log"
                acc = parse_acc(log)
                if acc is not None:
                    out[algo][x].append(acc)
    return out

def collect_axis3(root: Path, model: str, sub_root_name=None):
    base = root / (sub_root_name or f"{model}_sweep") / "axis3_presets"
    out = defaultdict(lambda: defaultdict(list))
    for algo in ALGOS:
        for preset in PRESETS:
            for s in [1, 2, 3]:
                log = base / algo / f"{preset}_seed{s}.log"
                acc = parse_acc(log)
                if acc is not None:
                    out[algo][preset].append(acc)
    return out

def mean_std(accs):
    if not accs: return None, None
    m = float(np.mean(accs))
    s = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    return m, s

def panel_line(ax, data, x_values, title, xlabel, log_x=True):
    for algo in ALGOS:
        means, stds, xs = [], [], []
        for x in x_values:
            accs = data.get(algo, {}).get(x, [])
            m, s = mean_std(accs)
            if m is not None:
                xs.append(x); means.append(m); stds.append(s)
        if xs:
            ax.errorbar(xs, means, yerr=stds,
                        color=ALGO_COLOR[algo], marker=ALGO_MARKER[algo],
                        markersize=6, linewidth=1.6, capsize=3,
                        label=ALGO_LABEL[algo])
    if log_x: ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

def panel_bar(ax, data, categories, title, xlabel):
    n_algo = len(ALGOS); n_cat = len(categories)
    width = 0.8 / n_algo
    x_pos = np.arange(n_cat)
    for i, algo in enumerate(ALGOS):
        means, stds = [], []
        for cat in categories:
            accs = data.get(algo, {}).get(cat, [])
            m, s = mean_std(accs)
            means.append(m or 0); stds.append(s or 0)
        offset = (i - (n_algo - 1) / 2) * width
        ax.bar(x_pos + offset, means, width, yerr=stds,
               color=ALGO_COLOR[algo], capsize=2,
               label=ALGO_LABEL[algo], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

def make_axis12_comparison(model: str, axis_no: int):
    if axis_no == 1:
        axis_name = "axis1_granularity"
        x_values = RS_VALUES
        xlabel = "res-state"
        param_pattern = lambda rs: f"pow_rs{rs}_tau0.7_rg0.5"
        full_name = "Axis 1: Pow granularity (rs sweep)"
    else:
        axis_name = "axis2_pow_response"
        x_values = RG_VALUES
        xlabel = "res-gamma"
        param_pattern = lambda rg: f"pow_rg{rg}_tau0.7_rs140"
        full_name = "Axis 2: Pow nonlinearity (rg sweep)"

    primary_data  = collect_axis_line(PRIMARY_ROOT,  model, axis_name, param_pattern, x_values)
    baseline_data = collect_axis_line(BASELINE_ROOT, model, axis_name, param_pattern, x_values,
                                    sub_root_name=f"{model}_sweep_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panel_line(axes[0], baseline_data, x_values, "Baseline", xlabel)
    panel_line(axes[1], primary_data, x_values, "Primary (α=0.05, β=0.01)", xlabel)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{model.upper()} MNIST — {full_name}: baseline vs primary", fontsize=12, y=1.00)
    plt.tight_layout()
    out_path = OUT_ROOT / f"comparison_axis{axis_no}_{model}.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  saved {out_path.name}")

def make_axis3_comparison(model: str):
    primary_data  = collect_axis3(PRIMARY_ROOT,  model)
    baseline_data = collect_axis3(BASELINE_ROOT, model, sub_root_name=f"{model}_sweep_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panel_bar(axes[0], baseline_data, PRESETS, "Baseline", "Preset")
    panel_bar(axes[1], primary_data, PRESETS, "Primary (α=0.05, β=0.01)", "Preset")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{model.upper()} MNIST — Axis 3: Measurement presets, baseline vs primary", fontsize=12, y=1.00)
    plt.tight_layout()
    out_path = OUT_ROOT / f"comparison_axis3_{model}.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  saved {out_path.name}")

def main():
    print("Generating comparison plots (baseline vs primary)")
    for model in ["fcn", "cnn"]:
        print(f"\n--- {model.upper()} ---")
        make_axis12_comparison(model, 1)
        make_axis12_comparison(model, 2)
        make_axis3_comparison(model)
    print(f"\nDone. 6 comparison plots in {OUT_ROOT}/")

if __name__ == "__main__":
    main()
