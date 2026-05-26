#!/usr/bin/env python3
"""Sweep runner for FCN MNIST."""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

SCRIPT_PATH = "NeurIPS-2025/S2-mnist-FCN.py"     # relative to TRAIN_ROOT
TRAIN_ROOT  = Path("~/analog_stress_test/analog-training").expanduser()
LOG_ROOT    = Path("~/analog_stress_test/logs/fcn_sweep").expanduser()

ALL_ALGOS = ["Analog SGD", "TT-v1", "TT-v2", "RL-v2"]

# Paper FCN MNIST hyperparams (per summary): tau=0.7, res_gamma=0.5
TAU       = 0.7
RES_GAMMA = 0.5

# Axis 1 : Pow granularity sweep
GRAN_STATES = [14000, 1400, 280, 140, 28, 14]

# Axis 2 : Pow response curve sweep (was Exp — replaced for orthogonality vs Axis 1)
POW_RES_GAMMAS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
POW_RES_STATE  = 140

# Axis 3 : Measurement-based device presets
PRESETS = ["HfO2", "EcRam", "EcRamMO", "OM"]

# Axis 4 : I/O quantization grid
# aihwkit IOParameters.inp_res / out_res convention: step size as fraction of bound.
# N-bit → 1/(2**N - 2).  Default aihwkit values: 7-bit input, 9-bit output, out_noise=0.06.
# Base device for this axis = Pow synthetic at the SAME baseline as Axis 1/2 (tau=0.7,
# rg=0.5, rs=140), so that Axis 4 isolates IO effect from device-side non-idealities.
def _bitres(n):
    return 1.0 / (2**n - 2)

# (label, INP_RES, OUT_RES, INP_NOISE, OUT_NOISE)
IO_GRID = [
    # Reference: matches aihwkit default IO (control, ~= axis 1 baseline)
    ("io_default",      _bitres(7), _bitres(9), 0.0,  0.06),
    # Mild: 6-bit input + low noise
    ("io_6b_mild",      _bitres(6), _bitres(8), 0.06, 0.06),
    # Extreme: 4-bit input + 10% noise
    ("io_4b_extreme",   _bitres(4), _bitres(7), 0.10, 0.10),
]
IO_BASE_TAU       = 0.7
IO_BASE_RES_GAMMA = 0.5
IO_BASE_RES_STATE = 140

DEFAULT_SEEDS  = [1, 2, 3]
DEFAULT_EPOCHS = 30

# AXIS BUILDERS — each returns a list of run dicts
# Each run dict has: algo, name, args (list[str]), env (dict[str,str])

def build_axis1_granularity(algos, seeds, epochs):
    runs = []
    for algo, rs, seed in product(algos, GRAN_STATES, seeds):
        runs.append(dict(
            axis="axis1_granularity",
            algo=algo,
            name=f"pow_rs{rs}_tau{TAU}_rg{RES_GAMMA}_seed{seed}",
            args=[
                "--SETTING", algo, "--RPU", "Pow",
                "--tau", str(TAU), "--res-gamma", str(RES_GAMMA),
                "--res-state", str(rs),
            ],
            env={"SEED": str(seed)},
        ))
    return runs

def build_axis2_pow_response(algos, seeds, epochs):
    """Axis 2: Pow nonlinearity (res-gamma) sweep at fixed res-state=140.
    Was Exp originally; switched to Pow so Axis 1 (Pow rs) and Axis 2 (Pow rg)
    isolate two orthogonal Pow knobs."""
    runs = []
    for algo, rg, seed in product(algos, POW_RES_GAMMAS, seeds):
        runs.append(dict(
            axis="axis2_pow_response",
            algo=algo,
            name=f"pow_rg{rg}_tau{TAU}_rs{POW_RES_STATE}_seed{seed}",
            args=[
                "--SETTING", algo, "--RPU", "Pow",
                "--tau", str(TAU), "--res-gamma", str(rg),
                "--res-state", str(POW_RES_STATE),
            ],
            env={"SEED": str(seed)},
        ))
    return runs

def build_axis3_presets(algos, seeds, epochs):
    runs = []
    for algo, preset, seed in product(algos, PRESETS, seeds):
        runs.append(dict(
            axis="axis3_presets",
            algo=algo,
            name=f"{preset}_seed{seed}",
            args=[
                "--SETTING", algo, "--RPU", preset,
            ],
            env={"SEED": str(seed)},
        ))
    return runs

def build_axis4_io(algos, seeds, epochs):
    """Axis 4: IO quantization sweep on Pow baseline device (tau=0.7, rg=0.5, rs=140).
    Reference: Axis 1's rs=140, rg=0.5 point uses aihwkit default IO (7b/9b/0.06)."""
    runs = []
    for algo, (label, ir, or_, in_, on), seed in product(algos, IO_GRID, seeds):
        runs.append(dict(
            axis="axis4_io",
            algo=algo,
            name=f"pow_rs{IO_BASE_RES_STATE}_{label}_seed{seed}",
            args=[
                "--SETTING", algo, "--RPU", "Pow",
                "--tau", str(IO_BASE_TAU),
                "--res-gamma", str(IO_BASE_RES_GAMMA),
                "--res-state", str(IO_BASE_RES_STATE),
            ],
            env={
                "SEED": str(seed),
                "INP_RES": str(ir), "OUT_RES": str(or_),
                "INP_NOISE": str(in_), "OUT_NOISE": str(on),
            },
        ))
    return runs

AXIS_BUILDERS = {
    1: ("granularity",  build_axis1_granularity),
    2: ("pow_response", build_axis2_pow_response),
    3: ("presets",      build_axis3_presets),
    4: ("io",           build_axis4_io),
}

def algo_to_dir(algo: str) -> str:
    """'Analog SGD' -> 'Analog_SGD' for filesystem-safe dir names."""
    return algo.replace(" ", "_")

def log_path_for(run: dict) -> Path:
    return LOG_ROOT / run["axis"] / algo_to_dir(run["algo"]) / f"{run['name']}.log"

def is_complete(log: Path) -> bool:
    """Run is complete iff log exists AND contains 'Training Time' near end."""
    if not log.exists():
        return False
    try:
        with open(log, "r", errors="replace") as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read()
        return "Training Time" in tail
    except Exception:
        return False

def build_command(run: dict, gpu: int) -> list:
    # NOTE: CUDA_VISIBLE_DEVICES=<gpu> remaps the chosen physical GPU to
    # logical index 0 inside the subprocess. So --CUDA 0 is correct here
    # (passing the physical index would yield "invalid device ordinal").
    return [
        "python", SCRIPT_PATH,
        "--CUDA", "0",
        *run["args"],
    ]

def env_str(run: dict, gpu: int) -> str:
    e = {"CUDA_VISIBLE_DEVICES": str(gpu),
         "PYTHONPATH": str(TRAIN_ROOT),
         **run["env"]}
    return " ".join(f"{k}={v}" for k, v in e.items())

# PARALLEL EXECUTOR — up to `parallel` processes on the same GPU

class ParallelRunner:
    def __init__(self, gpu: int, parallel: int, dry_run: bool):
        self.gpu = gpu
        self.parallel = parallel
        self.dry_run = dry_run
        self.active = []      # [(popen, run, log_path, t0, fh)]
        self.completed = []   # [(run, status, dt)]
        self._stop = False
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *a):
        print("\n[SIGINT] terminating active processes...", flush=True)
        self._stop = True
        for p, run, log, t0, fh in self.active:
            try: p.terminate()
            except Exception: pass

    def _launch(self, run, idx, total):
        log = log_path_for(run)
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_command(run, self.gpu)
        env = os.environ.copy()
        env.update(run["env"])
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        # Ensure analog-training/ is on PYTHONPATH so `from utils.logger import Logger`
        # (used by S2-mnist-FCN.py at module top) resolves.
        env["PYTHONPATH"] = str(TRAIN_ROOT) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        prefix = f"[{idx}/{total}]"
        if self.dry_run:
            print(f"{prefix} [DRY] {env_str(run, self.gpu)} {' '.join(cmd)}")
            print(f"           log -> {log}")
            self.completed.append((run, "dry", 0.0))
            return

        fh = open(log, "w")
        fh.write(f"# CMD: {env_str(run, self.gpu)} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=TRAIN_ROOT, env=env,
                              stdout=fh, stderr=subprocess.STDOUT)
        self.active.append((p, run, log, time.time(), fh))
        print(f"{prefix} [LAUNCH gpu={self.gpu} pid={p.pid}] "
              f"{run['algo']} :: {run['name']}", flush=True)

    def _reap(self):
        still = []
        for p, run, log, t0, fh in self.active:
            rc = p.poll()
            if rc is None:
                still.append((p, run, log, t0, fh))
            else:
                try: fh.close()
                except Exception: pass
                dt = time.time() - t0
                status = "ok" if rc == 0 else f"fail(rc={rc})"
                print(f"   [DONE {status} {dt:5.1f}s] {run['algo']} :: {run['name']}",
                      flush=True)
                self.completed.append((run, status, dt))
        self.active = still

    def run_all(self, runs):
        pending = list(runs)
        total = len(runs)
        idx = 0
        while (pending or self.active) and not self._stop:
            # launch up to parallel
            while pending and len(self.active) < self.parallel and not self._stop:
                idx += 1
                self._launch(pending.pop(0), idx, total)
                if self.dry_run:
                    continue
            if self.dry_run:
                continue
            time.sleep(2)
            self._reap()
        # final reap
        while self.active:
            time.sleep(1); self._reap()
        return self.completed

EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*-.*Test Accuracy:\s*([0-9.]+)")

def parse_final_accuracy(log: Path):
    if not log.exists():
        return (None, None, None)
    last_epoch, last_acc, max_acc = None, None, None
    with open(log, "r", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                e, a = int(m.group(1)), float(m.group(2))
                last_epoch, last_acc = e, a
                max_acc = a if max_acc is None else max(max_acc, a)
    return (last_epoch, last_acc, max_acc)

def print_summary(all_runs):
    print("\n" + "=" * 92)
    print("SUMMARY  [model=FCN]  last / max test accuracy per run")
    print("=" * 92)
    by_axis = {}
    for r in all_runs:
        by_axis.setdefault(r["axis"], []).append(r)
    for ax in sorted(by_axis):
        print(f"\n--- {ax} ---")
        for r in by_axis[ax]:
            log = log_path_for(r)
            _, la, ma = parse_final_accuracy(log)
            done = "OK " if is_complete(log) else "INC"
            la_s = f"{la:.4f}" if la is not None else "  -   "
            ma_s = f"{ma:.4f}" if ma is not None else "  -   "
            print(f"  [{done}] {r['algo']:<11} {r['name']:<48} last={la_s} max={ma_s}")

def parse_axes(spec):
    if not spec: return sorted(AXIS_BUILDERS.keys())
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok: continue
        try: out.append(int(tok))
        except ValueError:
            print(f"[ERR] axis '{tok}' not int", file=sys.stderr); sys.exit(2)
    return sorted(set(out))

def parse_seeds(spec):
    return [int(s.strip()) for s in spec.split(",") if s.strip()]

def parse_algos(spec):
    if not spec: return list(ALL_ALGOS)
    raw = [a.strip() for a in spec.split(",") if a.strip()]
    valid = []
    for a in raw:
        if a in ALL_ALGOS: valid.append(a)
        else:
            print(f"[ERR] algo '{a}' not in {ALL_ALGOS}", file=sys.stderr); sys.exit(2)
    return valid

def main():
    ap = argparse.ArgumentParser(description="FCN @ MNIST sweep (per-task logs, in-GPU parallelism)")
    ap.add_argument("--gpu", type=int, default=3, help="GPU index (default: 3)")
    ap.add_argument("--parallel", type=int, default=2,
                    help="Concurrent processes on the same GPU (default: 2)")
    ap.add_argument("--axes", type=str, default="",
                    help='Comma list, e.g. "1,3". Default: all (1,2,3,4).')
    ap.add_argument("--algos", type=str, default="",
                    help="Comma list. Default: all 4 algos.")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip runs whose log already has 'Training Time'.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=-1)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    axes  = parse_axes(args.axes)
    algos = parse_algos(args.algos)
    seeds = parse_seeds(args.seeds)

    all_runs = []
    for ax in axes:
        if ax not in AXIS_BUILDERS:
            print(f"[ERR] unknown axis {ax}", file=sys.stderr); sys.exit(2)
        _, builder = AXIS_BUILDERS[ax]
        all_runs.extend(builder(algos, seeds, args.epochs))

    print(f"[INFO] model=FCN gpu={args.gpu} parallel={args.parallel} "
          f"axes={axes} algos={algos} seeds={seeds} epochs={args.epochs}")
    print(f"[INFO] total runs: {len(all_runs)}")

    if args.summary_only:
        print_summary(all_runs); return

    # filter resume + start/max-runs
    runs = all_runs[args.start:]
    if args.max_runs > 0:
        runs = runs[:args.max_runs]
    if args.resume:
        runs = [r for r in runs if not is_complete(log_path_for(r))]
        print(f"[INFO] after --resume filter: {len(runs)} runs to execute")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    runner = ParallelRunner(args.gpu, args.parallel, args.dry_run)
    runner.run_all(runs)

    if not args.dry_run:
        ok = sum(1 for _, s, _ in runner.completed if s == "ok")
        fail = sum(1 for _, s, _ in runner.completed if s.startswith("fail"))
        print(f"\n[INFO] completed: ok={ok}  fail={fail}  total={len(runner.completed)}")
        print_summary(all_runs)

if __name__ == "__main__":
    main()
