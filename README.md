# analog stress-test

Sweep + analysis code for an internal stress-test on analog in-memory training.

## Layout

    scripts/   sweep runners and plotting code
    logs/      raw .log per run, grouped by axis/algorithm

## Running a sweep

The sweep drivers shell out to the upstream training scripts in
Zhaoxian-Wu/analog-training. Set them up once with the steps in
patches/APPLY_PATCHES.md.

After that:

    python scripts/sweep_fcn.py --gpu 0          # all axes, parallel=2
    python scripts/sweep_fcn.py --gpu 0 --resume # skip completed logs
    python scripts/sweep_cnn.py --gpu 0

## Building plots from logs

    python scripts/plot_results.py        # per-axis + summary figures
    python scripts/plot_device_curves.py  # device pulse-response curves
    python scripts/plot_comparison.py     # side-by-side comparison
