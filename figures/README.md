# Figures

Pre-rendered figures referenced from the 6-page report.

## Layout

    device_curves/                 16 PNGs
        Δw (up/down step) vs w for each device the sweep visits.
        Naming: axis<N>_<family>_<params>.png
        Used to inspect the F/G asymmetry that drives the symmetric-point shift.

    sweep_results/                 10 PNGs
        Per-axis test accuracy plots, FCN and CNN, plus a 2x2 overview per model.
        axis1_*  Pow granularity (res-state sweep)
        axis2_*  Pow nonlinearity (res-gamma sweep)
        axis3_*  measurement-based device presets
        axis4_*  IO quantization
        summary_*  4-axis overview

    library_default_comparison/    6 PNGs
        Side-by-side per-axis comparison of two configurations:
        aihwkit fork library default (TT-v1/v2 transfer_lr=1.0 scaled,
        RL-v2 transfer_lr=0.111 absolute) vs paper-calibrated β=0.01.

## Reproducing

All figures regenerate from the raw .log files in this repo:

    python scripts/plot_results.py        # per-axis + summary
    python scripts/plot_device_curves.py  # device pulse-response
    python scripts/plot_comparison.py     # library-default vs calibrated
