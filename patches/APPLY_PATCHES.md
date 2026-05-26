# Upstream training scripts

The sweep drivers in `scripts/` shell out to the upstream training scripts
at `NeurIPS-2025/S2-mnist-FCN.py` and `NeurIPS-2025/S2-mnist-CNN.py`.
Those live in:

    https://github.com/Zhaoxian-Wu/analog-training

pinned at commit `92f26cceb70b93119c088b236c9197065d5082dd`
(`feat/chopper-gamma` branch).

## One-time setup

From the parent directory you want everything to live under
(default: `~/analog_stress_test`):

```bash
mkdir -p ~/analog_stress_test
cd ~/analog_stress_test
git clone https://github.com/Zhaoxian-Wu/analog-training.git
cd analog-training
git checkout 92f26cceb70b93119c088b236c9197065d5082dd
```

## Apply the patches

From this repo's root, run:

```bash
cd ~/analog_stress_test/analog-training
git apply <THIS_REPO>/patches/S2-mnist-FCN.patch
git apply <THIS_REPO>/patches/S2-mnist-CNN.patch
```

Replace `<THIS_REPO>` with wherever you cloned this repo.

## What the patches change

- Add seed control driven by the `SEED` environment variable.
- Wire DAC/ADC bit resolution and IO noise sigma from `INP_RES`, `OUT_RES`,
  `INP_NOISE`, `OUT_NOISE` env vars into the main forward/backward IO config.
- Set the algorithm hyperparameters to alpha = 0.05, beta = 0.01
  (RL-v2 also gets gamma = 0.4, the residual mixing factor).

About 70 changed lines per file. No new dependencies introduced.

## Run a sweep

```bash
cd <THIS_REPO>
python scripts/sweep_fcn.py --gpu 0
python scripts/sweep_cnn.py --gpu 0
```
