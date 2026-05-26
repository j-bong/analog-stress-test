# Append this section to README.md

Replace the existing "Running a sweep" section in README.md with the following.

---

## Running a sweep

The sweep drivers shell out to the upstream training scripts in
`Zhaoxian-Wu/analog-training`. Set them up once with the steps in
`patches/APPLY_PATCHES.md`.

After that:

```bash
python scripts/sweep_fcn.py --gpu 0          # all axes, parallel=2
python scripts/sweep_fcn.py --gpu 0 --resume # skip completed logs
python scripts/sweep_cnn.py --gpu 0
```
