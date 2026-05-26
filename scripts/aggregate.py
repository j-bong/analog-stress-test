#!/usr/bin/env python3
"""Sweep results -> mean/std table."""
import json, sys, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / 'analog_stress_test' / 'results' / 'sweep_day2'

def main(axis=None):
    base = ROOT / axis if axis else ROOT
    if not base.exists():
        sys.exit(f"not found: {base}")
    grp = defaultdict(list)
    for cfg in base.rglob('config.json'):
        c = json.loads(cfg.read_text())
        f = cfg.parent / 'final.json'
        if not f.exists(): continue
        final = json.loads(f.read_text())
        key = (c['model'], c['algo'], c.get('res_state',''), c.get('rpu',''), c.get('res_gamma',''))
        grp[key].append(final['final_accuracy'])

    print(f"{'model':5} {'algo':12} {'state':6} {'rpu':10} {'γres':5} {'n':>3} {'mean%':>8} {'std%':>8}")
    print('-'*65)
    for key in sorted(grp.keys()):
        m, a, rs, rpu, rg = key
        accs = grp[key]
        mn = statistics.mean(accs)
        sd = statistics.stdev(accs) if len(accs)>1 else 0
        print(f"{m:5} {a:12} {rs!s:6} {rpu:10} {rg!s:5} {len(accs):>3} {mn*100:>7.3f}% {sd*100:>7.3f}%")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else None)
