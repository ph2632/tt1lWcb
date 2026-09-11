#!/usr/bin/env python3
"""
Write the cumulative raw-sum ROC baselines (cmp_raw_cb.npz, cmp_raw_cb_bb.npz)
for a run whose train.py predates them.

train.py now emits these itself; this only backfills an existing output dir so
the ROC can show raw cb / raw cb+bb / raw cb+bb+t2(b'c) without a retrain.
Reads the TEST split only (4 columns), so it is cheap.

  ./.venv/bin/python S1_tagger/backfill_raw_baselines.py [--config ...]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    outdir = HERE / cfg["output_dir"] / cfg.get("run_tag", cfg["dataset_tag"])
    raw = cfg["raw_sum_baseline"]                       # [bc, bb, topbwc]
    cols = ["topology", "weight"] + raw

    paths = sorted((dsdir / "test").glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no test parquet in {dsdir/'test'}")
    te = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)

    # identical to train.py: test weights stay at raw physics weight
    y = (te["topology"] > 0).astype(np.int8).to_numpy()
    w = te["weight"].to_numpy().astype(np.float32)
    print(f"[backfill] test jets {len(te):,}  (sig {int(y.sum()):,})")

    for key, sub in (("raw_cb", raw[:1]), ("raw_cb_bb", raw[:2])):
        s = te[sub].sum(axis=1).to_numpy().astype(np.float32)
        f = outdir / f"cmp_{key}.npz"
        np.savez_compressed(f, score=s, y=y, w=w)
        print(f"  wrote {f.name}  ({'+'.join(c.replace('ak8_gpt_','') for c in sub)})")


if __name__ == "__main__":
    main()
