#!/usr/bin/env python3
"""
S1 / S1' boosted-cb tagger -- single command-line driver.

    ./run_s1_tagger.py                 # train + plots (dataset reused if built)
    ./run_s1_tagger.py all             # dataset + train + plots
    ./run_s1_tagger.py dataset         # rebuild the per-jet training set only
    ./run_s1_tagger.py train           # retrain only
    ./run_s1_tagger.py plots           # re-render the figures only (no retrain)

    ./run_s1_tagger.py train --run-tag presel_v3 --set max_depth=3 min_child_weight=50
    ./run_s1_tagger.py presel          # just show the preselection it will use

The preselection is READ FROM Make_plots.py (the `_SELECTIONS["PRE"]` block),
so changing a threshold there propagates here automatically -- nothing to keep
in sync by hand.

Drives the 3-step pipeline (each step also runs standalone):
  dataset -> 01_build_trainset.py   per-jet training parquet (venv)
  train   -> 02_train_tagger.py     XGBoost S1 tagger + summary.json (venv)
  plots   -> 03_render_report.py    PyROOT panel + figures (self-bootstraps
                                     into an LCG view; no separate interpreter
                                     handling needed here since 2026-09-13)

config.json is pure input (preselection fallback, class_weight_share,
hyperparameters) -- nothing in this pipeline writes back to it.
class_weight_multiplier is derived fresh every run inside 02_train_tagger.py
from class_weight_share + the real per-class train sumw, so it can never go
stale.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "S1_tagger"
CONFIG = PKG / "config.json"
VENV = HERE / ".venv" / "bin" / "python"

STEPS = ("dataset", "train", "plots")
STEP_SCRIPT = {"dataset": HERE / "01_build_trainset.py",
               "train": HERE / "02_train_tagger.py",
               "plots": HERE / "03_render_report.py"}


def sh(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd[:3])} ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(f"step failed (exit {r.returncode})")
    print(f"    ({time.time() - t0:.1f}s)", flush=True)


def show_presel():
    sys.path.insert(0, str(PKG))
    from presel import parse_preselection
    p, cut, missing = parse_preselection()
    print("preselection read from Make_plots.py:\n  " + cut + "\n")
    for k, v in p.items():
        print(f"  {k:32s} {v}")
    if missing:
        print(f"\n  not found -> config.json fallback: {missing}")


def apply_overrides(cfg_path, run_tag, sets):
    """Write a temporary config with --run-tag / --set applied."""
    cfg = json.loads(cfg_path.read_text())
    if run_tag:
        cfg["run_tag"] = run_tag
    for kv in sets:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        (cfg["xgboost"] if k in cfg["xgboost"] else cfg)[k] = v
    if not run_tag and not sets:
        return cfg_path
    out = PKG / "config.effective.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[config] overrides -> {out}  (run_tag={cfg.get('run_tag')})")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["train", "plots"],
                    help="any of: all, dataset, train, plots, presel "
                         "(default: train plots)")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--run-tag", default="", help="name this training run")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="override a config or xgboost key, e.g. max_depth=3")
    ap.add_argument("--overwrite-dataset", action="store_true")
    args = ap.parse_args()
    t_wall0 = time.time()

    steps = args.steps
    if "presel" in steps:
        show_presel()
        if len(steps) == 1:
            return
        steps = [s for s in steps if s != "presel"]
    if "all" in steps:
        steps = list(STEPS)
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"unknown step(s) {bad}; choose from {STEPS} / all / presel")

    if not VENV.exists():
        sys.exit(f"missing interpreter {VENV}")
    cfg_path = apply_overrides(args.config, args.run_tag, args.set)
    show_presel()

    if "dataset" in steps:
        cmd = [str(VENV), str(STEP_SCRIPT["dataset"]), "--config", str(cfg_path)]
        if args.overwrite_dataset:
            cmd.append("--overwrite")
        sh(cmd)
    if "train" in steps:
        sh([str(VENV), str(STEP_SCRIPT["train"]), "--config", str(cfg_path), "--no-report"])
    if "plots" in steps:
        # 03_render_report.py self-bootstraps into an LCG view if PyROOT isn't
        # importable in VENV, so no separate interpreter handling is needed
        # here (2026-09-13; previously this step forced its own LCG source).
        sh([str(VENV), str(STEP_SCRIPT["plots"]), "--config", str(cfg_path)])

    cfg = json.loads(cfg_path.read_text())
    print(f"\ndone -> {PKG / cfg['output_dir'] / cfg.get('run_tag', cfg['dataset_tag'])}")
    print(f"[TIMER] run_s1_tagger.py total: {time.time() - t_wall0:.1f}s")


if __name__ == "__main__":
    main()
