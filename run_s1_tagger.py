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

Two interpreters are needed and are handled for you:
  training / dataset -> ./.venv/bin/python   (xgboost, awkward, zstd parquet)
  plots              -> an LCG view          (PyROOT; the venv has no ROOT)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "S1_tagger"
CONFIG = PKG / "config.json"
VENV = HERE / ".venv" / "bin" / "python"
LCG = "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh"

STEPS = ("dataset", "train", "plots")


def sh(cmd, lcg=False):
    if lcg:
        cmd = ["bash", "-lc", f"source {LCG} >/dev/null 2>&1; " + " ".join(cmd)]
    print(f"\n>>> {' '.join(str(c) for c in cmd[:3])} ...", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(f"step failed (exit {r.returncode})")


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
        cmd = [str(VENV), str(PKG / "build_trainset.py"), "--config", str(cfg_path)]
        if args.overwrite_dataset:
            cmd.append("--overwrite")
        sh(cmd)
    if "train" in steps:
        sh([str(VENV), str(PKG / "train.py"), "--config", str(cfg_path), "--no-report"])
    if "plots" in steps:
        if not Path(LCG).exists():
            sys.exit(f"LCG view not found: {LCG}")
        sh(["python3", str(PKG / "make_plots_root.py"), "--config", str(cfg_path)],
           lcg=True)

    cfg = json.loads(cfg_path.read_text())
    print(f"\ndone -> {PKG / cfg['output_dir'] / cfg.get('run_tag', cfg['dataset_tag'])}")


if __name__ == "__main__":
    main()
