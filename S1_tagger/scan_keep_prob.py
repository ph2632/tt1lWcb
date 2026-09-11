#!/usr/bin/env python3
"""
Scan background_keep_prob in {0.2, 0.3, 0.4, 0.5} to find the minimum that
keeps the S1 background training bias below ~5%.

Reads the dataset built ONCE at background_keep_prob=0.5 (config dataset_tag,
e.g. presel_v3_kp50) and, for each smaller target p, further Bernoulli-thins
ONLY the true-background rows (topology == 0) within the TRAIN split by
ratio = target_p / 0.5, reweighting the survivors by 1/ratio so the combined
factor is 1/target_p -- statistically equivalent to having built the dataset
at target_p directly, without re-reading the 17 ROOT files. Signal classes
1-5 (incl. the t2(b'c) proxy, class 5, which is never thinned) are left
untouched at every scan point. Evaluation always uses the FULL (untouched)
valid/test splits, so AUC differences reflect only the training-set change.

Same bias metric as scan_regularisation.py: |test-train|/test on the weighted
fraction above score > overtrain_score_cut (0.6), separately for signal (S)
and background (B) -- B is what background_keep_prob controls.

Each point runs in its OWN subprocess (--single-point) so memory is fully
released back to the OS before the next point starts -- this is a shared
lxplus node and a long-lived process holding train/valid/test + per-point
copies across 4 sequential fits was observed to get OOM-killed by the
system. scan_keep_prob.json is rewritten after every point, so a kill
mid-scan loses at most the one in-flight point, not the whole run.

  ./.venv/bin/python S1_tagger/scan_keep_prob.py [--config ...] [--points 0.2,0.3,0.4,0.5]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
RESULT_TAG = "RESULT_JSON: "


def bias(tr_s, tr_y, tr_w, te_s, te_y, te_w, cut=0.6):
    out = {}
    for cls, tag in ((1, "S"), (0, "B")):
        f_tr = tr_w[(tr_y == cls) & (tr_s > cut)].sum() / max(tr_w[tr_y == cls].sum(), 1e-12)
        f_te = te_w[(te_y == cls) & (te_s > cut)].sum() / max(te_w[te_y == cls].sum(), 1e-12)
        out[tag] = 100.0 * abs(f_te - f_tr) / max(f_te, 1e-12)
    return out


def class_weight_lookup(cfg):
    cwm = cfg.get("class_weight_multiplier")
    if not cwm:
        return None
    lut = np.ones(max(6, max(int(k) for k in cwm) + 1), dtype=np.float64)
    for k, v in cwm.items():
        lut[int(k)] = float(v)
    return lut


def run_single_point(cfg, target_p):
    """Load the dataset, thin to target_p, train S1 once, print RESULT_JSON."""
    base_p = float(cfg["background_keep_prob"])
    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    feats = cfg["features_S1"]
    derived = cfg.get("derived_features", {})
    derived_raw = sorted({c for comps in derived.values() for c in comps})
    cols = sorted((set(["topology", "weight"] + feats) - set(derived)) | set(derived_raw))

    def load(sub):
        ps = sorted((dsdir / sub).glob("*.parquet"))
        if not ps:
            raise SystemExit(f"no parquet in {dsdir/sub} -- run build_trainset.py first")
        df = pd.concat((pd.read_parquet(p, columns=cols) for p in ps), ignore_index=True)
        for name, comps in derived.items():
            df[name] = df[comps].sum(axis=1)
        return df

    print(f"[load] {dsdir}  target_p={target_p}  (base {base_p})", flush=True)
    tr, va, te = load("train"), load("valid"), load("test")

    y_tr_full = (tr["topology"] > 0).astype(np.int8).to_numpy()
    w_tr_full = tr["weight"].to_numpy()
    topo_tr = tr["topology"].to_numpy()
    y_va = (va["topology"] > 0).astype(np.int8).to_numpy()
    w_va = va["weight"].to_numpy()
    y_te = (te["topology"] > 0).astype(np.int8).to_numpy()
    w_te = te["weight"].to_numpy()
    n_bkg_full = int((topo_tr == 0).sum())

    # per-class sig sumw retarget -- TRAIN + VALID only, TEST stays physical
    cwm = class_weight_lookup(cfg)
    if cwm is not None:
        w_tr_full = w_tr_full * cwm[topo_tr]
        w_va = w_va * cwm[va["topology"].to_numpy()]

    rng = np.random.default_rng(cfg["seed"])
    ratio = target_p / base_p
    is_bkg = topo_tr == 0
    if ratio >= 1.0 - 1e-9:
        keep = np.ones(len(tr), dtype=bool)
        w_tr = w_tr_full.copy()
    else:
        thin = rng.random(len(tr)) < ratio
        keep = (~is_bkg) | thin
        w_tr = w_tr_full.copy()
        w_tr[is_bkg] = w_tr[is_bkg] / ratio  # combined with base 1/base_p -> 1/target_p

    d_tr = tr.loc[keep, feats]
    y_tr = y_tr_full[keep]
    w_tr = w_tr[keep]
    n_bkg_kept = int((~y_tr.astype(bool)).sum())

    params = dict(cfg["xgboost"], random_state=cfg["seed"],
                 scale_pos_weight=float(w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-12)))
    t0 = time.time()
    m = XGBClassifier(**params)
    m.fit(d_tr, y_tr, sample_weight=w_tr,
          eval_set=[(va[feats], y_va)], sample_weight_eval_set=[w_va], verbose=False)
    s_tr = m.predict_proba(d_tr)[:, 1]
    s_te = m.predict_proba(te[feats])[:, 1]
    auc = roc_auc_score(y_te, s_te, sample_weight=w_te)
    b = bias(s_tr, y_tr, w_tr, s_te, y_te, w_te, cfg.get("overtrain_score_cut", 0.6))
    it = int(getattr(m, "best_iteration", params["n_estimators"] - 1))
    dt = time.time() - t0

    row = {"target_keep_prob": target_p, "n_train_background": n_bkg_kept,
           "n_train_background_full": n_bkg_full, "test_auc": float(auc),
           "bias_S_pct": b["S"], "bias_B_pct": b["B"],
           "best_iteration": it, "seconds": round(dt, 1)}
    print(f"{target_p:9.2f} {n_bkg_kept:12,} {auc:9.4f} {b['S']:8.1f} {b['B']:8.1f} "
          f"{it:6d} {dt:5.0f}", flush=True)
    print(RESULT_TAG + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--points", default="0.2,0.3,0.4,0.5")
    ap.add_argument("--single-point", type=float, default=None,
                    help=argparse.SUPPRESS)  # internal: run exactly one point, no subprocess
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    if args.single_point is not None:
        run_single_point(cfg, args.single_point)
        return

    base_p = float(cfg["background_keep_prob"])
    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    marker = dsdir / "dataset_complete.json"
    if not marker.exists():
        raise SystemExit(
            f"[scan_keep_prob.py] {marker} not found -- build_trainset.py for "
            f"dataset_tag='{cfg['dataset_tag']}' has not finished (or not started). "
            f"Refusing to scan a partial/absent dataset -- check "
            f"'ps aux | grep build_trainset' and wait for '==== dataset built ===='.")

    out_path = HERE / "scan_keep_prob.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["target_keep_prob"] for r in rows}

    points = sorted(float(p) for p in args.points.split(","))
    print(f"[scan] base background_keep_prob = {base_p}, points = {points}")
    print(f"\n{'target_p':>9} {'n_train_bkg':>12} {'AUC test':>9} {'bias S%':>8} "
          f"{'bias B%':>8} {'iters':>6} {'s':>5}")
    print("-" * 68)
    for target_p in points:
        if target_p > base_p + 1e-9:
            print(f"{target_p:9.2f}  skipped -- above base keep_prob {base_p}")
            continue
        if target_p in done:
            r = next(r for r in rows if r["target_keep_prob"] == target_p)
            print(f"{target_p:9.2f} {r['n_train_background']:12,} {r['test_auc']:9.4f} "
                  f"{r['bias_S_pct']:8.1f} {r['bias_B_pct']:8.1f} {r['best_iteration']:6d} "
                  f"{r['seconds']:5.0f}  (cached)")
            continue

        row = None
        for attempt in range(1, 4):
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--config", str(args.config), "--single-point", str(target_p)],
                capture_output=True, text=True)
            sys.stdout.write(proc.stdout)
            if proc.returncode == 0:
                line = next((l for l in proc.stdout.splitlines()
                            if l.startswith(RESULT_TAG)), None)
                if line is not None:
                    row = json.loads(line[len(RESULT_TAG):])
                    break
                print(f"[scan] point {target_p} attempt {attempt}: "
                      f"exit 0 but no result line")
            else:
                # a shared-node OOM kill is transient (seen twice: system had
                # 27-47 GB free moments later) -- retry a couple of times
                # with a short cooldown before giving up on this point.
                print(proc.stderr[-2000:], file=sys.stderr)
                print(f"[scan] point {target_p} attempt {attempt}/3 FAILED "
                      f"(exit {proc.returncode}, likely OOM) ", flush=True)
            if attempt < 3:
                time.sleep(60)
        if row is None:
            print(f"[scan] point {target_p} FAILED after 3 attempts -- "
                  f"stopping; {len(rows)}/{len(points)} points saved so far.")
            break
        rows.append(row)
        out_path.write_text(json.dumps(rows, indent=2) + "\n")  # save after EVERY point

    ok = [r for r in rows if r["bias_B_pct"] < 5.0]
    if ok:
        best = min(ok, key=lambda r: r["target_keep_prob"])
        print(f"\nsmallest keep_prob with bias_B < 5%: {best['target_keep_prob']}"
              f"  (bias_B {best['bias_B_pct']:.1f}%, AUC {best['test_auc']:.4f})")
    else:
        print("\nno scanned point reached bias_B < 5% (yet)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
