#!/usr/bin/env python3
"""
Scan XGBoost regularisation to push the BACKGROUND training bias below ~5%.

Metric (the one train.py quotes):
    bias_B = |test - train| / test   on the weighted background fraction
             above score > 0.6, i.e. in the region the analysis cuts on.

Trains S1 only and skips permutation importance, so a point costs ~2-3 min
instead of ~10.  Prints a table and writes scan_regularisation.json.

  ./.venv/bin/python S1_tagger/scan_regularisation.py [--config ...] [--only 1,3]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent

# name -> params overriding the config's xgboost block.
# Rationale: the bias sits in the sparse, heavily-weighted high-score
# background tail, so the levers that matter most are min_child_weight
# (forces many weighted events per leaf) and depth.
# --- scan 1: how hard can regularisation alone push the background bias? ---
GRID1 = {
    "A current (d3 mcw100 L20 cs.5)": {},
    "B mcw 300":                      {"min_child_weight": 300},
    "C mcw 1000 + L50":               {"min_child_weight": 1000, "reg_lambda": 50},
    "D depth 2 + mcw 300":            {"max_depth": 2, "min_child_weight": 300},
    "E d3 mcw1000 L100 cs.4 sub.5":   {"min_child_weight": 1000, "reg_lambda": 100,
                                       "colsample_bytree": 0.4, "subsample": 0.5},
    "F d2 mcw3000 L100 lr.03":        {"max_depth": 2, "min_child_weight": 3000,
                                       "reg_lambda": 100, "learning_rate": 0.03},
    "G half background (diagnostic)": {"_half_bkg": True},
}

# --- scan 2: other knobs, all on top of the chosen point D ----------------
# D was ceiling-limited (stopped at 1494/1500), so every point here gets a
# high n_estimators and lets early stopping decide.
_D = {"max_depth": 2, "min_child_weight": 300, "n_estimators": 4000}
GRID2 = {
    "D0 baseline (ceiling raised)":   dict(_D),
    "D1 lr 0.03":                     dict(_D, learning_rate=0.03),
    "D2 lr 0.08":                     dict(_D, learning_rate=0.08),
    "D3 subsample 0.5":               dict(_D, subsample=0.5),
    "D4 colsample 0.35":              dict(_D, colsample_bytree=0.35),
    "D5 gamma 5":                     dict(_D, gamma=5.0),
    "D6 max_delta_step 1":            dict(_D, max_delta_step=1),
    "D7 reg_alpha 5":                 dict(_D, reg_alpha=5.0),
}
GRIDS = {"1": GRID1, "2": GRID2}



def bias_bkg(tr_s, tr_y, tr_w, te_s, te_y, te_w, cut=0.6):
    out = {}
    for cls, tag in ((1, "S"), (0, "B")):
        f_tr = tr_w[(tr_y == cls) & (tr_s > cut)].sum() / max(tr_w[tr_y == cls].sum(), 1e-12)
        f_te = te_w[(te_y == cls) & (te_s > cut)].sum() / max(te_w[te_y == cls].sum(), 1e-12)
        out[tag] = 100.0 * abs(f_te - f_tr) / max(f_te, 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--only", default="", help="comma-separated point letters, e.g. B,C")
    ap.add_argument("--grid", default="1", choices=sorted(GRIDS),
                    help="1 = regularisation scan, 2 = other knobs around point D")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    feats = cfg["features_S1"]
    cols = ["topology", "weight"] + feats

    def load(sub):
        ps = sorted((dsdir / sub).glob("*.parquet"))
        if not ps:
            raise SystemExit(f"no parquet in {dsdir/sub}")
        return pd.concat((pd.read_parquet(p, columns=cols) for p in ps), ignore_index=True)

    print(f"[load] {dsdir}")
    tr, va, te = load("train"), load("valid"), load("test")
    print(f"       train {len(tr):,}  valid {len(va):,}  test {len(te):,}")

    y = {k: (d["topology"] > 0).astype(np.int8).to_numpy() for k, d in
         (("tr", tr), ("va", va), ("te", te))}
    w = {k: d["weight"].to_numpy() for k, d in (("tr", tr), ("va", va), ("te", te))}

    GRID = GRIDS[args.grid]
    keep = [k for k in GRID if not args.only
            or k.split()[0] in {s.strip() for s in args.only.split(",")}]
    rows = []
    print(f"\n{'point':34s} {'AUC test':>9} {'bias S%':>8} {'bias B%':>8} "
          f"{'iters':>6} {'s':>5}")
    print("-" * 78)
    for name in keep:
        over = dict(GRID[name])
        half = over.pop("_half_bkg", False)
        d_tr, y_tr, w_tr = tr, y["tr"], w["tr"]
        if half:
            rng = np.random.default_rng(cfg["seed"])
            sig = np.flatnonzero(y["tr"] == 1)
            bkg = np.flatnonzero(y["tr"] == 0)
            bkg = rng.choice(bkg, len(bkg) // 2, replace=False)
            idx = np.concatenate([sig, bkg])
            d_tr, y_tr, w_tr = tr.iloc[idx], y["tr"][idx], w["tr"][idx] * 1.0

        params = dict(cfg["xgboost"], **over)
        params["random_state"] = cfg["seed"]
        params["scale_pos_weight"] = float(
            w_tr[y_tr == 0].sum() / max(w_tr[y_tr == 1].sum(), 1e-12))
        t0 = time.time()
        m = XGBClassifier(**params)
        m.fit(d_tr[feats], y_tr, sample_weight=w_tr,
              eval_set=[(va[feats], y["va"])], sample_weight_eval_set=[w["va"]],
              verbose=False)
        s_tr = m.predict_proba(d_tr[feats])[:, 1]
        s_te = m.predict_proba(te[feats])[:, 1]
        auc = roc_auc_score(y["te"], s_te, sample_weight=w["te"])
        b = bias_bkg(s_tr, y_tr, w_tr, s_te, y["te"], w["te"],
                     cfg.get("overtrain_score_cut", 0.6))
        it = int(getattr(m, "best_iteration", params["n_estimators"] - 1))
        dt = time.time() - t0
        mark = "  <== target met" if b["B"] < 5.0 else ""
        print(f"{name:34s} {auc:9.4f} {b['S']:8.1f} {b['B']:8.1f} {it:6d} {dt:5.0f}{mark}")
        rows.append({"point": name, "params": over, "test_auc": float(auc),
                     "bias_S_pct": b["S"], "bias_B_pct": b["B"],
                     "best_iteration": it, "seconds": round(dt, 1)})

    (HERE / f"scan_regularisation_{args.grid}.json").write_text(json.dumps(rows, indent=2) + "\n")
    ok = [r for r in rows if r["bias_B_pct"] < 5.0]
    print("\nbest AUC with bias_B < 5%: " +
          (max(ok, key=lambda r: r["test_auc"])["point"] if ok else "none reached it"))
    print(f"wrote {HERE}/scan_regularisation_{args.grid}.json")


if __name__ == "__main__":
    main()
