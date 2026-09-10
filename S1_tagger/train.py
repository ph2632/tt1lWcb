#!/usr/bin/env python3
"""
Train the S1 and S1' boosted-cb taggers, benchmark them against the existing
Dbc models, and check for overtraining on a held-out test set.

  S1   = XGBoost, binary:logistic, over the 30 independent GloParT nodes
  S1'  = S1 + ak8_tau21 + ak8_tau32
  signal    = jet topology in {W(cb), t2(b'c), t2(b'b), t3(b'bc)}
  background = every stored background jet

Data split (per event, hashed -> no leakage): train 70 / valid 15 / test 15.
  valid  -> early stopping only
  test   -> never seen in any way; the overtraining reference

Per run everything lands in  S1_tagger/output/<dataset_tag>/ :
  {S1,S1p}/model.json, training_history.json, eval.npz
  summary.json      metrics, params, working points, per-class eff,
                    feature_importance, overtraining (KS + bias), comparisons
  + the plots and 2x2 panels written by make_report_panel.build_report()

Usage:
  ./.venv/bin/python S1_tagger/train.py [--config S1_tagger/config.json]
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent


def load_split(dsdir, sub, columns):
    paths = sorted((dsdir / sub).glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet parts in {dsdir/sub} -- run build_trainset.py first")
    return pd.concat((pd.read_parquet(p, columns=columns) for p in paths),
                     ignore_index=True)


def wp_table(y, score, w, bkg_effs):
    fpr, tpr, thr = roc_curve(y, score, sample_weight=w)
    out = []
    for be in bkg_effs:
        i = int(np.argmin(np.abs(fpr - be)))
        out.append({"target_bkg_eff": be, "bkg_eff": float(fpr[i]),
                    "threshold": float(thr[i]), "signal_eff": float(tpr[i])})
    return out


def per_class_eff(topo, score, w, thr, sig_codes):
    return {name: float(w[(topo == int(c)) & (score >= thr)].sum()
                        / max(w[topo == int(c)].sum(), 1e-12))
            for c, name in sig_codes.items()}


def overtrain_metrics(tr_s, tr_y, tr_w, te_s, te_y, te_w, thr):
    """KS train-vs-test for signal and background score shapes (unweighted,
    the standard overtraining check) + the signal-efficiency bias at `thr`."""
    d_sig, p_sig = ks_2samp(tr_s[tr_y == 1], te_s[te_y == 1])
    d_bkg, p_bkg = ks_2samp(tr_s[tr_y == 0], te_s[te_y == 0])
    eff_tr = tr_w[(tr_y == 1) & (tr_s >= thr)].sum() / max(tr_w[tr_y == 1].sum(), 1e-12)
    eff_te = te_w[(te_y == 1) & (te_s >= thr)].sum() / max(te_w[te_y == 1].sum(), 1e-12)
    bias = 100.0 * (eff_tr - eff_te) / max(eff_te, 1e-12)
    verdict = "OK" if min(p_sig, p_bkg) > 0.05 else \
              ("WARN" if min(p_sig, p_bkg) > 0.01 else "FAIL")
    return {"ks_sig_D": float(d_sig), "ks_sig_p": float(p_sig),
            "ks_bkg_D": float(d_bkg), "ks_bkg_p": float(p_bkg),
            "sig_eff_train": float(eff_tr), "sig_eff_test": float(eff_te),
            "sig_eff_bias_pct": float(bias), "verdict": verdict}


def eval_old_dbc(pkl_path, df):
    import joblib
    feats = ["ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
             "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"]
    if not Path(pkl_path).exists() or not all(f in df.columns for f in feats):
        return None
    return joblib.load(pkl_path).predict_proba(df[feats])[:, 1]


def eval_dbc_3class(model_path, df):
    if not Path(model_path).exists():
        return None
    m = XGBClassifier()
    m.load_model(model_path)
    feats = m.get_booster().feature_names
    if feats is None or not all(f in df.columns for f in feats):
        return None
    return m.predict_proba(df[feats])[:, 0]


def feature_importance(model, feats):
    b = model.get_booster()
    per = {k: b.get_score(importance_type=k) for k in ("gain", "weight", "cover")}
    norm = {k: (max(v.values()) if v else 1.0) for k, v in per.items()}
    rows = [{"feature": f,
             "gain": per["gain"].get(f, 0.0),
             "weight": per["weight"].get(f, 0.0),
             "cover": per["cover"].get(f, 0.0),
             "score": sum(per[k].get(f, 0.0) / norm[k] for k in per) / 3.0}
            for f in feats]
    return sorted(rows, key=lambda d: d["gain"], reverse=True)


def train_one(name, feats, tr, va, te, xgb_params, seed, sig_codes, bkg_effs, outdir):
    y = {s: (d["topology"] > 0).astype(np.int8).to_numpy() for s, d in
         (("train", tr), ("valid", va), ("test", te))}
    w = {s: d["weight"].to_numpy() for s, d in
         (("train", tr), ("valid", va), ("test", te))}

    spw = float(w["train"][y["train"] == 0].sum() / max(w["train"][y["train"] == 1].sum(), 1e-12))
    params = dict(xgb_params, random_state=seed, scale_pos_weight=spw)
    print(f"  [{name}] {len(feats)} features | train {len(tr):,} "
          f"(sig {int(y['train'].sum()):,}) | scale_pos_weight {spw:.0f}")

    model = XGBClassifier(**params)
    t0 = time.time()
    model.fit(tr[feats], y["train"], sample_weight=w["train"],
              eval_set=[(tr[feats], y["train"]), (va[feats], y["valid"])],
              sample_weight_eval_set=[w["train"], w["valid"]], verbose=False)
    fit_s = time.time() - t0
    best_it = int(getattr(model, "best_iteration", params["n_estimators"] - 1))

    mdir = outdir / name
    mdir.mkdir(parents=True, exist_ok=True)
    model.save_model(mdir / "model.json")
    (mdir / "training_history.json").write_text(json.dumps(model.evals_result(), indent=2))

    sc = {s: model.predict_proba(d[feats])[:, 1] for s, d in
          (("train", tr), ("valid", va), ("test", te))}
    ranking = feature_importance(model, feats)

    res = {"features": feats, "n_features": len(feats), "scale_pos_weight": spw,
           "best_iteration": best_it, "fit_seconds": round(fit_s, 1),
           "params": params, "feature_importance": ranking,
           "n_train": int(len(tr)), "n_train_signal": int(y["train"].sum()),
           "n_train_background": int((y["train"] == 0).sum())}
    for s in ("train", "valid", "test"):
        res[s] = {"auc": float(roc_auc_score(y[s], sc[s], sample_weight=w[s])),
                  "log_loss": float(log_loss(y[s], sc[s], sample_weight=w[s], labels=[0, 1])),
                  "n": int(len(y[s])), "n_signal": int(y[s].sum())}
    res["test"]["working_points"] = wp_table(y["test"], sc["test"], w["test"], bkg_effs)
    ref_thr = min(res["test"]["working_points"],
                  key=lambda x: abs(x["target_bkg_eff"] - 1e-3))["threshold"]
    res["test"]["per_class_signal_eff_at_1e-3"] = per_class_eff(
        te["topology"].to_numpy(), sc["test"], w["test"], ref_thr, sig_codes)
    res["overtraining"] = overtrain_metrics(
        sc["train"], y["train"], w["train"], sc["test"], y["test"], w["test"], ref_thr)

    # arrays for the plotting stage (train sub-sampled to keep npz small)
    rng = np.random.default_rng(seed)
    ntr = min(len(tr), 120_000)
    idx = rng.choice(len(tr), ntr, replace=False)
    np.savez_compressed(
        mdir / "eval.npz",
        test_score=sc["test"].astype(np.float32), test_y=y["test"],
        test_w=w["test"].astype(np.float32),
        test_topo=te["topology"].to_numpy().astype(np.int8),
        train_score=sc["train"][idx].astype(np.float32), train_y=y["train"][idx],
        train_w=w["train"][idx].astype(np.float32))

    ot = res["overtraining"]
    print(f"       AUC train {res['train']['auc']:.4f} / valid {res['valid']['auc']:.4f} "
          f"/ test {res['test']['auc']:.4f}   best_iter {best_it}   ({fit_s:.0f}s)")
    print(f"       overtraining: KS_sig p={ot['ks_sig_p']:.3f}  KS_bkg p={ot['ks_bkg_p']:.3f}  "
          f"sig-eff bias {ot['sig_eff_bias_pct']:+.1f}%   -> {ot['verdict']}")
    print(f"       top-5 by gain: "
          + ", ".join(f"{r['feature'].replace('ak8_gpt_','')}({r['gain']:.0f})"
                      for r in ranking[:5]))
    return sc["test"], y["test"], w["test"], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--no-report", action="store_true", help="skip panel rendering")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    outdir = HERE / cfg["output_dir"] / cfg["dataset_tag"]
    outdir.mkdir(parents=True, exist_ok=True)
    ds_meta = json.loads((dsdir / "dataset_complete.json").read_text())
    sig_codes = {int(k): v for k, v in ds_meta["signal_codes"].items()}
    bkg_effs = cfg["benchmark_bkg_eff"]

    feats_s1 = cfg["features_S1"]
    feats_s1p = cfg["features_S1"] + cfg["features_S1p_extra"]
    gpt_all = sorted(set(feats_s1p) | {"ak8_gpt_topbw", "ak8_gpt_topw",
                                       "ak8_gpt_qcd", "ak8_gpt_cc"})
    load_cols = sorted(set(["topology", "weight"] + feats_s1p + gpt_all))

    print(f"[load] {dsdir}")
    tr = load_split(dsdir, "train", load_cols)
    va = load_split(dsdir, "valid", load_cols)
    te = load_split(dsdir, "test", load_cols)
    print(f"       train {len(tr):,}  valid {len(va):,}  test {len(te):,}")

    summary = {
        "run_stamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "dataset_dir": str(dsdir),
        "dataset_counts": ds_meta["counts_per_split"],
        "dataset_sumw": ds_meta["sumw_per_split"],
        "split_note": "train 70 / valid 15 (early stop) / test 15 (held out)",
        "models": {}, "comparison_test_auc": {}}

    for name, feats in (("S1", feats_s1), ("S1p", feats_s1p)):
        _, _, _, res = train_one(name, feats, tr, va, te, cfg["xgboost"],
                                 cfg["seed"], sig_codes, bkg_effs, outdir)
        summary["models"][name] = res

    y_te = (te["topology"] > 0).astype(np.int8).to_numpy()
    w_te = te["weight"].to_numpy()
    cmp_cfg = cfg.get("compare_models", {})
    for key, fn in (("Dbc_old_8node", eval_old_dbc),
                    ("Dbc_3class_expanded", eval_dbc_3class)):
        s = fn(cmp_cfg.get(key, ""), te)
        if s is not None:
            summary["comparison_test_auc"][key] = float(
                roc_auc_score(y_te, s, sample_weight=w_te))
            np.savez_compressed(outdir / f"cmp_{key}.npz",
                                score=s.astype(np.float32), y=y_te,
                                w=w_te.astype(np.float32))

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n==== SUMMARY ====")
    for name in ("S1", "S1p"):
        r = summary["models"][name]
        ot = r["overtraining"]
        print(f"  {name:4s} AUC train {r['train']['auc']:.4f} / test {r['test']['auc']:.4f}"
              f"   overtraining {ot['verdict']} (bias {ot['sig_eff_bias_pct']:+.1f}%)"
              f"   best_iter {r['best_iteration']}")
        for wp in r["test"]["working_points"]:
            print(f"       bkg_eff {wp['target_bkg_eff']:<6} -> sig_eff {wp['signal_eff']:.3f}")
        pc = r["test"]["per_class_signal_eff_at_1e-3"]
        print("       per-class @1e-3: " + "  ".join(f"{k} {v:.3f}" for k, v in pc.items()))
    for k, v in summary["comparison_test_auc"].items():
        print(f"  {k:22s} test AUC {v:.4f}")
    print("\n  S1 top-10 nodes by gain:")
    for i, rk in enumerate(summary["models"]["S1"]["feature_importance"][:10], 1):
        print(f"    {i:>2}  {rk['feature'].replace('ak8_gpt_',''):<14} "
              f"gain {rk['gain']:>9.1f}  splits {rk['weight']:>5.0f}")
    print("  XGBoost: " + ", ".join(f"{k}={v}" for k, v in cfg["xgboost"].items()))

    if not args.no_report:
        import make_report_panel
        make_report_panel.build_report(cfg, outdir)
    print(f"\n  outputs -> {outdir}")


if __name__ == "__main__":
    main()
