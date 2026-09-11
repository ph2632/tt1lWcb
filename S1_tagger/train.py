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


def class_weight_lookup(cfg):
    """topology-code -> multiplier array (index 0..5), 1.0 where unspecified
    (background and any un-listed class pass through unchanged)."""
    cwm = cfg.get("class_weight_multiplier")
    if not cwm:
        return None
    lut = np.ones(max(6, max(int(k) for k in cwm) + 1), dtype=np.float64)
    for k, v in cwm.items():
        lut[int(k)] = float(v)
    return lut


def add_derived_features(df, cfg):
    """Sum mutually-exclusive raw nodes into train-time derived features
    (e.g. ak8_gpt_lepq = the 10 leptonic top/W-decay categories). Raw
    components stay in the parquet; this is a load-time-only reduction."""
    for name, comps in cfg.get("derived_features", {}).items():
        df[name] = df[comps].sum(axis=1)
    return df


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


def overtrain_metrics(tr_s, tr_y, tr_w, te_s, te_y, te_w, cut=0.6):
    """Overtraining check on the fully held-out test set.

    Primary metric (same as ../Hgg/Plot.py):  |test - train| / test  on the
    weighted fraction of each class above `cut`, i.e. in the region the
    analysis actually cuts on.  KS is kept as a secondary, shape-wide number
    but it is dominated by the low-score bulk and misses tail overtraining.
    """
    out = {"score_cut": float(cut)}
    worst = 0.0
    for cls, tag in ((1, "sig"), (0, "bkg")):
        f_tr = tr_w[(tr_y == cls) & (tr_s > cut)].sum() / max(tr_w[tr_y == cls].sum(), 1e-12)
        f_te = te_w[(te_y == cls) & (te_s > cut)].sum() / max(te_w[te_y == cls].sum(), 1e-12)
        rel = 100.0 * abs(f_te - f_tr) / max(f_te, 1e-12)
        out[f"frac_train_{tag}"] = float(f_tr)
        out[f"frac_test_{tag}"] = float(f_te)
        out[f"rel_diff_{tag}_pct"] = float(rel)
        worst = max(worst, rel)
        d, p = ks_2samp(tr_s[tr_y == cls], te_s[te_y == cls])
        out[f"ks_{tag}_D"] = float(d)
        out[f"ks_{tag}_p"] = float(p)
    out["worst_rel_diff_pct"] = float(worst)
    out["verdict"] = "OK" if worst < 5.0 else ("WARN" if worst < 15.0 else "FAIL")
    return out


def permutation_importance(model, X, y, w, feats, repeats, seed):
    """AUC drop when each input is shuffled -- the discrimination each input
    actually provides (same definition as ../Hgg/permutation_importance.py).
    XGBoost `gain` is a split-time bookkeeping number and is badly distorted
    by the correlated GloParT nodes, so this is the ranking we quote."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1], sample_weight=w)
    rows = []
    for i, f in enumerate(feats):
        d = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, i])
            d.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1], sample_weight=w))
        rows.append({"feature": f, "auc_drop": float(np.mean(d)),
                     "auc_drop_err": float(np.std(d))})
    rows.sort(key=lambda r: -r["auc_drop"])
    return base, rows


def eval_old_dbc(pkl_path, df):
    import joblib
    feats = ["ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
             "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"]
    if not Path(pkl_path).exists() or not all(f in df.columns for f in feats):
        return None
    return joblib.load(pkl_path).predict_proba(df[feats])[:, 1]


def eval_dbc_3class(model_path, df):
    """Youpeng's 3-class model -> the three class probabilities.

    Class order is (bc, bb, other).  Our signal deliberately contains the
    t2(b'b) topology, which is a bb 2-prong, so Dbc alone cannot describe it
    -- Dbc+Dbb (= 1 - Dother) is the fair single discriminant to compare.
    """
    if not Path(model_path).exists():
        return None
    m = XGBClassifier()
    m.load_model(model_path)
    feats = m.get_booster().feature_names
    if feats is None or not all(f in df.columns for f in feats):
        return None
    p = m.predict_proba(df[feats])
    return {"Dbc3_bc": p[:, 0], "Dbc3_bb": p[:, 1], "Dbc3_bc_plus_bb": p[:, 0] + p[:, 1]}


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


def train_one(name, feats, tr, va, te, xgb_params, seed, sig_codes, bkg_effs,
              outdir, cfg):
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
    X_te = te[feats].to_numpy(dtype=np.float32)
    _, perm = permutation_importance(model, X_te, y["test"], w["test"], feats,
                                     cfg.get("permutation_repeats", 3), seed)

    res = {"features": feats, "n_features": len(feats), "scale_pos_weight": spw,
           "best_iteration": best_it, "fit_seconds": round(fit_s, 1),
           "params": params, "feature_importance": ranking,
           "permutation_importance": perm,
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
    # Overtraining must compare LIKE WITH LIKE.  train weights are class-
    # reweighted (signal ~58% Wcb) while test weights are raw physics (signal
    # ~99% proxy); since Wcb scores higher than the proxy, that mixture
    # difference alone showed up as a ~9% "bias" that is not overfitting at
    # all.  So the test side is reweighted with the same class multipliers
    # HERE ONLY -- AUC, working points and per-class efficiency above keep the
    # raw physics weights and stay physically meaningful.
    cwm = class_weight_lookup(cfg)
    w_te_ot = (w["test"] * cwm[te["topology"].to_numpy()]
               if cwm is not None else w["test"])
    res["overtraining"] = overtrain_metrics(
        sc["train"], y["train"], w["train"], sc["test"], y["test"], w_te_ot,
        cfg.get("overtrain_score_cut", 0.6))
    res["overtraining"]["weighting"] = (
        "train and test both class-reweighted (like-for-like)"
        if cwm is not None else "raw physics weights")

    # Arrays for the plotting stage.  Keep EVERY training signal jet -- a flat
    # subsample left the train-signal curve with ~1/3 the statistics of the
    # test one and made it visibly noisier than the 70% split implies.  Only
    # the (far more numerous) background is thinned, and its weights are
    # rescaled so the drawn shape is unbiased.
    rng = np.random.default_rng(seed)
    sig_i = np.flatnonzero(y["train"] == 1)
    bkg_i = np.flatnonzero(y["train"] == 0)
    keep_b = min(len(bkg_i), 400_000)
    bkg_i = rng.choice(bkg_i, keep_b, replace=False)
    idx = np.concatenate([sig_i, bkg_i])
    w_plot = w["train"][idx].astype(np.float64).copy()
    w_plot[len(sig_i):] *= len(np.flatnonzero(y["train"] == 0)) / keep_b
    np.savez_compressed(
        mdir / "eval.npz",
        test_score=sc["test"].astype(np.float32), test_y=y["test"],
        test_w=w["test"].astype(np.float32),
        test_w_ot=np.asarray(w_te_ot, dtype=np.float32),   # like-for-like vs train
        test_topo=te["topology"].to_numpy().astype(np.int8),
        train_score=sc["train"][idx].astype(np.float32), train_y=y["train"][idx],
        train_w=w_plot.astype(np.float32),
        # per-jet topology of the plotted train subset, so the overtraining
        # figure can be restricted to a signal subset (S2/S3) the same way
        # the test side already can
        train_topo=tr["topology"].to_numpy()[idx].astype(np.int8))

    ot = res["overtraining"]
    print(f"       AUC train {res['train']['auc']:.3f} / valid {res['valid']['auc']:.3f} "
          f"/ test {res['test']['auc']:.3f}   best_iter {best_it}   ({fit_s:.0f}s)")
    print(f"       overtraining |test-train|/test @score>{ot['score_cut']}:  "
          f"sig {ot['rel_diff_sig_pct']:.1f}%   bkg {ot['rel_diff_bkg_pct']:.1f}%"
          f"   -> {ot['verdict']}   (KS p: sig {ot['ks_sig_p']:.3f} bkg {ot['ks_bkg_p']:.3f})")
    print("       top-5 by permutation (AUC drop): "
          + ", ".join(f"{r['feature'].replace('ak8_gpt_','')}({r['auc_drop']:.4f})"
                      for r in perm[:5]))
    return sc["test"], y["test"], w["test"], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--no-report", action="store_true", help="skip panel rendering")
    ap.add_argument("--only", default=None,
                    help="comma-separated model(s) to train, e.g. S1 or S1,S1p "
                         "(default: config 'train_models', else both)")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    outdir = HERE / cfg["output_dir"] / cfg.get("run_tag", cfg["dataset_tag"])
    outdir.mkdir(parents=True, exist_ok=True)
    marker = dsdir / "dataset_complete.json"
    if not marker.exists():
        in_progress = any((dsdir / sub).exists() and any((dsdir / sub).glob("*.parquet"))
                          for sub in ("train", "valid", "test"))
        if in_progress:
            raise SystemExit(
                f"[train.py] {marker} not found -- build_trainset.py for "
                f"dataset_tag='{cfg['dataset_tag']}' looks IN PROGRESS "
                f"(partial parquet already under {dsdir}).\n"
                f"  Check it's still running:  ps aux | grep build_trainset\n"
                f"  It finishes when its log prints '==== dataset built ===='. "
                f"Re-run train.py once that shows up.")
        raise SystemExit(
            f"[train.py] no dataset at {dsdir} -- run build_trainset.py first:\n"
            f"  ./.venv/bin/python S1_tagger/build_trainset.py --config {args.config}")
    ds_meta = json.loads(marker.read_text())
    sig_codes = {int(k): v for k, v in ds_meta["signal_codes"].items()}
    bkg_effs = cfg["benchmark_bkg_eff"]

    derived = cfg.get("derived_features", {})
    derived_raw = sorted({c for comps in derived.values() for c in comps})

    feats_s1 = cfg["features_S1"]
    feats_s1p = cfg["features_S1"] + cfg["features_S1p_extra"]
    gpt_all = sorted(set(feats_s1p) | {"ak8_gpt_topbw", "ak8_gpt_topw",
                                       "ak8_gpt_qcd", "ak8_gpt_cc"})
    # merged names (e.g. ak8_gpt_lepq) aren't real parquet columns -- swap them
    # for their raw components on load, then sum into the merged column below.
    load_cols = sorted((set(["topology", "weight"] + feats_s1p + gpt_all) - set(derived))
                       | set(derived_raw))

    print(f"[load] {dsdir}")
    tr = add_derived_features(load_split(dsdir, "train", load_cols), cfg)
    va = add_derived_features(load_split(dsdir, "valid", load_cols), cfg)
    te = add_derived_features(load_split(dsdir, "test", load_cols), cfg)

    # per-class sig sumw retarget (e.g. Wcb was 0.4%, proxy 99% unweighted) --
    # TRAIN + VALID only, so the loss/early-stopping objective is reshaped but
    # TEST (reported AUC, working points, per-class eff) stays physical.
    cwm = class_weight_lookup(cfg)
    if cwm is not None:
        tr["weight"] = tr["weight"].to_numpy() * cwm[tr["topology"].to_numpy()]
        va["weight"] = va["weight"].to_numpy() * cwm[va["topology"].to_numpy()]
        print(f"[class_weight_multiplier] applied to train+valid: "
              + ", ".join(f"{c}={m:.4g}" for c, m in enumerate(cwm) if m != 1.0))
    print(f"       train {len(tr):,}  valid {len(va):,}  test {len(te):,}")

    summary = {
        "run_stamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "dataset_dir": str(dsdir),
        "dataset_counts": ds_meta["counts_per_split"],
        "dataset_sumw": ds_meta["sumw_per_split"],
        "split_note": "train 70 / valid 15 (early stop) / test 15 (held out)",
        "models": {}, "comparison_test_auc": {}}

    model_list = ([m.strip() for m in args.only.split(",")] if args.only
                  else cfg.get("train_models", ["S1", "S1p"]))
    for name, feats in (("S1", feats_s1), ("S1p", feats_s1p)):
        if name not in model_list:
            print(f"  [{name}] skipped (not in {model_list} -- pass --only S1,S1p to include)")
            continue
        _, _, _, res = train_one(name, feats, tr, va, te, cfg["xgboost"],
                                 cfg["seed"], sig_codes, bkg_effs, outdir, cfg)
        summary["models"][name] = res

    y_te = (te["topology"] > 0).astype(np.int8).to_numpy()
    w_te = te["weight"].to_numpy()
    cmp_cfg = cfg.get("compare_models", {})
    comparisons = {"Dbc_old_8node": eval_old_dbc(cmp_cfg.get("Dbc_old_8node", ""), te)}
    three = eval_dbc_3class(cmp_cfg.get("Dbc_3class_expanded", ""), te)
    if three:
        comparisons.update(three)
    # untrained baselines: the raw GloParT sums, added cumulatively, so the ROC
    # shows what each extra node buys before any training (bc -> +bb -> +topbwc,
    # the last being the sum used to pick J).
    raw = cfg.get("raw_sum_baseline", [])
    if raw and all(c in te.columns for c in raw):
        for key, cols in (("raw_cb", raw[:1]), ("raw_cb_bb", raw[:2]),
                          ("raw_sum_bc_bb_topbwc", raw)):
            comparisons[key] = te[cols].sum(axis=1).to_numpy()
    for key, s in comparisons.items():
        if s is None:
            continue
        summary["comparison_test_auc"][key] = float(
            roc_auc_score(y_te, s, sample_weight=w_te))
        np.savez_compressed(outdir / f"cmp_{key}.npz",
                            score=np.asarray(s, dtype=np.float32), y=y_te,
                            w=w_te.astype(np.float32))

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n==== SUMMARY ====")
    for name in ("S1", "S1p"):
        if name not in summary["models"]:
            continue
        r = summary["models"][name]
        ot = r["overtraining"]
        print(f"  {name:4s} AUC train {r['train']['auc']:.3f} / test {r['test']['auc']:.3f}"
              f"   overtraining {ot['verdict']} "
              f"(sig {ot['rel_diff_sig_pct']:.1f}% / bkg {ot['rel_diff_bkg_pct']:.1f}%)"
              f"   best_iter {r['best_iteration']}")
        for wp in r["test"]["working_points"]:
            print(f"       bkg_eff {wp['target_bkg_eff']:<6} -> sig_eff {wp['signal_eff']:.3f}")
        pc = r["test"]["per_class_signal_eff_at_1e-3"]
        print("       per-class @1e-3: " + "  ".join(f"{k} {v:.3f}" for k, v in pc.items()))
    for k, v in summary["comparison_test_auc"].items():
        print(f"  {k:24s} test AUC {v:.3f}")
    if "S1" in summary["models"]:
        print("\n  S1 top-10 by permutation importance (AUC drop) [gain rank in brackets]:")
        grank = {r["feature"]: i for i, r in
                 enumerate(summary["models"]["S1"]["feature_importance"], 1)}
        for i, rk in enumerate(summary["models"]["S1"]["permutation_importance"][:10], 1):
            print(f"    {i:>2}  {rk['feature'].replace('ak8_gpt_',''):<14} "
                  f"dAUC {rk['auc_drop']:+.5f}   [gain #{grank[rk['feature']]}]")
    print("  XGBoost: " + ", ".join(f"{k}={v}" for k, v in cfg["xgboost"].items()
                                    if not k.startswith("_")))
    print(f"\n  outputs -> {outdir}")
    if not args.no_report:
        print("  (render plots with:  source LCG; python3 XGBoost_training.py)")


if __name__ == "__main__":
    main()
