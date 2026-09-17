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

class_weight_multiplier is no longer read from config.json: it is DERIVED
here, every run, from config.json's class_weight_share and the just-loaded
train split's real per-class sumw (2026-09-13 -- previously a separate shell
step wrote it into config.json, which could silently go stale if
class_weight_share was edited without re-running that step). config.json
therefore only carries class_weight_share now; the derived numbers are
printed below and baked into summary.json's "config" for 03_Training_report_plots.py.

Usage:
  ./.venv/bin/python 02_train_tagger.py [--config S1_tagger/config.json]

Step 2/3 of the S1 tagger pipeline: 01_build_trainset.py -> 02_train_tagger.py
-> 03_Training_report_plots.py (or run_s1_tagger.py to drive all three).
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
from xgboost.callback import TrainingCallback


class TimedEval(TrainingCallback):
    """Same per-N-round eval line XGBoost's own verbose=N already prints,
    PLUS wall-clock elapsed time since fit() started (2026-09-15, user:
    "place the outputs with the run time every 100 cycles"). Replaces
    verbose=100 (pass verbose=False alongside this callback, or XGBoost
    prints its own line too and you get duplicates)."""

    def __init__(self, period=100):
        self.period = period
        self.t0 = None

    def before_training(self, model):
        self.t0 = time.time()
        return model

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            dt = time.time() - self.t0
            parts = [f"{dname}-{mname}:{vals[-1]:.5f}"
                     for dname, metrics in evals_log.items()
                     for mname, vals in metrics.items()]
            print(f"[{epoch}]\t" + "\t".join(parts) + f"\tt={dt:6.1f}s")
        return False   # never stop training ourselves

HERE = Path(__file__).resolve().parent
PKG = HERE / "S1_tagger"          # config.json / output live here


def _done_banner(t0, output):
    """One unmissable line at the very end of a run: wall time + where the
    output landed (2026-09-13, user -- the per-step timers were easy to miss
    scrolled past in a long log)."""
    line = f"[DONE  time {(time.time() - t0) / 60.0:.1f} min  output: {output}]"
    print("-" * len(line)); print(line); print("-" * len(line))


def load_split(dsdir, sub, columns):
    paths = sorted((dsdir / sub).glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet parts in {dsdir/sub} -- run 01_build_trainset.py first")
    return pd.concat((pd.read_parquet(p, columns=columns) for p in paths),
                     ignore_index=True)


def cap_topo0_weight(tr, va, te, k=20.0):
    """2026-09-16, user: investigate + heal the wild (>100%) overtraining
    bias on M3's bb-class check. Root cause (confirmed by cross-referencing
    the raw parquet's "sample" column against per-event weight): topology 0
    ("true bkg") pools MULTIPLE background MC samples (QCD, Z+jet, W+jet,
    diboson, single-top, ttbar variants, ttW) with wildly different
    xsec/Nevents ratios, unlike every SIGNAL topology (1-7), which all come
    from the single ttbar-powheg sample and have a uniform weight scale.
    A handful of QCD_merged_Skim.root events survive preselection (101 raw
    events total) with weight up to 3807 vs the topo-0 median of 0.28 --
    Kish's effective-N for topo=0 collapses from 1.41M raw train events to
    only ~5,400 EFFECTIVE events (sumw^2/sumw2) because 2 single events
    carry as much weight as ~150k typical ones. That statistical collapse
    (not a BDT hyperparameter problem) is what makes any bias/significance
    number touching topo=0 wildly unstable between train and test splits --
    whichever split a given freak event happens to land in swings the
    result by itself.

    Fix: cap any topo=0 event's weight at k times topo=0's OWN train-split
    median (measured: k=20 already recovers Neff 5.4k -> ~650k while only
    reweighting ~0.1% of events and removing ~18% of topo=0's total sumw --
    that 18% was itself an artifact of the same handful of freak events, not
    real background yield). The cap is measured ONCE on train and applied
    identically (same absolute threshold) to valid/test, so no split-
    dependent leakage. Signal topologies are untouched -- they don't have
    this problem (checked: their max/median ratios are O(1-4), not O(10^4)).
    """
    med = float(tr.loc[tr["topology"] == 0, "weight"].median())
    cap = med * k
    for name, df in (("train", tr), ("valid", va), ("test", te)):
        m = df["topology"] == 0
        n_hit = int((df.loc[m, "weight"] > cap).sum())
        sumw_before = float(df.loc[m, "weight"].sum())
        df.loc[m, "weight"] = df.loc[m, "weight"].clip(upper=cap)
        sumw_after = float(df.loc[m, "weight"].sum())
        print(f"  [weight-cap] {name}: topo=0 cap={cap:.3f} (20x train median {med:.3f}) "
              f"-- capped {n_hit}/{int(m.sum())} events, sumw {sumw_before:.1f} -> {sumw_after:.1f} "
              f"({100*(1-sumw_after/max(sumw_before,1e-12)):.1f}% removed)")
    return cap


def class_weight_lookup(cfg):
    """topology-code -> multiplier array, 1.0 where unspecified (background
    and any un-listed class pass through unchanged). Reads cfg["class_weight_
    multiplier"] -- populated in-memory by derive_class_weight_multiplier(),
    never persisted to config.json."""
    cwm = cfg.get("class_weight_multiplier")
    if not cwm:
        return None
    # floor 8 (2026-09-13): topology 7 (QCD(bb)) now exists in the dataset
    # but isn't in class_weight_share (S1 doesn't split it out specially) --
    # sizing only from cwm's OWN keys would make lut[topo] IndexError on any
    # jet with topology>=len(lut); un-listed topologies default to 1.0.
    lut = np.ones(max(8, max(int(k) for k in cwm) + 1), dtype=np.float64)
    for k, v in cwm.items():
        lut[int(k)] = float(v)
    return lut


def derive_class_weight_multiplier(train_df, cfg):
    """Derive {topology -> multiplier} from class_weight_share (the CHOSEN
    target sumw fraction per signal topology) and the train split's REAL
    per-class raw sumw:  multiplier_c = share_c * total_raw_sumw / raw_sumw_c.
    Sets cfg["class_weight_multiplier"] IN MEMORY ONLY (never written back to
    config.json) so class_weight_lookup(cfg) and summary.json's "config"
    field both see it.

    Folds in what used to be a separate shell step (run against the freshly
    built parquet, then written back into config.json) -- computing it here
    means it can never go stale relative to class_weight_share.
    """
    share = cfg.get("class_weight_share")
    if not share:
        return
    g = (train_df.loc[train_df["topology"] > 0]
                 .groupby("topology")["weight"].sum())
    tot = g.sum()
    mult = {k: round(float(v) * tot / g[int(k)], 4) for k, v in share.items()
            if int(k) in g.index}
    cfg["class_weight_multiplier"] = mult
    print(f"[class_weight_multiplier] derived from real train sumw: "
          + ", ".join(f"{k}={v:g}" for k, v in mult.items()))


# --------------------------------------------------------------------------- #
# M3: 4-class multiclass model (bkg/cb/bb/bbc), trained ALONGSIDE the binary
# S1 model in the same process on the SAME already-loaded tr/va/te -- no
# second parquet read.  See train_one_multiclass() for why this needs its
# own weight-derivation path rather than reusing class_weight_multiplier.
# --------------------------------------------------------------------------- #
def build_multiclass_labels(topo, class_groups):
    """topology code array -> M3 class-label array, via {topology_code(str):
    class_label(int)} from config's M3_class_groups."""
    lut = np.zeros(max(int(k) for k in class_groups) + 1, dtype=np.int8)
    for k, v in class_groups.items():
        lut[int(k)] = int(v)
    return lut[topo]


def derive_multiclass_weight_multiplier(topo_train, w_train_raw, class_groups,
                                        share, subshare=None):
    """Derive {TOPOLOGY code -> multiplier} (2026-09-13: topology-keyed, not
    class-label-keyed as before) so a class with more than one constituent
    topology (e.g. bb = t2(b'b)/Zbb/QCD(bb)) can be weighted DIFFERENTLY
    WITHIN the class instead of just uniformly:

      multiplier_t = class_share_c * topo_subshare_t * total_raw_sumw / raw_sumw_t
      where c = class_groups[t]

    `subshare` is OPTIONAL, per class: {class_label(str): {topology(str):
    sub_share}}. A class (or a topology inside a listed class) with no entry
    there falls back to ITS NATURAL raw-sumw proportion within the class --
    i.e. exactly the old single-multiplier-per-class behaviour, generalised
    rather than replaced (so classes nobody has asked to split, e.g. cb and
    bbc right now, are numerically unaffected by this change).

    Binary training gets background/signal balance for free from
    scale_pos_weight = sum(w_bkg)/sum(w_sig); multi:softprob has no such
    knob, so the background/signal balance has to be baked into
    sample_weight directly here -- unlike the binary path, this NEVER
    touches cfg["class_weight_multiplier"] (topology-keyed for S1/S1p, but
    different numbers entirely); it returns its own lookup instead.
    """
    subshare = subshare or {}
    topo_to_class = {int(k): int(v) for k, v in class_groups.items()}
    g_topo = pd.Series(w_train_raw).groupby(topo_train).sum()
    tot = g_topo.sum()

    g_class = {}
    for t, w in g_topo.items():
        c = topo_to_class.get(int(t))
        if c is not None:
            g_class[c] = g_class.get(c, 0.0) + float(w)

    mult = {}
    for k_str, class_share in share.items():
        c = int(k_str)
        if c not in g_class or g_class[c] <= 0:
            continue
        topos = sorted(t for t, cc in topo_to_class.items() if cc == c and t in g_topo.index)
        sub = subshare.get(k_str, {})
        for t in topos:
            s_t = float(sub[str(t)]) if str(t) in sub else float(g_topo[t]) / g_class[c]
            mult[t] = round(float(class_share) * s_t * tot / float(g_topo[t]), 4)

    # sized to cover every topology class_groups knows about, not just the
    # ones with a derived multiplier (unlisted topologies default to 1.0 --
    # e.g. background, code 0, is intentionally handled this way already)
    lut = np.ones(max(list(mult) + list(topo_to_class) + [0]) + 1, dtype=np.float64)
    for t, v in mult.items():
        lut[t] = v
    return lut, mult


def permutation_importance_multiclass(model, X, y, w, feats, repeats, seed, class_names):
    """Macro one-vs-rest AUC drop (direct multiclass analogue of
    permutation_importance() above) AND, in the SAME pass, the per-class
    one-vs-rest AUC drop for every non-background class -- one predict_proba
    call per (feature, repeat) feeds both, rather than running the shuffle
    loop 4 separate times (once per class + once for the macro average)."""
    rng = np.random.default_rng(seed)
    sig_classes = [(c, cn) for c, cn in enumerate(class_names) if c != 0]
    base_proba = model.predict_proba(X)
    base_macro = roc_auc_score(y, base_proba, multi_class="ovr",
                               average="macro", sample_weight=w)
    base_c = {cn: roc_auc_score((y == c).astype(np.int8), base_proba[:, c],
                                sample_weight=w) for c, cn in sig_classes}
    rows_macro = []
    rows_c = {cn: [] for _, cn in sig_classes}
    for i, f in enumerate(feats):
        d_macro, d_c = [], {cn: [] for _, cn in sig_classes}
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, i])
            proba_p = model.predict_proba(Xp)
            d_macro.append(base_macro - roc_auc_score(
                y, proba_p, multi_class="ovr", average="macro", sample_weight=w))
            for c, cn in sig_classes:
                d_c[cn].append(base_c[cn] - roc_auc_score(
                    (y == c).astype(np.int8), proba_p[:, c], sample_weight=w))
        rows_macro.append({"feature": f, "auc_drop": float(np.mean(d_macro)),
                           "auc_drop_err": float(np.std(d_macro))})
        for _, cn in sig_classes:
            rows_c[cn].append({"feature": f, "auc_drop": float(np.mean(d_c[cn])),
                               "auc_drop_err": float(np.std(d_c[cn]))})
    rows_macro.sort(key=lambda r: -r["auc_drop"])
    for cn in rows_c:
        rows_c[cn].sort(key=lambda r: -r["auc_drop"])
    return base_macro, rows_macro, rows_c


def train_one_multiclass(name, feats, tr, va, te, xgb_params, seed, class_groups,
                         class_names, share, bkg_effs, outdir, cfg,
                         w_tr_raw, w_va_raw, subshare=None):
    """4-class (bkg/cb/bb/bbc) sibling of train_one().  Shares tr/va/te with
    the binary model(s) -- caller passes w_tr_raw/w_va_raw captured BEFORE
    the binary path's in-place class_weight_multiplier reweight of
    tr["weight"]/va["weight"], so M3's own weighting starts from the
    untouched physics weight, not S1's already-reweighted one.

    subshare (optional): WITHIN-class topology sub-shares, e.g. bb =
    t2(b'b)/Zbb/QCD(bb) at 60/36/4 -- see derive_multiclass_weight_multiplier."""
    y = {"train": build_multiclass_labels(tr["topology"].to_numpy(), class_groups),
         "valid": build_multiclass_labels(va["topology"].to_numpy(), class_groups),
         "test": build_multiclass_labels(te["topology"].to_numpy(), class_groups)}
    topo_tr = tr["topology"].to_numpy()
    topo_va = va["topology"].to_numpy()

    lut, mult = derive_multiclass_weight_multiplier(topo_tr, w_tr_raw, class_groups,
                                                     share, subshare)
    w = {"train": w_tr_raw * lut[topo_tr],
         "valid": w_va_raw * lut[topo_va],
         "test": te["weight"].to_numpy()}          # physical, untouched (like S1's test)
    print(f"[{name} class_weight_multiplier] derived from real train sumw (by topology): "
          + ", ".join(f"topo{t}={v:g}" for t, v in sorted(mult.items())))

    params = dict(xgb_params, random_state=seed, objective="multi:softprob",
                  num_class=len(class_names), eval_metric="mlogloss")
    print(f"  [{name}] {len(feats)} features | train {len(tr):,} | "
          f"classes {class_names}")

    model = XGBClassifier(**params, callbacks=[TimedEval(100)])
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

    proba = {s: model.predict_proba(d[feats]) for s, d in
             (("train", tr), ("valid", va), ("test", te))}
    pred_te = proba["test"].argmax(axis=1)

    ranking = feature_importance(model, feats)
    X_te = te[feats].to_numpy(dtype=np.float32)
    macro_auc, perm, perm_per_class = permutation_importance_multiclass(
        model, X_te, y["test"], w["test"], feats,
        cfg.get("permutation_repeats", 3), seed, class_names)

    res = {"features": feats, "n_features": len(feats), "class_names": class_names,
           "class_weight_multiplier": mult,
           "best_iteration": best_it, "fit_seconds": round(fit_s, 1),
           "params": params, "feature_importance": ranking,
           "permutation_importance": perm, "macro_auc_ovr_test": float(macro_auc),
           "n_train": int(len(tr))}
    for s in ("train", "valid", "test"):
        res[s] = {"log_loss": float(log_loss(y[s], proba[s], sample_weight=w[s],
                                             labels=list(range(len(class_names))))),
                  "n": int(len(y[s]))}

    # Overtraining must compare LIKE WITH LIKE (same fix as S1's binary
    # w_te_ot, 2026-09-13 -- missed here originally). w["train"] pools
    # topologies with WILDLY different multipliers per class (e.g. bb=109x,
    # bbc=1249x) while w["test"] is raw physical weight; for a one-vs-rest
    # "rest" pool spanning multiple topologies, train's reweighted MIX and
    # test's raw MIX are then completely different compositions even for a
    # perfectly-generalising model, which reads as huge spurious "bias".
    # test_w_ot applies the SAME topology multiplier to test, for the
    # overtraining check ONLY -- AUC/working-points/sig_eff above stay on
    # raw physical test weight so they remain physically meaningful.
    topo_te = te["topology"].to_numpy()
    w_te_ot = te["weight"].to_numpy() * lut[topo_te]

    # per-class (cb/bb/bbc) one-vs-rest AUC + working points + overtraining,
    # reusing the SAME statistics helpers the binary model uses -- each is
    # just fed that class's probability column and a (label==c) binary mask,
    # so no new statistics code was needed for this part.
    per_class = {}
    for c, cname in enumerate(class_names):
        if c == 0:
            continue    # background is not itself a signal to report a WP for
        yb = {s: (y[s] == c).astype(np.int8) for s in ("train", "test")}
        sb = {s: proba[s][:, c] for s in ("train", "test")}
        wp = wp_table(yb["test"], sb["test"], w["test"], bkg_effs)
        ref_thr = min(wp, key=lambda x: abs(x["target_bkg_eff"] - 1e-3))["threshold"]
        ot = overtrain_metrics(sb["train"], yb["train"], w["train"],
                               sb["test"], yb["test"], w_te_ot,
                               cfg.get("overtrain_score_cut", 0.6))
        per_class[cname] = {
            "auc_ovr": float(roc_auc_score(yb["test"], sb["test"], sample_weight=w["test"])),
            "working_points": wp,
            "sig_eff_at_1e-3_bkg": float(
                w["test"][(yb["test"] == 1) & (sb["test"] >= ref_thr)].sum()
                / max(w["test"][yb["test"] == 1].sum(), 1e-12)),
            "overtraining": ot,
            "permutation_importance": perm_per_class[cname],
        }
    res["per_class"] = per_class

    # weighted confusion matrix (rows=true class, cols=predicted=argmax),
    # row-normalised -- "of the true-label-c test jets, what fraction did the
    # model call each class".  Physical (raw) test weights throughout.
    K = len(class_names)
    conf = np.zeros((K, K))
    for t in range(K):
        m = y["test"] == t
        wsum = w["test"][m].sum()
        if wsum > 0:
            for p in range(K):
                conf[t, p] = float(w["test"][m & (pred_te == p)].sum() / wsum)
    res["confusion_matrix_row_normalised"] = conf.tolist()

    np.savez_compressed(
        mdir / "eval.npz",
        test_proba=proba["test"].astype(np.float32), test_y=y["test"],
        test_w=w["test"].astype(np.float32),
        test_w_ot=w_te_ot.astype(np.float32),   # like-for-like vs train (see above)
        test_topo=topo_te.astype(np.int8),
        # t3(b'cq) proxy flag (2026-09-13): plain annotation, NOT a topology
        # code -- see build_trainset.py. bbc has no real proxy of its own
        # (unlike cb's t2(b'c) proxy), so the score plot uses this as a
        # rough, explicitly-imperfect reference instead.
        test_t3bcq_proxy=te["t3bcq_proxy_flag"].to_numpy().astype(bool),
        train_proba=proba["train"].astype(np.float32), train_y=y["train"],
        train_w=w["train"].astype(np.float32),
        train_topo=tr["topology"].to_numpy().astype(np.int8),
        class_names=np.array(class_names))

    print(f"       macro AUC (OVR, test) {macro_auc:.3f}   best_iter {best_it}   ({fit_s:.0f}s)")
    for cname, pc in per_class.items():
        ot = pc["overtraining"]
        print(f"       {cname:5s} AUC(vs rest) {pc['auc_ovr']:.3f}   "
              f"sig_eff@1e-3 {pc['sig_eff_at_1e-3_bkg']:.3f}   "
              f"overtrain {ot['verdict']} (sig {ot['rel_diff_sig_pct']:.1f}% / "
              f"bkg {ot['rel_diff_bkg_pct']:.1f}%)")
    print("       top-5 by permutation (macro-AUC drop): "
          + ", ".join(f"{r['feature'].replace('ak8_gpt_','')}({r['auc_drop']:.4f})"
                      for r in perm[:5]))
    return res


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

    model = XGBClassifier(**params, callbacks=[TimedEval(100)])
    t0 = time.time()
    # TimedEval (2026-09-15, was verbose=100 since 2026-09-13): fit on 5M+
    # rows for ~2500-2900 rounds takes 45-65 min with ZERO console output
    # otherwise, which reads as "hung" -- this prints eval AUC + wall time
    # every 100 rounds, no effect on the fit itself.
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
    t_wall0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "config.json")
    ap.add_argument("--no-report", action="store_true", help="skip panel rendering")
    ap.add_argument("--only", default=None,
                    help="comma-separated model(s) to train, e.g. S1 or S1,S1p "
                         "(default: config 'train_models', else both)")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    # 2026-09-15: n_jobs is NOT "more is better" for this workload -- measured
    # directly on this node (M3_xgboost params, 14% train subsample, 400
    # rounds each): n_jobs=6 -> 0.104s/round (best), 8 -> 0.115s, but
    # 12 -> 0.316s (3x slower) and 16 -> 0.837s (8x slower than 6!). depth=2
    # trees are cheap per split, so this is memory-bandwidth-bound, not
    # core-bound -- more threads past a small number just adds contention/
    # false-sharing overhead. (Earlier same-day attempt used
    # os.sched_getaffinity(0) directly, i.e. ALL detected cores -- that was
    # the wrong direction entirely; the fix isn't "detect the node's core
    # count", it's "cap well below it regardless of node size".) Still uses
    # sched_getaffinity so a genuinely core-constrained slot (<6 available)
    # doesn't request more than it has.
    _n_jobs = min(len(os.sched_getaffinity(0)), 6)
    for _blk in ("xgboost", "M3_xgboost"):
        if _blk in cfg:
            cfg[_blk]["n_jobs"] = _n_jobs
    print(f"[cpu] n_jobs={_n_jobs} (capped at 6 -- measured optimum, see comment)")

    dsdir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    outdir = PKG / cfg["output_dir"] / cfg.get("run_tag", cfg["dataset_tag"])
    outdir.mkdir(parents=True, exist_ok=True)
    marker = dsdir / "dataset_complete.json"
    if not marker.exists():
        in_progress = any((dsdir / sub).exists() and any((dsdir / sub).glob("*.parquet"))
                          for sub in ("train", "valid", "test"))
        if in_progress:
            raise SystemExit(
                f"[02_train_tagger] {marker} not found -- 01_build_trainset.py for "
                f"dataset_tag='{cfg['dataset_tag']}' looks IN PROGRESS "
                f"(partial parquet already under {dsdir}).\n"
                f"  Check it's still running:  ps aux | grep 01_build_trainset\n"
                f"  It finishes when its log prints '==== dataset built ===='. "
                f"Re-run 02_train_tagger.py once that shows up.")
        raise SystemExit(
            f"[02_train_tagger] no dataset at {dsdir} -- run 01_build_trainset.py first:\n"
            f"  ./.venv/bin/python 01_build_trainset.py --config {args.config}")
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
    load_cols = sorted((set(["topology", "weight", "t3bcq_proxy_flag"] + feats_s1p + gpt_all)
                        - set(derived)) | set(derived_raw))

    print(f"[load] {dsdir}")
    tr = add_derived_features(load_split(dsdir, "train", load_cols), cfg)
    va = add_derived_features(load_split(dsdir, "valid", load_cols), cfg)
    te = add_derived_features(load_split(dsdir, "test", load_cols), cfg)
    # 2026-09-16, user: investigate + heal M3 bb-class's wild overtraining
    # bias -- see cap_topo0_weight() docstring for the full root-cause
    # analysis (a handful of freak-weight QCD/Z+jet/W+jet events collapsing
    # topo=0's effective statistics). Applied BEFORE any reweighting below,
    # so every downstream sumw-derived multiplier sees the healed weight.
    cap_topo0_weight(tr, va, te)
    # snapshotted BEFORE any reweighting below -- M3 (multiclass) needs its
    # OWN independent weight-derivation starting from the untouched physics
    # weight, not S1's already-reweighted tr["weight"]/va["weight"] (see
    # train_one_multiclass() docstring).
    w_tr_raw = tr["weight"].to_numpy().copy()
    w_va_raw = va["weight"].to_numpy().copy()

    # per-class sig sumw retarget (e.g. Wcb was 0.4%, proxy 99% unweighted) --
    # TRAIN + VALID only, so the loss/early-stopping objective is reshaped but
    # TEST (reported AUC, working points, per-class eff) stays physical.
    # Multiplier is DERIVED here from tr's real (pre-reweight) per-class sumw
    # -- see derive_class_weight_multiplier() docstring.
    derive_class_weight_multiplier(tr, cfg)
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

    # 2026-09-15 bugfix: summary.json is written FRESH every run (the dict
    # literal above), so a model skipped THIS run (e.g. S1, parked below)
    # simply has no entry -- even though its model.json/eval.npz are still
    # sitting on disk untouched. 03_Training_report_plots.py hard-requires an "S1"
    # entry and exits immediately without one. Carry forward any PRIOR
    # summary.json's entry for a model not in model_list this run, so the
    # render step keeps working off the still-valid last-trained artifacts
    # instead of breaking every time a model is intentionally skipped.
    _prev_summary_f = outdir / "summary.json"
    if _prev_summary_f.exists():
        try:
            _prev = json.loads(_prev_summary_f.read_text())
            for _mname, _mres in _prev.get("models", {}).items():
                if _mname not in model_list and (outdir / _mname / "eval.npz").exists():
                    summary["models"][_mname] = _mres
                    print(f"  [{_mname}] carried forward from prior summary.json "
                          f"(not retrained this run, artifacts on disk unchanged)")
        except Exception as _e:
            print(f"[WARN] could not carry forward prior summary.json ({_e})")

    # 2026-09-15, user: "Comment out the Train S1 model completely. I want to
    # keep only the M3 for the moment (3 classes, each binary)." -- M3's 3
    # one-vs-rest columns (cb/bb/bbc) now cover what S1 targeted, so the
    # separate binary S1 tagger is parked. The already-trained S1/model.json
    # is untouched and still usable by 04_Make_plots.py (score_S1) -- this
    # only stops RE-training it. Re-enable by uncommenting + restoring
    # "S1" to config.json's train_models (or --only S1) when needed again.
    # for name, feats in (("S1", feats_s1), ("S1p", feats_s1p)):
    #     if name not in model_list:
    #         print(f"  [{name}] skipped (not in {model_list} -- pass --only S1,M3 to include)")
    #         continue
    #     _, _, _, res = train_one(name, feats, tr, va, te, cfg["xgboost"],
    #                              cfg["seed"], sig_codes, bkg_effs, outdir, cfg)
    #     summary["models"][name] = res
    print("  [S1] disabled (2026-09-15, user) -- training M3 only this run")

    # M3: 4-class multiclass (bkg/cb/bb/bbc), same tr/va/te already in memory
    # -- no second parquet read.  Uses the SAME feature list as S1 (feats_s1)
    # so the two architectures are compared on identical inputs.
    if "M3" in model_list:
        class_groups = cfg.get("M3_class_groups")
        class_names = cfg.get("M3_class_names")
        share = cfg.get("M3_class_weight_share")
        if not (class_groups and class_names and share):
            print("  [M3] skipped -- M3_class_groups/M3_class_names/"
                  "M3_class_weight_share missing from config")
        else:
            # M3 gets its OWN xgboost params (more regularised than S1's,
            # 2026-09-13) -- falls back to S1's if M3_xgboost isn't set.
            m3_xgb = cfg.get("M3_xgboost", cfg["xgboost"])
            subshare = cfg.get("M3_topo_subshare")
            summary["models"]["M3"] = train_one_multiclass(
                "M3", feats_s1, tr, va, te, m3_xgb, cfg["seed"],
                class_groups, class_names, share, bkg_effs, outdir, cfg,
                w_tr_raw, w_va_raw, subshare)
    else:
        print(f"  [M3] skipped (not in {model_list} -- pass --only S1,M3 to include)")

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
    if "M3" in summary["models"]:
        r = summary["models"]["M3"]
        print(f"  M3   macro AUC (OVR, test) {r['macro_auc_ovr_test']:.3f}"
              f"   best_iter {r['best_iteration']}")
        for cname, pc in r["per_class"].items():
            ot = pc["overtraining"]
            print(f"       {cname:5s} AUC(vs rest) {pc['auc_ovr']:.3f}   "
                  f"sig_eff@1e-3 {pc['sig_eff_at_1e-3_bkg']:.3f}   "
                  f"overtrain {ot['verdict']} (sig {ot['rel_diff_sig_pct']:.1f}% / "
                  f"bkg {ot['rel_diff_bkg_pct']:.1f}%)")
        print("       confusion matrix (row=true, col=pred, row-normalised, test):")
        names = r["class_names"]
        print("            " + "".join(f"{n:>8s}" for n in names))
        for i, n in enumerate(names):
            print(f"       {n:5s}" + "".join(
                f"{r['confusion_matrix_row_normalised'][i][j]:8.3f}" for j in range(len(names))))
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
    if not args.no_report:
        print("\n  (render plots with:  ./.venv/bin/python 03_Training_report_plots.py)")
    print()
    _done_banner(t_wall0, outdir)


if __name__ == "__main__":
    main()
