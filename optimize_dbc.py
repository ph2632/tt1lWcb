#!/usr/bin/env python3
"""Optimize the Boosted Dbc BDT.

Trains label/feature variants of the Dbc BDT on the scored MC sample
(loaded from the plot_shape_comparison v3 parquet cache, so no ROOT I/O),
then scores every event with each variant and evaluates:

  - discrimination : ROC AUC of true W->cb vs all other jets, and
                     W->cb vs top(bc) (the in-situ proxy tension);
  - in-situ validity: chi2/ndof and KS of the normalized S_EVT
                     distributions of W->cb vs top(bc) after Dbc > thr
                     (thr in {0.95, 0.985}) with pt_J1 > 200;
  - a naive SR significance proxy s/sqrt(b) in S_EVT>0.7 & Dbc>thr.

Reference rows for the deployed BDT and the ratio Dbc are included.
Trained models are saved (joblib .pkl) for the Combine step.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "shape_comparison")
import plot_shape_comparison as psc

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
FEATURES8 = ["ak8_gpt_bc_0", "ak8_gpt_bb_0", "ak8_gpt_cc_0", "ak8_gpt_qcd_0",
             "ak8_gpt_bs_0", "ak8_gpt_qq_0", "ak8_gpt_cs_0", "ak8_gpt_topbw_0"]
FEATURES10 = FEATURES8 + ["ak8_gpt_bqq_0", "ak8_gpt_topw_0"]

XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric=["logloss", "auc"],
    tree_method="hist", random_state=42,
)

# label: 'geo' geometric (1,1,2) [reproduces current]; 'union' Wcb U top(bc);
#        'wcb' Wcb only (topbc in bg); 'excltopbc' Wcb only, topbc excluded.
VARIANTS = [
    dict(name="geo8",      label="geo",      feats=FEATURES8),
    dict(name="union8",    label="union",    feats=FEATURES8),
    dict(name="union10",   label="union",    feats=FEATURES10),
    dict(name="wcb8",      label="wcb",      feats=FEATURES8),
    dict(name="excltopbc8", label="excltopbc", feats=FEATURES8),
]

BG_CAP = 4_000_000   # subsample background for training speed
RNG = np.random.default_rng(42)
OUT_DIR = "./train_bdt/dbc_bdt_variants"
THRESHOLDS = [0.95, 0.985]
SEVT_CUT = 0.7


# ----------------------------------------------------------------------
def build_label(lbl, is_wcb, is_topbc, n_b, n_c, n_in):
    if lbl == "geo":
        return (n_b == 1) & (n_c == 1) & (n_in == 2)
    if lbl == "union":
        return is_wcb | is_topbc
    if lbl == "wcb":
        return is_wcb
    if lbl == "excltopbc":
        return is_wcb  # background selection handled in train()
    raise ValueError(lbl)


# ----------------------------------------------------------------------
def normalized_hist(values, weights, bins):
    h_raw, _ = np.histogram(values, bins=bins, weights=weights)
    h_w2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    total = np.sum(h_raw)
    if total <= 0:
        return None, None, None
    h_norm = h_raw / total
    err_norm = np.sqrt(np.maximum(h_w2, 0)) / total
    return h_norm, err_norm


def chi2_ks(v1, w1, v2, w2, bins=np.linspace(0, 1, 11)):
    """stat-only chi2/ndof and weighted-KS between two distributions."""
    h1, e1 = normalized_hist(v1, w1, bins)
    h2, e2 = normalized_hist(v2, w2, bins)
    if h1 is None or h2 is None:
        return float("nan"), float("nan")
    m = (e1 > 0) & (e2 > 0) & (h1 + h2 > 0)
    if m.sum() == 0:
        return float("nan"), float("nan")
    chi2 = float(np.sum((h1[m] - h2[m]) ** 2 / (e1[m] ** 2 + e2[m] ** 2)))
    ndof = int(m.sum())
    # weighted KS
    x = np.concatenate([v1, v2])
    o = np.argsort(x)
    ww = np.concatenate([w1, w2])[o]
    c1 = np.cumsum(np.where(o < len(v1), ww, 0.0))
    c2 = np.cumsum(np.where(o >= len(v1), ww, 0.0))
    ks = float(np.max(np.abs(c1 / c1[-1] - c2 / c2[-1])))
    return chi2 / ndof, ks


def eval_score(score, w, sevt, pt, is_wcb, is_topbc, tag):
    """Metrics for one Dbc score on the full sample."""
    row = {"tag": tag}
    aw = np.abs(w)
    if is_wcb.sum() > 1 and (~is_wcb).sum() > 1:
        row["auc_wcb_vs_rest"] = float(roc_auc_score(
            is_wcb, score, sample_weight=aw))
    sel = is_wcb | is_topbc
    if is_wcb[sel].sum() > 1 and is_topbc[sel].sum() > 1:
        row["auc_wcb_vs_topbc"] = float(roc_auc_score(
            is_wcb[sel], score[sel], sample_weight=aw[sel]))
    for thr in THRESHOLDS:
        m = (pt > 200) & (score > thr)
        ms = m & is_wcb
        mt = m & is_topbc
        row[f"nwcb_{thr}"] = int(ms.sum())
        row[f"ntopbc_{thr}"] = int(mt.sum())
        row[f"wwcb_{thr}"] = float(np.sum(w[ms]))
        row[f"wtopbc_{thr}"] = float(np.sum(w[mt]))
        if ms.sum() >= 3 and mt.sum() >= 3:
            c2, ks = chi2_ks(sevt[ms], w[ms], sevt[mt], w[mt])
            row[f"chi2ndof_{thr}"] = c2
            row[f"ks_{thr}"] = ks
        else:
            row[f"chi2ndof_{thr}"] = float("nan")
            row[f"ks_{thr}"] = float("nan")
        # naive SR significance proxy: S_EVT>cut
        sr = m & (sevt > SEVT_CUT)
        s = float(np.sum(w[sr & is_wcb]))
        b = float(np.sum(w[sr & ~is_wcb]))
        row[f"s_over_sqrtb_{thr}"] = s / np.sqrt(b) if b > 0 else float("nan")
    return row


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    cfg = psc.Config()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("[INFO] Optimize Boosted Dbc BDT")
    print(f"[INFO] Loading cached arrays (v3 parquet) ...")
    samples = psc.load_all_events(cfg)

    # --- concatenate the needed per-event fields across samples ---
    cols = {k: [] for k in
            ["weights", "score_SC", "ak8_pt_0", "is_signal", "is_top_bc",
             "score_Dbc", "score_Dbc_ratio",
             "ak8_n_b_in_jet_0", "ak8_n_c_in_jet_0", "ak8_n_in_jet_0"]
            + FEATURES10}
    for s in samples:
        a = s["arr"]
        for k in cols:
            cols[k].append(np.asarray(a[k], dtype=np.float32 if k.startswith("ak8_gpt") else np.float64))
    C = {k: np.concatenate(v) for k, v in cols.items()}
    n_all = len(C["weights"])
    del cols
    del samples
    import gc
    gc.collect()
    print(f"[INFO] Total events: {n_all}  (load {time.time()-t0:.0f}s)")

    w = C["weights"]
    sevt = C["score_SC"]
    pt = C["ak8_pt_0"]
    is_wcb = C["is_signal"].astype(bool)
    is_topbc = C["is_top_bc"].astype(bool)
    n_b = C["ak8_n_b_in_jet_0"].astype(np.int32)
    n_c = C["ak8_n_c_in_jet_0"].astype(np.int32)
    n_in = C["ak8_n_in_jet_0"].astype(np.int32)

    results = []

    # --- reference scores (deployed BDT + ratio) ---
    print("\n--- reference scores ---")
    for key, tag in [("score_Dbc", "bdt_deployed"),
                     ("score_Dbc_ratio", "ratio")]:
        score = C[key]
        r = eval_score(score, w, sevt, pt, is_wcb, is_topbc, tag)
        results.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r.items()}))

    # --- training set ---
    X10 = np.column_stack([C[f] for f in FEATURES10]).astype(np.float32)
    X8 = X10[:, :8]  # view, avoids a second copy

    for v in VARIANTS:
        t1 = time.time()
        name, lbl = v["name"], v["label"]
        X = X10 if len(v["feats"]) == 10 else X8
        y = build_label(lbl, is_wcb, is_topbc, n_b, n_c, n_in)

        if lbl == "excltopbc":
            train_mask = y | (~is_topbc)  # topbc excluded from both classes
        else:
            train_mask = np.ones(n_all, dtype=bool)

        # subsample background for training speed
        neg = train_mask & ~y
        pos = train_mask & y
        n_neg = int(np.sum(neg))
        if n_neg > BG_CAP:
            keep = np.zeros(n_neg, dtype=bool)
            keep[:BG_CAP] = True
            RNG.shuffle(keep)
            neg_idx = np.where(neg)[0][keep]
        else:
            neg_idx = np.where(neg)[0]
        pos_idx = np.where(pos)[0]
        idx = np.concatenate([pos_idx, neg_idx])
        RNG.shuffle(idx)

        y_tr = y[idx].astype(np.int32)
        w_tr = np.abs(w[idx]).astype(np.float64)
        sum_w_pos = w_tr[y_tr == 1].sum()
        sum_w_neg = w_tr[y_tr == 0].sum()
        spw = sum_w_neg / max(sum_w_pos, 1e-12)
        params = dict(XGB_PARAMS)
        params["scale_pos_weight"] = spw

        model = XGBClassifier(**params)
        model.fit(X[idx], y_tr, sample_weight=w_tr, verbose=False)
        print(f"[TRAIN] {name}: {len(idx)} rows (pos {pos_idx.size}, "
              f"spw {spw:.1f}) in {time.time()-t1:.0f}s")

        # save model
        pkl = os.path.join(OUT_DIR, f"bdt_dbc_model_{name}.pkl")
        import joblib
        joblib.dump(model, pkl)

        # score all events
        score = model.predict_proba(X)[:, 1].astype(np.float32)
        r = eval_score(score, w, sevt, pt, is_wcb, is_topbc, name)
        results.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r.items()}))

    # --- summary table ---
    print("\n" + "=" * 100)
    hdr = (f"{'tag':16s} {'auc_rest':>9s} {'auc_vsTbc':>10s} "
           f"{'chi2_985':>9s} {'KS_985':>7s} {'chi2_95':>9s} {'KS_95':>7s} "
           f"{'s/sqrtb_985':>11s} {'s/sqrtb_95':>11s} "
           f"{'wWcb_985':>9s} {'wTbc_985':>9s}")
    print(hdr)
    print("-" * 100)
    for r in results:
        def g(k):
            v = r.get(k, float("nan"))
            return f"{v:9.3f}" if isinstance(v, float) else f"{v:9}"
        print(f"{r['tag']:16s} {g('auc_wcb_vs_rest'):>9s} "
              f"{g('auc_wcb_vs_topbc'):>10s} {g('chi2ndof_0.985'):>9s} "
              f"{g('ks_0.985'):>7s} {g('chi2ndof_0.95'):>9s} "
              f"{g('ks_0.95'):>7s} {g('s_over_sqrtb_0.985'):>11s} "
              f"{g('s_over_sqrtb_0.95'):>11s} {g('wwcb_0.985'):>9s} "
              f"{g('wtopbc_0.985'):>9s}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n[DONE] models + summary in {OUT_DIR} "
          f"(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
