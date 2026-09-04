#!/usr/bin/env python3
"""Scan (phase-space, Dbc, S_EVT) working points for the Wcb in-situ analysis.

Runs entirely on the cached derived arrays (plot_shape_comparison v4 parquet),
so no ROOT I/O. For each grid point it builds the per-bin (50 S_EVT bins)
expected yields of sig/bkg_wqq/bkg_topbc/bkg_other in the calib CR and SR,
using the same process classification and weight formula as
make_shape_asimov.py, then computes the expected signal significance from an
Asimov profile likelihood that mirrors VcbModel:

    nu_i = lambda*mu*s_i + lambda*t_i + (1-mu*r)/(1-r)*w_i + o_i,   r = 0.00084

with mu and lambda both free (q0 test: mu=0 vs mu free, lambda profiled).
This reproduces the Combine Significance method (stats-only) and was shown
to rank working points much more reliably than a naive s/sqrt(b).

Reference row: the baseline working point (ps, Dbc=0.985, S_EVT=0.7) should
come out near the Combine value ~0.5 sigma.

Optional --models wcb8,excltopbc8 adds those retrained BDT scores (computed
on the fly from the cached GloParT features) to the scan.
"""
import os, sys, time, json, argparse
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "shape_comparison")
import plot_shape_comparison as psc

from scipy.optimize import minimize

R_VAL = 0.00084          # r = 0.5*|Vcb_SM|^2, fixed in VcbModel
BINS = np.linspace(0, 1, 51)   # 50 S_EVT bins

FEATURES8 = ["ak8_gpt_bc_0", "ak8_gpt_bb_0", "ak8_gpt_cc_0", "ak8_gpt_qcd_0",
             "ak8_gpt_bs_0", "ak8_gpt_qq_0", "ak8_gpt_cs_0", "ak8_gpt_topbw_0"]

# ----------------------------------------------------------------------
# phase-space variants (all on top of the tree-level PS pt>200)
# ----------------------------------------------------------------------
def build_variants():
    def base(pt_lo=200, msd=(60, 110), tau21=None):
        m = (pt > pt_lo) & (sdmass > msd[0]) & (sdmass < msd[1])
        if tau21 is not None:
            m = m & (tau21v < tau21)
        return m

    return {
        "ps":             lambda: base(),
        "pt300":          lambda: base(pt_lo=300),
        "pt350":          lambda: base(pt_lo=350),
        "pt400":          lambda: base(pt_lo=400),
        "msd70_100":      lambda: base(msd=(70, 100)),
        "tau21_045":      lambda: base(tau21=0.45),
        "pt300_msd70_100": lambda: base(pt_lo=300, msd=(70, 100)),
        "pt350_msd70_100": lambda: base(pt_lo=350, msd=(70, 100)),
        "pt300_tau21_045": lambda: base(pt_lo=300, tau21=0.45),
        "pt350_tau21_045": lambda: base(pt_lo=350, tau21=0.45),
    }


# ----------------------------------------------------------------------
# likelihood
# ----------------------------------------------------------------------
def nll(params, s, t, w, o, n):
    mu, lam = params
    if mu < 0 or lam < 0:
        return 1e10
    wscale = (1 - mu * R_VAL) / (1 - R_VAL)
    nu = lam * mu * s + lam * t + wscale * w + o
    nu = np.maximum(nu, 1e-12)
    return -float(np.sum(n * np.log(nu) - nu))


def significance(s_c, t_c, w_c, o_c, s_s, t_s, w_s, o_s):
    """Asimov q0 significance (sqrt of profile-likelihood ratio)."""
    s = np.concatenate([s_c, s_s])
    t = np.concatenate([t_c, t_s])
    w = np.concatenate([w_c, w_s])
    o = np.concatenate([o_c, o_s])
    n = s + t + w + o
    if n.sum() < 5:
        return float("nan")

    r1 = minimize(nll, x0=[1.0, 1.0], args=(s, t, w, o, n),
                  method="Nelder-Mead",
                  options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
    r0 = minimize(lambda lam: nll([0.0, lam[0]], s, t, w, o, n),
                  x0=[1.0], method="Nelder-Mead",
                  options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
    q0 = 2.0 * (r0.fun - r1.fun)
    return float(np.sqrt(max(q0, 0.0)))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="deployed",
                    help="comma list: deployed[,wcb8[,excltopbc8]]")
    ap.add_argument("--variants", default=None,
                    help="comma list of phase-space variants (default: all)")
    ap.add_argument("--dbc-cuts", default="0.90,0.95,0.97,0.985,0.99")
    ap.add_argument("--cv-cuts", default="0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    t0 = time.time()
    cfg = psc.Config()
    samples = psc.load_all_events(cfg)

    global pt, sdmass, tau21v
    C = {}
    keys = ["weights", "score_SC", "score_Dbc", "ak8_pt_0", "ak8_sdmass_0",
            "ak8_tau21_0", "is_qcd", "ak8_type_0", "ak8_is_wbc_0",
            "ak8_n_c_in_jet_0"] + FEATURES8
    cols = {k: [] for k in keys}
    for s in samples:
        a = s["arr"]
        for k in cols:
            cols[k].append(np.asarray(a[k]))
    for k, v in cols.items():
        C[k] = np.concatenate(v)
    n_all = len(C["weights"])
    del cols, samples
    import gc; gc.collect()
    print(f"[INFO] {n_all} events loaded in {time.time()-t0:.0f}s")

    w = C["weights"].astype(np.float64)
    sevt = C["score_SC"].astype(np.float64)
    pt = C["ak8_pt_0"].astype(np.float64)
    sdmass = C["ak8_sdmass_0"].astype(np.float64)
    tau21v = C["ak8_tau21_0"].astype(np.float64)
    is_qcd = C["is_qcd"].astype(bool)
    type0 = C["ak8_type_0"].astype(np.int32)
    is_wbc = C["ak8_is_wbc_0"].astype(np.int32)
    nc0 = C["ak8_n_c_in_jet_0"].astype(np.int32)

    # process classification (mirror make_shape_asimov.py)
    is_sig = (type0 == 1) & (is_wbc == 1)
    is_wqq = (type0 == 1) & (~is_qcd) & (~is_sig)
    is_topbc = (type0 == 2) & (nc0 == 1) & (~is_qcd)
    is_other = ~(is_sig | is_wqq | is_topbc)
    print(f"[INFO] sig={is_sig.sum()} wqq={is_wqq.sum()} "
          f"topbc={is_topbc.sum()} other={is_other.sum()}")

    # score per model
    scores = {"deployed": C["score_Dbc"].astype(np.float64)}
    if "wcb8" in args.models or "excltopbc8" in args.models:
        import joblib
        X8 = np.column_stack([C[f] for f in FEATURES8]).astype(np.float32)
        del C
        gc.collect()
        for m in ("wcb8", "excltopbc8"):
            if m in args.models:
                pkl = f"train_bdt/dbc_bdt_variants/bdt_dbc_model_{m}.pkl"
                model = joblib.load(pkl)
                scores[m] = model.predict_proba(X8)[:, 1].astype(np.float64)
                print(f"[INFO] scored {m}")
        del X8
        gc.collect()
    else:
        del C
        gc.collect()

    variants = build_variants()
    if args.variants:
        keep = set(args.variants.split(","))
        variants = {k: v for k, v in variants.items() if k in keep}

    dbc_cuts = [float(x) for x in args.dbc_cuts.split(",")]
    cv_cuts = [float(x) for x in args.cv_cuts.split(",")]

    rows = []
    for mname, score in scores.items():
        for vname, vmask in variants.items():
            m_ps = vmask()
            for dbc in dbc_cuts:
                m_dbc = m_ps & (score > dbc)
                for cv in cv_cuts:
                    m_sr = m_dbc & (sevt > cv)
                    m_cal = m_dbc & (sevt <= cv)
                    h = lambda m, p: np.histogram(sevt[m], bins=BINS,
                                                  weights=w[m])[0]
                    s_c, t_c, w_c, o_c = (h(m_cal & p, p) for p in
                                          (is_sig, is_topbc, is_wqq, is_other))
                    s_s, t_s, w_s, o_s = (h(m_sr & p, p) for p in
                                          (is_sig, is_topbc, is_wqq, is_other))
                    sig = significance(s_c, t_c, w_c, o_c, s_s, t_s, w_s, o_s)
                    w_sig_sr = float(s_s.sum())
                    w_bkg_sr = float((t_s + w_s + o_s).sum())
                    w_sig_cal = float(s_c.sum())
                    w_topbc_cal = float(t_c.sum())
                    rows.append(dict(model=mname, var=vname, dbc=dbc, cv=cv,
                                     sig=sig, wsig_sr=w_sig_sr,
                                     wbkg_sr=w_bkg_sr, wsig_cal=w_sig_cal,
                                     wtopbc_cal=w_topbc_cal))
                    print(f"[{mname:12s} {vname:16s} dbc={dbc:.3f} cv={cv:.2f}] "
                          f"sig={sig:.3f}  (SR sig {w_sig_sr:.1f}, "
                          f"bkg {w_bkg_sr:.1f}; calib sig {w_sig_cal:.2f}, "
                          f"topbc {w_topbc_cal:.1f})")

    rows.sort(key=lambda r: r["sig"], reverse=True)
    out = "train_bdt/dbc_bdt_variants/workpoint_scan.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n" + "=" * 110)
    print(f"{'model':12s} {'variant':16s} {'Dbc':>6s} {'S_EVT':>6s} "
          f"{'signif':>8s} {'SRsig':>7s} {'SRbkg':>7s} {'Calsig':>7s} "
          f"{'Caltopbc':>9s}")
    print("-" * 110)
    for r in rows[:40]:
        print(f"{r['model']:12s} {r['var']:16s} {r['dbc']:6.3f} {r['cv']:6.2f} "
              f"{r['sig']:8.3f} {r['wsig_sr']:7.2f} {r['wbkg_sr']:7.2f} "
              f"{r['wsig_cal']:7.2f} {r['wtopbc_cal']:9.1f}")
    print("=" * 110)
    print(f"[DONE] scan saved to {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
