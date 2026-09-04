#!/usr/bin/env python3
"""Scan kinematic cuts for D_bc spectrum similarity (Wcb signal vs Topbc).

Goal: find a set of cuts on several variables that makes the two D_bc
spectra (top(bc) background and W->cb signal) consistent within statistics.

For each kinematic cut (from plot_shape_comparison.build_kinematic_cuts)
we compute, on the v7 cached arrays:
  - weighted chi2/ndof and KS between the normalized D_bc spectra
    (bins over [0.5, 1.0], where D_bc carries statistics);
  - signal/topbc weighted yields.

Then a greedy combination builds a cut set, adding at each step the cut
that most reduces chi2/ndof, until no meaningful improvement remains.

Outputs: train_bdt/dbc_bdt_variants/dbc_sim_scan.json (single-variable
table + greedy chain) and printed tables.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "shape_comparison")
import plot_shape_comparison as psc

BINS = np.linspace(0.5, 1.0, 31)          # D_bc spectra, statistics region
BASE_PT = 200.0


def norm_hist(v, w, bins):
    h, _ = np.histogram(v, bins=bins, weights=w)
    h2, _ = np.histogram(v, bins=bins, weights=w ** 2)
    tot = h.sum()
    if tot <= 0:
        return None, None
    return h / tot, np.sqrt(np.maximum(h2, 0)) / tot


def chi2_ks(v1, w1, v2, w2):
    h1, e1 = norm_hist(v1, w1, BINS)
    h2, e2 = norm_hist(v2, w2, BINS)
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


def main():
    t0 = time.time()
    cfg = psc.Config()
    samples = psc.load_all_events(cfg)

    # assemble arrays
    keys = ["weights", "score_Dbc", "score_Dbc_ratio", "is_signal", "is_top_bc",
            "ak8_pt_0", "ak8_sdmass_0", "ak8_tau21_0", "n_ak4", "n_btagM",
            "n_ctagM", "ht", "met", "lep1_pt", "mTW", "top_mass_fromJ1"]
    cols = {k: [] for k in keys}
    for s in samples:
        a = s["arr"]
        for k in keys:
            cols[k].append(np.asarray(a[k]))
    C = {k: np.concatenate(v) for k, v in cols.items()}
    del cols, samples
    import gc; gc.collect()
    n_all = len(C["weights"])
    print(f"[INFO] {n_all} events in {time.time()-t0:.0f}s")

    w = C["weights"].astype(np.float64)
    is_sig = C["is_signal"].astype(bool)
    is_top = C["is_top_bc"].astype(bool) & ~is_sig
    base = C["ak8_pt_0"] > BASE_PT

    cuts = psc.build_kinematic_cuts()
    print(f"[INFO] {len(cuts)} kinematic cuts")

    results = {"single": [], "greedy": []}

    for mode, dvar in [("bdt", "score_Dbc"), ("ratio", "score_Dbc_ratio")]:
        dbc = C[dvar].astype(np.float64)
        print(f"\n===== Dbc mode: {mode} =====")
        print(f"{'cut':32s} {'chi2/ndof':>9s} {'KS':>6s} {'sig_w':>8s} {'topbc_w':>8s}")
        rows = []
        for tag, label, cond in cuts:
            arr = {k: C[k] for k in keys}
            m = base & cond(arr)
            ms = m & is_sig
            mt = m & is_top
            if ms.sum() < 10 or mt.sum() < 10:
                continue
            c2, ks = chi2_ks(dbc[ms], w[ms], dbc[mt], w[mt])
            ws = float(w[ms].sum())
            wt = float(w[mt].sum())
            rows.append(dict(cut=tag, label=label, chi2=c2, ks=ks,
                             wsig=ws, wtopbc=wt))
            print(f"{tag:32s} {c2:9.3f} {ks:6.3f} {ws:8.2f} {wt:8.2f}")
        rows.sort(key=lambda r: r["chi2"])
        results["single"].append({"mode": mode, "rows": rows})
        print(f"  best: {rows[0]['cut']} chi2={rows[0]['chi2']:.3f} "
              f"KS={rows[0]['ks']:.3f}")

        # greedy combination (single-variable cuts only)
        singles = [r for r in rows if r["cut"].startswith(
            ("pt", "msd", "tau21", "nak4", "nbtagM", "nbcM", "ht_", "met_",
             "lep1pt", "mtw"))]
        cond_of = {c[0]: c[2] for c in cuts}
        arr = {k: C[k] for k in keys}
        cur = base.copy()
        chosen = []
        chi2_cur, ks_cur = chi2_ks(dbc[cur & is_sig], w[cur & is_sig],
                                   dbc[cur & is_top], w[cur & is_top])
        chain = [dict(step=0, cut="PS", chi2=chi2_cur, ks=ks_cur,
                      wsig=float(w[cur & is_sig].sum()),
                      wtopbc=float(w[cur & is_top].sum()))]
        print(f"\n[greedy] PS: chi2/ndof={chi2_cur:.3f} KS={ks_cur:.3f}")
        pool = {r["cut"]: r for r in singles}
        while pool:
            best = None
            for tag, r in pool.items():
                m = cur & cond_of[tag](arr)
                ms = m & is_sig
                mt = m & is_top
                if ms.sum() < 10 or mt.sum() < 10:
                    continue
                c2, ks = chi2_ks(dbc[ms], w[ms], dbc[mt], w[mt])
                if best is None or c2 < best[1]:
                    best = (tag, c2, ks, m, ms, mt)
            if best is None:
                break
            tag, c2, ks, m, ms, mt = best
            if c2 >= chi2_cur - 0.05:      # no meaningful improvement
                break
            chi2_cur = c2
            chosen.append(tag)
            cur = m
            chain.append(dict(step=len(chosen), cut=tag, chi2=c2, ks=ks,
                              wsig=float(w[ms].sum()),
                              wtopbc=float(w[mt].sum())))
            print(f"  + {tag:24s} chi2/ndof={c2:.3f} KS={ks:.3f} "
                  f"(sig {w[ms].sum():.1f}, topbc {w[mt].sum():.1f})")
            del pool[tag]
        results["greedy"].append({"mode": mode, "chain": chain,
                                  "chosen": chosen})
        print(f"  chosen: {chosen}")

    with open("train_bdt/dbc_bdt_variants/dbc_sim_scan.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n[DONE] saved dbc_sim_scan.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
