#!/usr/bin/env python3
"""
Is t3(b'cq) a usable proxy for t3(b'bc)?

Scores the extracted jets with S1 (and the dedicated S2/S3 taggers if present)
and compares, for each tagger, the proxy's score spectrum against the signal's.
The t2(b'c) signal/proxy pair is carried along as the reference: we already
know that pair is a good proxy, so it sets the scale for "how close is close".

Numerics per (tagger, pair):   KS p, mean shift, and the weighted efficiency
ratio proxy/signal above a set of score cuts -- the last is what actually
matters, since a calibration is applied at a working point.

  ./.venv/bin/python S1_tagger/t3_proxy_plots.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
PAIRS = [("t3(b'bc)", "sig_t3bbc", "prx_t3bcq"),
         ("t2(b'c)", "sig_t2bc", "prx_t2bc")]
CUTS = [0.5, 0.6, 0.7, 0.8, 0.9]


def wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, x))


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    df = pd.read_parquet(HERE / "t3_proxy_study.parquet")
    print(f"[load] {len(df):,} jets")
    print(df.groupby("cat").agg(n=("w", "size"), sumw=("w", "sum")).to_string(), "\n")

    taggers = {}
    for name, run in (("S1", "presel_v3_kp50"), ("S2", "presel_v3_kp50_S2"),
                      ("S3", "presel_v3_kp50_S3")):
        mp = HERE / "output" / run / "S1" / "model.json"
        if not mp.exists():
            continue
        m = XGBClassifier(); m.load_model(mp)
        df[f"score_{name}"] = m.predict_proba(df[m.get_booster().feature_names])[:, 1]
        taggers[name] = m
    print(f"[score] {', '.join(taggers)}\n")

    # kinematic comparison first -- if the objects differ here, the tagger
    # difference is a consequence, not a separate fact
    print("=" * 78)
    print("KINEMATICS / SUBSTRUCTURE  (weighted mean)")
    print("=" * 78)
    kv = ["ak8_pt", "ak8_sdmass", "ak8_tau21", "ak8_tau32"]
    print(f"{'category':12s}{'N':>9}" + "".join(f"{k.replace('ak8_',''):>11}" for k in kv))
    for c in ["sig_t3bbc", "prx_t3bcq", "sig_t2bc", "prx_t2bc"]:
        s = df[df.cat == c]
        if s.empty:
            continue
        row = "".join(f"{np.average(s[k], weights=s.w):11.2f}" for k in kv)
        print(f"{c:12s}{len(s):>9,}{row}")

    for tg in taggers:
        sc = f"score_{tg}"
        print("\n" + "=" * 78)
        print(f"TAGGER {tg}:  proxy vs signal score spectrum")
        print("=" * 78)
        for lab, cs, cp in PAIRS:
            a, b = df[df.cat == cs], df[df.cat == cp]
            if a.empty or b.empty:
                continue
            ks = ks_2samp(a[sc], b[sc])
            ma = np.average(a[sc], weights=a.w); mb = np.average(b[sc], weights=b.w)
            print(f"\n  {lab}:  signal {len(a):,}  proxy {len(b):,}")
            print(f"    mean score   sig {ma:.4f}   proxy {mb:.4f}   "
                  f"shift {100*(mb-ma)/ma:+.1f}%")
            print(f"    median       sig {wq(a[sc].values, a.w.values, .5):.4f}   "
                  f"proxy {wq(b[sc].values, b.w.values, .5):.4f}")
            print(f"    KS p-value   {ks.pvalue:.3g}   (D = {ks.statistic:.4f})")
            print(f"    {'cut':>8}{'eff sig':>10}{'eff prx':>10}{'prx/sig':>10}")
            for c in CUTS:
                ea = a.w[a[sc] > c].sum() / a.w.sum()
                eb = b.w[b[sc] > c].sum() / b.w.sum()
                r = eb / ea if ea > 0 else np.nan
                print(f"    {c:8.2f}{ea:10.4f}{eb:10.4f}{r:10.3f}")

    df.to_parquet(HERE / "t3_proxy_scored.parquet", index=False)
    print(f"\nwrote {HERE/'t3_proxy_scored.parquet'}")


if __name__ == "__main__":
    main()
