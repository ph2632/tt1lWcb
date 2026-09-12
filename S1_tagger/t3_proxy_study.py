#!/usr/bin/env python3
"""
Can t3(b'cq) stand in for t3(b'bc)?

The signal contains a fully-merged top t -> b' W(cb), i.e. THREE partons in
one jet (b' b c).  There is no GloParT node for it and no obvious proxy, so
this asks whether the Cabibbo-favoured t3(b'cq) -- same 3-prong top topology
with the W's b replaced by an s -- reproduces its tagger response well enough
to calibrate with.

Extracted per jet (PRESEL applied), from the gen-match flags:
  sig_t3bbc   ttbar-powheg   w_decay==5 & ak8_match_tbqq_wcb     (the target)
  prx_t3bcq   bkg samples    w_decay==4 & ak8_match_top_bcq      (candidate)
  sig_t2bc    ttbar-powheg   w_decay==5 & ak8_match_tbq_wcb & n_c>=1
  prx_t2bc    bkg samples    w_decay==4 & ak8_match_top_bc       (known-good
                             proxy, as the reference for "how close is close")
  bkg         bkg samples    everything else, thinned

Flag convention (verified): top_bXY = top's b' plus the listed W partons;
n_b / n_c count the W's partons only.

  ./.venv/bin/python S1_tagger/t3_proxy_study.py [--config ...]
"""
import argparse
import json
import os
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

HERE = Path(__file__).resolve().parent
PI = float(np.pi)
SIG_FILE = "ttbar-powheg_merged_Skim.root"
BKG_FILES = ["tt-semi_merged_Skim.root", "tt-had_merged_Skim.root",
             "ttbb_merged_Skim.root", "single-top_merged_Skim.root"]


def flat(a, d=0.0):
    return ak.to_numpy(ak.flatten(ak.fill_none(a, d), axis=1))


def bcast(ev, name, tmpl, d=0.0):
    return flat(ak.broadcast_arrays(ev[name], tmpl)[0], d)


def dphi(a, b):
    return (a - b + PI) % (2.0 * PI) - PI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--bkg-keep", type=float, default=0.05,
                    help="Bernoulli keep prob for the unmatched background")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    rng = np.random.default_rng(cfg["seed"])
    mc = Path(cfg["mc_dir"])
    presel = cfg["preselection"]

    derived = cfg.get("derived_features", {})
    raw_needed = sorted({c for v in derived.values() for c in v})
    feats = [f for f in cfg["features_S1"] if f not in derived]
    store = sorted(set(feats + raw_needed))
    kin = ["ak8_pt", "ak8_eta", "ak8_phi", "ak8_sdmass", "ak8_tau21", "ak8_tau32"]
    truth = ["w_decay", "ak8_type", "ak8_n_b_in_jet", "ak8_n_c_in_jet",
             "ak8_match_tbqq_wcb", "ak8_match_tbq_wcb", "ak8_match_wqq_wcb",
             "ak8_match_top_bc", "ak8_match_top_bcq"]
    evs = ["run", "luminosityBlock", "event", "lep1_pt", "lep1_eta", "lep1_phi"] + cfg["weights"]
    cols = sorted(set(store + kin + truth + evs))

    out, t0 = [], time.time()
    for fn in [SIG_FILE] + BKG_FILES:
        path = mc / fn
        if not path.exists():
            print(f"  [skip] {fn} not found"); continue
        is_sig = fn == SIG_FILE
        print(f"[{fn:34s}] {'SIGNAL' if is_sig else 'bkg'}", flush=True)
        for ev in uproot.iterate(f"{path}:{cfg['tree']}", cols, library="ak",
                                 step_size="300 MB"):
            if len(ev) == 0:
                continue
            tmpl = ev["ak8_pt"]
            w = np.ones(len(ev))
            for wn in cfg["weights"]:
                w *= ak.to_numpy(ak.fill_none(ev[wn], 0.0))
            w = flat(ak.broadcast_arrays(ak.Array(w), tmpl)[0], 0.0)

            pt, msd = flat(ev["ak8_pt"], -1.0), flat(ev["ak8_sdmass"], -1.0)
            t21, t32 = flat(ev["ak8_tau21"], -1.0), flat(ev["ak8_tau32"], -1.0)
            eta, phi = flat(ev["ak8_eta"], 0.0), flat(ev["ak8_phi"], 0.0)
            lpt = bcast(ev, "lep1_pt", tmpl, -1.0)
            dr = np.hypot(np.abs(bcast(ev, "lep1_eta", tmpl, 0.0) - eta),
                          dphi(bcast(ev, "lep1_phi", tmpl, 0.0), phi))
            srt = ak.sort(ev["ak8_sdmass"], axis=1, ascending=False)
            sub = flat(ak.broadcast_arrays(ak.Array(ak.to_numpy(ak.fill_none(
                ak.pad_none(srt, 2, axis=1)[:, 1], -np.inf))), tmpl)[0], -np.inf)
            keep = ((pt > presel["jet_pt_min"]) & (msd > presel["jet_sdmass_min"])
                    & (t21 > 0) & (t21 < presel["jet_tau21_max"])
                    & (lpt > 0) & (dr > presel["dr_lep_jet_min"])
                    & (sub < presel["event_subleading_sdmass_max"]))

            wd = bcast(ev, "w_decay", tmpl, -1).astype(int)
            g = lambda k: flat(ev[k], 0).astype(bool)
            nc = flat(ev["ak8_n_c_in_jet"], 0).astype(int)
            cat = np.full(len(pt), "bkg", dtype=object)
            if is_sig:
                cat[(wd == 5) & g("ak8_match_tbqq_wcb")] = "sig_t3bbc"
                cat[(wd == 5) & g("ak8_match_tbq_wcb") & (nc >= 1)] = "sig_t2bc"
                cat[(wd == 5) & g("ak8_match_wqq_wcb")] = "sig_wcb"
                keep &= cat != "bkg"
            else:
                cat[(wd == 4) & g("ak8_match_top_bcq")] = "prx_t3bcq"
                cat[(wd == 4) & g("ak8_match_top_bc")] = "prx_t2bc"
                thin = rng.random(len(pt)) < args.bkg_keep
                keep &= (cat != "bkg") | thin
                w = np.where(cat == "bkg", w / args.bkg_keep, w)
            if not keep.any():
                continue
            d = {"cat": cat[keep], "w": np.abs(w[keep]),
                 "ak8_pt": pt[keep], "ak8_sdmass": msd[keep],
                 "ak8_tau21": t21[keep], "ak8_tau32": t32[keep],
                 "ak8_type": flat(ev["ak8_type"], 0).astype(np.int8)[keep]}
            for f in store:
                d[f] = flat(ev[f], np.nan)[keep].astype(np.float32)
            out.append(pd.DataFrame(d))

    df = pd.concat(out, ignore_index=True)
    for name, comps in derived.items():
        df[name] = df[comps].sum(axis=1)
    dest = HERE / "t3_proxy_study.parquet"
    df.to_parquet(dest, index=False)
    print(f"\nwrote {dest}  ({len(df):,} jets, {time.time()-t0:.0f}s)")
    print(df.groupby("cat").size().to_string())


if __name__ == "__main__":
    main()
