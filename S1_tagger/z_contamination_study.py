#!/usr/bin/env python3
"""
How much of the current BACKGROUND pool is actually a gen-matched hadronic Z
(Z->bb, Z->cc, Z->qq light), and where does it sit in the S1 score?

Retraction: an earlier claim that "Z->bb cannot be identified from truth" was
wrong -- z_decay + genZ_pt/eta/phi exist (2026-09-02 rerun) and 04_Make_plots.py
already builds a real geometric match from them (dR(J, Z_gen) < 0.8, same
convention followed here): z_decay 1/2/3 = light, 4 = cc, 5 = bb.

Streams all 16 non-Vcb background samples, applies the S1 PRESEL per jet, and
tags every jet with:
  z_bb / z_cc / z_light   gen-matched hadronic Z, by flavour
  t2bc_proxy              existing w_decay==4 & ak8_match_top_bc (for overlap check)
  bkg                     everything else (topology==0 today)
Scores all of them with the trained S1 model and reports, per Z-flavour
class: sample origin, sumw share of current background, and score spectrum
against t2(b'b) signal (for Z->bb) and t2(b'c)/proxy (for Z->cc).

  ./.venv/bin/python S1_tagger/z_contamination_study.py [--config ...]
"""
import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

HERE = Path(__file__).resolve().parent
PI = float(np.pi)
SIG_FILE = "ttbar-powheg_merged_Skim.root"


def flat(a, d=0.0):
    return ak.to_numpy(ak.flatten(ak.fill_none(a, d), axis=1))


def bcast(ev, name, tmpl, d=0.0):
    return flat(ak.broadcast_arrays(ev[name], tmpl)[0], d)


def dphi(a, b):
    return (a - b + PI) % (2.0 * PI) - PI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--bkg-keep", type=float, default=0.15,
                    help="Bernoulli keep prob for the unmatched/unflavoured background")
    ap.add_argument("--step-size", default="300 MB",
                    help="uproot.iterate chunk size (smaller = safer on a loaded node)")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    rng = np.random.default_rng(cfg["seed"])
    mc = Path(cfg["mc_dir"])
    presel = cfg["preselection"]

    files = sorted(f for f in mc.glob("*.root") if f.name != SIG_FILE)
    derived = cfg.get("derived_features", {})
    raw_needed = sorted({c for v in derived.values() for c in v})
    feats = [f for f in cfg["features_S1"] if f not in derived]
    store = sorted(set(feats + raw_needed))

    kin = ["ak8_pt", "ak8_eta", "ak8_phi", "ak8_sdmass", "ak8_tau21", "ak8_tau32"]
    zbr = ["z_decay", "genZ_pt", "genZ_eta", "genZ_phi"]
    truth = ["w_decay", "ak8_match_top_bc"]
    evs = ["lep1_pt", "lep1_eta", "lep1_phi"] + cfg["weights"]

    partdir = HERE / "z_contamination_parts"
    partdir.mkdir(exist_ok=True)
    t0 = time.time()
    for path in files:
        done_marker = partdir / (path.stem + ".done")
        if done_marker.exists():
            print(f"[{path.name:34s}] already done -- skipping"); continue
        # a kill mid-file leaves partial chunk files -- wipe them so the
        # retry cannot double-count events from the previous attempt
        for stale in partdir.glob(path.stem + "__*.parquet"):
            stale.unlink()
        with uproot.open(f"{path}:{cfg['tree']}") as t:
            have_z = all(b in t.keys() for b in zbr)
        if not have_z:
            print(f"[{path.name:34s}] no z_decay/genZ_* -- skipped")
            done_marker.touch()
            continue
        cols = sorted(set(store + kin + zbr + truth + evs))
        n_jets_flavZ = 0
        for ci, ev in enumerate(uproot.iterate(f"{path}:{cfg['tree']}", cols,
                                               library="ak", step_size=args.step_size)):
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
            dr_lep = np.hypot(np.abs(bcast(ev, "lep1_eta", tmpl, 0.0) - eta),
                              dphi(bcast(ev, "lep1_phi", tmpl, 0.0), phi))
            srt = ak.sort(ev["ak8_sdmass"], axis=1, ascending=False)
            sub = flat(ak.broadcast_arrays(ak.Array(ak.to_numpy(ak.fill_none(
                ak.pad_none(srt, 2, axis=1)[:, 1], -np.inf))), tmpl)[0], -np.inf)
            keep = ((pt > presel["jet_pt_min"]) & (msd > presel["jet_sdmass_min"])
                    & (t21 > 0) & (t21 < presel["jet_tau21_max"])
                    & (lpt > 0) & (dr_lep > presel["dr_lep_jet_min"])
                    & (sub < presel["event_subleading_sdmass_max"]))

            # Z geometric match, exactly 04_Make_plots.py's convention
            zdec = bcast(ev, "z_decay", tmpl, 0).astype(int)
            gzpt = bcast(ev, "genZ_pt", tmpl, -1.0)
            gzeta = bcast(ev, "genZ_eta", tmpl, 0.0)
            gzphi = bcast(ev, "genZ_phi", tmpl, 0.0)
            z_had = np.isin(zdec, (1, 2, 3, 4, 5))
            z_dr = np.hypot(gzeta - eta, dphi(gzphi, phi))
            z_merged = z_had & (gzpt > 0) & (pt > 0) & (z_dr < 0.8)

            wd = bcast(ev, "w_decay", tmpl, -1).astype(int)
            proxy = (wd == 4) & flat(ev["ak8_match_top_bc"], 0).astype(bool)

            cat = np.full(len(pt), "bkg", dtype=object)
            cat[proxy] = "t2bc_proxy"                 # existing definition, precedence check
            cat[z_merged & np.isin(zdec, (1, 2, 3))] = "z_light"
            cat[z_merged & (zdec == 4)] = "z_cc"
            cat[z_merged & (zdec == 5)] = "z_bb"       # highest precedence
            n_jets_flavZ += int((cat != "bkg").sum())

            thin = rng.random(len(pt)) < args.bkg_keep
            keep &= (cat != "bkg") | thin
            w = np.where(cat == "bkg", w / args.bkg_keep, w)
            if not keep.any():
                continue
            d = {"sample": path.name, "cat": cat[keep], "w": np.abs(w[keep]),
                 "ak8_pt": pt[keep], "ak8_sdmass": msd[keep],
                 "ak8_tau21": t21[keep], "ak8_tau32": t32[keep]}
            for f in store:
                d[f] = flat(ev[f], np.nan)[keep].astype(np.float32)
            chunk_part = partdir / f"{path.stem}__{ci:04d}.parquet"
            pd.DataFrame(d).to_parquet(chunk_part, index=False)  # write EVERY chunk immediately
        done_marker.touch()
        print(f"[{path.name:34s}] Z-flavoured jets (pre-presel-thin): {n_jets_flavZ:,}"
              f"  -> {ci+1} chunks written, marked done", flush=True)

    parts = sorted(partdir.glob("*__*.parquet"))
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    for name, comps in derived.items():
        if name not in df.columns:
            df[name] = df[comps].sum(axis=1) if len(df) else np.nan
    dest = HERE / "z_contamination.parquet"
    df.to_parquet(dest, index=False)
    print(f"\nwrote {dest}  ({len(df):,} jets, {time.time()-t0:.0f}s)")
    print(df.groupby("cat").agg(n=("w", "size"), sumw=("w", "sum")).to_string())


if __name__ == "__main__":
    main()
