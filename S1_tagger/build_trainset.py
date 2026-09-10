#!/usr/bin/env python3
"""
Build the per-jet training set for the S1 / S1' boosted-cb tagger.

Streams the 17 scored MC ROOT files, flattens every AK8 jet, applies the MAT
PRESELECTION per jet, assigns a gen-match label, and writes train/valid/test
parquet chunks.

  signal jets  : ttbar-powheg only, labelled by the W->cb jet topology
                 1 = W(cb)      merged 2-prong b+c               (ak8_type 1)
                 2 = t2(b'c)    top-b + W's c, partial merge      (ak8_type 2)
                 3 = t2(b'b)    top-b + W's b, partial merge      (ak8_type 2)
                 4 = t3(b'bc)   fully-merged top t->(b b c)       (ak8_type 4)
  background   : every AK8 jet from the other 16 samples (label 0), Bernoulli-
                 thinned with p = background_keep_prob and weight up-scaled by
                 1/p so the summed weight (effective luminosity) is preserved.

The non-signal jets of ttbar-powheg (leptonic-side b, ISR, and -- crucially --
the ~48% of Vcb events where W->cb is fully RESOLVED) are dropped entirely:
they are genuine W->cb physics and must not train the tagger to reject signal.

Structure and helpers follow
  /eos/user/y/youpeng/research/wcb/BoostedDbcTrain/dcb_versions_v1/run_dcb_versions.py

Usage:
  ./.venv/bin/python S1_tagger/build_trainset.py [--config S1_tagger/config.json] [--overwrite]
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import awkward as ak
import uproot

HERE = Path(__file__).resolve().parent
PI = float(np.pi)

# jet topology codes
SIG_CODES = {1: "Wcb", 2: "t2_bc", 3: "t2_bb", 4: "t3_bbc"}
BKG_CODE = 0


# --------------------------------------------------------------------------- #
# helpers (flatten / broadcast / event_split adapted from run_dcb_versions.py)
# --------------------------------------------------------------------------- #
def flatten(array, default=0.0):
    return ak.to_numpy(ak.flatten(ak.fill_none(array, default), axis=1))


def broadcast(events, name, template, default=0.0):
    return flatten(ak.broadcast_arrays(events[name], template)[0], default)


def event_split(source, run, lumi, event, split_cfg):
    """Deterministic per-event train/valid/test tag; every jet of an event
    lands in the same split."""
    keys = np.char.add(np.char.add(np.char.add(source + ":", run.astype(str)), ":"),
                       lumi.astype(str))
    keys = np.char.add(np.char.add(keys, ":"), event.astype(str))
    buckets = np.fromiter(
        (int(hashlib.blake2b(k.encode(), digest_size=8).hexdigest(), 16) % 100 for k in keys),
        dtype=np.uint64, count=len(keys),
    )
    lo = split_cfg["train"]
    hi = lo + split_cfg["valid"]
    return np.where(buckets < lo, "train", np.where(buckets < hi, "valid", "test"))


def dphi_pipi(a, b):
    d = a - b
    return (d + PI) % (2.0 * PI) - PI


def topology_codes(events, template):
    """Per-jet W->cb signal topology code (0 for everything else).

    Gated on the event-level hadronic W->cb decay (w_decay == 5) AND the
    dedicated jet match flag, exactly as run_dcb_versions.py does, so a
    stray jet match in a non-cb event cannot become signal.  The flags are
    mutually exclusive by construction (verified: zero overlap, 1:1 with
    ak8_type), but the ~ exclusions are kept as belt-and-braces.
    """
    w_decay = broadcast(events, "w_decay", template, -1).astype(np.int16)
    is_wcb = w_decay == 5
    m_full = flatten(events["ak8_match_tbqq_wcb"], 0).astype(bool)
    m_wqq = flatten(events["ak8_match_wqq_wcb"], 0).astype(bool)
    m_tbq = flatten(events["ak8_match_tbq_wcb"], 0).astype(bool)
    n_c = flatten(events["ak8_n_c_in_jet"], 0).astype(np.int16)
    n_b = flatten(events["ak8_n_b_in_jet"], 0).astype(np.int16)

    full = is_wcb & m_full
    merged = is_wcb & m_wqq & ~full
    tbq = is_wcb & m_tbq & ~full & ~merged
    t2bc = tbq & (n_c >= 1)
    t2bb = tbq & ~t2bc & (n_b >= 1)

    return np.select([full, merged, t2bc, t2bb], [4, 1, 2, 3], default=0).astype(np.int8)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if dataset_complete.json exists")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="debug: process only the first N ROOT files")
    ap.add_argument("--samples", default="",
                    help="debug: comma-separated name substrings to keep")
    ap.add_argument("--max-chunks-per-file", type=int, default=0,
                    help="debug: stop each file after N streamed chunks")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    rng = np.random.default_rng(cfg["seed"])

    data_dir = Path(os.environ.get("S1_DATA_DIR", cfg["data_dir"])) / cfg["dataset_tag"]
    marker = data_dir / "dataset_complete.json"
    if marker.exists() and not args.overwrite:
        print(f"[skip] {marker} exists -- pass --overwrite to rebuild")
        return

    mc_dir = Path(cfg["mc_dir"])
    files = sorted(mc_dir.glob("*.root"))
    if args.samples:
        subs = [s.strip() for s in args.samples.split(",") if s.strip()]
        files = [f for f in files if any(s in f.name for s in subs)]
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"no ROOT files under {mc_dir}")
    sig_name = cfg["signal_sample"]
    if not any(f.name == sig_name for f in files):
        raise SystemExit(f"signal sample {sig_name} not found in {mc_dir}")

    # store every ak8_gpt_* node (all 33, incl. the 3 parent sums) plus the
    # two subjettiness ratios, so comparison models (old 8-node Dbc, Youpeng's
    # 3-class) and future feature-set studies need no rebuild.  S1 / S1' pick
    # their subsets at train time from config.
    gpt_parents = ["ak8_gpt_topbw", "ak8_gpt_topw", "ak8_gpt_qcd"]
    feats_all = sorted(set(cfg["features_S1"] + cfg["features_S1p_extra"] + gpt_parents))
    presel = cfg["preselection"]
    keep_p = float(cfg["background_keep_prob"])
    psel = cfg["split"]

    # branches to read (jagged + event-scalar)
    kin = ["ak8_pt", "ak8_eta", "ak8_phi", "ak8_sdmass"]
    truth = ["ak8_type", "ak8_n_b_in_jet", "ak8_n_c_in_jet",
             "ak8_match_wqq_wcb", "ak8_match_tbq_wcb", "ak8_match_tbqq_wcb", "w_decay"]
    ev_scalar = ["run", "luminosityBlock", "event", "n_ak8",
                 "lep1_pt", "lep1_eta", "lep1_phi"] + cfg["weights"]
    read_cols = sorted(set(kin + truth + ev_scalar + feats_all))

    for sub in ("train", "valid", "test"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    counts = {s: {c: 0 for c in list(SIG_CODES) + [BKG_CODE]}
              for s in ("train", "valid", "test")}
    sumw = {s: {"sig": 0.0, "bkg": 0.0} for s in ("train", "valid", "test")}
    part_idx = 0
    t0 = time.time()

    for fi, path in enumerate(files):
        is_signal = path.name == sig_name
        role = "SIGNAL" if is_signal else "bkg"
        print(f"[{fi + 1:2d}/{len(files)}] {path.name:38s} ({role})", flush=True)

        for ci, events in enumerate(uproot.iterate(
                f"{path}:{cfg['tree']}", read_cols, library="ak", step_size="300 MB")):
            if args.max_chunks_per_file and ci >= args.max_chunks_per_file:
                break
            n_ev = len(events)
            if n_ev == 0:
                continue
            template = events["ak8_pt"]

            ev_w = np.ones(n_ev, dtype=np.float64)
            for wname in cfg["weights"]:
                ev_w *= ak.to_numpy(ak.fill_none(events[wname], 0.0))
            weight = flatten(ak.broadcast_arrays(ak.Array(ev_w), template)[0], 0.0)

            ids = {k: broadcast(events, k, template, -1) for k in ("run", "luminosityBlock", "event")}
            split = event_split(path.name, ids["run"], ids["luminosityBlock"],
                                ids["event"], psel)

            jpt = flatten(events["ak8_pt"], -1.0)
            jeta = flatten(events["ak8_eta"], 0.0)
            jphi = flatten(events["ak8_phi"], 0.0)
            jmsd = flatten(events["ak8_sdmass"], -1.0)
            jt21 = flatten(events["ak8_tau21"], -1.0)

            lep_pt = broadcast(events, "lep1_pt", template, -1.0)
            lep_eta = broadcast(events, "lep1_eta", template, 0.0)
            lep_phi = broadcast(events, "lep1_phi", template, 0.0)
            dr_lep = np.hypot(np.abs(lep_eta - jeta), dphi_pipi(lep_phi, jphi))

            # event-level: soft-drop mass of the 2nd-heaviest AK8 (-inf if <2)
            msd_sorted = ak.sort(events["ak8_sdmass"], axis=1, ascending=False)
            sub_msd_ev = ak.to_numpy(ak.fill_none(
                ak.pad_none(msd_sorted, 2, axis=1)[:, 1], -np.inf))
            sub_msd = flatten(ak.broadcast_arrays(ak.Array(sub_msd_ev), template)[0], -np.inf)

            keep = (
                (jpt > presel["jet_pt_min"])
                & (jmsd > presel["jet_sdmass_min"])
                & (jt21 > 0.0) & (jt21 < presel["jet_tau21_max"])
                & (lep_pt > 0.0) & (dr_lep > presel["dr_lep_jet_min"])
                & (sub_msd < presel["event_subleading_sdmass_max"])
            )

            topo = topology_codes(events, template)

            if is_signal:
                keep &= topo > 0                       # signal jets only
            else:
                topo[:] = BKG_CODE
                keep &= rng.random(len(keep)) < keep_p  # thin the background
                weight = weight / keep_p                # ... keep sum(w) intact

            if not keep.any():
                continue

            cols = {"topology": topo[keep].astype(np.int8),
                    "weight": np.abs(weight[keep]).astype(np.float64),
                    "weight_signed": weight[keep].astype(np.float64),
                    "sample": np.repeat(path.name, int(keep.sum())),
                    "ak8_pt": jpt[keep].astype(np.float32),
                    "ak8_sdmass": jmsd[keep].astype(np.float32)}
            for fname in feats_all:
                cols[fname] = flatten(events[fname], np.nan)[keep].astype(np.float32)

            frame = pd.DataFrame(cols)
            finite = np.isfinite(frame[feats_all]).all(axis=1).to_numpy()
            frame = frame.loc[finite].reset_index(drop=True)
            ksplit = split[keep][finite]

            for sub in ("train", "valid", "test"):
                part = frame.loc[ksplit == sub]
                if part.empty:
                    continue
                part.to_parquet(
                    data_dir / sub / f"part-{fi:02d}-{part_idx:05d}.parquet", index=False)
                for c in list(SIG_CODES) + [BKG_CODE]:
                    counts[sub][c] += int((part["topology"] == c).sum())
                m_sig = part["topology"] > 0
                sumw[sub]["sig"] += float(part.loc[m_sig, "weight"].sum())
                sumw[sub]["bkg"] += float(part.loc[~m_sig, "weight"].sum())
            part_idx += 1

    summary = {
        "config": cfg, "dataset_dir": str(data_dir),
        "n_input_files": len(files), "n_parquet_chunks": part_idx,
        "counts_per_split": counts, "sumw_per_split": sumw,
        "signal_codes": SIG_CODES,
        "build_seconds": round(time.time() - t0, 1),
    }
    marker.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n==== dataset built ====")
    for sub in ("train", "valid", "test"):
        tot_sig = sum(counts[sub][c] for c in SIG_CODES)
        print(f"  {sub:5s}: sig {tot_sig:>9d}  bkg {counts[sub][BKG_CODE]:>10d}   "
              f"(sumw sig {sumw[sub]['sig']:.1f}  bkg {sumw[sub]['bkg']:.1f})")
    print("  signal by class (all splits):")
    for c, name in SIG_CODES.items():
        tot = sum(counts[s][c] for s in counts)
        print(f"      {c} {name:8s} {tot:>9d}")
    print(f"  wrote {part_idx} chunks to {data_dir}  in {summary['build_seconds']}s")


if __name__ == "__main__":
    main()
