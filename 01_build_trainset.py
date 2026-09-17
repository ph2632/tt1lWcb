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
                 5 = t2(b'c) PROXY: top-b' + W's c merged in a W->cs event of
                     the 16 non-Vcb samples (ak8_match_top_bc & w_decay==4).
                     Confirmed statistically identical to class 2 (tau21/32,
                     mSD, all gpt nodes agree to 1-2%) -- promoted to signal
                     instead of contaminating the background pool with
                     mislabeled duplicates of class 2. Never thinned.
                 6 = Z(bb)-adj: gen-matched hadronic Z->bb in the 16 non-Vcb
                     samples (z_decay==5 & dR(J, Z_gen)<0.8, the SAME
                     geometric match 04_Make_plots.py already uses for its
                     mat_cat==15).  Quantified in z_contamination_study.py:
                     mean S1 response 0.797 vs true t2(b'b) signal's 0.834,
                     tracking it closely up to and including the highest
                     score bin -- a b'+b jet from a top and a b+b jet from a
                     Z are the same object to a bb-content tagger.  8.1x
                     LARGER in sumw than genuine t2(b'b) (real Zbb yield vs a
                     Cabibbo-suppressed partial-merge topology), so this is a
                     substantial addition, not a rounding correction. Never
                     thinned. Z->cc / Z->light stay BACKGROUND: Z->cc scores
                     0.452, far from the bc-like region -- the "fakes bc"
                     hypothesis is not supported by the tagger response.
  background   : every remaining AK8 jet from the 16 non-Vcb samples (label
                 0), Bernoulli-thinned with p = background_keep_prob and
                 weight up-scaled by 1/p so the summed weight (effective
                 luminosity) is preserved.

The non-signal jets of ttbar-powheg (leptonic-side b, ISR, and -- crucially --
the ~48% of Vcb events where W->cb is fully RESOLVED) are dropped entirely:
they are genuine W->cb physics and must not train the tagger to reject signal.

Structure and helpers follow
  /eos/user/y/youpeng/research/wcb/BoostedDbcTrain/dcb_versions_v1/run_dcb_versions.py

Usage:
  ./.venv/bin/python 01_build_trainset.py [--config S1_tagger/config.json] [--overwrite]

Step 1/3 of the S1 tagger pipeline: 01_build_trainset.py -> 02_train_tagger.py
-> 03_Training_report_plots.py (or run_s1_tagger.py to drive all three).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import awkward as ak
import uproot

HERE = Path(__file__).resolve().parent
PKG = HERE / "S1_tagger"          # config.json / output / presel.py live here
PI = float(np.pi)

# jet topology codes
SIG_CODES = {1: "Wcb", 2: "t2_bc", 3: "t2_bb", 4: "t3_bbc", 5: "t2_bc_proxy",
             6: "z_bb", 7: "qcd_bb"}
BKG_CODE = 0
PROXY_CODE = 5
ZBB_CODE = 6
QCDBB_CODE = 7


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


def _done_banner(t0, output):
    """One unmissable line at the very end of a run: wall time + where the
    output landed (2026-09-13, user -- the per-step timers were easy to miss
    scrolled past in a long log)."""
    line = f"[DONE  time {(time.time() - t0) / 60.0:.1f} min  output: {output}]"
    print("-" * len(line)); print(line); print("-" * len(line))


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
    t_wall0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "config.json")
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
        _done_banner(t_wall0, data_dir)
        return
    if args.overwrite:
        # 2026-09-13 bugfix: --overwrite previously only bypassed the skip
        # check above -- it never cleared old part-*.parquet files, so a
        # rebuild (especially a partial one via --samples/--limit-files, used
        # for quick testing) left STALE files with a different schema sitting
        # alongside the new ones. pd.concat/read_parquet across the directory
        # then either crashes (missing column) or silently mixes old+new
        # rows. Wipe every existing chunk first so --overwrite means what it
        # says: start clean.
        n_removed = 0
        for sub in ("train", "valid", "test"):
            for p in (data_dir / sub).glob("part-*.parquet"):
                p.unlink()
                n_removed += 1
        if n_removed:
            print(f"[overwrite] removed {n_removed} stale part-*.parquet files under {data_dir}")

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
    # raw components of any train-time derived feature (e.g. ak8_gpt_lepq)
    # must still be streamed/stored -- the merge happens at load time, not here.
    derived_raw = sorted({c for comps in cfg.get("derived_features", {}).values() for c in comps})
    feats_all = sorted(set(cfg["features_S1"] + cfg["features_S1p_extra"] + gpt_parents
                           + derived_raw) - set(cfg.get("derived_features", {})))

    # Preselection is read from 04_Make_plots.py so a threshold change there
    # propagates here automatically; config.json only supplies fallbacks.
    presel = dict(cfg["preselection"])
    try:
        sys.path.insert(0, str(PKG))
        from presel import parse_preselection
        parsed, cut_expr, missing = parse_preselection()
        presel.update(parsed)
        print(f"[presel] from 04_Make_plots.py: {cut_expr}")
        if missing:
            print(f"[presel] not found, using config fallback for: {missing}")
    except Exception as exc:                                   # noqa: BLE001
        cut_expr = "(config fallback)"
        print(f"[presel] WARNING could not read 04_Make_plots.py ({exc}); "
              f"using config.json preselection")
    print("[presel] " + "  ".join(f"{k}={v}" for k, v in presel.items()))
    keep_p = float(cfg["background_keep_prob"])
    psel = cfg["split"]

    # branches to read (jagged + event-scalar)
    kin = ["ak8_pt", "ak8_eta", "ak8_phi", "ak8_sdmass"]
    truth = ["ak8_type", "ak8_n_b_in_jet", "ak8_n_c_in_jet",
             "ak8_match_wqq_wcb", "ak8_match_tbq_wcb", "ak8_match_tbqq_wcb",
             "ak8_match_top_bc", "ak8_match_top_bcq", "w_decay",
             "z_decay", "genZ_pt", "genZ_eta", "genZ_phi",
             # per-AK4-jet hadron flavour (0/4/5, standard convention) -- the
             # ONLY genuine gen-truth flavour info in this production; used
             # to gen-match a QCD-origin bb-bar jet (dR<0.8 to the AK8),
             # since ak8_n_b_in_jet is a top-decay-only sentinel (-1) for
             # every non-top sample (2026-09-13).
             "ak4_pt", "ak4_eta", "ak4_phi", "ak4_hflav"]
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
            t3bcq_proxy_flag = np.zeros(len(keep), dtype=bool)  # bkg-only, see below

            if is_signal:
                keep &= topo > 0                       # signal jets only
            else:
                # t2(b'c) PROXY: top-b' + W's c merged in a W->cs event (the
                # background mirror image of signal class 2, b<->s swapped).
                # Confirmed statistically identical to true t2(b'c) (tau21,
                # tau32, mSD, all gpt nodes agree to 1-2%), so it is promoted
                # to a 5th signal class rather than left as contaminating
                # background -- and, like true signal, is never thinned.
                w_decay_bg = broadcast(events, "w_decay", template, -1).astype(np.int16)
                m_top_bc = flatten(events["ak8_match_top_bc"], 0).astype(bool)
                proxy = (w_decay_bg == 4) & m_top_bc

                # Z(bb)-adj: gen-matched hadronic Z->bb, same geometric match
                # 04_Make_plots.py uses for mat_cat==15 (dR(J,Z_gen)<0.8).
                # Quantified in z_contamination_study.py: mean S1 response
                # 0.797 vs true t2(b'b)'s 0.834 -- the tagger cannot tell a
                # b'+b top pairing from a Z->bb pairing, so this is promoted
                # to signal for the same reason the t2(b'c) proxy was.
                # z_bb takes precedence over the (rarer) proxy match on the
                # few jets where both could apply, matching the precedence
                # already established in z_contamination_study.py.
                z_decay = broadcast(events, "z_decay", template, 0).astype(np.int16)
                gz_pt = broadcast(events, "genZ_pt", template, -1.0)
                gz_eta = broadcast(events, "genZ_eta", template, 0.0)
                gz_phi = broadcast(events, "genZ_phi", template, 0.0)
                z_dr = np.hypot(gz_eta - jeta, dphi_pipi(gz_phi, jphi))
                z_bb = (z_decay == 5) & (gz_pt > 0.0) & (jpt > 0.0) & (z_dr < 0.8)

                # QCD(bb): genuine QCD-origin bb-bar (e.g. gluon splitting),
                # gen-matched via >=2 AK4 jets with real hadron-flavour truth
                # (ak4_hflav==5) geometrically inside the AK8 cone (dR<0.8,
                # same convention as z_bb above). ak8_n_b_in_jet cannot be
                # used here -- it is a top-decay-only sentinel (-1) for every
                # jet in this sample -- so this is rebuilt from the AK4
                # collection directly. Rare by construction (2026-09-13,
                # user: "QCDbb has poor statistics, accept all these
                # events"); z_bb/proxy take precedence on any jet where more
                # than one condition could apply (same precedence chain as
                # the z_bb/proxy overlap above).
                a8 = ak.zip({"eta": events["ak8_eta"], "phi": events["ak8_phi"]})
                a4 = ak.zip({"eta": events["ak4_eta"], "phi": events["ak4_phi"],
                            "pt": events["ak4_pt"], "hflav": events["ak4_hflav"]})
                p8, p4 = ak.unzip(ak.cartesian([a8, a4], nested=True))
                q_dr = np.hypot(p4.eta - p8.eta, dphi_pipi(p4.phi, p8.phi))
                q_match = (q_dr < 0.8) & (p4.pt > 20.0) & (p4.hflav == 5)
                n_bflav_in_cone = flatten(ak.sum(q_match, axis=-1), 0)

                special_zp = proxy | z_bb
                qcd_bb = (n_bflav_in_cone >= 2) & ~special_zp

                special = special_zp | qcd_bb
                topo = np.select([z_bb, proxy, qcd_bb],
                                [ZBB_CODE, PROXY_CODE, QCDBB_CODE],
                                default=BKG_CODE).astype(np.int8)
                thin = rng.random(len(keep)) < keep_p   # thin true background only
                keep &= (special | thin)
                weight = np.where(special, weight, weight / keep_p)  # ... keep sum(w) intact for bkg

                # Weight-tail capping for Zbb/QCD(bb) (2026-09-14, user: "Now
                # M3 is overtrained ... investigate what is wrong"): these two
                # topologies' per-event weight is severely skewed -- verified
                # on the built dataset (train split): topo6 (z_bb) top 1% of
                # events carry ~39% of total sumw (effective sample size only
                # 1820/136000 = 1.3% of raw N); topo7 (qcd_bb) top 1% carry
                # ~47% (ESS 157/179609 = 0.09%). Almost certainly a pT-hat/HT
                # generator-weight-tail artifact, not physics -- a handful of
                # extreme-weight jets land differently in every train/test
                # split by pure chance, which is what was driving M3 bb's
                # overtraining bias (topo6 sig bias 9.3%, topo7 15.3%, vs
                # topo3's clean 0.4%). Capped at 15x the per-topology median
                # and rescaled to restore each topology's original total
                # sumw -- both constants derived ONCE from the full
                # presel_v3_kp50_zbb train split (this build streams
                # file-by-file, so an exact percentile isn't available at
                # write time; see the investigation in the 2026-09-14 chat
                # turn for the derivation). Affects 0.24% of z_bb jets and
                # 0.37% of qcd_bb jets; raises effective sample size to
                # ~70000/~69000 (near the raw event count -- the skew is
                # resolved). Applied on |weight| to correctly handle any
                # signed (NLO negative-weight) events.
                ZBB_WCAP, ZBB_WRESCALE = 0.00913, 1.6892
                QCDBB_WCAP, QCDBB_WRESCALE = 0.60331, 2.0121
                for code, cap, rescale in ((ZBB_CODE, ZBB_WCAP, ZBB_WRESCALE),
                                           (QCDBB_CODE, QCDBB_WCAP, QCDBB_WRESCALE)):
                    m = topo == code
                    capped = np.minimum(np.abs(weight), cap) * rescale
                    weight = np.where(m, np.sign(weight) * capped, weight)

                # t3(b'cq) proxy flag (2026-09-13, user): same idea as the
                # t2(b'c) proxy, but for the fully-merged t3(b'bc) signal --
                # a Cabibbo-favoured W->cs event with the SAME 3-prong
                # top-decay match pattern (w_decay==4 & ak8_match_top_bcq),
                # exactly as studied standalone in t3_proxy_study.py. This is
                # a PLAIN ANNOTATION column, not a topology/signal promotion:
                # it does not touch topo/special/keep/weight above, so it
                # cannot change S1 or M3 training at all -- purely for a
                # reference curve on the bbc score plot (no adequate real
                # proxy exists there, unlike cb's t2(b'c) proxy).
                m_top_bcq = flatten(events["ak8_match_top_bcq"], 0).astype(bool)
                t3bcq_proxy_flag = (w_decay_bg == 4) & m_top_bcq

            if not keep.any():
                continue

            cols = {"topology": topo[keep].astype(np.int8),
                    "weight": np.abs(weight[keep]).astype(np.float64),
                    "weight_signed": weight[keep].astype(np.float64),
                    "sample": np.repeat(path.name, int(keep.sum())),
                    "ak8_pt": jpt[keep].astype(np.float32),
                    "ak8_sdmass": jmsd[keep].astype(np.float32),
                    "t3bcq_proxy_flag": t3bcq_proxy_flag[keep]}
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
        "preselection_used": presel, "preselection_source": cut_expr,
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
    _done_banner(t_wall0, data_dir)


if __name__ == "__main__":
    main()
