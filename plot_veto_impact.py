#!/usr/bin/env python3
"""Jet Veto Map impact study.

Computes and plots the fraction of events vetoed per sample category
for 2017 data.

Data source: scored ROOT samples from RunTTH/_2017_1L/
"""

import os, time
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import mplhep as hep

# ======================================================================
# Configuration
# ======================================================================
class Config:
    def __init__(self):
        self.mc_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_1merged_/"
        self.data_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/Data/scored_samples_1merged_/"
        self.tree_name = "Events"
        self.figure_path = "./plots/plot/"
        self.lumi = 41.479
        self.enable_data = True

        # Colors unified with plot_all.py
        self.colors = {
            "Data": "#333333",
            "tt": "#5790FC",
            "QCD": "#F89C20",
            "single-top": "#E42536",
            "Rare": "#964A8B",
            "Diboson": "#7A21DD",
            "Wcb": "#0B3603",
            "Other": "#999999",
        }

        self.sample_map = {
            "ttbar-powheg": "Wcb",
            "tt-semi": "tt",
            "tt-lep": "tt",
            "tt-had": "tt",
            "single-top": "single-top",
            "QCD": "QCD",
            "ttbb": "Rare",
            "ttW": "Rare",
            "ttZ": "Rare",
            "twZ": "Rare",
            "ttHToTauTau": "Rare",
            "ttHNonbb": "Rare",
            "WW": "Diboson",
            "WZ": "Diboson",
            "ZZ": "Diboson",
            "Wjet": "Other",
            "Zjet": "Other",
        }

        self.group_order = ["Wcb", "tt", "single-top", "QCD", "Rare", "Diboson", "Other"]
        self.group_labels = {
            "Wcb": r"$W_{cb}$ (signal)",
            "tt": r"$t\bar{t}$",
            "QCD": "QCD multijet",
            "single-top": "Single top",
            "Rare": "Rare",
            "Diboson": "Diboson",
            "Other": "Other",
            "Data": "Data",
        }


def draw_cms_label(cfg, ax):
    hep.cms.label("Preliminary", data=cfg.enable_data, lumi=cfg.lumi, ax=ax)


def load_events(path, branches, sample_map, is_mc):
    files = sorted(f for f in os.listdir(path) if f.endswith(".root"))
    all_data = []
    for fname in files:
        filepath = os.path.join(path, fname)
        try:
            f = uproot.open(filepath)
            tree = f["Events"]
            available = [b for b in branches if b in tree.keys()]
            evt = tree.arrays(available, library="ak")
            if is_mc:
                group = "Other"
                for pat, grp in sample_map.items():
                    if pat in fname.replace("_merged.root", ""):
                        group = grp
                        break
                evt["_group"] = ak.Array(np.full(len(evt), group))
            all_data.append(evt)
        except Exception as e:
            print(f"[WARN] Failed {fname}: {e}")
    if not all_data:
        return None
    return ak.concatenate(all_data)


def main():
    start = time.time()
    cfg = Config()
    hep.style.use("CMS")

    branches = [
        "jetVetoMapEventVeto", "ak8_pt", "ak8_sdmass", "ak8_eta",
    ]

    print("[INFO] Loading MC with jetVetoMapEventVeto...")
    mc = load_events(cfg.mc_path, branches, cfg.sample_map, is_mc=True)
    print("[INFO] Loading Data...")
    data = load_events(cfg.data_path, branches, {}, is_mc=False)

    results = {}
    # Process MC by group
    if mc is not None:
        for group in cfg.group_order:
            mask = mc["_group"] == group
            evts = mc[mask]
            has_ak8 = ak.num(evts.ak8_pt) > 0
            ak8_pt_0 = ak.fill_none(ak.firsts(evts.ak8_pt), -1)
            ak8_sdmass_0 = ak.fill_none(ak.firsts(evts.ak8_sdmass), -1)
            pass_ps = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30)
            n_ps = int(ak.sum(pass_ps))
            veto = ak.to_numpy(evts.jetVetoMapEventVeto)[ak.to_numpy(pass_ps)]
            n_vetoed = int(np.sum(veto))
            frac = n_vetoed / n_ps * 100 if n_ps > 0 else 0
            results[group] = {"n_ps": n_ps, "n_vetoed": n_vetoed, "frac": frac}

    # Process Data
    if data is not None:
        evts = data
        has_ak8 = ak.num(evts.ak8_pt) > 0
        ak8_pt_0 = ak.fill_none(ak.firsts(evts.ak8_pt), -1)
        ak8_sdmass_0 = ak.fill_none(ak.firsts(evts.ak8_sdmass), -1)
        pass_ps = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30)
        n_ps = int(ak.sum(pass_ps))
        veto = ak.to_numpy(evts.jetVetoMapEventVeto)[ak.to_numpy(pass_ps)]
        n_vetoed = int(np.sum(veto))
        frac = n_vetoed / n_ps * 100 if n_ps > 0 else 0
        results["Data"] = {"n_ps": n_ps, "n_vetoed": n_vetoed, "frac": frac}

    # Draw
    plot_order = ["Data"] + cfg.group_order
    x_labels = []
    veto_fracs = []
    colors_use = []
    for group in plot_order:
        if group in results:
            x_labels.append(cfg.group_labels.get(group, group))
            veto_fracs.append(results[group]["frac"])
            colors_use.append(cfg.colors.get(group, cfg.colors["Other"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(x_labels)), veto_fracs, color=colors_use, alpha=0.85)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=11, rotation=20)
    ax.set_ylabel("Veto efficiency [%]", fontsize=13)
    ax.yaxis.grid(True, alpha=0.3)

    for i, (label, frac) in enumerate(zip(x_labels, veto_fracs)):
        group_key = plot_order[i]
        if group_key in results:
            r = results[group_key]
            ax.text(i, frac + 0.15, f"{frac:.2f}%\n({r['n_vetoed']}/{r['n_ps']})",
                    ha="center", va="bottom", fontsize=8)

    draw_cms_label(cfg, ax)
    plt.tight_layout()

    os.makedirs(cfg.figure_path, exist_ok=True)
    out = os.path.join(cfg.figure_path, "jet_veto_impact.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[SAVE] {out}")

    # Summary
    print("\n=== Jet Veto Impact Summary ===")
    print(f"{'Group':<20} {'N(PS)':>12} {'N(vetoed)':>12} {'Fraction':>10}")
    for group in plot_order:
        if group in results:
            r = results[group]
            print(f"{cfg.group_labels.get(group, group):<20} {r['n_ps']:>12} {r['n_vetoed']:>12} {r['frac']:>8.2f}%")

    elapsed = time.time() - start
    print(f"\n[DONE] {elapsed:.1f}s")


if __name__ == "__main__":
    main()
