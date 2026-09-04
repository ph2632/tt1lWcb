#!/usr/bin/env python3
"""Generate calibration-related figures (FIG-CAL-1 through FIG-CAL-5).

FIG-CAL-1: Generator-level category fractions after preselection
FIG-CAL-2: Dbc score distribution in calib CR, decomposed by category
FIG-CAL-3: EC score distribution in calib CR, decomposed by category
FIG-CAL-4: Dbc shape comparison: signal vs top(bc) background
FIG-CAL-5: Non-top(bc) contamination purity vs Dbc threshold

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
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_/")
        self.tree_name = "Events"
        self.figure_path = "./plots/plot/"
        self.lumi = 41.479
        self.enable_data = True
        self.cms_label = "Preliminary"

        # Colors unified with plot_all.py
        self.colors = {
            "signal": "#0B3603",
            "Wcb": "#0B3603",
            "top_bc": "#964A8B",
            "top_bq": "#5790FC",
            "top_bqq": "#E42536",
            "W_qq": "#7A21DD",
            "QCD": "#F89C20",
            "Other": "#9C9CA1",
            "ratio_k": "#333333",
        }


def draw_cms_label(cfg, ax):
    hep.cms.label(cfg.cms_label, data=cfg.enable_data, lumi=cfg.lumi, ax=ax)


# ======================================================================
# Data Loading
# ======================================================================
def load_mc_data(cfg):
    files = sorted(f for f in os.listdir(cfg.mc_path) if f.endswith(".root"))
    print(f"[INFO] Loading {len(files)} MC ROOT files...")

    branches = [
        "ak8_pt", "ak8_eta", "ak8_sdmass",
        "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
        "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw",
        "ak8_match_wqq", "ak8_match_wcq", "ak8_match_wqq_wcb",
        "ak8_match_tbq_wcb", "ak8_match_tbqq_wcb",
        "ak8_match_top_bc", "ak8_match_top_bq", "ak8_match_top_bqq",
        "ak8_match_top_bcq", "ak8_match_non", "ak8_match_qcd",
        "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
        "score_cata_top_bc", "score_cata_top_bq", "score_cata_non",
    ]

    all_data = []
    for fname in files:
        path = os.path.join(cfg.mc_path, fname)
        try:
            f = uproot.open(path)
            tree = f[cfg.tree_name]
            available = [b for b in branches if b in tree.keys()]
            events = tree.arrays(available, library="ak")
            events["_sample"] = fname
            all_data.append(events)
        except Exception as e:
            print(f"[WARN] Failed to load {fname}: {e}")

    if not all_data:
        raise RuntimeError("No MC files loaded!")
    data = ak.concatenate(all_data)
    print(f"[INFO] Loaded {len(data)} MC events")
    return data


# ======================================================================
# Common Dbc computation
# ======================================================================
def compute_dbc(data):
    g_bc = ak.fill_none(ak.firsts(data.ak8_gpt_bc), 0)
    g_bb = ak.fill_none(ak.firsts(data.ak8_gpt_bb), 0)
    g_cc = ak.fill_none(ak.firsts(data.ak8_gpt_cc), 0)
    g_qcd = ak.fill_none(ak.firsts(data.ak8_gpt_qcd), 0)
    g_bs = ak.fill_none(ak.firsts(data.ak8_gpt_bs), 0)
    g_qq = ak.fill_none(ak.firsts(data.ak8_gpt_qq), 0)
    g_cs = ak.fill_none(ak.firsts(data.ak8_gpt_cs), 0)
    g_topbw = ak.fill_none(ak.firsts(data.ak8_gpt_topbw), 0)
    return g_bc / (g_bc + g_qcd + g_cc + g_cs + g_qq + g_bs + g_bb + g_topbw + 1e-10)


# ======================================================================
# FIG-CAL-1: Generator-level category fractions after preselection
# ======================================================================
def plot_fig_cal_1(cfg, data):
    print("[INFO] Generating FIG-CAL-1...")

    has_ak8 = ak.num(data.ak8_pt) > 0
    ak8_pt_0 = ak.fill_none(ak.firsts(data.ak8_pt), -1)
    ak8_sdmass_0 = ak.fill_none(ak.firsts(data.ak8_sdmass), -1)
    mask_ps = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30)
    dps = data[mask_ps]

    cat_items = [
        ("match_top_bc",   r"Top ($bc$)",           cfg.colors["top_bc"]),
        ("match_top_bq",   r"Top ($bq$)",           cfg.colors["top_bq"]),
        ("match_top_bqq",  r"Top ($bqq'$)",         cfg.colors["top_bqq"]),
        ("match_wqq_wcb",  r"$W\to cb$ (signal)",   cfg.colors["signal"]),
        ("match_wqq",      r"$W\to qq'$ (non-$cb$)", cfg.colors["W_qq"]),
        ("match_qcd",      "QCD-originated",         cfg.colors["QCD"]),
        ("match_non",      "Non-matched",             cfg.colors["Other"]),
    ]

    counts = {}
    for flag, label, _ in cat_items:
        counts[label] = int(ak.sum(ak.fill_none(ak.firsts(dps[f"ak8_{flag}"]), 0) > 0))

    labels = list(counts.keys())
    vals = list(counts.values())
    colors_use = [c for _, _, c in cat_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), vals, color=colors_use, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Number of jets passing PS", fontsize=13)
    ax.xaxis.grid(True, alpha=0.3)

    total = sum(vals)
    for i, c in enumerate(vals):
        frac = c / total * 100 if total > 0 else 0
        ax.text(c + total * 0.01, i, f"{c} ({frac:.1f}%)", va="center", fontsize=10)

    draw_cms_label(cfg, ax)
    plt.tight_layout()
    out = os.path.join(cfg.figure_path, "FIG-CAL-1_category_fractions_PS.pdf")
    os.makedirs(cfg.figure_path, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[SAVE] {out}")
    return counts


# ======================================================================
# FIG-CAL-2: Dbc score distribution in calib CR, by category
# ======================================================================
def plot_fig_cal_2(cfg, data):
    print("[INFO] Generating FIG-CAL-2...")

    has_ak8 = ak.num(data.ak8_pt) > 0
    ak8_pt_0 = ak.fill_none(ak.firsts(data.ak8_pt), -1)
    ak8_sdmass_0 = ak.fill_none(ak.firsts(data.ak8_sdmass), -1)
    dbc = compute_dbc(data)
    mask = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30) & (dbc > 0.985)
    dbc_filt = ak.to_numpy(dbc[mask])

    def get_flag(flag):
        return ak.to_numpy(ak.fill_none(ak.firsts(data[f"ak8_{flag}"][mask]), 0) > 0)

    cat_specs = [
        ("W (bc)",        get_flag("match_wqq_wcb"), cfg.colors["signal"]),
        ("Top (bc)",      get_flag("match_top_bc"),  cfg.colors["top_bc"]),
        ("Top (bq)",      get_flag("match_top_bq"),  cfg.colors["top_bq"]),
        ("Top (bqq)",     get_flag("match_top_bqq"), cfg.colors["top_bqq"]),
        ("W (qq) non-cb", get_flag("match_wqq") & ~get_flag("match_wqq_wcb"), cfg.colors["W_qq"]),
        ("QCD",           get_flag("match_qcd"),     cfg.colors["QCD"]),
        ("Other",         get_flag("match_non"),     cfg.colors["Other"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.985, 1.0, 31)
    for label, mask, color in cat_specs:
        vals = dbc_filt[mask]
        if len(vals) < 5:
            continue
        ax.hist(vals, bins=bins, histtype="step", linewidth=2,
                color=color, label=f"{label} ({len(vals)})")

    ax.legend(fontsize=10, ncol=2)
    ax.set_xlabel(r"$D_{bc}$", fontsize=13)
    ax.set_ylabel("Jets", fontsize=13)
    draw_cms_label(cfg, ax)
    plt.tight_layout()
    out = os.path.join(cfg.figure_path, "FIG-CAL-2_dbc_per_category_calibCR.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[SAVE] {out}")


# ======================================================================
# FIG-CAL-3: EC score distribution in calib CR
# ======================================================================
def plot_fig_cal_3(cfg, data):
    print("[INFO] Generating FIG-CAL-3...")

    has_ak8 = ak.num(data.ak8_pt) > 0
    ak8_pt_0 = ak.fill_none(ak.firsts(data.ak8_pt), -1)
    ak8_sdmass_0 = ak.fill_none(ak.firsts(data.ak8_sdmass), -1)
    dbc = compute_dbc(data)
    mask = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30) & (dbc > 0.985)

    def safe_get(arr, default=0.0):
        return ak.to_numpy(ak.fill_none(arr, default)).astype(float)

    s_w_qq = safe_get(data.score_cata_w_qq)
    s_qcd = safe_get(data.score_cata_qcd)
    s_tbqq = safe_get(data.score_cata_top_bqq)
    s_tbc = safe_get(data.score_cata_top_bc)
    s_tbq = safe_get(data.score_cata_top_bq)
    s_non = safe_get(data.score_cata_non)
    ec = s_w_qq / (s_w_qq + s_qcd + s_tbqq + s_tbc + s_tbq + s_non + 1e-10)
    ec_filt = ec[ak.to_numpy(mask)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 51)
    ax.hist(ec_filt, bins=bins, histtype="step", linewidth=2,
            color=cfg.colors["top_bq"], label=f"$D_{{bc}} > 0.985$ ({len(ec_filt)} jets)")
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1.5, label=r"$S_{\mathrm{EVT}} = 0.7$")
    ax.legend(fontsize=12)
    ax.set_xlabel(r"$S_{\mathrm{EVT}}$ (EventClassifier)", fontsize=13)
    ax.set_ylabel("Jets", fontsize=13)
    ax.set_xlim(0.0, 1.0)
    draw_cms_label(cfg, ax)
    plt.tight_layout()
    out = os.path.join(cfg.figure_path, "FIG-CAL-3_ec_distribution_calibCR.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[SAVE] {out}")


# ======================================================================
# FIG-CAL-4: Dbc shape comparison: signal W->cb vs top(bc) background
# ======================================================================
def plot_fig_cal_4(cfg, data):
    print("[INFO] Generating FIG-CAL-4...")

    has_ak8 = ak.num(data.ak8_pt) > 0
    ak8_pt_0 = ak.fill_none(ak.firsts(data.ak8_pt), -1)
    ak8_sdmass_0 = ak.fill_none(ak.firsts(data.ak8_sdmass), -1)
    dbc = compute_dbc(data)

    mask_ps = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30)
    d_dbc = ak.to_numpy(dbc[mask_ps])
    is_sig = ak.to_numpy(ak.fill_none(ak.firsts(data.ak8_match_wqq_wcb[mask_ps]), 0) > 0)
    is_topbc = ak.to_numpy(ak.fill_none(ak.firsts(data.ak8_match_top_bc[mask_ps]), 0) > 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    bins = np.linspace(0, 1, 41)

    for label, mask, color in [
        (r"Signal $W\to cb$", is_sig, cfg.colors["signal"]),
        (r"Top ($bc$) background", is_topbc, cfg.colors["top_bc"]),
    ]:
        vals = d_dbc[mask]
        axes[0].hist(vals, bins=bins, histtype="step", linewidth=2, color=color,
                     label=f"{label} ({len(vals)})", density=True)
    axes[0].legend(fontsize=10)
    axes[0].set_xlabel(r"$D_{bc}$", fontsize=12)
    axes[0].set_ylabel("Normalised entries", fontsize=12)
    axes[0].set_xlim(0.0, 1.0)

    h_sig, _ = np.histogram(d_dbc[is_sig], bins=bins, density=True)
    h_topbc, _ = np.histogram(d_dbc[is_topbc], bins=bins, density=True)
    ratio = np.where(h_topbc > 0, h_sig / h_topbc, 1.0)
    bc = (bins[:-1] + bins[1:]) / 2
    axes[1].step(bc, ratio, where="mid", linewidth=2, color=cfg.colors["ratio_k"])
    axes[1].axhline(1, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"$D_{bc}$", fontsize=12)
    axes[1].set_ylabel(r"Signal / Top($bc$)", fontsize=12)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0, 2)

    draw_cms_label(cfg, axes[0])
    plt.tight_layout()
    out = os.path.join(cfg.figure_path, "FIG-CAL-4_dbc_shape_signal_vs_topbc.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[SAVE] {out}")


# ======================================================================
# FIG-CAL-5: Non-top(bc) contamination purity vs Dbc threshold
# ======================================================================
def plot_fig_cal_5(cfg, data):
    print("[INFO] Generating FIG-CAL-5...")

    has_ak8 = ak.num(data.ak8_pt) > 0
    ak8_pt_0 = ak.fill_none(ak.firsts(data.ak8_pt), -1)
    ak8_sdmass_0 = ak.fill_none(ak.firsts(data.ak8_sdmass), -1)
    dbc = compute_dbc(data)

    mask_ps = has_ak8 & (ak8_pt_0 > 200) & (ak8_sdmass_0 > 30)
    d_dbc = ak.to_numpy(dbc[mask_ps])
    is_topbc = ak.to_numpy(ak.fill_none(ak.firsts(data.ak8_match_top_bc[mask_ps]), 0) > 0)

    if len(d_dbc) < 10:
        print("[WARN] Not enough events for FIG-CAL-5")
        return

    thresholds = np.linspace(0.9, 0.999, 50)
    purities = []
    totals = []
    for th in thresholds:
        sel = d_dbc > th
        total = sel.sum()
        topbc = (sel & is_topbc).sum()
        purities.append(topbc / total if total > 0 else 0)
        totals.append(total)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, purities, linewidth=2, color=cfg.colors["top_bc"])
    ax.axvline(0.985, color="red", linestyle="--", linewidth=1, label=r"$D_{bc}=0.985$")

    ax2 = ax.twinx()
    ax2.plot(thresholds, totals, linewidth=1.5, color=cfg.colors["top_bq"], linestyle="dotted")
    ax2.set_ylabel("Total jets above threshold", fontsize=12, color=cfg.colors["top_bq"])

    ax.legend(fontsize=10)
    ax.set_xlabel(r"$D_{bc}$ threshold", fontsize=12)
    ax.set_ylabel(r"Top($bc$) purity", fontsize=12, color=cfg.colors["top_bc"])
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(0, 1.05)
    draw_cms_label(cfg, ax)
    plt.tight_layout()
    out = os.path.join(cfg.figure_path, "FIG-CAL-5_topbc_purity_vs_dbc.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[SAVE] {out}")
    idx = np.argmin(np.abs(thresholds - 0.985))
    print(f"[INFO] Top(bc) purity at Dbc>0.985: {purities[idx]:.1%}")


# ======================================================================
# Main
# ======================================================================
def main():
    start = time.time()
    cfg = Config()
    hep.style.use("CMS")

    print("[INFO] Loading MC data...")
    data = load_mc_data(cfg)

    print("[INFO] Creating calibration figures...")
    cat_counts = plot_fig_cal_1(cfg, data)
    plot_fig_cal_2(cfg, data)
    plot_fig_cal_3(cfg, data)
    plot_fig_cal_4(cfg, data)
    plot_fig_cal_5(cfg, data)

    elapsed = time.time() - start
    print(f"[DONE] All calibration figures generated in {elapsed:.1f}s")
    print(f"[DONE] Output directory: {cfg.figure_path}")

    total = sum(cat_counts.values())
    topbc_ps = cat_counts.get(r"Top ($bc$)", 0)
    print(f"\n=== Summary for AN ===")
    print(f"Total jets after PS: {total}")
    print(f"Top(bc) jets after PS: {topbc_ps} ({topbc_ps/max(total,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
