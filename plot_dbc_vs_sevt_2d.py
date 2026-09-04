#!/usr/bin/env python3
"""2D distribution in (Dbc, S_EVT) plane — 4-panel by calibration category.

Dbc uses BDT-based score (same as Combine). All PS events shown (no Dbc pre-cut).
Power-normalised coloring for sparse visibility.
"""

import os, sys, time
import numpy as np
import awkward as ak
import uproot
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep

# ======================================================================
class Config:
    def __init__(self):
        self.mc_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_1merged_/"
        self.tree_name = "Events"
        self.figure_path = "./plots/plot/"
        self.lumi = 41.479

        # BDT Dbc model (same as Combine)
        self.bdt_model_path = "./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"

        self.colors = {
            "sig":       "#00AA00",
            "bkg_wqq":   "#D41159",
            "bkg_topbc": "#FF7F00",
            "bkg_other": "#377EB8",
        }
        self.cat_labels = {
            "sig":       r"$W\to cb$ signal",
            "bkg_wqq":   r"$W\to qq'$ (non-$cb$)",
            "bkg_topbc": r"Top($bc$) bkg.",
            "bkg_other": "Other bkg.",
        }
        self.cat_order = ["sig", "bkg_wqq", "bkg_topbc", "bkg_other"]


def draw_cms_label(cfg, ax):
    hep.cms.label("Preliminary", data=False, lumi=cfg.lumi, ax=ax)


# ======================================================================
def make_dbc_bins(n_left=35, n_right=35):
    """Smooth non-uniform bins with single transition at Dbc cut."""
    b1 = np.linspace(0.0, 0.985, n_left, endpoint=False)
    b2 = np.linspace(0.985, 1.0, n_right + 1)
    return np.concatenate([b1, b2])

EPS = 1e-5
def dbc_forward(x):
    return -np.log10(np.maximum(1.0 + EPS - x, EPS))
def dbc_inverse(t):
    return 1.0 + EPS - np.power(10.0, -t)

# ======================================================================
def count_str(n):
    if abs(n) < 1e-4:
        return "0"
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a < 1000:
        return f"{sign}{a:.1f}"
    elif a < 1e6:
        return f"{sign}{a/1e3:.1f}k"
    else:
        return f"{sign}{a/1e6:.1f}M"

# ======================================================================
def load_mc_data(cfg):
    files = sorted(f for f in os.listdir(cfg.mc_path) if f.endswith(".root"))
    print(f"[INFO] Loading {len(files)} MC ROOT files...")
    branches = [
        "ak8_pt", "ak8_sdmass",
        "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
        "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw",
        "ak8_is_wbc", "ak8_match_wqq", "ak8_match_top_bc",
        "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
        "score_cata_top_bc", "score_cata_top_bq", "score_cata_non",
        "xsecWeight", "genWeight", "lumiwgt",
        "puWeight", "trigEffWeight", "l1PreFiringWeight",
        "flavTagWeight", "elEffWeight", "muEffWeight", "topptWeight",
    ]
    all_data = []
    for fname in files:
        path = os.path.join(cfg.mc_path, fname)
        try:
            f = uproot.open(path)
            tree = f[cfg.tree_name]
            available = [b for b in branches if b in tree.keys()]
            events = tree.arrays(available, library="ak")
            all_data.append(events)
        except Exception as e:
            print(f"[WARN] Failed {fname}: {e}")
    return ak.concatenate(all_data)


def compute_dbc_bdt(data, bdt_model):
    """BDT-based Dbc, matching the Combine."""
    g_bc = ak.fill_none(ak.firsts(data.ak8_gpt_bc), 0.0)
    g_bb = ak.fill_none(ak.firsts(data.ak8_gpt_bb), 0.0)
    g_cc = ak.fill_none(ak.firsts(data.ak8_gpt_cc), 0.0)
    g_qcd = ak.fill_none(ak.firsts(data.ak8_gpt_qcd), 0.0)
    g_bs = ak.fill_none(ak.firsts(data.ak8_gpt_bs), 0.0)
    g_qq = ak.fill_none(ak.firsts(data.ak8_gpt_qq), 0.0)
    g_cs = ak.fill_none(ak.firsts(data.ak8_gpt_cs), 0.0)
    g_topbw = ak.fill_none(ak.firsts(data.ak8_gpt_topbw), 0.0)

    import pandas as pd
    X = pd.DataFrame({
        "ak8_gpt_bc":    ak.to_numpy(g_bc).astype(np.float32),
        "ak8_gpt_bb":    ak.to_numpy(g_bb).astype(np.float32),
        "ak8_gpt_cc":    ak.to_numpy(g_cc).astype(np.float32),
        "ak8_gpt_qcd":   ak.to_numpy(g_qcd).astype(np.float32),
        "ak8_gpt_bs":    ak.to_numpy(g_bs).astype(np.float32),
        "ak8_gpt_qq":    ak.to_numpy(g_qq).astype(np.float32),
        "ak8_gpt_cs":    ak.to_numpy(g_cs).astype(np.float32),
        "ak8_gpt_topbw": ak.to_numpy(g_topbw).astype(np.float32),
    })
    return bdt_model.predict_proba(X[list(X.columns)])[:, 1].astype(np.float32)


def compute_weight(events):
    """Exact match to make_shape_asimov.py."""
    w = ak.fill_none(events.genWeight, 1.0)
    w = w * ak.fill_none(events.lumiwgt, 1.0)
    w = w * ak.fill_none(events.xsecWeight, 1.0)
    w = w * ak.fill_none(events.puWeight, 1.0)
    w = w * ak.fill_none(events.trigEffWeight, 1.0)
    w = w * ak.fill_none(events.elEffWeight, 1.0)
    w = w * ak.fill_none(events.muEffWeight, 1.0)
    if "l1PreFiringWeight" in events.fields:
        w = w * ak.fill_none(events.l1PreFiringWeight, 1.0)
    w = w * ak.fill_none(events.flavTagWeight, 1.0)
    if "topptWeight" in events.fields:
        w = w * ak.fill_none(events.topptWeight, 1.0)
    return w

# ======================================================================
def main():
    start = time.time()
    cfg = Config()
    hep.style.use("CMS")

    # Load BDT model
    if not os.path.exists(cfg.bdt_model_path):
        print(f"[ERROR] BDT model not found: {cfg.bdt_model_path}")
        sys.exit(1)
    bdt_model = joblib.load(cfg.bdt_model_path)
    print(f"[INFO] BDT model loaded from {cfg.bdt_model_path}")

    data = load_mc_data(cfg)

    # --- Preselection ---
    has_ak8 = ak.num(data.ak8_pt) > 0
    ak8_pt0 = ak.fill_none(ak.firsts(data.ak8_pt), -1)
    ak8_sdmass0 = ak.fill_none(ak.firsts(data.ak8_sdmass), -1)
    mask_ps = has_ak8 & (ak8_pt0 > 200) & (ak8_sdmass0 > 30)
    d = data[mask_ps]
    print(f"[INFO] Events after PS: {int(ak.sum(mask_ps))}")

    evt_w = ak.to_numpy(compute_weight(d)).astype(np.float64)
    evt_w = np.where(np.isfinite(evt_w), evt_w, 0.0)

    # BDT-based Dbc
    dbc = compute_dbc_bdt(d, bdt_model)
    mask_dbc_valid = (dbc >= 0) & (dbc <= 1) & np.isfinite(dbc)

    def safe_get(arr, default=0.0):
        return ak.to_numpy(ak.fill_none(arr, default)).astype(float)

    s_w_qq = safe_get(d.score_cata_w_qq)
    s_qcd = safe_get(d.score_cata_qcd)
    s_tbqq = safe_get(d.score_cata_top_bqq)
    s_tbc = safe_get(d.score_cata_top_bc)
    s_tbq = safe_get(d.score_cata_top_bq)
    s_non = safe_get(d.score_cata_non)
    ec = s_w_qq / (s_w_qq + s_qcd + s_tbqq + s_tbc + s_tbq + s_non + 1e-10)
    mask_ec_valid = (ec >= 0) & (ec <= 1) & np.isfinite(ec)

    # --- Categories (sequential priority) ---
    match_wqq   = ak.to_numpy(ak.fill_none(ak.firsts(d.ak8_match_wqq), 0)) > 0
    match_topbc = ak.to_numpy(ak.fill_none(ak.firsts(d.ak8_match_top_bc), 0)) > 0
    is_wbc      = ak.to_numpy(ak.fill_none(ak.firsts(d.ak8_is_wbc), 0)) == 1

    is_sig     = is_wbc
    is_wqq_bkg = (~is_sig) & match_wqq
    is_topbc   = (~is_sig) & (~is_wqq_bkg) & match_topbc
    is_other   = (~is_sig) & (~is_wqq_bkg) & (~is_topbc)

    valid = mask_dbc_valid & mask_ec_valid
    dbc = dbc[valid]
    ec = ec[valid]
    evt_w = evt_w[valid]
    cat_masks = {
        "sig":       is_sig[valid],
        "bkg_wqq":   is_wqq_bkg[valid],
        "bkg_topbc": is_topbc[valid],
        "bkg_other": is_other[valid],
    }

    for k, m in cat_masks.items():
        print(f"[INFO]   {k}: unweighted={m.sum():.0f}  weighted={evt_w[m].sum():.2f}")

    # ==================================================================
    # FIGURE: 2x2 subplots
    # ==================================================================
    DBCCUT = 0.985
    ECCUT = 0.7
    dbc_bins = make_dbc_bins(35, 35)
    ec_bins = np.linspace(0, 1, 41)
    major_ticks = [0.0, 0.5, 0.9, 0.95, 0.98, 0.985, 0.99, 0.995, 0.998, 0.999, 0.9995]

    from matplotlib.colors import PowerNorm, to_rgba, LinearSegmentedColormap

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, cat_name in enumerate(cfg.cat_order):
        ax = axes[idx]
        mask = cat_masks[cat_name]
        if mask.sum() < 5:
            ax.text(0.5, 0.5, "Insufficient events", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            continue

        h, xe, ye = np.histogram2d(dbc[mask], ec[mask],
                                    bins=[dbc_bins, ec_bins],
                                    weights=evt_w[mask])
        # Suppress negative bins, keep positive
        h_pos = np.maximum(h, 0)
        nonzero = h_pos[h_pos > 0]
        vmin = nonzero.min() if len(nonzero) > 0 else 1e-10
        vmax = h_pos.max()

        rgb = to_rgba(cfg.colors[cat_name])[:3]
        cmap = LinearSegmentedColormap.from_list(
            f'cmap_{cat_name}',
            [(0, (*rgb, 0.0)), (0.05, (*rgb, 0.1)), (0.3, (*rgb, 0.4)), (1, (*rgb, 1.0))])

        ax.pcolormesh(xe, ye, h_pos.T,
                      norm=PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax),
                      cmap=cmap, shading='auto', rasterized=True)

        # Cut lines
        ax.axvline(DBCCUT, color='black', linewidth=1.8, linestyle='-')
        ax.axhline(ECCUT, color='black', linewidth=1.8, linestyle='-')

        # Log-like x-axis
        ax.set_xscale('function', functions=(dbc_forward, dbc_inverse))
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
        def fmt(x, pos):
            return f"{x:.3f}" if x > 0.985 else f"{x:.2f}" if x > 0.1 else f"{x:.1f}"
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
        ax.tick_params(axis='x', labelsize=10, rotation=30)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(0, 1)

        # --- Annotate 4 quadrants ---
        tot_w = evt_w[mask].sum()
        m_sr  = (dbc > DBCCUT) & (ec > ECCUT)
        m_cr  = (dbc > DBCCUT) & (ec <= ECCUT)
        m_tl  = (dbc <= DBCCUT) & (ec > ECCUT)
        m_bl  = (dbc <= DBCCUT) & (ec <= ECCUT)

        def eff_str(n):
            return f"{n/max(abs(tot_w),1e-12)*100:.2f}%"

        cat_color = cfg.colors[cat_name]

        # SR (top-right) — inside plot
        ax.text(0.993, 0.94, "SR", ha='left', va='top', fontsize=12, fontweight='bold',
                color=cat_color)
        sr_w = evt_w[mask & m_sr].sum()
        ax.text(0.993, 0.88, rf"$\epsilon$={eff_str(sr_w)}  [{count_str(sr_w)}]",
                ha='left', va='top', fontsize=10, fontfamily='monospace', color='#333333')

        # calib CR (bottom-right) — inside plot
        ax.text(0.993, 0.44, "calib CR", ha='left', va='center', fontsize=12, fontweight='bold',
                color=cat_color)
        cr_w = evt_w[mask & m_cr].sum()
        ax.text(0.993, 0.38, rf"$\epsilon$={eff_str(cr_w)}  [{count_str(cr_w)}]",
                ha='left', va='center', fontsize=10, fontfamily='monospace', color='#333333')

        # Top-left
        tl_w = evt_w[mask & m_tl].sum()
        ax.text(0.007, 0.94, "passing", ha='left', va='top', fontsize=9,
                color='#999999', fontstyle='italic')
        ax.text(0.007, 0.88, rf"{eff_str(tl_w)}  [{count_str(tl_w)}]",
                ha='left', va='top', fontsize=9, fontfamily='monospace', color='#999999')

        # Bottom-left
        bl_w = evt_w[mask & m_bl].sum()
        ax.text(0.007, 0.44, "failing", ha='left', va='center', fontsize=9,
                color='#999999', fontstyle='italic')
        ax.text(0.007, 0.38, rf"{eff_str(bl_w)}  [{count_str(bl_w)}]",
                ha='left', va='center', fontsize=9, fontfamily='monospace', color='#999999')

        # Title — inside plot, top-center
        tot_label = cfg.cat_labels[cat_name]
        ax.text(0.50, 0.97, f"{tot_label}  ($\\Sigma$ = {count_str(tot_w)})",
                ha='center', va='top', transform=ax.transAxes,
                fontsize=14, color=cat_color, fontweight='bold')

        draw_cms_label(cfg, ax)

    fig.supxlabel(r"$D_{bc}^{\mathrm{BDT}}$", fontsize=18, y=0.02)
    fig.supylabel(r"$S_{\mathrm{EVT}}$ (EventClassifier)", fontsize=18, x=0.02)
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.98])
    fig.subplots_adjust(wspace=0.25, hspace=0.30)

    os.makedirs(cfg.figure_path, exist_ok=True)
    out = os.path.join(cfg.figure_path, "dbc_vs_sevt_2d_4panel.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"[SAVE] {out}")

    elapsed = time.time() - start
    print(f"[DONE] {elapsed:.1f}s")


if __name__ == "__main__":
    main()
