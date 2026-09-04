#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

from sklearn.metrics import roc_curve, roc_auc_score

# ============================================================
# Config
# ============================================================
INPUT_DIR = "./dbc_bdt_output"
INPUT_FILE = os.path.join(INPUT_DIR, "valid_prediction.parquet")

FIG_DIR = os.path.join(INPUT_DIR, "figures_compare_ratio_vs_bdt")
os.makedirs(FIG_DIR, exist_ok=True)

hep.style.use("CMS")

CMS_LABEL = "Simulation"

# mass windows
MASS_WINDOWS = [
    ("m50", 40.0, 60.0),
    ("m80", 65.0, 95.0),
    ("m125", 105.0, 145.0),
]

# pt windows
PT_WINDOWS = [
    ("pt200to300", 200.0, 300.0),
    ("pt300to450", 300.0, 450.0),
    ("pt450toInf", 450.0, np.inf),
]

# ratio definition option
USE_BB_BS = False
EPS = 1e-10

def add_working_points(ax, y, score, weight, cuts, color="blue", label_prefix="cut"):
    wp_df = get_working_points_from_cuts(y, score, weight, cuts)

    ax.scatter(
        wp_df["sig_eff"],
        wp_df["bkg_eff"],
        color=color,
        s=30,
        zorder=5
    )

    for _, row in wp_df.iterrows():
        ax.annotate(
            f"{label_prefix}={row['cut']:.2f}",
            (row["sig_eff"], row["bkg_eff"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=14,
            color=color
        )

    return wp_df

def get_working_points_from_cuts(y, score, weight, cuts):
    """
    对给定 cuts，返回每个 cut 对应的 signal/background efficiency
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    cuts = np.asarray(cuts, dtype=np.float64)

    sig_mask = (y == 1)
    bkg_mask = (y == 0)

    w_sig_total = weight[sig_mask].sum()
    w_bkg_total = weight[bkg_mask].sum()

    rows = []
    for cut in cuts:
        pass_mask = (score >= cut)

        sig_eff = weight[sig_mask & pass_mask].sum() / max(w_sig_total, 1e-12)
        bkg_eff = weight[bkg_mask & pass_mask].sum() / max(w_bkg_total, 1e-12)

        rows.append({
            "cut": cut,
            "sig_eff": sig_eff,
            "bkg_eff": bkg_eff,
        })

    return pd.DataFrame(rows)


# ============================================================
# Score helpers
# ============================================================
def get_ratio_score(df, use_bb_bs=False, eps=1e-10):
    g_bc = df["ak8_gpt_bc"].to_numpy(dtype=np.float64)
    g_bb = df["ak8_gpt_bb"].to_numpy(dtype=np.float64)
    g_cc = df["ak8_gpt_cc"].to_numpy(dtype=np.float64)
    g_qcd = df["ak8_gpt_qcd"].to_numpy(dtype=np.float64)
    g_bs = df["ak8_gpt_bs"].to_numpy(dtype=np.float64)
    g_qq = df["ak8_gpt_qq"].to_numpy(dtype=np.float64)
    g_cs = df["ak8_gpt_cs"].to_numpy(dtype=np.float64)
    g_topbw = df["ak8_gpt_topbw"].to_numpy(dtype=np.float64)

    if use_bb_bs:
        denom = g_bc + g_bb + g_bs + g_qcd + g_cc + g_cs + g_qq + g_topbw + eps
    else:
        denom = g_bc + g_qcd + g_cc + g_cs + g_qq + g_topbw + eps

    return (g_bc / denom).astype(np.float64)

def compute_roc(y, score, weight):
    if len(y) == 0:
        return None
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return None

    fpr, tpr, thr = roc_curve(y, score, sample_weight=weight)
    auc = roc_auc_score(y, score, sample_weight=weight)
    return fpr, tpr, thr, auc

# ============================================================
# Plot helpers
# ============================================================
def add_cms_label(ax):
    hep.cms.label(
        # "Simulation",
        data=False,
        com=13,
        ax=ax
    )

def add_text_box(ax, text_lines):
    text = "\n".join(text_lines)
    ax.text(
        0.05, 0.98, text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9)
    )

def plot_comparison_roc(df, outpath, extra_lines=None, use_bb_bs=False):
    y = df["label"].to_numpy(dtype=np.int32)
    w = np.abs(df["weight"].to_numpy(dtype=np.float64))

    score_ratio = get_ratio_score(df, use_bb_bs=use_bb_bs, eps=EPS)
    score_bdt = df["bdt_score"].to_numpy(dtype=np.float64)

    roc_ratio = compute_roc(y, score_ratio, w)
    roc_bdt = compute_roc(y, score_bdt, w)

    if (roc_ratio is None) or (roc_bdt is None):
        print(f"[WARN] Skip ROC comparison, not enough signal/background: {outpath}")
        return

    fpr_r, tpr_r, _, auc_r = roc_ratio
    fpr_b, tpr_b, _, auc_b = roc_bdt

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.plot(
        tpr_r, fpr_r,
        color="black",
        linewidth=2,
        label=f"Ratio ROC (AUC={auc_r:.4f})"
    )
    ax.plot(
        tpr_b, fpr_b,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Boosted ROC (AUC={auc_b:.4f})"
    )

    cuts_to_mark = [0.7, 0.85, 0.95, 0.98, 0.99]
    add_working_points(
        ax,
        y,
        score_bdt,
        w,
        cuts_to_mark,
        color="red",
        label_prefix="BDT"
    )

    cuts_to_mark = [0.7, 0.85, 0.9]
    add_working_points(
        ax,
        y,
        score_ratio,
        w,
        cuts_to_mark,
        color="black",
        label_prefix="Ratio"
    )

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    add_cms_label(ax)
    ax.legend(loc="lower right", frameon=True)

    if extra_lines is not None and len(extra_lines) > 0:
        add_text_box(ax, extra_lines)

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {outpath}")

# ============================================================
# Main
# ============================================================
def main():
    df = pd.read_parquet(INPUT_FILE)

    # inclusive
    inclusive_lines = [
        "Signal: bc FatJets",
        "Background: non-bc FatJets",
        "Inclusive in mSD and pT",
        f"Ratio: bc / (bc + qcd + cc + cs + qq + topbw{' + bb + bs' if USE_BB_BS else ''})",
    ]
    plot_comparison_roc(
        df=df,
        outpath=os.path.join(FIG_DIR, "roc_compare_inclusive.png"),
        extra_lines=inclusive_lines,
        use_bb_bs=USE_BB_BS
    )

    # 9 categorized ROC
    for m_name, m_low, m_high in MASS_WINDOWS:
        for pt_name, pt_low, pt_high in PT_WINDOWS:
            mask = (
                (df["ak8_sdmass"] >= m_low) &
                (df["ak8_sdmass"] < m_high) &
                (df["ak8_pt"] >= pt_low) &
                (df["ak8_pt"] < pt_high)
            )

            sub_df = df[mask].copy()

            text_lines = [
                "Signal: bc FatJets",
                "Background: non-bc FatJets",
                f"{m_low:.0f} < mSD < {m_high:.0f} GeV",
                f"{pt_low:.0f} < pT < {'∞' if np.isinf(pt_high) else f'{pt_high:.0f}'} GeV",
                f"Ratio: bc / (bc + qcd + cc + cs + qq + topbw{' + bb + bs' if USE_BB_BS else ''})",
            ]

            outname = f"roc_compare_{m_name}_{pt_name}.png"
            plot_comparison_roc(
                df=sub_df,
                outpath=os.path.join(FIG_DIR, outname),
                extra_lines=text_lines,
                use_bb_bs=USE_BB_BS
            )

if __name__ == "__main__":
    main()
