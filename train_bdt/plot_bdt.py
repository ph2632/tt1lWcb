#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

from sklearn.metrics import roc_curve, roc_auc_score

# ============================================================
# Config
# ============================================================
INPUT_DIR = "./dbc_bdt_output"
FIG_DIR = os.path.join(INPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

hep.style.use("CMS")

# 你可以在这里改质量窗口定义
MASS_WINDOWS = [
    ("m50", 40.0, 60.0),
    ("m80", 65.0, 95.0),
    ("m125", 105.0, 145.0),
]

PT_WINDOWS = [
    ("pt200to300", 200.0, 300.0),
    ("pt300to450", 300.0, 450.0),
    ("pt450toInf", 450.0, np.inf),
]

# CMS label options
CMS_LABEL = "Simulation"
COM_ENERGY = "13 TeV"

# ============================================================
# Helpers
# ============================================================
def weighted_hist(ax, values, weights, bins, label, color, histtype="step", linestyle="-", linewidth=2):
    ax.hist(
        values,
        bins=bins,
        weights=weights,
        histtype=histtype,
        density=True,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth
    )

def add_cms_label(ax):
    hep.cms.label(
        # CMS_LABEL,
        data=False,
        com=13,
        ax=ax
    )

def add_text_below_legend(ax, text_lines):
    """
    在坐标轴内、图例下方放多行文字
    """
    text = "\n".join(text_lines)
    ax.text(
        0.05, 0.98, text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9)
    )

def compute_roc(df):
    if len(df) == 0:
        return None

    y = df["label"].to_numpy()
    s = df["bdt_score"].to_numpy()
    w = np.abs(df["weight"].to_numpy())

    # 需要同时存在 signal 和 background
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return None

    fpr, tpr, thr = roc_curve(y, s, sample_weight=w)
    auc = roc_auc_score(y, s, sample_weight=w)
    return fpr, tpr, thr, auc

def plot_single_roc(df, outpath, extra_lines=None, include_train=None, label_valid="Valid", label_train="Train"):
    fig, ax = plt.subplots(figsize=(10, 9))

    roc_valid = compute_roc(df)
    if roc_valid is None:
        print(f"[WARN] Skip ROC, not enough signal/background: {outpath}")
        plt.close(fig)
        return

    fpr_v, tpr_v, _, auc_v = roc_valid
    ax.plot(tpr_v, fpr_v, color="black", linewidth=2, label=f"{label_valid} ROC (AUC={auc_v:.4f})")

    if include_train is not None:
        roc_train = compute_roc(include_train)
        if roc_train is not None:
            fpr_t, tpr_t, _, auc_t = roc_train
            ax.plot(tpr_t, fpr_t, color="red", linestyle="--", linewidth=2, label=f"{label_train} ROC (AUC={auc_t:.4f})")

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    add_cms_label(ax)

    leg = ax.legend(loc="lower right", frameon=True)

    if extra_lines is not None and len(extra_lines) > 0:
        add_text_below_legend(ax, extra_lines)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {outpath}")

def plot_all_rocs(train_df, valid_df):
    # Inclusive ROC
    inclusive_text = [
        "Signal: bc FatJets",
        "Background: non-bc FatJets",
        "Inclusive in mSD and pT",
    ]
    plot_single_roc(
        df=valid_df,
        include_train=train_df,
        outpath=os.path.join(FIG_DIR, "roc_inclusive.png"),
        extra_lines=inclusive_text
    )

    # 9 categorized ROC
    for m_name, m_low, m_high in MASS_WINDOWS:
        for pt_name, pt_low, pt_high in PT_WINDOWS:
            valid_mask = (
                (valid_df["ak8_sdmass"] >= m_low) &
                (valid_df["ak8_sdmass"] < m_high) &
                (valid_df["ak8_pt"] >= pt_low) &
                (valid_df["ak8_pt"] < pt_high)
            )
            train_mask = (
                (train_df["ak8_sdmass"] >= m_low) &
                (train_df["ak8_sdmass"] < m_high) &
                (train_df["ak8_pt"] >= pt_low) &
                (train_df["ak8_pt"] < pt_high)
            )

            sub_valid = valid_df[valid_mask].copy()
            sub_train = train_df[train_mask].copy()

            text_lines = [
                "Signal: bc FatJets",
                "Background: non-bc FatJets",
                f"{m_low:.0f} < mSD < {m_high:.0f} GeV",
                f"{pt_low:.0f} < pT < {'∞' if np.isinf(pt_high) else f'{pt_high:.0f}'} GeV",
            ]

            outname = f"roc_{m_name}_{pt_name}.png"
            plot_single_roc(
                df=sub_valid,
                include_train=sub_train,
                outpath=os.path.join(FIG_DIR, outname),
                extra_lines=text_lines
            )

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
            fontsize=9,
            color=color
        )

    return wp_df


def plot_score_distribution(train_df, valid_df):
    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(10, 9))

    sig_train = train_df["label"] == 1
    bkg_train = train_df["label"] == 0
    sig_valid = valid_df["label"] == 1
    bkg_valid = valid_df["label"] == 0

    weighted_hist(
        ax,
        train_df.loc[sig_train, "bdt_score"],
        np.abs(train_df.loc[sig_train, "weight"]),
        bins,
        label="Train Signal",
        color="red",
        linestyle="-"
    )
    weighted_hist(
        ax,
        train_df.loc[bkg_train, "bdt_score"],
        np.abs(train_df.loc[bkg_train, "weight"]),
        bins,
        label="Train Background",
        color="blue",
        linestyle="-"
    )
    weighted_hist(
        ax,
        valid_df.loc[sig_valid, "bdt_score"],
        np.abs(valid_df.loc[sig_valid, "weight"]),
        bins,
        label="Valid Signal",
        color="red",
        linestyle="--"
    )
    weighted_hist(
        ax,
        valid_df.loc[bkg_valid, "bdt_score"],
        np.abs(valid_df.loc[bkg_valid, "weight"]),
        bins,
        label="Valid Background",
        color="blue",
        linestyle="--"
    )

    ax.set_xlabel("BDT score")
    ax.set_ylabel("Arbitrary units")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    add_cms_label(ax)
    ax.legend(loc="upper center", ncol=2, frameon=True)

    add_text_below_legend(ax, [
        "Signal: bc FatJets",
        "Background: non-bc FatJets",
        "Inclusive in mSD and pT",
    ])

    outpath = os.path.join(FIG_DIR, "bdt_score_distribution.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {outpath}")

def plot_loss_curve(history):
    train_logloss = history["validation_0"]["logloss"]
    valid_logloss = history["validation_1"]["logloss"]
    epochs = np.arange(1, len(train_logloss) + 1)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(epochs, train_logloss, color="red", linewidth=2, label="Train logloss")
    ax.plot(epochs, valid_logloss, color="black", linewidth=2, label="Valid logloss")

    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Logloss")
    ax.grid(True, linestyle="--", alpha=0.4)

    add_cms_label(ax)
    ax.legend(loc="upper right", frameon=True)

    add_text_below_legend(ax, [
        "Binary classification: bc vs non-bc FatJets",
        "Loss monitored during training",
    ])

    outpath = os.path.join(FIG_DIR, "loss_curve.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {outpath}")

def plot_auc_curve(history):
    train_auc = history["validation_0"]["auc"]
    valid_auc = history["validation_1"]["auc"]
    epochs = np.arange(1, len(train_auc) + 1)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(epochs, train_auc, color="red", linewidth=2, label="Train AUC")
    ax.plot(epochs, valid_auc, color="black", linewidth=2, label="Valid AUC")

    ax.set_xlabel("Boosting round")
    ax.set_ylabel("AUC")
    ax.grid(True, linestyle="--", alpha=0.4)

    add_cms_label(ax)
    ax.legend(loc="lower right", frameon=True)

    add_text_below_legend(ax, [
        "Binary classification: bc vs non-bc FatJets",
        "AUC monitored during training",
    ])

    outpath = os.path.join(FIG_DIR, "auc_curve.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {outpath}")

# ============================================================
# Main
# ============================================================
def main():
    train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train_prediction.parquet"))
    valid_df = pd.read_parquet(os.path.join(INPUT_DIR, "valid_prediction.parquet"))

    with open(os.path.join(INPUT_DIR, "training_history.json"), "r") as f:
        history = json.load(f)

    plot_all_rocs(train_df, valid_df)
    plot_score_distribution(train_df, valid_df)
    plot_loss_curve(history)
    plot_auc_curve(history)

if __name__ == "__main__":
    main()
