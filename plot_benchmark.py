#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import gc
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import awkward as ak
import psutil
import mplhep as hep
from train_bdt.dbc_tools import DbcEvaluator

dbc_eval = DbcEvaluator(
    mode="bdt",
    model_path="./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"
)

# ======================================================================================
# 0. Configuration
# ======================================================================================
BASE_PATH = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
MC_PATH = os.path.join(BASE_PATH, "MC/scored_samples_1merged_/")
PARQUET_PATH = "./score_parquet/"
FIGURE_PATH = "./figures/scan/"
os.makedirs(FIGURE_PATH, exist_ok=True)

TREE_NAME = "Events"
LUMI = 41.479   # fb^-1

CATALOG = {
    "Wcb": ["ttbar-powheg"],
    "tt": ["tt-semi", "tt-lep", "tt-had"],
    "QCD": ["QCD"],
    "SingleTop": ["single-top"],
    "Rare": ["ttbb", "ttW", "ttZ", "twZ", "ttHToTauTau", "ttHNonbb"],
    "Diboson": ["WW", "WZ", "ZZ"],
}

BRANCHES = [
    "xsecWeight", "genWeight", "lumiwgt", "puWeight",
    "trigEffWeight", "l1PreFiringWeight",
    "ak8_type", "ak8_n_c_in_jet",
    "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
    "score_cata_top_bc", "score_cata_top_bq", "score_cata_non",
    "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
    "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
]

DBC_WORKING_POINTS = {
    "Loose": 0.50,
    "Medium": 0.85,
    "Tight": 0.95,
    "TightII": 0.98,
    "TightIII": 0.985,
}

# validity requirement
MIN_BKG_YIELD = 0.0
MIN_BKG_RAW = 10
MIN_BKG_NEFF = 0.0

# ======================================================================================
# 1. Helpers
# ======================================================================================
def get_leading(arr, default=-999):
    try:
        return ak.fill_none(ak.firsts(arr, axis=1), default)
    except Exception as e:
        if "axis" in str(e).lower() or "depth" in str(e).lower():
            return ak.fill_none(arr, default)
        raise e

def get_group(filename):
    shortname = filename.replace("_tree.root", "")
    for cat, keywords in CATALOG.items():
        if any(k in shortname for k in keywords):
            return cat
    return "Other"

def asimov_z(S, B, eps=1e-6):
    B = np.maximum(B, eps)
    S = np.maximum(S, 0.0)
    return np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S))

def effective_n(weights):
    weights = np.asarray(weights, dtype=float)
    sw = np.sum(weights)
    sw2 = np.sum(weights**2)
    if sw2 <= 0:
        return 0.0
    return (sw**2) / sw2

def centers_to_edges(arr, domain=(0.0, 1.0)):
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 1:
        raise ValueError("arr must be 1D")
    if len(arr) < 2:
        raise ValueError("Need at least 2 points to infer edges")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("arr must be strictly increasing")

    mids = 0.5 * (arr[:-1] + arr[1:])
    left = arr[0] - 0.5 * (arr[1] - arr[0])
    right = arr[-1] + 0.5 * (arr[-1] - arr[-2])

    edges = np.concatenate([[left], mids, [right]])

    if domain is not None:
        edges[0] = max(edges[0], domain[0])
        edges[-1] = min(edges[-1], domain[1])

    return edges

def make_heatmap(ax, data, x_arr, y_arr, cmap, norm, xlabel, ylabel):
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    data = np.asarray(data)

    if data.shape != (len(y_arr), len(x_arr)):
        raise ValueError(f"data.shape={data.shape}, expected ({len(y_arr)}, {len(x_arr)})")

    x_edges = centers_to_edges(x_arr, domain=(0.0, 1.0))
    y_edges = centers_to_edges(y_arr, domain=(0.0, 1.0))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgrey")

    data_masked = np.ma.masked_invalid(data)

    im = ax.pcolormesh(
        x_edges, y_edges, data_masked,
        cmap=cmap_obj, norm=norm,
        shading="flat"
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)

    return im

# ======================================================================================
# 2. Load file
# ======================================================================================
def load_file(root_filepath):
    filename = os.path.basename(root_filepath)
    parquet_fp = os.path.join(PARQUET_PATH, filename.replace(".root", ".parquet"))

    if os.path.exists(parquet_fp):
        try:
            ev = ak.from_parquet(parquet_fp)
            return (
                np.array(ev["score"]),
                np.array(ev["dbc"]),
                np.array(ev["weights"])
            )
        except Exception:
            print(f"[WARN] Corrupt parquet {filename}, reloading from ROOT.")

    try:
        with uproot.open(root_filepath) as f:
            raw = f[TREE_NAME].arrays(BRANCHES, library="ak")
    except Exception as e:
        print(f"[WARN] Cannot read {root_filepath}: {e}")
        return None

    if len(raw) == 0:
        return None

    weights = np.array(
        raw["xsecWeight"] * raw["genWeight"] * raw["lumiwgt"]
        * raw["puWeight"] * raw["trigEffWeight"] * raw["l1PreFiringWeight"],
        dtype=np.float32
    )

    s_w_qq = np.array(get_leading(raw["score_cata_w_qq"], 0))
    s_qcd = np.array(get_leading(raw["score_cata_qcd"], 0))
    s_top_bqq = np.array(get_leading(raw["score_cata_top_bqq"], 0))
    s_top_bc = np.array(get_leading(raw["score_cata_top_bc"], 0))
    s_top_bq = np.array(get_leading(raw["score_cata_top_bq"], 0))
    s_non = np.array(get_leading(raw["score_cata_non"], 0))

    denom = s_w_qq + s_qcd + s_top_bqq + s_top_bc + s_top_bq + s_non + 1e-10
    score = s_w_qq / denom

    g_bc = np.array(get_leading(raw["ak8_gpt_bc"], 0))
    g_bb = np.array(get_leading(raw["ak8_gpt_bb"], 0))
    g_cc = np.array(get_leading(raw["ak8_gpt_cc"], 0))
    g_qcd = np.array(get_leading(raw["ak8_gpt_qcd"], 0))
    g_bs = np.array(get_leading(raw["ak8_gpt_bs"], 0))
    g_qq = np.array(get_leading(raw["ak8_gpt_qq"], 0))
    g_cs = np.array(get_leading(raw["ak8_gpt_cs"], 0))
    g_topbw = np.array(get_leading(raw["ak8_gpt_topbw"], 0))

    dbc = dbc_eval.get_Dbc(g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw)

    return score, dbc, weights

# ======================================================================================
# 3. Collect arrays
# ======================================================================================
def collect_arrays():
    mc_files = [f for f in os.listdir(MC_PATH) if f.endswith(".root")]
    print(f"[INFO] Found {len(mc_files)} MC files.")

    sig_score, sig_dbc, sig_w = [], [], []
    bkg_score, bkg_dbc, bkg_w = [], [], []

    for i, fname in enumerate(mc_files):
        group = get_group(fname)
        result = load_file(os.path.join(MC_PATH, fname))
        if result is None:
            continue

        score, dbc, weights = result

        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"\r[PROG] {i+1}/{len(mc_files)} | group={group:10s} | Mem={mem:.0f}MB", end="")

        if group == "Wcb":
            sig_score.append(score)
            sig_dbc.append(dbc)
            sig_w.append(weights)
        else:
            bkg_score.append(score)
            bkg_dbc.append(dbc)
            bkg_w.append(weights)

        if i % 10 == 0:
            gc.collect()

    print("\n[INFO] Concatenating arrays...")

    return (
        np.concatenate(sig_score), np.concatenate(sig_dbc), np.concatenate(sig_w),
        np.concatenate(bkg_score), np.concatenate(bkg_dbc), np.concatenate(bkg_w),
    )

# ======================================================================================
# 4. Scan
# ======================================================================================
def run_scan(sig_score, sig_dbc, sig_w,
             bkg_score, bkg_dbc, bkg_w,
             dbc_cuts, score_cuts,
             min_bkg_yield=0.0,
             min_bkg_raw=10,
             min_bkg_neff=0.0):

    Z = np.full((len(dbc_cuts), len(score_cuts)), np.nan)
    S = np.zeros((len(dbc_cuts), len(score_cuts)))
    B = np.zeros((len(dbc_cuts), len(score_cuts)))

    Nsig_raw = np.zeros((len(dbc_cuts), len(score_cuts)), dtype=int)
    Nbkg_raw = np.zeros((len(dbc_cuts), len(score_cuts)), dtype=int)
    Neff_bkg = np.zeros((len(dbc_cuts), len(score_cuts)), dtype=float)

    for i, dc in enumerate(dbc_cuts):
        sig_mask_dbc = sig_dbc > dc
        bkg_mask_dbc = bkg_dbc > dc

        ss = sig_score[sig_mask_dbc]
        sw = sig_w[sig_mask_dbc]

        bs = bkg_score[bkg_mask_dbc]
        bw = bkg_w[bkg_mask_dbc]

        for j, sc in enumerate(score_cuts):
            s_pass = ss > sc
            b_pass = bs > sc

            sw_pass = sw[s_pass]
            bw_pass = bw[b_pass]

            s_val = sw_pass.sum()
            b_val = bw_pass.sum()

            n_sig_raw = np.count_nonzero(s_pass)
            n_bkg_raw = np.count_nonzero(b_pass)
            n_eff_bkg = effective_n(bw_pass) if n_bkg_raw > 0 else 0.0

            S[i, j] = s_val
            B[i, j] = b_val
            Nsig_raw[i, j] = n_sig_raw
            Nbkg_raw[i, j] = n_bkg_raw
            Neff_bkg[i, j] = n_eff_bkg

            is_valid = True
            if b_val < min_bkg_yield:
                is_valid = False
            if n_bkg_raw < min_bkg_raw:
                is_valid = False
            if n_eff_bkg < min_bkg_neff:
                is_valid = False

            if is_valid:
                Z[i, j] = asimov_z(s_val, b_val)
            else:
                Z[i, j] = np.nan

    return Z, S, B, Nsig_raw, Nbkg_raw, Neff_bkg

# ======================================================================================
# 5. Plot scan
# ======================================================================================
def plot_scan(Z, S, B, dbc_cuts, score_cuts):
    valid_mask = np.isfinite(Z)
    n_valid = np.count_nonzero(valid_mask)
    print(f"[INFO] Valid scan points: {n_valid} / {Z.size}")

    if n_valid == 0:
        raise RuntimeError("No valid scan points left after background-statistics requirements.")

    best_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    best_dbc = dbc_cuts[best_idx[0]]
    best_score = score_cuts[best_idx[1]]
    best_Z = Z[best_idx]
    best_S = S[best_idx]
    best_B = B[best_idx]

    print(f"\n{'='*60}")
    print(f"  Best Dbc   cut : {best_dbc:.4f}")
    print(f"  Best Score cut : {best_score:.4f}")
    print(f"  Best Z         : {best_Z:.4f}")
    print(f"  S = {best_S:.4f}, B = {best_B:.4f}")
    print(f"{'='*60}")

    hep.style.use("CMS")
    fig, axes = plt.subplots(3, 1, figsize=(9, 20))

    LABEL_FS = 14
    TICK_FS = 12
    LEGEND_FS = 11

    ax = axes[0]
    im = make_heatmap(
        ax, Z,
        x_arr=score_cuts, y_arr=dbc_cuts,
        cmap="viridis", norm=None,
        xlabel="EventClassifier cut",
        ylabel="$D_{bc}$ cut"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Asimov Z", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    ax.scatter(
        best_score, best_dbc,
        color="red", marker="*", s=400, zorder=5,
        label=f"Best: Z={best_Z:.3f}\n$D_{{bc}}$>{best_dbc:.3f}, Sc>{best_score:.3f}"
    )

    ax.set_xlabel("EventClassifier cut", fontsize=LABEL_FS)
    ax.set_ylabel("$D_{bc}$ cut", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="lower right", framealpha=0.85, edgecolor="grey")
    hep.cms.label("Preliminary", data=False, lumi=LUMI, ax=ax, fontsize=14, loc=0)

    ax = axes[1]
    ax.plot(score_cuts, Z[best_idx[0], :], color="steelblue", lw=2)
    ax.axvline(
        best_score, color="red", ls="--", lw=1.5,
        label=f"Best Score={best_score:.3f}\nZ={best_Z:.3f}"
    )
    ax.set_xlabel("EventClassifier cut", fontsize=LABEL_FS)
    ax.set_ylabel("Asimov Z", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FS, loc="upper left", framealpha=0.85, edgecolor="grey")
    hep.cms.label("Preliminary", data=False, lumi=LUMI, ax=ax, fontsize=14, loc=0)
    ax.set_title(
        f"$D_{{bc}}$ fixed = {best_dbc:.3f}",
        fontsize=13, pad=4, loc="center", style="italic", color="grey"
    )

    ax = axes[2]
    ax.plot(dbc_cuts, Z[:, best_idx[1]], color="darkorange", lw=2)
    ax.axvline(
        best_dbc, color="red", ls="--", lw=1.5,
        label=f"Best $D_{{bc}}$={best_dbc:.3f}\nZ={best_Z:.3f}"
    )
    ax.set_xlabel("$D_{bc}$ cut", fontsize=LABEL_FS)
    ax.set_ylabel("Asimov Z", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FS, loc="upper left", framealpha=0.85, edgecolor="grey")
    hep.cms.label("Preliminary", data=False, lumi=LUMI, ax=ax, fontsize=14, loc=0)
    ax.set_title(
        f"Score fixed = {best_score:.3f}",
        fontsize=13, pad=4, loc="right", style="italic", color="grey"
    )

    fig.subplots_adjust(hspace=0.38, top=0.97, bottom=0.06, left=0.10, right=0.97)

    out = os.path.join(FIGURE_PATH, "significance_2d_scan.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[PLOT] Saved {out}")
    plt.close(fig)

    return best_dbc, best_score, best_Z

# ======================================================================================
# 6. Plot ROC
# ======================================================================================
def plot_roc(sig_score, sig_dbc, sig_w,
             bkg_score, bkg_dbc, bkg_w,
             wp_dict=DBC_WORKING_POINTS):

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 7))

    LINESTYLES = {
        "Loose": (0, (5, 1)),
        "Medium": (0, (3, 1, 1, 1)),
        "Tight": "-",
        "TightII": (0, (1, 1)),
        "TightIII": (0, (1, 1)),
    }
    COLORS = {
        "Loose": "steelblue",
        "Medium": "darkorange",
        "Tight": "crimson",
        "TightII": "purple",
        "TightIII": "green",
    }

    def make_weighted_roc(labels, scores, weights, n_bins=500):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.digitize(scores, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        sig_w_bin = np.zeros(n_bins)
        bkg_w_bin = np.zeros(n_bins)

        for i in range(len(labels)):
            b = bin_idx[i]
            if labels[i] == 1:
                sig_w_bin[b] += weights[i]
            else:
                bkg_w_bin[b] += weights[i]

        sig_w_bin = np.where(sig_w_bin < 0, 0.0, sig_w_bin)
        bkg_w_bin = np.where(bkg_w_bin < 0, 0.0, bkg_w_bin)

        sig_cum = np.cumsum(sig_w_bin[::-1])[::-1]
        bkg_cum = np.cumsum(bkg_w_bin[::-1])[::-1]

        sig_total = sig_w_bin.sum()
        bkg_total = bkg_w_bin.sum()

        if sig_total <= 0 or bkg_total <= 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan

        tpr = sig_cum / sig_total
        fpr = bkg_cum / bkg_total

        tpr = np.concatenate([[0.0], tpr, [1.0]])
        fpr = np.concatenate([[0.0], fpr, [1.0]])

        roc_auc = np.trapz(fpr, tpr)
        return tpr, fpr, roc_auc

    for wp_name, dbc_cut in wp_dict.items():
        s_mask = sig_dbc > dbc_cut
        b_mask = bkg_dbc > dbc_cut

        scores = np.concatenate([sig_score[s_mask], bkg_score[b_mask]])
        labels = np.concatenate([np.ones(s_mask.sum()), np.zeros(b_mask.sum())])
        weights = np.concatenate([sig_w[s_mask], bkg_w[b_mask]])

        tpr, fpr, roc_auc = make_weighted_roc(labels, scores, weights)

        ax.plot(
            tpr, fpr,
            linestyle=LINESTYLES[wp_name],
            color=COLORS[wp_name],
            lw=2,
            label=f"{wp_name} ($D_{{bc}}>{dbc_cut:.3f}$) AUC={roc_auc:.3f}" if np.isfinite(roc_auc)
                  else f"{wp_name} ($D_{{bc}}>{dbc_cut:.3f}$) AUC=nan"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

    ax.set_yscale("log")
    ax.set_xlabel("Signal Efficiency", fontsize=14)
    ax.set_ylabel("Background Efficiency", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-4, 1.0)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.25)
    hep.cms.label("Preliminary", data=False, lumi=LUMI, ax=ax, fontsize=14, loc=0)

    out = os.path.join(FIGURE_PATH, "roc_dbc_wp.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"[PLOT] Saved {out}")
    plt.close(fig)

# ======================================================================================
# 7. Plot S/B maps
# ======================================================================================
def plot_sb_2d(S, B, dbc_cuts, score_cuts, valid_mask=None):
    hep.style.use("CMS")

    if valid_mask is None:
        valid_mask = np.ones_like(S, dtype=bool)

    S_plot = np.where(valid_mask, S, np.nan)
    B_plot = np.where(valid_mask, B, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        SoSqrtB = np.where((B_plot > 0) & valid_mask, S_plot / np.sqrt(B_plot), np.nan)

    panels = [
        (S_plot, "Signal Yield ($S$)", "Blues"),
        (B_plot, "Background Yield ($B$)", "Reds"),
        (SoSqrtB, "$S / \\sqrt{B}$", "viridis"),
    ]

    LABEL_FS = 14
    TICK_FS = 12

    fig, axes = plt.subplots(3, 1, figsize=(9, 20))

    for ax, (data, cblabel, cmap) in zip(axes, panels):
        positive = data[np.isfinite(data) & (data > 0)]
        if len(positive) == 0:
            vmin, vmax = 1e-6, 1.0
        else:
            vmin = positive.min()
            vmax = positive.max()
            if vmax <= vmin:
                vmax = vmin * 1.01

        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        im = make_heatmap(
            ax, data,
            x_arr=score_cuts, y_arr=dbc_cuts,
            cmap=cmap, norm=norm,
            xlabel="EventClassifier cut",
            ylabel="$D_{bc}$ cut"
        )

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cblabel, fontsize=LABEL_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)

        ax.set_xlabel("EventClassifier cut", fontsize=LABEL_FS)
        ax.set_ylabel("$D_{bc}$ cut", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        hep.cms.label("Preliminary", data=False, lumi=LUMI, ax=ax, fontsize=14, loc=0)

    fig.subplots_adjust(hspace=0.38, top=0.97, bottom=0.06, left=0.10, right=0.97)

    out = os.path.join(FIGURE_PATH, "sb_2d_scan.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[PLOT] Saved {out}")
    plt.close(fig)

# ======================================================================================
# 8. Main
# ======================================================================================
if __name__ == "__main__":
    sig_score, sig_dbc, sig_w, bkg_score, bkg_dbc, bkg_w = collect_arrays()
    print(f"[INFO] Signal events: {len(sig_score)}, Background events: {len(bkg_score)}")

    dbc_cuts = np.concatenate([
        np.linspace(0.000, 0.900, 30, endpoint=False),
        np.linspace(0.900, 0.990, 30, endpoint=False),
        np.linspace(0.990, 0.999, 30)
    ])
    score_cuts = np.linspace(0.00, 0.99, 60)

    Z, S, B, Nsig_raw, Nbkg_raw, Neff_bkg = run_scan(
        sig_score, sig_dbc, sig_w,
        bkg_score, bkg_dbc, bkg_w,
        dbc_cuts, score_cuts,
        min_bkg_yield=MIN_BKG_YIELD,
        min_bkg_raw=MIN_BKG_RAW,
        min_bkg_neff=MIN_BKG_NEFF,
    )

    best_dbc, best_score, best_Z = plot_scan(Z, S, B, dbc_cuts, score_cuts)

    valid_mask = np.isfinite(Z)
    plot_sb_2d(S, B, dbc_cuts, score_cuts, valid_mask=valid_mask)

    plot_roc(sig_score, sig_dbc, sig_w, bkg_score, bkg_dbc, bkg_w)

    print("\n[SUMMARY]")
    print(f"  Valid points requirement: B >= {MIN_BKG_YIELD}, N_bkg_raw >= {MIN_BKG_RAW}, N_eff_bkg >= {MIN_BKG_NEFF}")
    print(f"  Best Dbc cut   : {best_dbc:.4f}")
    print(f"  Best Score cut : {best_score:.4f}")
    print(f"  Best Z         : {best_Z:.4f}")
