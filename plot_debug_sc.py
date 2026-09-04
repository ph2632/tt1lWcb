#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import gc
import time
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from train_bdt.dbc_tools import DbcEvaluator

hep.style.use("CMS")

# =========================================================
# 1. Basic config
# =========================================================

dbc_eval = DbcEvaluator(
    mode="bdt",
    model_path="./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"
)

BASE_PATH = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
MC_PATH = os.path.join(BASE_PATH, "MC/scored_samples_1merged_/")
OUTDIR = "./debug_figures/"
TREE_NAME = "Events"
LUMI = 41.479

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

BRANCHES = [
    "xsecWeight", "genWeight", "lumiwgt", "puWeight",
    "trigEffWeight", "l1PreFiringWeight",
    "ak8_sdmass", "ak8_pt", "ak8_eta",
    "ak8_type", "ak8_n_c_in_jet",

    "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
    "score_cata_top_bc", "score_cata_top_bq", "score_cata_non",

    "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
    "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
]

SIGNAL_KEYWORDS = ["ttbar-powheg"]


# =========================================================
# 2. Helpers
# =========================================================

def get_leading(arr, default=-999):
    try:
        return ak.fill_none(ak.firsts(arr, axis=1), default)
    except Exception:
        return ak.fill_none(arr, default)

def is_signal_file(fname):
    low = fname.lower()
    return any(k in low for k in SIGNAL_KEYWORDS)

def read_one_file(path):
    with uproot.open(path) as f:
        tree = f[TREE_NAME]
        raw = tree.arrays(BRANCHES, library="ak")

    # basic selection
    # 先取 leading，变成 event-level 标量
        lead_sdmass = get_leading(raw["ak8_sdmass"], default=-999)
        # lead_pt     = get_leading(raw["ak8_pt"], default=-999)
        # lead_eta    = get_leading(raw["ak8_eta"], default=-999)
        # lead_type   = get_leading(raw["ak8_type"], default=-999)
        # lead_nc     = get_leading(raw["ak8_n_c_in_jet"], default=-999)

        # 再基于 leading mass 做 event mask
        event_mask = (lead_sdmass > 60) & (lead_sdmass < 120)
        raw = raw[event_mask]
    if len(raw) == 0:
        return None

    ak8_sdmass = np.array(get_leading(raw["ak8_sdmass"]), dtype=np.float32)
    ak8_pt     = np.array(get_leading(raw["ak8_pt"]), dtype=np.float32)
    ak8_eta    = np.array(get_leading(raw["ak8_eta"]), dtype=np.float32)
    ak8_type   = np.array(get_leading(raw["ak8_type"]), dtype=np.int32)
    n_c        = np.array(get_leading(raw["ak8_n_c_in_jet"]), dtype=np.int32)

    weights = (
        raw["xsecWeight"] *
        raw["genWeight"] *
        raw["lumiwgt"] *
        raw["puWeight"] *
        raw["trigEffWeight"] *
        raw["l1PreFiringWeight"]
    )
    weights = np.array(weights, dtype=np.float32)

    # score
    s_w_qq    = np.array(get_leading(raw["score_cata_w_qq"], 0), dtype=np.float32)
    s_qcd     = np.array(get_leading(raw["score_cata_qcd"], 0), dtype=np.float32)
    s_top_bqq = np.array(get_leading(raw["score_cata_top_bqq"], 0), dtype=np.float32)
    s_top_bc  = np.array(get_leading(raw["score_cata_top_bc"], 0), dtype=np.float32)
    s_top_bq  = np.array(get_leading(raw["score_cata_top_bq"], 0), dtype=np.float32)
    s_non     = np.array(get_leading(raw["score_cata_non"], 0), dtype=np.float32)

    denom = s_w_qq + s_qcd + s_top_bqq + s_top_bc + s_top_bq + s_non + 1e-10
    score = s_w_qq / denom

    # dbc inputs
    g_bc    = np.array(get_leading(raw["ak8_gpt_bc"], 0), dtype=np.float32)
    g_bb    = np.array(get_leading(raw["ak8_gpt_bb"], 0), dtype=np.float32)
    g_cc    = np.array(get_leading(raw["ak8_gpt_cc"], 0), dtype=np.float32)
    g_qcd   = np.array(get_leading(raw["ak8_gpt_qcd"], 0), dtype=np.float32)
    g_bs    = np.array(get_leading(raw["ak8_gpt_bs"], 0), dtype=np.float32)
    g_qq    = np.array(get_leading(raw["ak8_gpt_qq"], 0), dtype=np.float32)
    g_cs    = np.array(get_leading(raw["ak8_gpt_cs"], 0), dtype=np.float32)
    g_topbw = np.array(get_leading(raw["ak8_gpt_topbw"], 0), dtype=np.float32)

    dbc = dbc_eval.get_Dbc(g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw)

    out = {
        "mass": ak8_sdmass,
        "pt": ak8_pt,
        "eta": ak8_eta,
        "ak8_type": ak8_type,
        "n_c": n_c,
        "weight": weights,
        "score": score,
        "dbc": dbc,
    }

    del raw
    gc.collect()
    return out

def concat_dicts(dict_list):
    if len(dict_list) == 0:
        return None
    keys = dict_list[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([d[k] for d in dict_list if d is not None and len(d[k]) > 0])
    return out

def weighted_hist(ax, values, weights, bins, label, color=None, histtype="step", lw=2):
    ax.hist(values, bins=bins, weights=weights, histtype=histtype, linewidth=lw, label=label, color=color)

def savefig(fig, name):
    hep.cms.label("Preliminary", data=False, lumi=LUMI, ax=fig.axes[0])
    fig.tight_layout()
    out = os.path.join(OUTDIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVE] {out}")


# =========================================================
# 3. Load signal-only sample
# =========================================================

def load_signal():
    files = [f for f in os.listdir(MC_PATH) if f.endswith(".root")]
    sig_files = [f for f in files if is_signal_file(f)]

    print(f"[INFO] Found {len(sig_files)} signal-like files")
    chunks = []

    for i, fname in enumerate(sig_files, 1):
        full = os.path.join(MC_PATH, fname)
        print(f"[INFO] ({i}/{len(sig_files)}) reading {fname}")
        arr = read_one_file(full)
        if arr is not None:
            arr["source_file"] = np.array([fname] * len(arr["score"]))
            chunks.append(arr)

    if len(chunks) == 0:
        raise RuntimeError("No signal events loaded.")

    # concat numeric keys
    numeric_keys = ["mass", "pt", "eta", "ak8_type", "n_c", "weight", "score", "dbc"]
    out = {}
    for k in numeric_keys:
        out[k] = np.concatenate([c[k] for c in chunks])
    out["source_file"] = np.concatenate([c["source_file"] for c in chunks])

    print(f"[INFO] Total signal events loaded: {len(out['score'])}")
    return out


# =========================================================
# 4. Make masks
# =========================================================

def make_score_masks(score):
    return {
        "low":  score < 0.10,
        "mid":  (score >= 0.40) & (score < 0.60),
        "high": score >= 0.90,
    }


# =========================================================
# 5. Plots
# =========================================================

def plot_score_distribution(sig):
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)
    weighted_hist(ax, sig["score"], sig["weight"], bins, label="Wcb signal", color="red")
    ax.set_xlabel("Score")
    ax.set_ylabel("Events / 0.02")
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.legend()
    savefig(fig, "01_signal_score.pdf")

def plot_dbc_distribution(sig):
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)
    weighted_hist(ax, sig["dbc"], sig["weight"], bins, label="Wcb signal", color="blue")
    ax.set_xlabel("Dbc")
    ax.set_ylabel("Events / 0.02")
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.legend()
    savefig(fig, "02_signal_dbc.pdf")

def plot_score_vs_dbc_2d(sig):
    fig, ax = plt.subplots(figsize=(8, 7))
    h = ax.hist2d(
        sig["score"], sig["dbc"],
        bins=[np.linspace(0, 1, 51), np.linspace(0, 1, 51)],
        weights=sig["weight"],
        cmap="viridis"
    )
    plt.colorbar(h[3], ax=ax, label="Weighted events")
    ax.set_xlabel("Score")
    ax.set_ylabel("Dbc")
    savefig(fig, "03_signal_score_vs_dbc_2d.pdf")

def plot_compare_regions(sig, var, bins, xlabel, outname, logy=False):
    masks = make_score_masks(sig["score"])
    colors = {"low": "red", "mid": "orange", "high": "green"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for region in ["low", "mid", "high"]:
        m = masks[region]
        if np.sum(m) == 0:
            continue
        weighted_hist(
            ax,
            sig[var][m],
            sig["weight"][m],
            bins=bins,
            label=f"{region} score",
            color=colors[region],
            histtype="step",
            lw=2
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Weighted events")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(0.1, None)
    ax.legend()
    savefig(fig, outname)

def plot_ak8_type_breakdown(sig):
    masks = make_score_masks(sig["score"])
    regions = ["low", "mid", "high"]

    type_values = sorted(np.unique(sig["ak8_type"]))
    x = np.arange(len(type_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, region in enumerate(regions):
        m = masks[region]
        y = []
        for t in type_values:
            mt = m & (sig["ak8_type"] == t)
            y.append(np.sum(sig["weight"][mt]))
        ax.bar(x + (i - 1) * width, y, width=width, label=region)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in type_values])
    ax.set_xlabel("ak8_type")
    ax.set_ylabel("Weighted events")
    ax.legend()
    savefig(fig, "04_signal_ak8_type_by_score_region.pdf")

def plot_nc_breakdown(sig):
    masks = make_score_masks(sig["score"])
    regions = ["low", "mid", "high"]

    nc_values = sorted(np.unique(sig["n_c"]))
    x = np.arange(len(nc_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, region in enumerate(regions):
        m = masks[region]
        y = []
        for n in nc_values:
            mn = m & (sig["n_c"] == n)
            y.append(np.sum(sig["weight"][mn]))
        ax.bar(x + (i - 1) * width, y, width=width, label=region)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nc_values])
    ax.set_xlabel("ak8_n_c_in_jet")
    ax.set_ylabel("Weighted events")
    ax.legend()
    savefig(fig, "05_signal_nc_by_score_region.pdf")

def plot_score_split_by_nc(sig):
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)

    for n, color in zip(sorted(np.unique(sig["n_c"])), ["black", "red", "blue", "green", "orange", "purple"]):
        m = (sig["n_c"] == n)
        if np.sum(m) == 0:
            continue
        weighted_hist(
            ax,
            sig["score"][m],
            sig["weight"][m],
            bins,
            label=f"n_c = {n}",
            color=color,
            histtype="step",
            lw=2
        )

    ax.set_xlabel("Score")
    ax.set_ylabel("Events / 0.02")
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.legend()
    savefig(fig, "06_signal_score_split_by_nc.pdf")

def plot_score_split_by_type(sig):
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)

    colors = ["black", "red", "blue", "green", "orange", "purple", "brown", "cyan"]
    for t, color in zip(sorted(np.unique(sig["ak8_type"])), colors):
        m = (sig["ak8_type"] == t)
        if np.sum(m) == 0:
            continue
        weighted_hist(
            ax,
            sig["score"][m],
            sig["weight"][m],
            bins,
            label=f"type = {t}",
            color=color,
            histtype="step",
            lw=2
        )

    ax.set_xlabel("Score")
    ax.set_ylabel("Events / 0.02")
    ax.set_yscale("log")
    ax.set_ylim(0.1, None)
    ax.legend()
    savefig(fig, "07_signal_score_split_by_ak8type.pdf")

def print_region_summary(sig):
    masks = make_score_masks(sig["score"])
    total_w = np.sum(sig["weight"])

    print("\n========== Signal region summary ==========")
    for name, m in masks.items():
        w = np.sum(sig["weight"][m])
        frac = w / total_w if total_w > 0 else 0
        print(f"{name:>5s}: weighted = {w:.3f}, fraction = {frac:.4f}")

        if np.sum(m) > 0:
            print(f"       mean pt   = {np.average(sig['pt'][m], weights=sig['weight'][m]):.3f}")
            print(f"       mean mass = {np.average(sig['mass'][m], weights=sig['weight'][m]):.3f}")
            print(f"       mean eta  = {np.average(sig['eta'][m], weights=sig['weight'][m]):.3f}")
            print(f"       mean dbc  = {np.average(sig['dbc'][m], weights=sig['weight'][m]):.3f}")

def print_file_contribution_in_low_region(sig, topn=10):
    low = sig["score"] < 0.10
    files = sig["source_file"][low]
    weights = sig["weight"][low]

    uniq = np.unique(files)
    stats = []
    for f in uniq:
        m = (files == f)
        stats.append((f, np.sum(weights[m])))

    stats = sorted(stats, key=lambda x: x[1], reverse=True)

    print("\n========== Top source files in low-score signal ==========")
    for f, w in stats[:topn]:
        print(f"{f:60s}  {w:.3f}")


# =========================================================
# 6. Main
# =========================================================

if __name__ == "__main__":
    t0 = time.time()

    sig = load_signal()

    print_region_summary(sig)
    print_file_contribution_in_low_region(sig, topn=20)

    plot_score_distribution(sig)
    plot_dbc_distribution(sig)
    plot_score_vs_dbc_2d(sig)

    plot_compare_regions(sig, "pt",   np.linspace(200, 1000, 41), "AK8 $p_T$ [GeV]", "08_compare_pt_low_mid_high.pdf", logy=False)
    plot_compare_regions(sig, "mass", np.linspace(60, 120, 31),   "AK8 soft drop mass [GeV]", "09_compare_mass_low_mid_high.pdf", logy=False)
    plot_compare_regions(sig, "eta",  np.linspace(-2.5, 2.5, 31), r"AK8 $\eta$", "10_compare_eta_low_mid_high.pdf", logy=False)
    plot_compare_regions(sig, "dbc",  np.linspace(0, 1, 51),      "Dbc", "11_compare_dbc_low_mid_high.pdf", logy=True)

    plot_ak8_type_breakdown(sig)
    plot_nc_breakdown(sig)
    plot_score_split_by_nc(sig)
    plot_score_split_by_type(sig)

    print(f"\n[DONE] Total time: {time.time() - t0:.2f} s")
