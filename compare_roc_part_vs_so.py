#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import gc
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import roc_curve, auc

# ======================================================================================
# 1. Configuration
# ======================================================================================

class Config:
    def __init__(self):
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_/")

        self.tree_name = "Events"
        self.figure_path = "./figures/roc_compare/"
        self.sophon_roc_dir = "./extern/roc_outputs/"
        self.use_weights = False  # ROC 一般先不加权；如需加权可改 True

        self.mass_windows = [
            ("m50", 40.0, 60.0),
            ("m80", 65.0, 95.0),
            ("m125", 105.0, 145.0),
        ]

        self.pt_windows = [
            ("pt200to300", 200.0, 300.0),
            ("pt300to450", 300.0, 450.0),
            ("pt450toInf", 450.0, np.inf),
        ]

        self.branches = [
            "xsecWeight", "genWeight", "lumiwgt", "puWeight",
            "trigEffWeight", "l1PreFiringWeight",
            "ak8_sdmass", "ak8_pt",
            "ak8_type", "ak8_n_c_in_jet",
            "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
            "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw",
        ]

        # 如果某些 ntuple 还是旧名字，可在这里兼容
        self.alt_branch_map = {
            "ak8_gpt_topbw": "ak8_gpt_topw",
        }

# ======================================================================================
# 2. Helper
# ======================================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def safe_get(arrs, key, alt_map=None):
    fields = set(ak.fields(arrs))
    if key in fields:
        return ak.fill_none(arrs[key], 0)
    if alt_map is not None and key in alt_map and alt_map[key] in fields:
        return ak.fill_none(arrs[alt_map[key]], 0)
    raise KeyError(f"Missing branch: {key}")

def flatten_event_weights_to_jets(weights, jagged_ref):
    counts = ak.to_numpy(ak.num(jagged_ref))
    weights_np = np.asarray(weights, dtype=np.float64)
    return np.repeat(weights_np, counts)

def compute_dbc(arrs, alt_map=None):
    g_bc    = safe_get(arrs, "ak8_gpt_bc", alt_map)
    g_bb    = safe_get(arrs, "ak8_gpt_bb", alt_map)
    g_cc    = safe_get(arrs, "ak8_gpt_cc", alt_map)
    g_qcd   = safe_get(arrs, "ak8_gpt_qcd", alt_map)
    g_bs    = safe_get(arrs, "ak8_gpt_bs", alt_map)
    g_qq    = safe_get(arrs, "ak8_gpt_qq", alt_map)
    g_cs    = safe_get(arrs, "ak8_gpt_cs", alt_map)
    g_topbw = safe_get(arrs, "ak8_gpt_topbw", alt_map)

    denom = g_bc + g_bb + g_cc + g_qcd + g_bs + g_qq + g_cs + g_topbw + 1e-10
    dbc = g_bc / denom
    return dbc

def load_mc_file(filepath, cfg):
    with uproot.open(filepath) as f:
        tree = f[cfg.tree_name]

        requested = []
        available = set(tree.keys())

        for b in cfg.branches:
            if b in available:
                requested.append(b)
            elif b in cfg.alt_branch_map and cfg.alt_branch_map[b] in available:
                requested.append(cfg.alt_branch_map[b])
            else:
                print(f"[WARN] Missing branch {b} in {os.path.basename(filepath)}")

        arrs = tree.arrays(requested, library="ak")

    # rename alt branch -> canonical branch name
    for canon, alt in cfg.alt_branch_map.items():
        if canon not in ak.fields(arrs) and alt in ak.fields(arrs):
            arrs[canon] = arrs[alt]
            del arrs[alt]

    return arrs

def build_scores_labels_from_file(arrs, cfg):
    ak8_sdmass = safe_get(arrs, "ak8_sdmass", cfg.alt_branch_map)
    ak8_pt     = safe_get(arrs, "ak8_pt", cfg.alt_branch_map)
    ak8_type   = safe_get(arrs, "ak8_type", cfg.alt_branch_map)
    n_c        = safe_get(arrs, "ak8_n_c_in_jet", cfg.alt_branch_map)
    dbc        = compute_dbc(arrs, cfg.alt_branch_map)

    # event weights
    if cfg.use_weights:
        w_evt = (
            safe_get(arrs, "xsecWeight", cfg.alt_branch_map)
            * safe_get(arrs, "genWeight", cfg.alt_branch_map)
            * safe_get(arrs, "lumiwgt", cfg.alt_branch_map)
            * safe_get(arrs, "puWeight", cfg.alt_branch_map)
            * safe_get(arrs, "trigEffWeight", cfg.alt_branch_map)
            * safe_get(arrs, "l1PreFiringWeight", cfg.alt_branch_map)
        )
    else:
        w_evt = np.ones(len(arrs), dtype=np.float64)

    # flatten
    weights_flat = flatten_event_weights_to_jets(w_evt, ak8_sdmass)
    sdmass_flat  = ak.to_numpy(ak.flatten(ak8_sdmass))
    pt_flat      = ak.to_numpy(ak.flatten(ak8_pt))
    dbc_flat     = ak.to_numpy(ak.flatten(dbc))
    type_flat    = ak.to_numpy(ak.flatten(ak8_type))
    nc_flat      = ak.to_numpy(ak.flatten(n_c))

    # signal definition
    sig_mask = (type_flat == 2) & (nc_flat == 1)
    bkg_mask = ~sig_mask

    return {
        "sdmass": sdmass_flat,
        "pt": pt_flat,
        "dbc": dbc_flat,
        "sig_mask": sig_mask,
        "bkg_mask": bkg_mask,
        "weights": weights_flat,
    }

def select_window(data_dict, mass_low, mass_high, pt_low, pt_high):
    mass_mask = (data_dict["sdmass"] >= mass_low) & (data_dict["sdmass"] < mass_high)
    if np.isinf(pt_high):
        pt_mask = (data_dict["pt"] >= pt_low)
    else:
        pt_mask = (data_dict["pt"] >= pt_low) & (data_dict["pt"] < pt_high)

    win_mask = mass_mask & pt_mask

    sig = win_mask & data_dict["sig_mask"]
    bkg = win_mask & data_dict["bkg_mask"]

    score = np.concatenate([
        data_dict["dbc"][sig],
        data_dict["dbc"][bkg],
    ])

    label = np.concatenate([
        np.ones(np.sum(sig), dtype=int),
        np.zeros(np.sum(bkg), dtype=int),
    ])

    if len(score) == 0:
        weight = np.array([])
    else:
        weight = np.concatenate([
            data_dict["weights"][sig],
            data_dict["weights"][bkg],
        ])

    return score, label, weight, int(np.sum(sig)), int(np.sum(bkg))

def read_sophon_roc_csv(csv_path):
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return {
            "fpr": np.array([arr["fpr"]]),
            "tpr": np.array([arr["tpr"]]),
            "threshold": np.array([arr["threshold"]]),
        }
    return {
        "fpr": np.asarray(arr["fpr"], dtype=float),
        "tpr": np.asarray(arr["tpr"], dtype=float),
        "threshold": np.asarray(arr["threshold"], dtype=float),
    }

# ======================================================================================
# 3. Main
# ======================================================================================

def main():
    cfg = Config()
    ensure_dir(cfg.figure_path)
    hep.style.use("CMS")

    mc_files = [
        os.path.join(cfg.mc_path, f)
        for f in os.listdir(cfg.mc_path)
        if f.endswith(".root")
    ]
    mc_files = sorted(mc_files)

    print(f"[INFO] Found {len(mc_files)} MC files.")

    # collect all flattened jet-level info from all files
    all_sdmass = []
    all_pt = []
    all_dbc = []
    all_sig_mask = []
    all_bkg_mask = []
    all_weights = []

    for i, filepath in enumerate(mc_files):
        try:
            arrs = load_mc_file(filepath, cfg)
            out = build_scores_labels_from_file(arrs, cfg)

            all_sdmass.append(out["sdmass"])
            all_pt.append(out["pt"])
            all_dbc.append(out["dbc"])
            all_sig_mask.append(out["sig_mask"])
            all_bkg_mask.append(out["bkg_mask"])
            all_weights.append(out["weights"])

            if (i + 1) % 1 == 0:
                print(f"\r[PROG] {i+1}/{len(mc_files)} : {os.path.basename(filepath)}", end="")
        except Exception as e:
            print(f"\n[WARN] failed on {filepath}: {e}")

        gc.collect()

    print("")

    if len(all_dbc) == 0:
        raise RuntimeError("No valid MC data loaded.")

    merged = {
        "sdmass": np.concatenate(all_sdmass),
        "pt": np.concatenate(all_pt),
        "dbc": np.concatenate(all_dbc),
        "sig_mask": np.concatenate(all_sig_mask),
        "bkg_mask": np.concatenate(all_bkg_mask),
        "weights": np.concatenate(all_weights),
    }

    summary_txt = os.path.join(cfg.figure_path, "summary.txt")
    with open(summary_txt, "w") as fout:
        fout.write("window,n_sig,n_bkg,auc_glopart,auc_sophon\n")

        for mass_name, mass_low, mass_high in cfg.mass_windows:
            for pt_name, pt_low, pt_high in cfg.pt_windows:
                window_name = f"{mass_name}_{pt_name}"
                print(f"[INFO] Processing window: {window_name}")

                score, label, weight, n_sig, n_bkg = select_window(
                    merged, mass_low, mass_high, pt_low, pt_high
                )

                if n_sig == 0 or n_bkg == 0 or len(np.unique(label)) < 2:
                    print(f"[WARN] skip {window_name}: n_sig={n_sig}, n_bkg={n_bkg}")
                    fout.write(f"{window_name},{n_sig},{n_bkg},nan,nan\n")
                    continue

                if cfg.use_weights:
                    fpr_g, tpr_g, _ = roc_curve(label, score, sample_weight=weight)
                else:
                    fpr_g, tpr_g, _ = roc_curve(label, score)
                auc_g = auc(fpr_g, tpr_g)

                sophon_csv = os.path.join(cfg.sophon_roc_dir, f"roc_{window_name}.csv")
                sophon_data = None
                auc_s = np.nan
                if os.path.exists(sophon_csv):
                    sophon_data = read_sophon_roc_csv(sophon_csv)
                    if sophon_data is not None:
                        auc_s = auc(sophon_data["fpr"], sophon_data["tpr"])
                else:
                    print(f"[WARN] Sophon ROC csv not found: {sophon_csv}")

                # draw compare
                fig, ax = plt.subplots(figsize=(10, 10))

                ax.plot(
                    tpr_g, fpr_g, 
                    lw=2,
                    color="red",
                    label=f"GloParT Dbc (AUC={auc_g:.4f})"
                )

                if sophon_data is not None:
                    ax.plot(
                        sophon_data["tpr"], sophon_data["fpr"], 
                        lw=2,
                        color="blue",
                        linestyle="--",
                        label=f"Sophon (AUC={auc_s:.4f})"
                    )

                ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1)

                title_pt = f"[{pt_low}, inf)" if np.isinf(pt_high) else f"[{pt_low}, {pt_high})"
                # ax.set_title(
                #     f"ROC Comparison: {window_name}\n"
                #     f"mass=[{mass_low}, {mass_high}), pt={title_pt}"
                # )
                text = f"Mass: [{mass_low}, {mass_high}) GeV\nPt: {title_pt} GeV\nSignal: {n_sig}, Background: {n_bkg}"
                ax.text(
                    0.05, 0.95, text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9)
                )
                ax.set_xlabel("Signal efficiency")
                ax.set_ylabel("Background efficiency")
                ax.set_xlim(0, 1)
                ax.set_ylim(1e-4, 1)
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower right")
                hep.cms.label("Preliminary", data=False, ax=ax)

                fig.tight_layout()
                out_png = os.path.join(cfg.figure_path, f"roc_compare_{window_name}.png")
                out_pdf = os.path.join(cfg.figure_path, f"roc_compare_{window_name}.pdf")
                fig.savefig(out_png, dpi=150)
                fig.savefig(out_pdf)
                plt.close(fig)

                print(f"[SAVE] {out_png}")
                fout.write(f"{window_name},{n_sig},{n_bkg},{auc_g:.6f},{auc_s if not np.isnan(auc_s) else 'nan'}\n")

    print(f"[DONE] Outputs saved in: {cfg.figure_path}")
    print(f"[DONE] Summary saved in: {summary_txt}")

if __name__ == "__main__":
    main()
