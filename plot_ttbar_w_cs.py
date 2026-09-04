#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
from sklearn.metrics import roc_curve, auc
import time
import gc
import psutil

# ======================================================================================
# 1. Configuration
# ======================================================================================

class Config:
    def __init__(self):
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_v1/")
        self.figure_path = "./figures/roc_validation_ttbar_dcs/"
        self.tree_name = "Events"

        # 只处理 ttbar
        self.ttbar_keywords = ["ttbar-powheg", "tt-semi", "tt-lep", "tt-had"]

        self.branches = [
            "xsecWeight", "genWeight", "lumiwgt", "puWeight",
            "trigEffWeight", "l1PreFiringWeight",
            "ak8_type", "ak8_n_c_in_jet", "ak8_n_b_in_jet",
            "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
            "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
        ]


# ======================================================================================
# 2. Data Loader
# ======================================================================================

class ValidationROCLoader:
    def __init__(self, config):
        self.cfg = config

    def get_leading(self, arr, default=-999):
        try:
            return ak.fill_none(ak.firsts(arr, axis=1), default)
        except Exception:
            return ak.fill_none(arr, default)

    def is_ttbar_file(self, filename):
        fname_clean = filename.replace('_tree.root', '')
        return any(k in fname_clean for k in self.cfg.ttbar_keywords)

    def load_data(self):
        print(f"[INFO] Loading TTbar files from {self.cfg.mc_path} ...")
        mc_files = [f for f in os.listdir(self.cfg.mc_path) if f.endswith('.root')]
        ttbar_files = [f for f in mc_files if self.is_ttbar_file(f)]

        print(f"[INFO] Found {len(ttbar_files)} TTbar files.")

        data = {
            "weights": [],
            "ak8_type": [],
            "ak8_nc": [],
            "ak8_nb": [],
            "g_bc": [],
            "g_bb": [],
            "g_cc": [],
            "g_qcd": [],
            "g_bs": [],
            "g_qq": [],
            "g_cs": [],
            "g_topbw": []
        }

        for i, fname in enumerate(ttbar_files):
            if i % 5 == 0:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                print(f"\r[PROG] Reading {i+1}/{len(ttbar_files)} | Mem: {mem:.1f} MB", end="")

            file_path = os.path.join(self.cfg.mc_path, fname)

            try:
                with uproot.open(file_path) as f:
                    tree = f[self.cfg.tree_name]
                    events = tree.arrays(self.cfg.branches, library="ak")
            except Exception as e:
                print(f"\n[WARN] Failed to read {fname}: {e}")
                continue

            if len(events) == 0:
                continue

            weights = (
                events["xsecWeight"] * events["genWeight"] * events["lumiwgt"] *
                events["puWeight"] * events["trigEffWeight"] * events["l1PreFiringWeight"]
            )
            data["weights"].append(np.array(weights, dtype=np.float32))

            data["ak8_type"].append(np.array(self.get_leading(events["ak8_type"], -1)))
            data["ak8_nc"].append(np.array(self.get_leading(events["ak8_n_c_in_jet"], -1)))
            data["ak8_nb"].append(np.array(self.get_leading(events["ak8_n_b_in_jet"], -1)))

            data["g_bc"].append(np.array(self.get_leading(events["ak8_gpt_bc"], 0)))
            data["g_bb"].append(np.array(self.get_leading(events["ak8_gpt_bb"], 0)))
            data["g_cc"].append(np.array(self.get_leading(events["ak8_gpt_cc"], 0)))
            data["g_qcd"].append(np.array(self.get_leading(events["ak8_gpt_qcd"], 0)))
            data["g_bs"].append(np.array(self.get_leading(events["ak8_gpt_bs"], 0)))
            data["g_qq"].append(np.array(self.get_leading(events["ak8_gpt_qq"], 0)))
            data["g_cs"].append(np.array(self.get_leading(events["ak8_gpt_cs"], 0)))
            data["g_topbw"].append(np.array(self.get_leading(events["ak8_gpt_topbw"], 0)))

            del events, weights
            if i % 20 == 0:
                gc.collect()

        print("\n[INFO] Concatenating arrays...")
        final_data = {k: np.concatenate(v) for k, v in data.items()}
        return final_data


# ======================================================================================
# 3. ROC Plot
# ======================================================================================

def plot_validation_roc(data, cfg):
    print("[INFO] Preparing validation ROC for TTbar only, signal = (type=1, nc=1, nb=0)")

    if not os.path.exists(cfg.figure_path):
        os.makedirs(cfg.figure_path)

    weights = data["weights"]
    ak8_type = data["ak8_type"]
    ak8_nc = data["ak8_nc"]
    ak8_nb = data["ak8_nb"]

    g_bc = data["g_bc"]
    g_bb = data["g_bb"]
    g_cc = data["g_cc"]
    g_qcd = data["g_qcd"]
    g_bs = data["g_bs"]
    g_qq = data["g_qq"]
    g_cs = data["g_cs"]
    g_topbw = data["g_topbw"]

    # -------------------------
    # Signal / Background truth
    # -------------------------
    # mask_sig = (ak8_type == 1) & (ak8_nc == 1) & (ak8_nb == 0)
    mask_sig = (ak8_nc == 1) & (ak8_nb == 0)
    # mask_bkg = ~mask_sig
    mask_bkg = (ak8_type == 1) & ((ak8_nc != 1) | (ak8_nb == 0))

    # -------------------------
    # Dcs = g_cs / all
    # -------------------------
    denom = g_bc + g_bb + g_cc + g_qcd + g_bs + g_qq + g_cs + g_topbw + 1e-10
    dcs = g_cs / denom

    # 可选：只保留正权重点
    valid_mask = (weights > 0) & np.isfinite(dcs)

    y_true = mask_sig[valid_mask].astype(int)
    y_score = dcs[valid_mask]
    w = weights[valid_mask]

    n_sig = np.sum((y_true == 1))
    n_bkg = np.sum((y_true == 0))

    print(f"[INFO] Signal events     : {n_sig}")
    print(f"[INFO] Background events : {n_bkg}")

    if n_sig == 0 or n_bkg == 0:
        print("[ERROR] Need both signal and background events to make ROC.")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=w)
    roc_auc = auc(fpr, tpr)

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))

    # 横轴：signal efficiency
    # 纵轴：background efficiency
    ax.plot(
        tpr, fpr,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label=f"Dcs ROC (AUC={roc_auc:.4f})"
    )
    ## add pheno validation points
    pheno_data = np.load("extern/leading_fj_inclusive_c_roc.npy", allow_pickle=True).item()
    ph_item = pheno_data["score_cs_vs_w"]
    fpr_ph = ph_item["fpr"]
    tpr_ph = ph_item["tpr"]
    auc_ph = ph_item["auc"]
    ax.plot(
        tpr_ph, fpr_ph,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Pheno ROC (AUC={auc_ph:.4f})"
    )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 1.0)

    ax.set_xlabel("Signal efficiency", fontsize=18)
    ax.set_ylabel("Background efficiency", fontsize=18)

    hep.cms.label("Simulation", data=False, lumi=None, ax=ax, loc=0)

    truth_text = (
        "TTbar only\n"
        "Signal: ak8_type=1, n_c=1, n_b=0\n"
        "Background: all other TTbar jets"
    )
    ax.text(
        0.05, 0.20, truth_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    ax.legend(loc="lower right", fontsize=14, frameon=False)
    ax.grid(True, which="major", ls="-", alpha=0.2)



    out_pdf = os.path.join(cfg.figure_path, "ROC_ttbar_validation_Dcs.pdf")
    out_png = os.path.join(cfg.figure_path, "ROC_ttbar_validation_Dcs.png")

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    print(f"[SAVE] Saved {out_pdf}")
    print(f"[SAVE] Saved {out_png}")
    plt.close(fig)


# ======================================================================================
# Main
# ======================================================================================

if __name__ == "__main__":
    start_time = time.time()

    cfg = Config()
    loader = ValidationROCLoader(cfg)
    data = loader.load_data()
    plot_validation_roc(data, cfg)

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
