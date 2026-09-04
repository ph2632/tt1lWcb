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
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_/")
        self.figure_path = "./figures/roc_figures/"
        
        self.tree_name = "Events"
        
        self.catalog = {
            "Wcb": ["ttbar-powheg"],
            "single-top": ["single-top"],
            "tt":  ["tt-semi", "tt-lep", "tt-had"],
            "Rare": ["ttbb", "ttW", "ttZ", "twZ", "ttHToTauTau", "ttHNonbb", "ttbb"],
            "Diboson": ["WW", "WZ", "ZZ"],
            "QCD": ["QCD"],
        }

        self.base_branches = [
            "xsecWeight", "genWeight", "lumiwgt", "puWeight", 
            "trigEffWeight", "l1PreFiringWeight",
            "ak8_type", "ak8_n_c_in_jet"
        ]
        
        self.score_branches = [
            "score_cata_w_qq", 
            "score_cata_qcd", 
            "score_cata_top_bqq", 
            "score_cata_top_bc", 
            "score_cata_top_bq", 
            "score_cata_non"
        ]
        
        self.branches = self.base_branches + self.score_branches

# ======================================================================================
# 2. Data Loader
# ======================================================================================

class ROCLoader:
    def __init__(self, config):
        self.cfg = config

    def get_leading(self, arr, default=-999):
        try:
            return ak.fill_none(ak.firsts(arr, axis=1), default)
        except Exception:
            return ak.fill_none(arr, default)

    def identify_group(self, filename):
        fname_clean = filename.replace('_tree.root', '')
        for group, patterns in self.cfg.catalog.items():
            for p in patterns:
                if p in fname_clean:
                    return group
        return "Other"

    def load_data(self):
        print(f"[INFO] Loading MC files from {self.cfg.mc_path} ...")
        mc_files = [f for f in os.listdir(self.cfg.mc_path) if f.endswith('.root')]
        
        data = {
            "weights": [], "ak8_type": [], "ak8_nc": [], "is_qcd_sample": []
        }
        for s in self.cfg.score_branches:
            data[s] = []

        for i, fname in enumerate(mc_files):
            if i % 5 == 0:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                print(f"\r[PROG] Reading {i+1}/{len(mc_files)} | Mem: {mem:.1f} MB", end="")

            file_path = os.path.join(self.cfg.mc_path, fname)
            group_name = self.identify_group(fname)
            is_qcd_file = (group_name == "QCD")

            try:
                with uproot.open(file_path) as f:
                    tree = f[self.cfg.tree_name]
                    events = tree.arrays(self.cfg.branches, library="ak")
            except Exception as e:
                print(f"\n[WARN] Failed to read {fname}: {e}")
                continue

            if len(events) == 0: continue

            w = (events["xsecWeight"] * events["genWeight"] * events["lumiwgt"] * 
                 events["puWeight"] * events["trigEffWeight"] * events["l1PreFiringWeight"])
            data["weights"].append(np.array(w, dtype=np.float32))

            ak8_t = np.array(self.get_leading(events["ak8_type"], -1))
            data["ak8_type"].append(ak8_t)
            
            ak8_nc = np.array(self.get_leading(events["ak8_n_c_in_jet"], 0))
            data["ak8_nc"].append(ak8_nc)
            
            data["is_qcd_sample"].append(np.full(len(w), is_qcd_file, dtype=bool))

            for score_name in self.cfg.score_branches:
                s_val = np.array(self.get_leading(events[score_name], 0))
                data[score_name].append(s_val)
            
            del events, w, ak8_t, ak8_nc
            if i % 20 == 0: gc.collect()

        print("\n[INFO] Concatenating arrays...")
        final_data = {k: np.concatenate(v) for k, v in data.items()}
        return final_data

# ======================================================================================
# 3. Plotter
# ======================================================================================

def plot_six_figures(data, cfg):
    print("[INFO] Preparing 6 separate ROC figures (Black lines, varied styles)...")
    
    if not os.path.exists(cfg.figure_path):
        os.makedirs(cfg.figure_path)

    weights = data["weights"]
    ak8_type = data["ak8_type"]
    ak8_nc = data["ak8_nc"]
    is_qcd_sample = data["is_qcd_sample"]
    
    sum_scores = np.zeros_like(weights)
    for name in cfg.score_branches:
        sum_scores += data[name]
    sum_scores[sum_scores == 0] = 1e-10

    # Truth Definitions
    truth_masks = {}
    truth_masks["score_cata_qcd"] = is_qcd_sample
    truth_masks["score_cata_top_bqq"] = (ak8_type == 4) & (~truth_masks["score_cata_qcd"])
    truth_masks["score_cata_top_bc"] = (ak8_type == 2) & (ak8_nc == 1) & (~truth_masks["score_cata_qcd"])
    truth_masks["score_cata_top_bq"] = (ak8_type == 2) & (ak8_nc == 0) & (~truth_masks["score_cata_qcd"])
    truth_masks["score_cata_w_qq"] = (ak8_type == 1) & (~truth_masks["score_cata_qcd"])
    truth_masks["score_cata_non"] = ~(truth_masks["score_cata_qcd"] | truth_masks["score_cata_top_bqq"] | 
                                      truth_masks["score_cata_top_bc"] | truth_masks["score_cata_top_bq"] | 
                                      truth_masks["score_cata_w_qq"])

    label_map = {
        "score_cata_w_qq":    "W(qq)",
        "score_cata_qcd":     "QCD",
        "score_cata_top_bqq": "Top(bqq)",
        "score_cata_top_bc":  "Top(bc)",
        "score_cata_top_bq":  "Top(bq)",
        "score_cata_non":     "Non"
    }

    pos_w_mask = weights > 0
    
    # === [STYLE CONFIG] ===
    # 定义6种不同的线型 (Matplotlib format)
    # 0: Solid
    # 1: Dashed
    # 2: Dash-dot
    # 3: Dotted
    # 4: Long Dash
    # 5: Dash-Dot-Dot
    linestyles = [
        '-',                          # Solid
        '--',                         # Dashed
        '-.',                         # Dash-dot
        ':',                          # Dotted
        (0, (5, 5)),                  # Long Dash
        (0, (3, 5, 1, 5, 1, 5))       # Dash-Dot-Dot
    ]

    for sig_key in cfg.score_branches:
        sig_label = label_map[sig_key]
        print(f"\n[PLOT] Generating figure for Signal: {sig_label}")
        
        hep.style.use("CMS")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        sig_score_val = data[sig_key] / sum_scores
        
        # --- Curve 1: Inclusive (vs All) ---
        # 使用第一种线型 (Solid)
        style_idx = 0
        
        mask_sig = truth_masks[sig_key]
        final_mask = pos_w_mask
        
        y_true = mask_sig[final_mask].astype(int)
        y_score = sig_score_val[final_mask]
        w_curr = weights[final_mask]
        
        if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
            fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=w_curr)
            roc_auc = auc(fpr, tpr)
            
            # X=Signal Eff (TPR), Y=Background Eff (FPR)
            ax.plot(tpr, fpr, 
                    color='black', 
                    linestyle=linestyles[style_idx], 
                    lw=2, 
                    label=f'{sig_label} vs All (AUC={roc_auc:.2f})')
        
        style_idx += 1

        # --- Curves 2-6: Pairwise ---
        for bkg_key in cfg.score_branches:
            if sig_key == bkg_key: continue
            
            bkg_label = label_map[bkg_key]
            mask_bkg_specific = truth_masks[bkg_key]
            
            pair_mask = (mask_sig | mask_bkg_specific) & pos_w_mask
            
            if np.sum(pair_mask) < 10: continue
                
            y_true_pair = mask_sig[pair_mask].astype(int)
            y_score_pair = sig_score_val[pair_mask]
            w_pair = weights[pair_mask]
            
            if len(np.unique(y_true_pair)) < 2: continue
                
            fpr, tpr, _ = roc_curve(y_true_pair, y_score_pair, sample_weight=w_pair)
            roc_auc = auc(fpr, tpr)
            
            # 使用后续的线型
            current_ls = linestyles[style_idx % len(linestyles)]
            
            ax.plot(tpr, fpr, 
                    color='black', 
                    linestyle=current_ls, 
                    lw=2, 
                    label=f'{sig_label} vs {bkg_label} (AUC={roc_auc:.2f})')
            
            style_idx += 1

        # 装饰
        # Log Scale on Y-axis (Background Efficiency)
        ax.set_yscale('log')
        
        ax.set_xlim([0.0, 1.0])
        # Y轴范围建议：从 1e-3 或 1e-4 开始
        ax.set_ylim([1e-3, 1.0])
        
        ax.set_xlabel(r'Signal efficiency', fontsize=20)
        ax.set_ylabel(r'Background efficiency', fontsize=20)
        
        hep.cms.label("Simulation", data=False, lumi=None, ax=ax, loc=0)
        
        # 图例放在右下角
        ax.legend(loc="lower right", fontsize=14, frameon=False)
        ax.grid(True, which="major", ls="-", alpha=0.2)
        
        outname = os.path.join(cfg.figure_path, f"ROC_Signal_{sig_label}.pdf")
        outname = outname.replace("(", "").replace(")", "")
        
        plt.savefig(outname)
        plt.savefig(outname.replace(".pdf", ".png"))
        print(f"  [SAVE] Saved {outname}")
        plt.close(fig)

# ======================================================================================
# Main
# ======================================================================================

if __name__ == "__main__":
    start_time = time.time()
    cfg = Config()
    
    loader = ROCLoader(cfg)
    data = loader.load_data()
    
    plot_six_figures(data, cfg)
    
    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
