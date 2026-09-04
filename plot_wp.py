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
# 3. Helper functions for truth masks and combined ROC
# ======================================================================================

def build_truth_masks(data, cfg):
    """Return a dict of truth masks for each score branch."""
    is_qcd = data["is_qcd_sample"]
    ak8_type = data["ak8_type"]
    ak8_nc = data["ak8_nc"]
    
    masks = {}
    masks["score_cata_qcd"] = is_qcd
    masks["score_cata_top_bqq"] = (ak8_type == 4) & (~masks["score_cata_qcd"])
    masks["score_cata_top_bc"] = (ak8_type == 2) & (ak8_nc == 1) & (~masks["score_cata_qcd"])
    masks["score_cata_top_bq"] = (ak8_type == 2) & (ak8_nc == 0) & (~masks["score_cata_qcd"])
    masks["score_cata_w_qq"] = (ak8_type == 1) & (~masks["score_cata_qcd"])
    masks["score_cata_non"] = ~(masks["score_cata_qcd"] | masks["score_cata_top_bqq"] | 
                                masks["score_cata_top_bc"] | masks["score_cata_top_bq"] | 
                                masks["score_cata_w_qq"])
    return masks

def compute_roc(data, cfg, signal_branches, background_branches):
    """Compute ROC for combined signal vs combined background.
    
    Returns fpr, tpr, thresholds, roc_auc, signal_score
    """
    weights = data["weights"]
    sum_scores = np.zeros_like(weights)
    for name in cfg.score_branches:
        sum_scores += data[name]
    sum_scores[sum_scores == 0] = 1e-10

    truth_masks = build_truth_masks(data, cfg)

    # Combined truth masks (OR)
    signal_mask = np.zeros(len(weights), dtype=bool)
    for b in signal_branches:
        signal_mask |= truth_masks[b]
        
    background_mask = np.zeros(len(weights), dtype=bool)
    for b in background_branches:
        background_mask |= truth_masks[b]
    # exclude signal events from background
    background_mask &= ~signal_mask

    # discriminant: sum of selected signal branches / sum_scores
    signal_score = np.zeros_like(weights)
    for b in signal_branches:
        signal_score += data[b]
    signal_score /= sum_scores

    # apply selection: positive weights and (signal or background)
    pos_w = weights > 0
    final_mask = (signal_mask | background_mask) & pos_w

    y_true = signal_mask[final_mask].astype(int)
    y_score = signal_score[final_mask]
    sample_weight = weights[final_mask]

    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Not enough signal and background events to compute ROC.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc, signal_score

# ======================================================================================
# 4. Interactive Menu
# ======================================================================================

def print_branches(cfg):
    print("\nAvailable score branches:")
    for idx, branch in enumerate(cfg.score_branches):
        label_map = {
            "score_cata_w_qq":    "W(qq)",
            "score_cata_qcd":     "QCD",
            "score_cata_top_bqq": "Top(bqq)",
            "score_cata_top_bc":  "Top(bc)",
            "score_cata_top_bq":  "Top(bq)",
            "score_cata_non":     "Non"
        }
        label = label_map.get(branch, branch)
        print(f"  [{idx}] {branch} ({label})")

def select_branches(cfg, role):
    print_branches(cfg)
    while True:
        try:
            raw = input(f"Enter {role} branch indices (comma separated, e.g. 0,2): ")
            indices = [int(x.strip()) for x in raw.split(",")]
            selected = [cfg.score_branches[i] for i in indices]
            print(f"  Selected {role}: {selected}")
            return selected
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")

def find_work_point_by_efficiency(tpr, fpr, thresholds, target_eff):
    """Find working point closest to a given signal efficiency."""
    idx = np.argmin(np.abs(tpr - target_eff))
    return idx, tpr[idx], fpr[idx], thresholds[idx]

def find_work_point_by_threshold(tpr, fpr, thresholds, threshold_val):
    """Find working point closest to a given threshold."""
    idx = np.argmin(np.abs(thresholds - threshold_val))
    return idx, tpr[idx], fpr[idx]

def plot_roc_and_mark(fpr, tpr, roc_auc, mark_tpr, mark_fpr, 
                      sig_label, bkg_label, cfg, suffix="WP"):
    """Plot ROC curve (TPR vs FPR) with a marked working point."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw ROC curve: x = signal eff (tpr), y = background eff (fpr)
    ax.plot(tpr, fpr, 'k-', lw=2, label=f'{sig_label} vs {bkg_label} (AUC={roc_auc:.2f})')
    ax.scatter(mark_tpr, mark_fpr, color='red', s=80, zorder=10,
               label=f'WP: sig eff={mark_tpr:.3f}, bkg eff={mark_fpr:.3f}')
    
    ax.set_yscale('log')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel(r'Signal efficiency', fontsize=20)
    ax.set_ylabel(r'Background efficiency', fontsize=20)
    hep.cms.label("Simulation", data=False, lumi=None, ax=ax, loc=0)
    ax.legend(loc="lower right", fontsize=14, frameon=False)
    ax.grid(True, which="major", ls="-", alpha=0.2)
    
    os.makedirs(cfg.figure_path, exist_ok=True)
    fname = os.path.join(cfg.figure_path, 
                         f"ROC_{sig_label}_vs_{bkg_label}_{suffix}.pdf")
    fname = fname.replace("(", "").replace(")", "")
    plt.savefig(fname)
    plt.savefig(fname.replace(".pdf", ".png"))
    print(f"  [SAVE] {fname}")
    plt.close(fig)

# ======================================================================================
# 5. Main interactive session
# ======================================================================================

def main():
    cfg = Config()
    loader = ROCLoader(cfg)
    print("Loading data... This may take a while.")
    data = loader.load_data()
    print("Data loaded successfully.\n")
    
    # Pre‑compute truth masks (will be reused)
    truth_masks = build_truth_masks(data, cfg)
    
    # Main loop: select signal/background, then study working points
    while True:
        print("\n" + "="*60)
        print("Define signal and background categories:")
        sig_branches = select_branches(cfg, "signal")
        bkg_branches = select_branches(cfg, "background")
        
        sig_names = [s.split("_")[-1] for s in sig_branches]
        bkg_names = [s.split("_")[-1] for s in bkg_branches]
        sig_label = "+".join(sig_names)
        bkg_label = "+".join(bkg_names)
        
        print(f"\nComputing ROC: signal=({sig_label}) vs background=({bkg_label})")
        try:
            fpr, tpr, thresholds, roc_auc, _ = compute_roc(data, cfg, sig_branches, bkg_branches)
        except RuntimeError as e:
            print(f"Error: {e}")
            continue
        
        print(f"  AUC = {roc_auc:.4f}")
        
        # Sub‑menu for working point operations
        while True:
            print("\nOptions:")
            print("  1: Given signal efficiency, find working point")
            print("  2: Given threshold (working point), mark on ROC and save plot")
            print("  3: Choose different signal/background")
            print("  4: Quit")
            choice = input("Your choice: ").strip()
            
            if choice == '1':
                try:
                    eff = float(input("Enter desired signal efficiency (0-1): "))
                    if not 0 <= eff <= 1:
                        print("Please enter a value between 0 and 1.")
                        continue
                    idx, eff_tpr, eff_fpr, thr = find_work_point_by_efficiency(tpr, fpr, thresholds, eff)
                    print(f"  Closest working point:")
                    print(f"    Signal efficiency : {eff_tpr:.4f}")
                    print(f"    Background eff.   : {eff_fpr:.4f}  (rejection: {1.0/eff_fpr if eff_fpr>0 else np.inf:.1f})")
                    print(f"    Threshold         : {thr:.6f}")
                except ValueError:
                    print("Invalid input.")
                    
            elif choice == '2':
                try:
                    thr_val = float(input("Enter threshold value (working point): "))
                    idx, mark_tpr, mark_fpr = find_work_point_by_threshold(tpr, fpr, thresholds, thr_val)
                    print(f"  At threshold = {thr_val:.6f}:")
                    print(f"    Signal efficiency : {mark_tpr:.4f}")
                    print(f"    Background eff.   : {mark_fpr:.4f}")
                    plot_roc_and_mark(fpr, tpr, roc_auc, mark_tpr, mark_fpr,
                                      sig_label, bkg_label, cfg, suffix=f"thr_{thr_val:.4f}")
                except ValueError:
                    print("Invalid input.")
                    
            elif choice == '3':
                break  # go back to signal/bkg selection
            elif choice == '4':
                print("Exiting.")
                return
            else:
                print("Invalid option.")

if __name__ == "__main__":
    main()