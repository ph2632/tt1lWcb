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
        self.figure_path = "./figures/"
        
        self.tree_name = "Events"
        
        # 我们只需要计算分数和真值所需的变量
        self.branches = [
            "xsecWeight", "genWeight", "lumiwgt", "puWeight", 
            "trigEffWeight", "l1PreFiringWeight",
            "ak8_type", 
            "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq", 
            "score_cata_top_bc", "score_cata_top_bq", "score_cata_non"
        ]

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

    def load_data(self):
        print(f"[INFO] Loading MC files from {self.cfg.mc_path} ...")
        mc_files = [f for f in os.listdir(self.cfg.mc_path) if f.endswith('.root')]
        
        all_y_true = []   # 1 for Signal (True Wqq), 0 for Background
        all_y_score = []  # Discriminator score
        all_weights = []  # Event weights

        for i, fname in enumerate(mc_files):
            if i % 5 == 0:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                print(f"\r[PROG] Reading {i+1}/{len(mc_files)} | Mem: {mem:.1f} MB", end="")

            file_path = os.path.join(self.cfg.mc_path, fname)
            
            try:
                with uproot.open(file_path) as f:
                    tree = f[self.cfg.tree_name]
                    events = tree.arrays(self.cfg.branches, library="ak")
            except Exception as e:
                print(f"\n[WARN] Failed to read {fname}: {e}")
                continue

            if len(events) == 0: continue

            # --- 1. 计算权重 ---
            weights = (events["xsecWeight"] * events["genWeight"] * events["lumiwgt"] * 
                       events["puWeight"] * events["trigEffWeight"] * events["l1PreFiringWeight"])
            weights = np.array(weights, dtype=np.float32)

            # --- 2. 计算分数 (Discriminator) ---
            s_w_qq    = np.array(self.get_leading(events["score_cata_w_qq"], 0))
            s_qcd     = np.array(self.get_leading(events["score_cata_qcd"], 0))
            s_top_bqq = np.array(self.get_leading(events["score_cata_top_bqq"], 0))
            s_top_bc  = np.array(self.get_leading(events["score_cata_top_bc"], 0))
            s_top_bq  = np.array(self.get_leading(events["score_cata_top_bq"], 0))
            s_non     = np.array(self.get_leading(events["score_cata_non"], 0))
            
            denom = s_w_qq + s_qcd + s_top_bqq + s_top_bc + s_top_bq + s_non + 1e-10
            score_val = s_w_qq / denom

            # --- 3. 定义真值 (Truth Definition) ---
            ak8_type = np.array(self.get_leading(events["ak8_type"]))
            
            # 判断是否为 QCD 样本文件 (沿用之前的逻辑)
            fname_check = fname.lower()
            is_qcd_file = any(k in fname_check for k in ["qcd", "wjet", "zjet"])
            
            # 信号定义：Type 是 W (1) 且 不在 QCD 样本中
            # 背景定义：反之
            is_signal = (ak8_type == 1) & (not is_qcd_file)
            
            # 转换为 0/1 标签
            y_true = is_signal.astype(int)

            all_y_true.append(y_true)
            all_y_score.append(score_val)
            all_weights.append(weights)
            
            del events, weights, score_val, y_true
            if i % 20 == 0: gc.collect()

        print("\n[INFO] Concatenating arrays...")
        return (
            np.concatenate(all_y_true),
            np.concatenate(all_y_score),
            np.concatenate(all_weights)
        )

# ======================================================================================
# 3. Plotter
# ======================================================================================
def plot_roc(y_true, y_score, weights, cfg):
    print("[INFO] Calculating ROC curve...")
    
    # === [FIX] 处理负权重 ===
    # 负权重会导致 FPR/TPR 非单调，从而使 auc() 报错。
    # 这里我们将负权重置为 0 (或者直接丢弃这些事件)
    # 方法 A: 仅保留权重 > 0 的事件 (推荐)
    mask_pos = weights > 0
    n_total = len(weights)
    n_kept = np.sum(mask_pos)
    print(f"[INFO] Filtering negative weights: kept {n_kept}/{n_total} events.")
    
    y_true_clean = y_true[mask_pos]
    y_score_clean = y_score[mask_pos]
    weights_clean = weights[mask_pos]
    
    # 计算 ROC
    # pos_label=1 表示 y_true=1 是信号
    fpr, tpr, thresholds = roc_curve(y_true_clean, y_score_clean, 
                                     sample_weight=weights_clean, pos_label=1)
    
    # 计算 AUC
    try:
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(f"[WARN] AUC calculation failed directly: {e}")
        # 备用方案：强制排序（虽然过滤负权重后通常不需要）
        sort_idx = np.argsort(fpr)
        roc_auc = auc(fpr[sort_idx], tpr[sort_idx])
    
    # save roc
    np.savez(os.path.join(cfg.figure_path, "roc_data_Wqq_TrueCat_Dec29.npz"),
             fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
    print(f"[RESULT] ROC AUC = {roc_auc:.4f}")

    # 绘图
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制曲线
    ax.plot(tpr, fpr, color='darkorange', lw=2, 
            label=r'ROC curve (AUC = %0.3f)' % roc_auc)
    
    # 绘制对角线 (随机猜测)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置坐标轴
    ax.set_xlim([0.0, 1.0])
    # Y轴 Log 刻度
    ax.set_yscale('log')
    # 设置 Y 轴范围，避免 log(0)
    # 自动寻找非零的最小 FPR (忽略 0)
    min_fpr = np.min(fpr[fpr > 0]) if len(fpr[fpr > 0]) > 0 else 1e-4
    # 稍微给点余量，比如从最小值的 0.5 倍开始，或者固定 1e-4
    bottom_lim = max(1e-4, min_fpr * 0.5)
    ax.set_ylim([bottom_lim, 1.0])
    
    ax.set_xlabel(r'Signal Efficiency ($\epsilon_S$)')
    ax.set_ylabel(r'Background Efficiency ($\epsilon_B$)')
    
    # 添加网格
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # CMS Label
    hep.cms.label("Preliminary", data=False, lumi=41.48, ax=ax)
    
    ax.legend(loc="lower right")
    
    if not os.path.exists(cfg.figure_path):
        os.makedirs(cfg.figure_path)
        
    outname = os.path.join(cfg.figure_path, "ROC_Wqq_TrueCat.pdf")
    plt.savefig(outname)
    print(f"[PLOT] Saved to {outname}")
    plt.close(fig)


# ======================================================================================
# Main
# ======================================================================================

if __name__ == "__main__":
    start_time = time.time()
    cfg = Config()
    
    loader = ROCLoader(cfg)
    y_true, y_score, weights = loader.load_data()
    
    print(f"[INFO] Total Events: {len(y_true)}")
    print(f"[INFO] Signal Events (Weighted): {np.sum(weights[y_true==1]):.2f}")
    print(f"[INFO] Bkg Events (Weighted):    {np.sum(weights[y_true==0]):.2f}")
    
    plot_roc(y_true, y_score, weights, cfg)
    
    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
