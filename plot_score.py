#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import time
import gc
import psutil
import shutil
from train_bdt.dbc_tools import DbcEvaluator
# ======================================================================================
# 1. Configuration
# ======================================================================================

dbc_eval = DbcEvaluator(
    mode="bdt",
    model_path="./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"
)

class Config:
    def __init__(self):
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        # self.mc_path = os.path.join(self.base_path, "MC/pieces_1merged_/")
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_/")
        self.data_path = os.path.join(self.base_path, "data/scored_data/merged/")
        
        self.figure_path = "./figures/"
        self.parquet_path = "./score_parquet/"
        
        # --- 核心开关 ---
        self.enable_data = False 
        self.use_cache = True   
        self.force_reload = False 
        
        self.lumi = 41.479 
        self.tree_name = "Events"
        
        self.branches = [
            "xsecWeight", "genWeight", "lumiwgt", "puWeight", 
            "trigEffWeight", "l1PreFiringWeight",
            "ak8_sdmass", "ak8_pt", "ak8_eta", 
            "ak8_type", "ak8_n_c_in_jet", 
            "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq", 
            "score_cata_top_bc", "score_cata_top_bq", "score_cata_non",
            "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
            "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
        ]
        self.data_branches = [b for b in self.branches if b not in ["xsecWeight", "genWeight", "lumiwgt", "puWeight", "trigEffWeight", "l1PreFiringWeight", "ak8_type", "ak8_n_c_in_jet"]]

        self.catalog = {
            "Wcb":       ["ttbar-powheg"], 
            "tt":        ["tt-semi", "tt-lep", "tt-had"],
            "QCD":       ["QCD"],
            "SingleTop": ["single-top"],
            "Rare":      ["ttbb", "ttW", "ttZ", "twZ", "ttHToTauTau", "ttHNonbb"],
            "Diboson":   ["WW", "WZ", "ZZ"],
        }
        
        self.signal_groups = ["Wcb"] 

        self.labels = {
            "Wcb":       r"$W_{cb}$ (Signal)",
            "tt":        r"$t\bar{t}$",
            "QCD":       "QCD Multijet",
            "SingleTop": "Single Top",
            "Rare":      "Rare",
            "Diboson":   "VV",
            
            "Cat_QCD":     "QCD/V+Jets",
            "Cat_Top_bqq": r"Top ($bqq'$)",
            "Cat_Top_bc":  r"Top ($bc$)",
            "Cat_Top_bq":  r"Top ($bq$)",
            "Cat_W_qq":    r"$W \to qq'$",
            "Cat_Other":   "Other",
            
            "Data": "Data 2017"
        }

        self.colors = {
            "tt": '#DAA520', "QCD": '#CD5C5C', "SingleTop": '#4682B4',
            "Wcb": "red", "Rare": '#9ACD32', "Diboson": '#FF7F50',
            "Cat_Top_bqq": '#FF69B4', "Cat_Top_bc": '#8FBC8F', 
            "Cat_Top_bq": '#6A5ACD', "Cat_W_qq": '#F0E68C',
            "Cat_QCD": '#CD5C5C', "Cat_Other": 'grey'
        }

        self.plots = {
            "Plot1_Score_TrueCat": {
                "type": "score_by_true",
                "bins": np.linspace(0, 1, 51),
                "xlabel": "Discriminator Score ($S_{wqq} / \Sigma S$)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            "Plot2_Mass_TrueCat": {
                "type": "mass_by_true",
                "bins": np.linspace(0, 250, 26),
                "xlabel": "AK8 Soft Drop Mass [GeV]",
                "ylabel": "Events / 10 GeV",
                "logy": False,
                "ssf": 1000,
                "xlim": (0, 250),
            },
            "Plot3_Mass_Catalog": {
                "type": "mass_by_catalog",
                "bins": np.linspace(0, 250, 26),
                "xlabel": "AK8 Soft Drop Mass [GeV]",
                "ylabel": "Events / 10 GeV",
                "logy": False,
                "ssf": 1000,
                "xlim": (0, 250),
            },
            "Plot4_Score_Catalog_Cut": {
                "type": "score_by_catalog_cut",
                "bins": np.linspace(0, 1, 51),
                "cut_val": 0.85,
                "xlabel": "Discriminator Score ($S_{wqq} / \Sigma S$)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            "Plot5_Score_TrueCat_Cut": {
                "type": "score_by_true_cut",
                "bins": np.linspace(0, 1, 51),
                "cut_val": 0.85,
                "xlabel": "Discriminator Score ($S_{wqq} / \Sigma S$)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            # --- 新增 Plot 6: 混合模式 ---
            "Plot6_Score_Mixed": {
                "type": "score_by_true_mixed", # 新类型
                "bins": np.linspace(0, 1, 51),
                "xlabel": "Score (Bkg: True Cat, Sig: Wcb)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            "Plot7_Score_Mixed_Cut": {
                "type": "score_by_true_mixed_cut", # 定义新类型
                "bins": np.linspace(0.05, 1, 46),
                "cut_val": 0.985,  # 设置 Dbc 阈值
                "xlabel": "Score (Bkg: True Cat, Sig: Wcb) [Dbc > 0.985]",
                "ylabel": "Events / 0.02",
                "logy": False,
                "ssf": 1,
                "xlim": (0, 1),
                "ylim_bottom": 0.001,
            }
            
        }

# ======================================================================================
# 2. Processor
# ======================================================================================

class Processor:
    def __init__(self, config):
        self.cfg = config
        self.hist_data = {pid: {} for pid in self.cfg.plots.keys()}
        
        if self.cfg.use_cache and not os.path.exists(self.cfg.parquet_path):
            os.makedirs(self.cfg.parquet_path)

    def get_leading(self, arr, default=-999):
        try:
            return ak.fill_none(ak.firsts(arr, axis=1), default)
        except Exception as e:
            if "axis" in str(e).lower() or "depth" in str(e).lower():
                return ak.fill_none(arr, default)
            raise e

    def process_file(self, root_filepath, is_data=False, sample_group="Data"):
        filename = os.path.basename(root_filepath)
        parquet_name = filename.replace(".root", ".parquet")
        parquet_path = os.path.join(self.cfg.parquet_path, parquet_name)
        
        events_cache = None

        if self.cfg.use_cache and not self.cfg.force_reload and os.path.exists(parquet_path):
            try:
                events_cache = ak.from_parquet(parquet_path)
            except Exception as e:
                print(f"[WARN] Corrupt parquet {parquet_name}, reloading from ROOT.")
        
        if events_cache is None:
            try:
                with uproot.open(root_filepath) as f:
                    tree = f[self.cfg.tree_name]
                    if is_data:
                        raw_events = tree.arrays(self.cfg.data_branches, library="ak")
                    else:
                        raw_events = tree.arrays(self.cfg.branches, library="ak")
            except Exception as e:
                print(f"[WARN] Cannot read ROOT {root_filepath}: {e}")
                return

            #basic selection: 1. ak8 mass in [60,120]
            # raw_events = raw_events[(raw_events["ak8_sdmass"] > 60) & (raw_events["ak8_sdmass"] < 120)]

            if len(raw_events) == 0: return



            ak8_sdmass = np.array(self.get_leading(raw_events["ak8_sdmass"]))
            
            fname_check = filename.lower()
            is_qcd_file = any(k in fname_check for k in ["qcd", "wjet", "zjet"])
            is_qcd = np.full(len(raw_events), is_qcd_file, dtype=bool)

            if is_data:
                weights = np.ones(len(raw_events), dtype=np.float32)
                is_qcd = np.zeros(len(raw_events), dtype=bool) 
                ak8_type = np.full(len(raw_events), -1, dtype=int)
                n_c = np.full(len(raw_events), -1, dtype=int)
            else:
                ak8_type   = np.array(self.get_leading(raw_events["ak8_type"])) 
                n_c        = np.array(self.get_leading(raw_events["ak8_n_c_in_jet"]))
                weights = (raw_events["xsecWeight"] * raw_events["genWeight"] * raw_events["lumiwgt"] * 
                           raw_events["puWeight"] * raw_events["trigEffWeight"] * raw_events["l1PreFiringWeight"])
                weights = np.array(weights, dtype=np.float32)

            # Score 1
            s_w_qq    = np.array(self.get_leading(raw_events["score_cata_w_qq"], 0))
            s_qcd     = np.array(self.get_leading(raw_events["score_cata_qcd"], 0))
            s_top_bqq = np.array(self.get_leading(raw_events["score_cata_top_bqq"], 0))
            s_top_bc  = np.array(self.get_leading(raw_events["score_cata_top_bc"], 0))
            s_top_bq  = np.array(self.get_leading(raw_events["score_cata_top_bq"], 0))
            s_non     = np.array(self.get_leading(raw_events["score_cata_non"], 0))
            denom = s_w_qq + s_qcd + s_top_bqq + s_top_bc + s_top_bq + s_non + 1e-10
            score_val = s_w_qq / denom

            # Score 2 (Dbc)
            g_bc   = np.array(self.get_leading(raw_events["ak8_gpt_bc"], 0))
            g_bb   = np.array(self.get_leading(raw_events["ak8_gpt_bb"], 0))
            g_cc   = np.array(self.get_leading(raw_events["ak8_gpt_cc"], 0))
            g_qcd  = np.array(self.get_leading(raw_events["ak8_gpt_qcd"], 0))
            g_bs   = np.array(self.get_leading(raw_events["ak8_gpt_bs"], 0))
            g_qq   = np.array(self.get_leading(raw_events["ak8_gpt_qq"], 0))
            g_cs   = np.array(self.get_leading(raw_events["ak8_gpt_cs"], 0))
            g_topbw= np.array(self.get_leading(raw_events["ak8_gpt_topbw"], 0))
            denom_dbc = g_bc + g_qcd + (g_cc + g_bb + g_bs + g_cs + g_qq + g_topbw)  + 1e-10
            # dbc bc vs w non-bc
            # denom_dbc = g_bc + g_qcd + g_cc + g_cs + g_qq + g_topbw + 1e-10

            # dbc_val = g_bc / denom_dbc
            dbc_val = dbc_eval.get_Dbc(g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw)

            events_cache = ak.Array({
                "ak8_sdmass": ak8_sdmass,
                "weights": weights,
                "score": score_val,
                "dbc": dbc_val,
                "ak8_type": ak8_type,
                "n_c": n_c,
                "is_qcd": is_qcd
            })
            
            if self.cfg.use_cache:
                ak.to_parquet(events_cache, parquet_path)
            
            del raw_events, s_w_qq, s_qcd, denom, g_bc, g_bb, denom_dbc
        
        # --- 3. 填充直方图 ---
        ak8_sdmass = events_cache["ak8_sdmass"]
        weights    = events_cache["weights"]
        score_val  = events_cache["score"]
        dbc_val    = events_cache["dbc"]
        ak8_type   = events_cache["ak8_type"]
        n_c        = events_cache["n_c"]
        is_qcd     = events_cache["is_qcd"]

        mask_qcd     = is_qcd
        mask_top_bqq = (ak8_type == 4) & (~mask_qcd)
        mask_top_bc  = (ak8_type == 2) & (n_c == 1) & (~mask_qcd)
        mask_top_bq  = (ak8_type == 2) & (n_c == 0) & (~mask_qcd)
        mask_w_qq    = (ak8_type == 1) & (~mask_qcd)
        mask_other   = ~(mask_qcd | mask_top_bqq | mask_top_bc | mask_top_bq | mask_w_qq)

        masks_true_cat = {
            "Cat_QCD": mask_qcd,
            "Cat_Top_bqq": mask_top_bqq,
            "Cat_Top_bc": mask_top_bc,
            "Cat_Top_bq": mask_top_bq,
            "Cat_W_qq": mask_w_qq,
            "Cat_Other": mask_other
        }
        
        for plot_id, settings in self.cfg.plots.items():
            ptype = settings["type"]
            bins = settings["bins"]
            if sample_group == "Data":
                # 1. 确定是否需要 Cut
                use_cut = "cut" in ptype
                if use_cut:
                    cut_threshold = settings["cut_val"]

                    mask_data = (dbc_val > cut_threshold)
                else:
                    mask_data = slice(None) # 全选

                # 2. 确定画什么变量 (Mass 还是 Score)
                if "mass" in ptype:
                    val_to_plot = ak8_sdmass
                else:
                    val_to_plot = score_val

                # 3. 填充
                self._fill(plot_id, "Data", val_to_plot[mask_data], weights[mask_data], bins)
                continue # 跳过后续的 MC 逻辑
            
            if ptype == "score_by_true":
                for cat_name, mask in masks_true_cat.items():
                    self._fill(plot_id, cat_name, score_val[mask], weights[mask], bins)

            elif ptype == "mass_by_true":
                for cat_name, mask in masks_true_cat.items():
                    self._fill(plot_id, cat_name, ak8_sdmass[mask], weights[mask], bins)

            elif ptype == "mass_by_catalog":
                self._fill(plot_id, sample_group, ak8_sdmass, weights, bins)

            elif ptype == "score_by_catalog_cut":
                cut_threshold = settings["cut_val"]
                mask_cut = (dbc_val > cut_threshold)
                self._fill(plot_id, sample_group, score_val[mask_cut], weights[mask_cut], bins)

            elif ptype == "score_by_true_cut":
                cut_threshold = settings["cut_val"]
                mask_cut = (dbc_val > cut_threshold)
                for cat_name, mask_cat in masks_true_cat.items():
                    final_mask = mask_cat & mask_cut
                    self._fill(plot_id, cat_name, score_val[final_mask], weights[final_mask], bins)

            # === [NEW] 混合模式: 信号按组，背景按True Cat ===
            elif ptype == "score_by_true_mixed":
                # 如果当前文件属于信号组 (在 Config 中定义，如 Wcb)
                if sample_group in self.cfg.signal_groups:
                    # 信号作为一个整体填充，Key 为 "Wcb"
                    self._fill(plot_id, sample_group, score_val, weights, bins)
                else:
                    # 背景按 True Category 拆分填充，Key 为 "Cat_..."
                    for cat_name, mask in masks_true_cat.items():
                        self._fill(plot_id, cat_name, score_val[mask], weights[mask], bins)
                        # ... (接在 elif ptype == "score_by_true_mixed": ... 的块后面)

            # === [NEW] 混合模式 + Cut ===
            elif ptype == "score_by_true_mixed_cut":
                cut_threshold = settings["cut_val"]
                mask_cut = (dbc_val > cut_threshold)
                
                # 如果当前文件属于信号组 (如 Wcb)
                if sample_group in self.cfg.signal_groups:
                    # 信号作为一个整体填充，应用 Cut
                    self._fill(plot_id, sample_group, score_val[mask_cut], weights[mask_cut], bins)
                else:
                    # 背景按 True Category 拆分填充，应用 Cut
                    for cat_name, mask_cat in masks_true_cat.items():
                        final_mask = mask_cat & mask_cut
                        self._fill(plot_id, cat_name, score_val[final_mask], weights[final_mask], bins)

        del events_cache, weights, ak8_sdmass, score_val, masks_true_cat, dbc_val

    def _fill(self, plot_id, group, values, weights, bins):
        if len(values) == 0: return
        v_np = ak.to_numpy(values)
        w_np = ak.to_numpy(weights)
        counts, _ = np.histogram(v_np, bins=bins, weights=w_np)
        if group not in self.hist_data[plot_id]:
            self.hist_data[plot_id][group] = counts
        else:
            self.hist_data[plot_id][group] += counts

    def run(self):
        mc_files = [f for f in os.listdir(self.cfg.mc_path) if f.endswith('.root')]
        print(f"[INFO] Found {len(mc_files)} MC files.")
        
        for i, fname in enumerate(mc_files):
            shortname = fname.replace("_tree.root", "")
            group = "Other"
            for cat, keywords in self.cfg.catalog.items():
                if any(k in shortname for k in keywords):
                    group = cat
                    break
            
            if i % 1 == 0:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                print(f"\r[PROG] Processing MC {i+1}/{len(mc_files)} | Mem: {mem:.1f} MB", end="")
            
            self.process_file(os.path.join(self.cfg.mc_path, fname), is_data=False, sample_group=group)
            
            if i % 10 == 0: gc.collect()
        print("")

        if self.cfg.enable_data:
            data_files = [f for f in os.listdir(self.cfg.data_path) if f.endswith('.root')]
            print(f"[INFO] Found {len(data_files)} Data files.")
            for i, fname in enumerate(data_files):
                self.process_file(os.path.join(self.cfg.data_path, fname), is_data=True, sample_group="Data")
                if i % 10 == 0: gc.collect()

        return self.hist_data

# ======================================================================================
# 3. Plotter
# ======================================================================================

class Plotter:
    def __init__(self, config):
        self.cfg = config
        hep.style.use("CMS")

    def get_label(self, key):
        return self.cfg.labels.get(key, key)

    def plot_all(self, hist_data):
        if not os.path.exists(self.cfg.figure_path):
            os.makedirs(self.cfg.figure_path)

        for plot_id, settings in self.cfg.plots.items():
            self.draw_single_plot(plot_id, settings, hist_data[plot_id])

    def draw_single_plot(self, plot_id, settings, groups_data):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        bins = settings["bins"]
        ssf = settings.get("ssf", 1)
        
        bkg_hists = []
        bkg_labels = []
        bkg_colors = []
        
        sig_hists = [] 
        data_hist = None
        
        all_keys = sorted(groups_data.keys())
        
        for key in all_keys:
            counts = groups_data[key]
            
            if key == "Data":
                data_hist = counts
                continue
            
            # === [MODIFIED] 判断信号逻辑 ===
            # 只要 Key 在 signal_groups 中，就视为信号（画 Step 线）
            # 混合模式下，信号 Key 是 "Wcb"，背景 Key 是 "Cat_..."
            # 原有模式下，如果画 Catalog，信号 Key 也是 "Wcb"
            # 如果画 True Cat，信号文件会被拆成 "Cat_..."，此时 Key 不在 signal_groups，会被归为背景堆叠（符合预期）
            is_signal = (key in self.cfg.signal_groups)
            
            display_label = self.get_label(key)

            if is_signal:
                sig_hists.append((counts, display_label, self.cfg.colors.get(key, "blue")))
            else:
                bkg_hists.append(counts)
                bkg_labels.append(display_label)
                bkg_colors.append(self.cfg.colors.get(key, None))

        if bkg_hists:
            hep.histplot(bkg_hists, bins=bins, stack=True, histtype='fill',
                         label=bkg_labels, color=bkg_colors, ax=ax)

        for counts, label, color in sig_hists:
            hep.histplot(counts * ssf, bins=bins, label=f"{label} $\\times {ssf}$",
                         histtype='step', linewidth=2, color=color, ax=ax)

        if self.cfg.enable_data and data_hist is not None:
            hep.histplot(data_hist, bins=bins, yerr=True, histtype='errorbar',
                         color='black', label=self.get_label('Data'), marker='o', ax=ax)

        hep.cms.label("Preliminary", data=self.cfg.enable_data, lumi=self.cfg.lumi, ax=ax)
        ax.set_xlabel(settings.get("xlabel", ""))
        ax.set_ylabel(settings.get("ylabel", "Events"))
        ax.set_xlim(settings.get("xlim", (bins[0], bins[-1])))
        
        if settings.get("logy", False):
            ax.set_yscale("log")
            ax.set_ylim(bottom=settings.get("ylim_bottom", 0.1))
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 50)
        else:
             ymin, ymax = ax.get_ylim()
             ax.set_ylim(ymin, ymax * 1.4)

        ax.legend(ncol=2, loc='upper right')
        
        outname = os.path.join(self.cfg.figure_path, f"{plot_id}.pdf")
        plt.savefig(outname)
        print(f"[PLOT] Saved {outname}")
        plt.close(fig)

if __name__ == "__main__":
    start_time = time.time()
    cfg = Config()
    
    processor = Processor(cfg)
    hist_data = processor.run()
    
    plotter = Plotter(cfg)
    plotter.plot_all(hist_data)
    
    print(f"\n[DONE] Total time: {time.time() - start_time:.2f} s")
