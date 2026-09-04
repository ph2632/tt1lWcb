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

# ======================================================================================
# 1. Configuration
# ======================================================================================

class Config:
    def __init__(self):
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_/")
        self.data_path = os.path.join(self.base_path, "data/")
        
        self.figure_path = "./figures/"
        # Parquet 路径保持为 flatten 版本
        self.parquet_path = "./score_parquet_flatten/"
        
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
            "Plot1_Score_TrueCat_Flat": {
                "type": "score_by_true",
                "bins": np.linspace(0, 1, 51),
                "xlabel": "Discriminator Score ($S_{wqq} / \Sigma S$)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            "Plot2_Mass_TrueCat_Flat": {
                "type": "mass_by_true",
                "bins": np.linspace(0, 250, 26),
                "xlabel": "AK8 Soft Drop Mass [GeV]",
                "ylabel": "Events / 10 GeV",
                "logy": False,
                "ssf": 1000,
                "xlim": (0, 250),
            },
            "Plot3_Mass_Catalog": {
                "type": "mass_by_catalog_Flat",
                "bins": np.linspace(0, 250, 26),
                "xlabel": "AK8 Soft Drop Mass [GeV]",
                "ylabel": "Events / 10 GeV",
                "logy": False,
                "ssf": 1000,
                "xlim": (0, 250),
            },
            "Plot4_Score_Catalog_Cut_Flat": {
                "type": "score_by_catalog_cut",
                "bins": np.linspace(0, 1, 51),
                "cut_val": 0.7,
                "xlabel": "Discriminator Score ($S_{wqq} / \Sigma S$)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            "Plot5_Score_TrueCat_Cut_Flat": {
                "type": "score_by_true_cut",
                "bins": np.linspace(0, 1, 51),
                "cut_val": 0.7,
                "xlabel": "Discriminator Score ($S_{wqq} / \Sigma S$)",
                "ylabel": "Events / 0.02",
                "logy": False,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
            },
            "Plot6_Score_Mixed_Flat": {
                "type": "score_by_true_mixed",
                "bins": np.linspace(0, 1, 51),
                "xlabel": "Score (Bkg: True Cat, Sig: Wcb)",
                "ylabel": "Events / 0.02",
                "logy": True,
                "ssf": 1000,
                "xlim": (0, 1),
                "ylim_bottom": 0.1,
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

    def process_file(self, root_filepath, is_data=False, sample_group="Data"):
        filename = os.path.basename(root_filepath)
        parquet_name = filename.replace(".root", ".parquet")
        parquet_path = os.path.join(self.cfg.parquet_path, parquet_name)
        
        events_cache = None

        # --- 1. 读取或生成 Parquet (保存 Jagged Array) ---
        if self.cfg.use_cache and not self.cfg.force_reload and os.path.exists(parquet_path):
            try:
                events_cache = ak.from_parquet(parquet_path)
            except Exception as e:
                print(f"[WARN] Corrupt parquet {parquet_name}, reloading from ROOT.")
        
        if events_cache is None:
            try:
                with uproot.open(root_filepath) as f:
                    tree = f[self.cfg.tree_name]
                    raw_events = tree.arrays(self.cfg.branches, library="ak")
            except Exception as e:
                print(f"[WARN] Cannot read ROOT {root_filepath}: {e}")
                return

            if len(raw_events) == 0: return

            # 读取 Jagged Array
            ak8_sdmass = raw_events["ak8_sdmass"]
            ak8_type   = raw_events["ak8_type"]
            n_c        = raw_events["ak8_n_c_in_jet"]
            
            # Event 级别的变量
            fname_check = filename.lower()
            is_qcd_file = any(k in fname_check for k in ["qcd", "wjet", "zjet"])
            is_qcd = np.full(len(raw_events), is_qcd_file, dtype=bool)

            if is_data:
                weights = np.ones(len(raw_events), dtype=np.float32)
                is_qcd = np.zeros(len(raw_events), dtype=bool) 
            else:
                weights = (raw_events["xsecWeight"] * raw_events["genWeight"] * raw_events["lumiwgt"] * 
                           raw_events["puWeight"] * raw_events["trigEffWeight"] * raw_events["l1PreFiringWeight"])
                weights = np.array(weights, dtype=np.float32)

            # Score 计算
            def safe_get(key):
                return ak.fill_none(raw_events[key], 0)

            s_w_qq    = safe_get("score_cata_w_qq")
            s_qcd     = safe_get("score_cata_qcd")
            s_top_bqq = safe_get("score_cata_top_bqq")
            s_top_bc  = safe_get("score_cata_top_bc")
            s_top_bq  = safe_get("score_cata_top_bq")
            s_non     = safe_get("score_cata_non")
            
            denom = s_w_qq + s_qcd + s_top_bqq + s_top_bc + s_top_bq + s_non + 1e-10
            score_val = s_w_qq / denom

            # Score 2 (Dbc)
            g_bc   = safe_get("ak8_gpt_bc")
            g_bb   = safe_get("ak8_gpt_bb")
            g_cc   = safe_get("ak8_gpt_cc")
            g_qcd  = safe_get("ak8_gpt_qcd")
            g_bs   = safe_get("ak8_gpt_bs")
            g_qq   = safe_get("ak8_gpt_qq")
            g_cs   = safe_get("ak8_gpt_cs")
            g_topbw= safe_get("ak8_gpt_topbw")
            
            denom_dbc = g_bc + g_qcd + g_cc + g_bb + g_bs + g_cs + g_qq + g_topbw + 1e-10
            dbc_val = g_bc / denom_dbc
            
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
        
        # --- 2. 准备画图数据 (Flattening) ---
        j_sdmass = events_cache["ak8_sdmass"]
        
        # 统计每个 Event 有多少个 Jet
        counts = ak.num(j_sdmass)
        
        # [修正] 使用 np.repeat 替代 ak.repeat
        # 必须先转换为 numpy array 才能使用 np.repeat
        counts_np = ak.to_numpy(counts)
        weights_np = ak.to_numpy(events_cache["weights"])
        is_qcd_np  = ak.to_numpy(events_cache["is_qcd"])
        
        # 广播 Event 变量
        weights_flat = np.repeat(weights_np, counts_np)
        is_qcd_flat  = np.repeat(is_qcd_np, counts_np)
        
        # 将结果转回 Awkward Array，以保证与后续 ak8_type_flat (Awkward Array) 的兼容性
        weights_flat = ak.Array(weights_flat)
        is_qcd_flat  = ak.Array(is_qcd_flat)
        
        # 展平 Jet 变量
        ak8_sdmass_flat = ak.flatten(j_sdmass)
        score_val_flat  = ak.flatten(events_cache["score"])
        dbc_val_flat    = ak.flatten(events_cache["dbc"])
        ak8_type_flat   = ak.flatten(events_cache["ak8_type"])
        n_c_flat        = ak.flatten(events_cache["n_c"])
        
        if len(ak8_sdmass_flat) == 0:
            return

        # --- 3. 定义 Mask (基于展平后的数据) ---
        mask_qcd     = is_qcd_flat
        mask_top_bqq = (ak8_type_flat == 4) & (~mask_qcd)
        mask_top_bc  = (ak8_type_flat == 2) & (n_c_flat == 1) & (~mask_qcd)
        mask_top_bq  = (ak8_type_flat == 2) & (n_c_flat == 0) & (~mask_qcd)
        mask_w_qq    = (ak8_type_flat == 1) & (~mask_qcd)
        mask_other   = ~(mask_qcd | mask_top_bqq | mask_top_bc | mask_top_bq | mask_w_qq)

        masks_true_cat = {
            "Cat_QCD": mask_qcd,
            "Cat_Top_bqq": mask_top_bqq,
            "Cat_Top_bc": mask_top_bc,
            "Cat_Top_bq": mask_top_bq,
            "Cat_W_qq": mask_w_qq,
            "Cat_Other": mask_other
        }

        # --- 4. 填充直方图 ---
        for plot_id, settings in self.cfg.plots.items():
            ptype = settings["type"]
            bins = settings["bins"]

            if ptype == "score_by_true":
                for cat_name, mask in masks_true_cat.items():
                    self._fill(plot_id, cat_name, score_val_flat[mask], weights_flat[mask], bins)

            elif ptype == "mass_by_true":
                for cat_name, mask in masks_true_cat.items():
                    self._fill(plot_id, cat_name, ak8_sdmass_flat[mask], weights_flat[mask], bins)

            elif ptype == "mass_by_catalog":
                self._fill(plot_id, sample_group, ak8_sdmass_flat, weights_flat, bins)

            elif ptype == "score_by_catalog_cut":
                cut_threshold = settings["cut_val"]
                mask_cut = (dbc_val_flat > cut_threshold)
                self._fill(plot_id, sample_group, score_val_flat[mask_cut], weights_flat[mask_cut], bins)

            elif ptype == "score_by_true_cut":
                cut_threshold = settings["cut_val"]
                mask_cut = (dbc_val_flat > cut_threshold)
                for cat_name, mask_cat in masks_true_cat.items():
                    final_mask = mask_cat & mask_cut
                    self._fill(plot_id, cat_name, score_val_flat[final_mask], weights_flat[final_mask], bins)

            elif ptype == "score_by_true_mixed":
                if sample_group in self.cfg.signal_groups:
                    self._fill(plot_id, sample_group, score_val_flat, weights_flat, bins)
                else:
                    for cat_name, mask in masks_true_cat.items():
                        self._fill(plot_id, cat_name, score_val_flat[mask], weights_flat[mask], bins)

        del events_cache, weights_flat, ak8_sdmass_flat, score_val_flat, masks_true_cat, dbc_val_flat

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
        print(f"[INFO] Parquet output path: {self.cfg.parquet_path}")
        
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
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
        
        outname = os.path.join(self.cfg.figure_path, f"{plot_id}_Flatten.pdf")
        plt.savefig(outname)
        print(f"[PLOT] Saved {outname}")
        plt.close(fig)

if __name__ == "__main__":
    start_time = time.time()
    cfg = Config()
    
    print(f"Parquet files will be saved to: {cfg.parquet_path}")
    
    processor = Processor(cfg)
    hist_data = processor.run()
    
    plotter = Plotter(cfg)
    plotter.plot_all(hist_data)
    
    print(f"\n[DONE] Total time: {time.time() - start_time:.2f} s")
