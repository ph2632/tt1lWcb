#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
import time
import functools
import psutil

# decorator
def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"[TIMEIT] Function '{func.__name__}' executed in {end - start:.4f} seconds | Memory usage: {mem_mb:.2f} MB")
        return result
    return wrapper

# for plotting
class plotTemplate:
    #@timeit
    def __init__(self):
        self.mc_files = {}
        self.data_files = {}
        self.signal_dict = [
            "ttbar-powheg",
        ]
        self.luminosity = 41.479    # [pb^-1]
        self.enableHLTmask=True
        self.enableStackData=True
        self.mc_branches = [
                "xsecWeight","genWeight","PSWeight","puWeight",
                "trigEffWeight","l1PreFiringWeight",
                "lumiwgt",
                "n_ak8", "n_ak4", 
                "ak8_sdmass","ak8_pt","ak8_eta","ak8_phi",
                "ak8_gpt_bc","ak8_gpt_bb","ak8_gpt_cc","ak8_gpt_qcd",
                "ak8_gpt_bs","ak8_gpt_qq","ak8_gpt_cs",
                # "ak4_pn_b", "ak4_pn_c",
                "ak4_pt",#"ak4_eta","ak4_phi","ak4_mass","ak4_tag",
                "lep1_pt",#"lep1_eta","lep1_phi","lep1_mass",
                "lep1_pdgId",
                "genW_pt",
                "ak8jet_type", "ak8jet_n_b_in_jet", "ak8jet_n_c_in_jet", "ak8jet_n_in_jet",
                "ak8jet_is_wbc",
                # "LOMatch_ttWcb","LOMatch_wqq","LOMatch_wcq","LOMatch_top_bq",
                # "LOMatch_top_bc","LOMatch_top_bqq","LOMatch_top_bcq","LOMatch_non",
                # "ak8jet_match_wqq",
                # "ak8jet_match_wcq",
                # "ak8jet_match_top_bq",
                # "ak8jet_match_top_bc",
                # "ak8jet_match_top_bqq",
                # "ak8jet_match_top_bcq",
                # "ak8jet_match_non",
                # "ak8jet_match_ttWcb",
                "passTrigEl","passTrigMu",
        ]
        self.data_branches = [
                "n_ak8", "n_ak4", 
                #"ak4_pn_b", "ak4_pn_c",
                "ak4_pt",#"ak4_eta","ak4_phi","ak4_mass","ak4_tag",
                "ak8_sdmass","ak8_pt","ak8_eta","ak8_phi",
                "lep1_pt",#"lep1_eta","lep1_phi","lep1_mass",
                "lep1_pdgId",
                "ak8_gpt_bc","ak8_gpt_bb","ak8_gpt_cc","ak8_gpt_qcd",
                "ak8_gpt_bs","ak8_gpt_qq","ak8_gpt_cs",
                "passTrigEl","passTrigMu",
        ]
        self.tree_name = "Events"
        self.colors = [ '#DAA520', '#CD5C5C', '#8FBC8F', '#4682B4', '#6A5ACD', '#F0E68C', '#FF69B4', '#40E0D0' , '#D2691E', '#9ACD32' , '#FF7F50', '#6495ED' , '#DC143C' ]
    def get_paths(self, pathdir):
        # mc_pathdir = pathdir + "mc-T_Trig-F_JMC"
        mc_pathdir = pathdir + "MC/"
        data_pathdir = pathdir + "data/"
        mc_file_list = [os.path.join(mc_pathdir, f) for f in os.listdir(mc_pathdir) if f.endswith('.root')]
        data_file_list = [os.path.join(data_pathdir, f) for f in os.listdir(data_pathdir) if f.endswith('.root')]
        # use file name as dictionary keys
        mc_files = {os.path.basename(f).replace('_tree.root', ''): f for f in mc_file_list}
        data_files = {os.path.basename(f).replace('_tree.root', ''): f for f in data_file_list}
        self.mc_files = mc_files
        self.data_files = data_files
        return mc_files, data_files
    def load_data(self, file_path, tree_name, branches):
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            data = tree.arrays(branches, library="ak")
        return data
    def load_data_and_parquet(self, file_path, tree_name, branches, output_path):
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            data = tree.arrays(branches, library="ak")
        # save to parquet
        name=os.path.basename(file_path).replace('_tree.root','')+".parquet"
        print(os.path.join(output_path, name))
        ak.to_parquet(data, os.path.join(output_path, name))
        return data
    def load_parquet(self, parquet_path):
        data = ak.from_parquet(parquet_path)
        return data
    
    #@timeit
    def getData(self):
        mc_files = self.mc_files
        data_files = self.data_files
        mc_data = {}
        print(f"Loading parquet file for MC:", end="")
        for key, path in mc_files.items():
            try:
                if os.path.exists("./parquet/"+os.path.basename(path).replace('_tree.root','')+".parquet"):
                    print(f" {key}, ", end="")
                    if key in ["tt-semi", "tt-lep", "tt-had"]:
                        if "tt" not in mc_data:
                            mc_data["tt"] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_tree.root','')+".parquet")
                        else:
                            # Concatenate the entire awkward arrays
                            mc_data["tt"] = ak.concatenate([mc_data["tt"], self.load_parquet("./parquet/"+os.path.basename(path).replace('_tree.root','')+".parquet")])
                        continue
                    mc_data[key] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_tree.root','')+".parquet")
                else:
                    mc_data[key] = self.load_data_and_parquet(path, self.tree_name, self.mc_branches, "./parquet/")
            except Exception as e:
                raise RuntimeError(f"Error loading MC file {path}: {e}")
        data_data = {}
        print(f"\nLoading parquet file for Data:", end="")
        for key, path in data_files.items():
            try:
                if os.path.exists("./parquet/"+os.path.basename(path).replace('_tree.root','')+".parquet"):
                    print(f" {key}, ", end="")
                    data_data[key] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_tree.root','')+".parquet")
                else:
                    data_data[key] = self.load_data_and_parquet(path, self.tree_name, self.data_branches, "./parquet/")
            except Exception as e:
                raise RuntimeError(f"Error loading Data file {path}: {e}")
        self.mc_data = mc_data
        self.data_data = data_data
        return mc_data, data_data

    # @timeit
    def get_L_value(self, inputAkw, key, rank=0, refKey="ak8_pt"):
        data=inputAkw[key]
        refdata=inputAkw[refKey]
        if rank > 0:
            #[XXX]
            sorted_indice=ak.argsort(refdata, axis=1, ascending=False)
            selected_mask=sorted_indice == rank
        else:
            selected_mask=ak.argmax(refdata, keepdims=True, axis=1) 
        values=np.array(ak.fill_none(ak.flatten(data[selected_mask]), 1e-8))
        return values
    #@timeit
    def get_value(self, inputAkw, key):
        values = np.array(inputAkw[key])
        return values
    
    def flatten_value(self, inputAkw, key, weights):
        data = inputAkw[key]
        flat_data = ak.flatten(data)
        counts = ak.num(data)
        repeated_weights = np.repeat(weights, counts)
        return np.array(flat_data), np.array(repeated_weights)

    #@timeit
    def getHLTWeight(self, inputAkw):
        if self.enableHLTmask==False:
            return np.ones(len(inputAkw["passTrigEl"]))
        trigger_mask = ak.any([inputAkw["passTrigEl"], inputAkw["passTrigMu"]], axis=0)
        trigger_mask_int = np.asarray(trigger_mask, dtype=int)
        return trigger_mask_int
    
    # @timeit
    def getScore(self, datakey):
        sbc=self.get_L_value(datakey, "ak8_gpt_bc", rank=0)
        sbb=self.get_L_value(datakey, "ak8_gpt_bb", rank=0)
        scc=self.get_L_value(datakey, "ak8_gpt_cc", rank=0)
        sqcd=self.get_L_value(datakey, "ak8_gpt_qcd", rank=0)
        sbs=self.get_L_value(datakey, "ak8_gpt_bs", rank=0)
        sqq=self.get_L_value(datakey, "ak8_gpt_qq", rank=0)
        scs=self.get_L_value(datakey, "ak8_gpt_cs", rank=0)
        Dbc=sbc/(sbc+sqcd+scc+sbb+sbs+scs+sqq)
        return Dbc
    
    # @timeit
    def QuickMaskWeight(self, datakey, refBranch="ak8_sdmass", mass_min=0, mass_max=600):
        sdmass=self.get_L_value(datakey, refBranch, rank=0)
        quickmask = (sdmass > mass_min) & (sdmass < mass_max)
        quickmask_int = np.asarray(quickmask, dtype=int)
        print(f"\r[DEBUG] QuickMask applied: {mass_min} < {refBranch} < {mass_max} , length={len(quickmask_int)}", end="\r")
        return quickmask_int

    def calculate_significance(self, xlabel, bins_plot, figure_path="./figures/", figure_name="significance_dbc_cut.pdf"):
        
        # calculate data counts
        print(f"\n[INFO] [PLOTER] Plotting variable: {xlabel}")
        if self.enableStackData:
            counts_all = None
        else:
            counts_all = []
        for key in self.data_data.keys():
            cut = self.QuickMaskWeight(self.data_data[key])
            counts, bins = np.histogram(self.getScore(self.data_data[key]), bins=bins_plot, weights=cut)
            if self.enableStackData:
                counts_all = counts if counts_all is None else counts_all + counts
            else:
                counts_all.extend(counts)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        data_counts = counts_all if self.enableStackData else np.array(counts_all)

        # calculate mc counts (signal and background)
        mc_bkg_counts=[]
        cut=1
        for key in self.mc_data.keys():
            default_weights=self.mc_data[key]["xsecWeight"]*self.mc_data[key]["genWeight"]*self.mc_data[key]["lumiwgt"]\
                *self.mc_data[key]["puWeight"]*self.mc_data[key]["trigEffWeight"]*self.mc_data[key]["l1PreFiringWeight"]

            cut = self.QuickMaskWeight(self.mc_data[key])
            hist_values = self.getScore(self.mc_data[key])
            weights = default_weights*cut
            
            if key in self.signal_dict:
                mc_counts_signal, bins_signal = np.histogram(hist_values, bins=bins_plot, weights=weights)
                continue
            counts, bins = np.histogram(hist_values, bins=bins_plot, weights=weights)
            mc_bkg_counts.append(counts)
        # stack mc background
        for i, counts in enumerate(mc_bkg_counts):
            if i == 0:
                mc_bkg_stack = counts
            else:
                mc_bkg_stack = mc_bkg_stack + counts

        # print(f"[Debug] background : {mc_bkg_stack[-10:]}")
        # print(f"[Debug] signal     : {mc_counts_signal[-10:]}")
        # calculate significance in different Dbc cut
        print(f"\n[INFO] Calculating significance for different Dbc cuts...")
        binwidth = bins_plot[1] - bins_plot[0]
        dbc_cuts_list = np.linspace(0.05, 0.95, 51)
        significance_list = []
        for dbc_cut in dbc_cuts_list:
            # signal counts
            sig_mask = bin_centers >= dbc_cut
            # print(f"[Debug] Dbc cut: {dbc_cut}, sig_mask sum: {np.sum(sig_mask)}")
            # print(f"[Debug] sig_mask: {sig_mask}")
            S = np.sum(mc_counts_signal[sig_mask])
            # background counts
            B = np.sum(mc_bkg_stack[sig_mask])
            # print(f"[Debug] Signal counts S: {S}")
            # print(f"[Debug] Background counts B: {B}")
            if B > 0:
                significance = S / np.sqrt(B)
            else:
                significance = 0
            significance_list.append(significance)
        # plot significance vs Dbc cut
        style_set = {'color':'black', 'marker':'o', 'markersize':8, 'linestyle':'-', 'linewidth':2}

        hep.style.use("CMS")
        hep.cms.label("Preliminary(2017)", data=True)
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        ax.plot(dbc_cuts_list,
                significance_list,
                label="Significance",
                **style_set
                )
        ax.set_xlabel("$D_{bc}$ Cut", fontsize=20)
        ax.set_ylabel("Significance (S/$\sqrt{B}$)", fontsize=20)
        ax.set_xlim(0,1)
        ax.set_ylim(0, max(significance_list)*1.2)
        hep.style.use("CMS")
        hep.cms.label("Preliminary(2017)", data=True, lumi=41.479)
        fig.tight_layout()
        plt.savefig(os.path.join(figure_path, figure_name))
        plt.close()
        
        # lumiratio = 137.14/41.479
        # fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        # ax.plot(dbc_cuts_list,
        #         np.array(significance_list)*np.sqrt(lumiratio),
        #         label="Significance",
        #         **style_set
        #         )
        # ax.set_xlabel("$D_{bc}$ Cut", fontsize=20)
        # ax.set_xlim(0,1)
        # ax.set_ylabel("Significance (S/$\sqrt{B}$)", fontsize=20)
        # hep.style.use("CMS")
        # hep.cms.label("Preliminary (Run2)", data=True, lumi=137.14)
        # fig.tight_layout()
        # plt.savefig(os.path.join(figure_path, figure_name.replace('.pdf','_lumi_Run2.pdf')))
        # plt.close()



def __main__():
    pathdir = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
    analysis = plotTemplate()
    mc_files, data_files = analysis.get_paths(pathdir)

    print("[INFO] Paths retrieved.")
    print("MC files:", mc_files)
    print("Data files:", data_files)
    analysis.getData()
    print("\n[INFO] Data loaded.")
    figure_output_path = "./figures/"

    analysis.calculate_significance(
        xlabel="Score Dbc",
        bins_plot=np.linspace(0, 1, 101),
        figure_path=figure_output_path,
        figure_name="significance_dbc_cut_2017.pdf",
    )


    print(f"\n[SUMMARY] All plots saved to {figure_output_path} .")


if __name__ == "__main__":
    __main__()