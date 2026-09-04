#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import awkward as ak
import mplhep as hep
import time
import functools
import psutil
from train_bdt.dbc_tools import DbcEvaluator
dbc_eval = DbcEvaluator(
    mode="bdt",
    model_path="./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"
)
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
            "Wcb",
        ]
        self.luminosity = 41.479    # [pb^-1]
        self.signal_scale_factor=1000
        self.enableHLTmask=True
        self.enableStackData=True
        self.mc_branches = [
                "xsecWeight","genWeight","PSWeight","puWeight",
                "trigEffWeight","l1PreFiringWeight",
                "lumiwgt",
                "n_ak8", "n_ak4", 
                "ak8_sdmass","ak8_pt","ak8_eta","ak8_phi",
                "ak8_gpt_bc","ak8_gpt_bb","ak8_gpt_cc","ak8_gpt_qcd",
                "ak8_gpt_bs","ak8_gpt_qq","ak8_gpt_cs","ak8_gpt_topbw",
                # "ak4_pn_b", "ak4_pn_c",
                "ak4_pt",#"ak4_eta","ak4_phi","ak4_mass","ak4_tag",
                "lep1_pt",#"lep1_eta","lep1_phi","lep1_mass",
                "lep1_pdgId",
                "genW_pt",
                ["ak8jet_type","ak8_type"], ["ak8jet_n_b_in_jet","ak8_n_b_in_jet"], ["ak8jet_n_c_in_jet","ak8_n_c_in_jet"], ["ak8jet_n_in_jet","ak8_n_c_in_jet"],
                ["ak8jet_is_wbc","ak8_is_wbc"],
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
                "ak8_gpt_bs","ak8_gpt_qq","ak8_gpt_cs","ak8_gpt_topbw",
                "passTrigEl","passTrigMu",
        ]
        self.tree_name = "Events"
        self.colors = [ '#DAA520', '#CD5C5C', '#8FBC8F', '#4682B4', '#6A5ACD', '#F0E68C', '#FF69B4', '#40E0D0' , '#D2691E', '#9ACD32' , '#FF7F50', '#6495ED' , '#DC143C' ]
        self.colors_map ={
            "tt": '#DAA520',
            "QCD": '#CD5C5C',
            "single-top": '#4682B4',
        }
        self.catalog = {
            "Wcb": ["ttbar-powheg"],
            "single-top": ["single-top"],
            "tt":  ["tt-semi", "tt-lep", "tt-had"],
            "Rare": ["ttbb", "ttW", "ttZ", "twZ", "ttHToTauTau", "ttHNonbb", "ttbb"],
            "Diboson": ["WW", "WZ", "ZZ"],
            "QCD": ["QCD"],
        }
    def get_paths(self, pathdir):
        # mc_pathdir = pathdir + "mc-T_Trig-F_JMC"
        # mc_pathdir = pathdir + "MC/"
        mc_pathdir = pathdir + "MC/scored_samples_1merged_/"
        data_pathdir = pathdir + "Data/"
        mc_file_list = [os.path.join(mc_pathdir, f) for f in os.listdir(mc_pathdir) if f.endswith('.root')]
        data_file_list = [os.path.join(data_pathdir, f) for f in os.listdir(data_pathdir) if f.endswith('.root')]
        # use file name as dictionary keys
        # mc_files = {os.path.basename(f).replace('_merged.root', ''): f for f in mc_file_list}
        # data_files = {os.path.basename(f).replace('_merged.root', ''): f for f in data_file_list}
        mc_files = {os.path.basename(f).replace('_merged.root', ''): f for f in mc_file_list}
        data_files = {os.path.basename(f).replace('_merged.root', ''): f for f in data_file_list}
        self.mc_files = mc_files
        self.data_files = data_files
        return mc_files, data_files
    def load_data(self, file_path, tree_name, branches):
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            data = tree.arrays(branches, library="ak")
        return data
    def load_data_and_parquet(self, file_path, tree_name, branches_coll, output_path):
        branches = []
        branches_bak = []
        diff_name_dict = {}
        for item in branches_coll:
            # try to use the first element if it's a list
            if isinstance(item, list):
                branches.append(item[0])
                branches_bak.append(item[1])
                diff_name_dict[item[1]] = item[0]
            else:
                branches.append(item)
                branches_bak.append(item)
        try:
            with uproot.open(file_path) as file:
                tree = file[tree_name]
                data = tree.arrays(branches, library="ak")
        except:
            print(f"[WARNING] Some branches are missing, trying backup branches.")
            try:
                with uproot.open(file_path) as file:
                    tree = file[tree_name]
                    data = tree.arrays(branches_bak, library="ak")
                    # modify data keys to original branches
                    for old_name, new_name in diff_name_dict.items():
                        data[new_name] = data[old_name]
                        del data[old_name]
            except Exception as e2:
                print(f"Error loading backup branches: {branches_bak} from {file_path}: {e2}")
                raise
        # save to parquet
        name=os.path.basename(file_path).replace('_merged.root','')+".parquet"
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
                if os.path.exists("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet"):
                    print(f" {key}, ", end="")
                    # Use self.catalog to group MC samples
                    for group, keys in self.catalog.items():
                        if key in keys:
                            if group not in mc_data:
                                mc_data[group] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet")
                            else:
                                mc_data[group] = ak.concatenate([mc_data[group], self.load_parquet("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet")])
                            break
                    else:
                        mc_data[key] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet")
                    # mc_data[key] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet")
                else:
                    mc_data[key] = self.load_data_and_parquet(path, self.tree_name, self.mc_branches, "./parquet/")
            except Exception as e:
                raise RuntimeError(f"Error loading MC file {path}: {e}")
        data_data = {}
        print(f"\nLoading parquet file for Data:", end="")
        for key, path in data_files.items():
            try:
                if os.path.exists("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet"):
                    print(f" {key}, ", end="")
                    data_data[key] = self.load_parquet("./parquet/"+os.path.basename(path).replace('_merged.root','')+".parquet")
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
        sbqq=self.get_L_value(datakey, "ak8_gpt_topbw", rank=0)
        # Dbc=sbc/(sbc+sqcd+scc+sbb+sbs+scs+sqq+sbqq)
        # sbc vs w non-bc
        # Dbc = sbc/(sbc+scs+sqq+sbqq)
        Dbc = dbc_eval.get_Dbc(sbc, sbb, scc, sqcd, sbs, sqq, scs, sbqq)
        return Dbc
    
    # @timeit
    def QuickMaskWeight(self, datakey, refBranch="ak8_sdmass", mass_min=0, mass_max=600):
        sdmass=self.get_L_value(datakey, refBranch, rank=0)
        quickmask = (sdmass > mass_min) & (sdmass < mass_max)
        quickmask_int = np.asarray(quickmask, dtype=int)
        print(f"\r[DEBUG] QuickMask applied: {mass_min} < {refBranch} < {mass_max} , length={len(quickmask_int)}", end="\r")
        return quickmask_int
    # def QuickMaskWeight(self, datakey, refBranch="lep1_pdgId", mass_min=-12, mass_max=12):
    #     # lepid=self.get_L_value(datakey, refBranch, rank=0)
    #     lepid=datakey[refBranch]
    #     quickmask = (lepid > mass_min) & (lepid < mass_max)
    #     quickmask_int = np.asarray(quickmask, dtype=int)
    #     print(f"\r[DEBUG] QuickMask applied: {mass_min} < {refBranch} < {mass_max} , length={len(quickmask_int)}", end="\r")
    #     # print(quickmask_int[:20])
    #     print(f" effective entries: {np.sum(quickmask_int)} / {len(quickmask_int)} ", end="\r")
    #     return quickmask_int
    #@timeit
    def plot_variable(self, xlabel, xlabeltext, bins_plot, figure_name, figure_path="./", signal_scale_factor=1000, type='normal'):
        hep.style.use("CMS")
        hep.cms.label("Preliminary(2017)", data=True)
        if signal_scale_factor != self.signal_scale_factor:
            self.signal_scale_factor = signal_scale_factor
        fig, axs = plt.subplots(2, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax = axs[0]
        # plot data
        print(f"\n[INFO] [PLOTER] Plotting variable: {xlabel}")
        if self.enableStackData:
            counts_all = None
        else:
            counts_all = []
        for key in self.data_data.keys():
            if type=='normal':
                # cut = self.QuickMaskWeight(self.data_data[key])
                counts, bins = np.histogram(self.get_value(self.data_data[key], xlabel), bins=bins_plot)
            elif type=='ak8Flat':
                cut = self.QuickMaskWeight(self.data_data[key])
                data , weights = self.flatten_value(self.data_data[key], xlabel, cut)
                counts, bins = np.histogram(data, bins=bins_plot, weights=weights)
            elif type=='ak8L':
                cut = self.QuickMaskWeight(self.data_data[key])
                counts, bins = np.histogram(self.get_L_value(self.data_data[key], xlabel), bins=bins_plot , weights=cut)
                # counts, bins = np.histogram(self.get_L_value(self.data_data[key], xlabel), bins=bins_plot) #, weights=cut)
            elif type=='ak4L':
                cut = self.QuickMaskWeight(self.data_data[key])
                counts, bins = np.histogram(self.get_L_value(self.data_data[key], xlabel, refKey="ak4_pt"), bins=bins_plot , weights=cut)
            elif type=='ak4Flat':
                cut = self.QuickMaskWeight(self.data_data[key])
                data , weights = self.flatten_value(self.data_data[key], xlabel, cut)
                counts, bins = np.histogram(data, bins=bins_plot, weights=weights)
            elif type=='Dbc':
                cut = self.QuickMaskWeight(self.data_data[key])
                counts, bins = np.histogram(self.getScore(self.data_data[key]), bins=bins_plot, weights=cut)
                # counts, bins = np.histogram(self.getScore(self.data_data[key]), bins=bins_plot)#, weights=cut)
            
            if self.enableStackData:
                counts_all = counts if counts_all is None else counts_all + counts
            else:
                counts_all.extend(counts)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        ax.plot(
            bin_centers,
            counts_all,
            label=f"data",
            color="black",
            marker="o",
            linestyle="None"
        )

        # plot mc and signal
        hist_values_list=[]
        key_list=[]
        color_list=[]
        weights_list=[]
        # mc counts
        mc_counts_list = []

        # color_dict = { key: self.colors[i % len(self.colors)]  for i, key in enumerate(self.mc_data.keys()) }
        # color definition with preference
        color_dict = {}
        used_colors = set(self.colors_map.values())
        unused_colors = [c for c in self.colors if c not in used_colors]
        unused_idx = 0
        for key in self.mc_data.keys():
            if key in self.colors_map:
                color_dict[key] = self.colors_map[key]
            else:
                if unused_idx < len(unused_colors):
                    color_dict[key] = unused_colors[unused_idx]
                    unused_idx += 1
                else:
                    color_dict[key] = self.colors[unused_idx % len(self.colors)]
                    unused_idx += 1

        cut=1
        for key in self.mc_data.keys():
            default_weights=self.mc_data[key]["xsecWeight"]*self.mc_data[key]["genWeight"]*self.mc_data[key]["lumiwgt"]\
                *self.mc_data[key]["puWeight"]*self.mc_data[key]["trigEffWeight"]*self.mc_data[key]["l1PreFiringWeight"]

            if type=='normal':
                # cut = self.QuickMaskWeight(self.mc_data[key])
                hist_values = self.get_value(self.mc_data[key], xlabel)
                weights = default_weights * cut
            elif type=='ak4L':
                hist_values = self.get_L_value(self.mc_data[key], xlabel, refKey="ak4_pt")
                cut = self.QuickMaskWeight(self.mc_data[key])
                weights = default_weights*cut
            elif type=='ak4Flat':
                print(f"[DEBUG] {key} ak4Flat values loading.")
                weights = default_weights
                hist_values , weights = self.flatten_value(self.mc_data[key], xlabel, self.QuickMaskWeight(self.mc_data[key])*weights)
            elif type=='ak8L':
                hist_values = self.get_L_value(self.mc_data[key], xlabel)
                cut = self.QuickMaskWeight(self.mc_data[key])
                weights = default_weights*cut
            elif type=='ak8Flat':
                print(f"[DEBUG] {key} ak8Flat values loading.")
                weights = default_weights
                hist_values , weights = self.flatten_value(self.mc_data[key], xlabel, self.QuickMaskWeight(self.mc_data[key])*weights)
            elif type=='Dbc':
                cut = self.QuickMaskWeight(self.mc_data[key])
                hist_values = self.getScore(self.mc_data[key])
                weights = default_weights*cut
                print(f"\r[DEBUG] {key} Dbc values loaded.", end="\r")
            
            if key in self.signal_dict:
                ax.hist(
                    hist_values,
                    bins=bins_plot,
                    label=key+f"$\\times {signal_scale_factor}$ (signal)",
                    histtype='step',
                    linewidth=2,
                    stacked=False,
                    color="blue",
                    weights=weights*signal_scale_factor,
                )
                mc_counts_list.append(np.histogram(hist_values, bins=bins_plot, weights=weights)[0])
                continue
            hist_values_list.append(hist_values)
            key_list.append(key)
            color_list.append(color_dict[key])
            weights_list.append(weights)
            mc_counts_list.append(np.histogram(hist_values, bins=bins_plot, weights=weights)[0])
        
        ax.hist(
            hist_values_list,
            bins=bins_plot,
            label=key_list,
            stacked=True,
            color=color_list,
            weights=weights_list,
        )
       
        ax.set_ylabel("Events")
        ax.legend(ncol=2)
        if xlabel=="Dbc":
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax*2)
        origin_xtick=ax.get_xticklabels()

        ax.set_xlim(bins_plot[0], bins_plot[-1])
        ax_ratio = axs[1]
        
        # plot ratio
        mc_count = np.sum(mc_counts_list, axis=0)
        ratio = counts_all / (mc_count + 1e-8)
        ax_ratio.plot(
            bin_centers,
            ratio,
            label="Data/Pred",
            color="black",
            marker="o",
            linestyle="None"
        )
        ax_ratio.axhline(1, color='red', linestyle='--')
        ax_ratio.set_ylabel("Data/Pred")
        ax_ratio.set_ylim(0.5, 1.5)
    
        print(origin_xtick)
        ax_ratio.set_xlabel(xlabeltext)
        # ax_ratio.set_xlim(bins_plot[0], bins_plot[-1])  
        # ax.set_xticklabels([])
        hep.cms.label("Preliminary(2017)", data=True, ax=ax, lumi=41.479)
        fig.tight_layout()
        plt.savefig(os.path.join(figure_path, figure_name+".pdf"))
        plt.close(fig)


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

    plot_dict = {
        # xlabel     : [xmin,  xmax, nbins,        xlabeltext,                      figure_name,            ssf,  type   ],
        "lep1_pt"    : [  30,   300,    50, "Lepton $p_{T}$ [GeV]",        "lepton_pt_distribution_2017",  1000, 'normal'],
        "ak8_pt"     : [   200,   800,    30, "AK8 Leading $p_{T}$ [GeV]",        "ak8_pt_distribution_2017",     1000, 'ak8L' ],
        "lep1_pdgId" : [ -15,   15,    30, "Lepton PDG ID",                   "lepton_pdgId_distribution_2017",     100, 'normal' ],
        "n_ak8"      : [   0,    5,    5, "Number of AK8 jets",               "n_ak8_distribution_2017",             1000, 'normal' ],
        "n_ak4"     : [   2,   11,    9, "Number of AK4 jets",               "n_ak4_distribution_2017",            1000, 'normal' ],
        # "ak4_pn_b" : [   0,    1,    20, "AK4 Leading b-tagging prob.",      "ak4_pn_b_distribution_2017",          100, 'ak4L' ],
        # "ak4_pn_c" : [   0,    1,    20, "AK4 Leading c-tagging prob.",      "ak4_pn_c_distribution_2017",          100, 'ak4L' ],
        "ak4_pt"    : [   80,   500,    42, "AK4 Leading $p_{T}$ [GeV]",        "ak4_pt_distribution_2017",     1000, 'ak4L' ],
        # "ak4_eta"   : [-2.5,   2.5,    30, "AK4 Leading $\eta$",               "ak4_eta_distribution_2017",    1000, 'ak4L' ],
        # "ak4_phi"   : [-3.2,   3.2,    35, "AK4 Leading $\phi$",               "ak4_phi_distribution_2017",    1000, 'ak4L' ],
        # "ak4_mass"  : [   0,   200,    40, "AK4 Leading Mass [GeV]",           "ak4_mass_distribution_2017",   1000, 'ak4L' ],
        "ak8_sdmass" : [   0,   200,    20, "AK8 Leading Soft Drop Mass [GeV]", "ak8_sdmass_distribution_2017", 1000, 'ak8L' ],
        "ak8_eta"    : [-2.5,   2.5,    30, "AK8 Leading $\eta$",               "ak8_eta_distribution_2017",    1000, 'ak8L' ],
        "ak8_phi"    : [-3.2,   3.2,    35, "AK8 Leading $\phi$",               "ak8_phi_distribution_2017",    1000, 'ak8L' ],
        "Dbc"        : [   0,     1,    50, "D$_{bc}$",                    "Dbc_distribution_2017",        1000, 'Dbc'   ],
    }
    plot_dict_alias = {
        # xlabel     : [xmin,  xmax, nbins,        xlabeltext,                      figure_name,            ssf,  type   ],
        "ak8_pt"     : [   200,   800,    30, "AK8 cadidate $p_{T}$ [GeV]",        "ak8_pt_flatten_distribution_2017",     1000, 'ak8Flat' ],
        "ak4_pt"    : [   30,   300,    50, "AK4 cadidate $p_{T}$ [GeV]",        "ak4_pt_flatten_distribution_2017",     1000, 'ak4Flat' ],
    
    }
    
    for key in plot_dict:
        analysis.plot_variable(
            xlabel=key,
            xlabeltext=plot_dict[key][3],
            bins_plot=np.linspace(plot_dict[key][0], plot_dict[key][1], plot_dict[key][2]+1),
            figure_name=plot_dict[key][4],
            figure_path=figure_output_path,
            signal_scale_factor=plot_dict[key][5],
            type=plot_dict[key][6],
        )
        print(f"[INFO] Plotted {key} Done .", end=" "*10)
    
    for key in plot_dict_alias:
        analysis.plot_variable(
            xlabel=key,
            xlabeltext=plot_dict_alias[key][3],
            bins_plot=np.linspace(plot_dict_alias[key][0], plot_dict_alias[key][1], plot_dict_alias[key][2]+1),
            figure_name=plot_dict_alias[key][4],
            figure_path=figure_output_path,
            signal_scale_factor=plot_dict_alias[key][5],
            type=plot_dict_alias[key][6],
        )
        print(f"[INFO] Plotted {key} Done .", end=" "*10)

    print(f"\n[SUMMARY] All plots saved to {figure_output_path} .")


if __name__ == "__main__":
    __main__()