#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python

import os
import re
import gc
import time
import shutil
import argparse
import psutil
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")          # batch: only write PNGs, never open a window
import matplotlib.pyplot as plt
import mplhep as hep

from sklearn.metrics import roc_curve, auc
from train_bdt.dbc_tools import DbcEvaluator


# ======================================================================================
# 1. Configuration
# ======================================================================================

class Config:
    def __init__(self):
        # ------------------------------------------------------------------
        # Paths
        # ------------------------------------------------------------------
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        self.mc_path = os.path.join(self.base_path, "MC/scored_samples_1merged_/")
        self.data_path = os.path.join(self.base_path, "Data/scored_samples_1merged_/")

        # If the real Data dir is not reachable (e.g. running off youpeng's AFS),
        # fall back to ./fake_data/ : empty *.root stubs whose names resolve onto
        # the existing SingleElectron/SingleMuon parquet caches.
        try:
            os.listdir(self.data_path)
        except OSError:
            _stub = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_data")
            if os.path.isdir(_stub):
                print(f"[WARN] {self.data_path} not readable -> using stub {_stub}")
                self.data_path = _stub

        self.tree_name = "Events"

        self.figure_path = "."
        # ./cache_batch_parquet is a symlink onto youpeng's read-only EOS, so a
        # new cache_tag (below) cannot be written there. Point the cache at a
        # writable area; override with $WCB_CACHE_DIR.
        self.cache_path = os.environ.get(
            "WCB_CACHE_DIR", "/eos/user/a/agapitos/wcb_cache/"
        )

        # 修改变量定义、BDT 模型、cache 内容后请改 cache_tag
        # v5: + leading-AK8 substructure / GloParT raw scores / angular vars
        #     for the MAT (Wcb vs Top-bc) jet-kinematics scan -> full re-derive.
        # v6: + ak8_match_wqq (W->qq' light) for the 5th MAT overlay line.
        self.cache_tag = "derived_v6_matkin_wqq_v1"

        # ------------------------------------------------------------------
        # Switches
        # ------------------------------------------------------------------
        self.enable_data = True
        self.use_cache = True
        self.force_reload = False

        self.lumi = 41.479

        # ------------------------------------------------------------------
        # BDT Dbc
        # ------------------------------------------------------------------
        self.dbc_model_path = "./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"

        # ------------------------------------------------------------------
        # Sample catalog
        # ------------------------------------------------------------------
        self.signal_groups = ["Wcb"]

        self.catalog_sample = {
            "Wcb": ["ttbar-powheg"],
            "single-top": ["single-top"],
            "tt": ["tt-semi", "tt-lep", "tt-had"],
            "Rare": ["ttbb", "ttW", "ttZ", "twZ", "ttHToTauTau", "ttHNonbb"],
            "Diboson": ["WW", "WZ", "ZZ"],
            "QCD": ["QCD"],
        }

        # ------------------------------------------------------------------
        # True-category integer code
        # Do not store strings in awkward cache.
        # ------------------------------------------------------------------
        self.true_cat_codes = {
            "Cat_QCD": 0,
            "Cat_Top_bqq": 1,
            "Cat_Top_bc": 2,
            "Cat_Top_bq": 3,
            "Cat_W_qq": 4,
            "Cat_Other": 5,
        }

        self.true_cat_names = {
            0: "Cat_QCD",
            1: "Cat_Top_bqq",
            2: "Cat_Top_bc",
            3: "Cat_Top_bq",
            4: "Cat_W_qq",
            5: "Cat_Other",
        }

        self.true_cat_order = [
            "Cat_QCD",
            "Cat_Top_bqq",
            "Cat_Top_bc",
            "Cat_Top_bq",
            "Cat_W_qq",
            "Cat_Other",
        ]

        # ------------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------------
        self.labels = {
            "Wcb": r"$W_{cb}$",
            "tt": r"$t\bar{t}$",
            "QCD": "QCD multijet",
            "single-top": "Single top",
            "Rare": "Rare",
            "Diboson": "Diboson",
            "Other": "Other",

            "Cat_QCD": "QCD/V+Jets",
            "Cat_Top_bqq": r"Top ($bqq'$)",
            "Cat_Top_bc": r"Top ($bc$)",
            "Cat_Top_bq": r"Top ($bq$)",
            "Cat_W_qq": r"$W \to qq'$",
            "Cat_Other": "Other",
            "Rest": "Rest (all other jets)",

            "Data": "Data 2017",

            "score_Dbc": r"$D_{bc}$",
            "score_SC": r"$S_{\mathrm{EVT}}$",

            "score_cata_w_qq_norm": r"EC $W \to qq'$ score",
            "score_cata_qcd_norm": "EC QCD score",
            "score_cata_top_bqq_norm": r"EC Top $bqq'$ score",
            "score_cata_top_bc_norm": r"EC Top $bc$ score",
            "score_cata_top_bq_norm": r"EC Top $bq$ score",
            "score_cata_non_norm": "EventClassifier none-of-the-above score",
        }

        # ------------------------------------------------------------------
        # CMS-like colors
        # ------------------------------------------------------------------
        self.colors_sample = {
            "tt": "#5790FC",
            "QCD": "#F89C20",
            "single-top": "#E42536",
            "Rare": "#964A8B",
            "Diboson": "#7A21DD",
            "Wcb": "#0B3603",
            "Other": "#999999",
        }

        self.colors_true = {
            "Cat_QCD": "#F89C20",
            "Cat_Top_bqq": "#E42536",
            "Cat_Top_bc": "#964A8B",
            "Cat_Top_bq": "#5790FC",
            "Cat_W_qq": "#7A21DD",
            "Cat_Other": "#9C9CA1",
            "Wcb": "#0B3603",
            "Rest": "#5A5A5A",
        }

        # ------------------------------------------------------------------
        # EventClassifier branches
        # ------------------------------------------------------------------
        self.ec_score_branches = [
            "score_cata_w_qq",
            "score_cata_qcd",
            "score_cata_top_bqq",
            "score_cata_top_bc",
            "score_cata_top_bq",
            "score_cata_non",
        ]

        self.ec_score_norm_branches = [
            "score_cata_w_qq_norm",
            "score_cata_qcd_norm",
            "score_cata_top_bqq_norm",
            "score_cata_top_bc_norm",
            "score_cata_top_bq_norm",
            "score_cata_non_norm",
        ]

        self.ec_label_map = {
            "score_cata_w_qq": "W(qq)",
            "score_cata_qcd": "QCD",
            "score_cata_top_bqq": "Top(bqq)",
            "score_cata_top_bc": "Top(bc)",
            "score_cata_top_bq": "Top(bq)",
            "score_cata_non": "Non",
        }

        self.ec_truth_map = {
            "score_cata_w_qq": "Cat_W_qq",
            "score_cata_qcd": "Cat_QCD",
            "score_cata_top_bqq": "Cat_Top_bqq",
            "score_cata_top_bc": "Cat_Top_bc",
            "score_cata_top_bq": "Cat_Top_bq",
            "score_cata_non": "Cat_Other",
        }

        # ------------------------------------------------------------------
        # MET branch candidates
        # ------------------------------------------------------------------
        self.met_branch_candidates = [
            ("MET_pt", "MET_phi"),
            ("met_pt", "met_phi"),
            ("met", "met_phi"),
            ("pfMET_pt", "pfMET_phi"),
            ("PuppiMET_pt", "PuppiMET_phi"),
            ("METFixEE2017_pt", "METFixEE2017_phi"),
        ]

        # ------------------------------------------------------------------
        # Branches
        # ------------------------------------------------------------------
        self.weight_branches = [
            "xsecWeight",
            "genWeight",
            "lumiwgt",
            "puWeight",
            "trigEffWeight",
            "l1PreFiringWeight",
            "flavTagWeight",
            "elEffWeight",
            "muEffWeight",
            "topptWeight",
        ]

        self.common_branches = [
            # event-level
            "n_ak8",
            "n_ak4",
            "lep1_pt",
            "lep1_eta",
            "lep1_phi",
            "lep1_mass",
            "lep1_pdgId",

            # jagged jets
            "ak8_sdmass",
            "ak8_pt",
            "ak8_eta",
            "ak8_phi",
            "ak4_pt",
            "ak4_eta",
            "ak4_phi",

            # event classifier scores
            "score_cata_w_qq",
            "score_cata_qcd",
            "score_cata_top_bqq",
            "score_cata_top_bc",
            "score_cata_top_bq",
            "score_cata_non",

            # boosted Dbc inputs
            "ak8_gpt_bc",
            "ak8_gpt_bb",
            "ak8_gpt_cc",
            "ak8_gpt_qcd",
            "ak8_gpt_bs",
            "ak8_gpt_qq",
            "ak8_gpt_cs",
            "ak8_gpt_topbw",

            # extra GloParT raw scores (MAT jet-kinematics study)
            "ak8_gpt_bqq",
            "ak8_gpt_topw",

            # leading-AK8 substructure (MAT jet-kinematics study)
            "ak8_rawFactor",
            "ak8_tau21",
            "ak8_tau32",
            "ak8_nConstituents",

            # leading-AK4 (MAT jet-kinematics study)
            "ak4_mass",

            # reconstructed leptonic-W / vector boson
            "v_pt",
            "v_eta",
            "v_phi",
            "v_mass",

            # event activity / flavour multiplicities
            "ht",
            "ht_b",
            "ht_c",
            "ht_bc",
            "n_btag",
            "n_ctag",
            "n_btagM",
            "n_btagT",
            "n_ctagM",
            "n_ctagT",

            # dR / invariant-mass observables built from tagged AK4 jets
            "minDR_b",
            "mass_minDR_b",
            "maxMass_b",
            "minDR_c",
            "mass_minDR_c",
            "maxMass_c",
            "minDR_bc",
            "mass_minDR_bc",
            "maxMass_bc",

            # trigger
            "passTrigEl",
            "passTrigMu",
        ]

        self.mc_only_branches = [
            "ak8_type",
            "ak8_n_c_in_jet",
            "ak8_n_b_in_jet",
            "ak8_n_in_jet",
            # truth-match flags for the MAT overlay (W->cb / W->cq / W->qq split)
            "ak8_is_wbc",
            "ak8_match_wcq",
            "ak8_match_wqq",
        ]

        # ------------------------------------------------------------------
        # Default plot options
        # ------------------------------------------------------------------
        self.default_plot = {
            "ylabel": "Events",
            "logy": False,
            "signal_scale": 1000,
            "cut": "1",
            "ratio": True,
            "xlim": None,
            "ylim": None,
            "ylim_bottom": 0.1,
            "var_mode": "event",  # event, leading, flatten
        }


# ======================================================================================
# 2. Utilities
# ======================================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_leading(arr, default=-999):
    try:
        return ak.fill_none(ak.firsts(arr, axis=1), default)
    except Exception:
        return ak.fill_none(arr, default)


def get_sample_group(filename, catalog):
    shortname = os.path.basename(filename)
    shortname = shortname.replace("_tree.root", "")
    shortname = shortname.replace("_merged.root", "")

    for group, keys in catalog.items():
        if any(k in shortname for k in keys):
            return group

    return "Other"


def normalize_cut_expr(expr):
    """
    Normalize simple cut expression.

    Supported:
      ak8_pt[0] > 200
      (ak8_pt[0] > 200) and (score_Dbc > 0.985)
      (ak8_pt[0] > 200) && (score_Dbc > 0.985)
      (ak8_sdmass[0] > 50) and (ak8_sdmass[0] < 120)

    Internally convert:
      branch[0] -> branch_0
      and / &&  -> &
      or  / ||  -> |

    Important:
      For compound cuts, each comparison should be wrapped by parentheses.
      Good:
        (ak8_pt[0] > 200) and (score_Dbc > 0.985)
      Bad:
        ak8_pt[0] > 200 and score_Dbc > 0.985
    """

    if expr is None:
        return "1"

    expr = str(expr).strip()

    if expr == "":
        return "1"

    # branch[0] -> branch_0
    expr = re.sub(r"([A-Za-z0-9_]+)\[0\]", r"\1_0", expr)

    # Normalize logical operators for numpy arrays.
    # Must use bitwise operators for array masks.
    expr = expr.replace("&&", "&")
    expr = expr.replace("||", "|")

    # Replace standalone and/or, not inside variable names.
    expr = re.sub(r"\band\b", "&", expr)
    expr = re.sub(r"\bor\b", "|", expr)

    return expr



def eval_cut(expr, arrays):
    """
    Evaluate event-level simple cut.

    Supported examples:
      "1"
      "ak8_pt[0] > 200"
      "ak8_pt_0 > 200"
      "(ak8_pt[0] > 200) and (score_Dbc > 0.985)"
      "(ak8_sdmass[0] > 50) and (ak8_sdmass[0] < 120)"
      "abs(ak8_eta[0]) < 2.4"

    Notes:
      - Cuts are event-level only.
      - Jagged branches are not directly exposed to eval.
      - For branch[0], a leading alias branch_0 is created automatically.
    """

    expr = normalize_cut_expr(expr)

    n = len(arrays["weights"])

    if expr == "1":
        return np.ones(n, dtype=bool)

    local_dict = {}

    # ------------------------------------------------------------------
    # 1. Add all event-level flat fields
    # ------------------------------------------------------------------
    for key in arrays.fields:
        try:
            val = arrays[key]

            # Skip jagged arrays in direct eval.
            # They are handled via *_0 aliases below.
            if isinstance(val, ak.Array):
                try:
                    if ak.ndim(val) > 1:
                        continue
                except Exception:
                    pass

            local_dict[key] = ak.to_numpy(val)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. Automatically create leading aliases from jagged branches
    #    e.g. ak8_pt -> ak8_pt_0
    # ------------------------------------------------------------------
    leading_candidates = [
        "ak8_pt",
        "ak8_eta",
        "ak8_phi",
        "ak8_sdmass",
        "ak4_pt",
        "ak4_eta",
        "ak4_phi",
    ]

    for base in leading_candidates:
        alias = f"{base}_0"

        # If alias already exists, keep it.
        if alias in local_dict:
            continue

        # If jagged base exists, create alias.
        if base in arrays.fields:
            try:
                local_dict[alias] = ak.to_numpy(get_leading(arrays[base], -999))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 3. Debug helper if variable missing
    # ------------------------------------------------------------------
    allowed_globals = {
        "__builtins__": {},
        "np": np,
        "abs": np.abs,
        "sqrt": np.sqrt,
    }

    try:
        mask = eval(expr, allowed_globals, local_dict)
    except NameError as e:
        print("\n[ERROR] Failed to evaluate cut.")
        print(f"[ERROR] Cut expression after normalization: {expr}")
        print("[ERROR] Available variables in local_dict:")
        for k in sorted(local_dict.keys()):
            try:
                print(f"  {k:<30} shape={np.shape(local_dict[k])}")
            except Exception:
                print(f"  {k}")
        raise RuntimeError(f"Failed to evaluate cut: {expr}\nError: {e}")

    except Exception as e:
        raise RuntimeError(f"Failed to evaluate cut: {expr}\nError: {e}")

    if isinstance(mask, (int, float, bool)):
        return np.ones(n, dtype=bool) if bool(mask) else np.zeros(n, dtype=bool)

    mask = np.asarray(mask, dtype=bool)

    if len(mask) != n:
        raise RuntimeError(
            f"Cut mask length mismatch: len(mask)={len(mask)}, n_events={n}, expr={expr}"
        )

    return mask



def safe_name(text):
    text = str(text)
    text = text.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace(":", "")
    text = text.replace(",", "_")
    text = text.replace(".", "p")
    text = text.replace("$", "")
    text = text.replace("\\", "")
    text = text.replace("'", "")
    text = text.replace(">", "gt")
    text = text.replace("<", "lt")
    return text


def existing_branches(tree, requested):
    available = set(tree.keys())
    return [b for b in requested if b in available]


def find_met_branches(tree, candidates):
    available = set(tree.keys())

    for met_pt, met_phi in candidates:
        if met_pt in available and met_phi in available:
            return met_pt, met_phi

    return None, None


def numpy_float_array(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


def repeat_event_weights_for_jagged(jagged, event_weights):
    counts = ak.to_numpy(ak.num(jagged, axis=1))
    return np.repeat(event_weights, counts)


def flatten_jagged(jagged):
    return ak.to_numpy(ak.flatten(jagged, axis=None))


# Sentinel value used for missing / undefined quantities in the derived array
# (leading jet absent, mTW when no MET, ...). Kept out of the histograms.
SENTINEL = -999.0
SENTINEL_CUT = -998.0


def hist_with_flow(values, bins, weights=None):
    """np.histogram, ROOT-``UnderOverFlow1D`` style: entries strictly below the
    first edge are added to the first visible bin and entries strictly above the
    last edge to the last visible bin, so nothing is lost off the axis (matches
    the behaviour in Root_plot.py).

    Returns ``(counts, variance)`` where ``variance`` is the per-bin sum of
    squared weights (Poisson ``N`` when unweighted) with the same
    under/overflow folding applied -- i.e. the flow bin's statistical
    uncertainty is carried into the first/last visible bin in quadrature,
    exactly like ROOT's ``SetBinError(1, sqrt(e1**2 + e0**2))``.

    No double counting: ``np.histogram`` puts values on the left edge in bin 0
    and values on the right edge in the last bin, and the flow masks are
    strict (`< lo`, `> hi`), so every entry is counted once and
    ``counts.sum() == len(values)`` (or ``weights.sum()``).

    Values are assumed already cleaned of NaN/inf and of the SENTINEL fill.
    """
    values = np.asarray(values)

    counts, _ = np.histogram(values, bins=bins, weights=weights)
    counts = counts.astype(np.float64)

    if weights is None:
        variance = counts.copy()                       # Poisson: Var = N
    else:
        weights = np.asarray(weights, dtype=np.float64)
        variance, _ = np.histogram(values, bins=bins, weights=weights ** 2)
        variance = variance.astype(np.float64)

    if values.size == 0:
        return counts, variance

    lo, hi = bins[0], bins[-1]
    under = values < lo
    over = values > hi          # values == hi already land in the last bin

    if weights is None:
        n_u = float(np.count_nonzero(under))
        n_o = float(np.count_nonzero(over))
        counts[0] += n_u
        counts[-1] += n_o
        variance[0] += n_u
        variance[-1] += n_o
    else:
        counts[0] += weights[under].sum()
        counts[-1] += weights[over].sum()
        variance[0] += (weights[under] ** 2).sum()
        variance[-1] += (weights[over] ** 2).sum()

    return counts, variance


# ======================================================================================
# 3. Data manager
# ======================================================================================

class DataManager:
    def __init__(self, cfg):
        self.cfg = cfg
        ensure_dir(cfg.cache_path)

        self.dbc_eval = DbcEvaluator(
            mode="bdt",
            model_path=self.cfg.dbc_model_path,
        )

    def load_all(self):
        samples = []

        mc_files = [
            os.path.join(self.cfg.mc_path, f)
            for f in os.listdir(self.cfg.mc_path)
            if f.endswith(".root")
        ]

        print(f"[INFO] Found {len(mc_files)} MC files in {self.cfg.mc_path}")

        for i, path in enumerate(mc_files):
            group = get_sample_group(path, self.cfg.catalog_sample)

            print(
                f"\r[LOAD] MC {i + 1}/{len(mc_files)} | "
                f"group={group:<10} | Mem={get_memory_mb():.1f} MB",
                end="",
            )

            arr = self.load_one_file(path, is_data=False, sample_group=group)

            if arr is not None:
                samples.append({
                    "name": os.path.basename(path),
                    "group": group,
                    "is_data": False,
                    "array": arr,
                })

            if i % 10 == 0:
                gc.collect()

        print("")

        if self.cfg.enable_data:
            if not os.path.exists(self.cfg.data_path):
                print(f"[WARN] Data path does not exist: {self.cfg.data_path}")
            else:
                data_files = [
                    os.path.join(self.cfg.data_path, f)
                    for f in os.listdir(self.cfg.data_path)
                    if f.endswith(".root")
                ]

                print(f"[INFO] Found {len(data_files)} Data files in {self.cfg.data_path}")

                for i, path in enumerate(data_files):
                    print(
                        f"\r[LOAD] Data {i + 1}/{len(data_files)} | "
                        f"Mem={get_memory_mb():.1f} MB",
                        end="",
                    )

                    arr = self.load_one_file(path, is_data=True, sample_group="Data")

                    if arr is not None:
                        samples.append({
                            "name": os.path.basename(path),
                            "group": "Data",
                            "is_data": True,
                            "array": arr,
                        })

                    if i % 10 == 0:
                        gc.collect()

                print("")

        return samples

    def load_one_file(self, root_path, is_data=False, sample_group="Other"):
        basename = os.path.basename(root_path)
        cache_name = basename.replace(".root", f"__{self.cfg.cache_tag}.parquet")
        cache_file = os.path.join(self.cfg.cache_path, cache_name)

        if self.cfg.use_cache and (not self.cfg.force_reload) and os.path.exists(cache_file):
            try:
                return ak.from_parquet(cache_file)
            except Exception as e:
                print(f"\n[WARN] Broken cache: {cache_file}")
                print(f"[WARN] {e}")
                print("[WARN] Reloading from ROOT.")

        try:
            with uproot.open(root_path) as f:
                tree = f[self.cfg.tree_name]

                met_pt_branch, met_phi_branch = find_met_branches(
                    tree,
                    self.cfg.met_branch_candidates,
                )

                branches = list(self.cfg.common_branches)

                if not is_data:
                    branches += self.cfg.weight_branches
                    branches += self.cfg.mc_only_branches

                if met_pt_branch is not None:
                    branches += [met_pt_branch, met_phi_branch]
                else:
                    print(f"\n[WARN] No MET branch found in {basename}. Fill MET=-999.")

                branches = existing_branches(tree, branches)
                raw = tree.arrays(branches, library="ak")

        except Exception as e:
            print(f"\n[WARN] Cannot read {root_path}: {e}")
            return None

        if len(raw) == 0:
            return None

        arr = self.build_derived_array(
            raw=raw,
            is_data=is_data,
            sample_group=sample_group,
            filename=basename,
            met_pt_branch=met_pt_branch,
            met_phi_branch=met_phi_branch,
        )

        if self.cfg.use_cache:
            try:
                ak.to_parquet(arr, cache_file)
            except (OSError, PermissionError) as e:
                print(f"\n[WARN] Could not write cache {cache_file}: {e}")
                print("[WARN] Continuing without caching this file "
                      "(set $WCB_CACHE_DIR to a writable path).")

        return arr

    def build_derived_array(
        self,
        raw,
        is_data=False,
        sample_group="Other",
        filename="",
        met_pt_branch=None,
        met_phi_branch=None,
    ):
        n = len(raw)

        # ------------------------------------------------------------------
        # Weights and truth
        # ------------------------------------------------------------------
        if is_data:
            weights = np.ones(n, dtype=np.float32)
            ak8_type = np.full(n, -1, dtype=np.int32)
            n_b = np.full(n, -1, dtype=np.int32)
            n_c = np.full(n, -1, dtype=np.int32)
            n_in_jet = np.full(n, -1, dtype=np.int32)
            is_qcd = np.zeros(n, dtype=bool)
        else:
            weight_terms = []

            for b in self.cfg.weight_branches:
                if b in raw.fields:
                    weight_terms.append(raw[b])
                else:
                    print(f"\n[WARN] Missing weight branch {b} in {filename}. Use 1.")
                    weight_terms.append(np.ones(n, dtype=np.float32))

            weights = weight_terms[0]
            for w in weight_terms[1:]:
                weights = weights * w
            weights = np.asarray(weights, dtype=np.float32)

            ak8_type = np.asarray(
                get_leading(raw["ak8_type"], -1) if "ak8_type" in raw.fields else np.full(n, -1),
                dtype=np.int32,
            )

            n_c = np.asarray(
                get_leading(raw["ak8_n_c_in_jet"], -1)
                if "ak8_n_c_in_jet" in raw.fields
                else np.full(n, -1),
                dtype=np.int32,
            )

            n_b = np.asarray(
                get_leading(raw["ak8_n_b_in_jet"], -1)
                if "ak8_n_b_in_jet" in raw.fields
                else np.full(n, -1),
                dtype=np.int32,
            )

            n_in_jet = np.asarray(
                get_leading(raw["ak8_n_in_jet"], -1)
                if "ak8_n_in_jet" in raw.fields
                else np.full(n, -1),
                dtype=np.int32,
            )

            group_lower = sample_group.lower()
            fname_lower = filename.lower()
            is_qcd_file = (
                group_lower == "qcd"
                or ("qcd" in fname_lower)
                or ("wjet" in fname_lower)
                or ("zjet" in fname_lower)
            )
            is_qcd = np.full(n, is_qcd_file, dtype=bool)

        # ------------------------------------------------------------------
        # Event-level variables
        # ------------------------------------------------------------------
        n_ak8 = np.asarray(raw["n_ak8"], dtype=np.float32) if "n_ak8" in raw.fields else np.full(n, -999, dtype=np.float32)
        n_ak4 = np.asarray(raw["n_ak4"], dtype=np.float32) if "n_ak4" in raw.fields else np.full(n, -999, dtype=np.float32)

        lep1_pt = np.asarray(raw["lep1_pt"], dtype=np.float32) if "lep1_pt" in raw.fields else np.full(n, -999, dtype=np.float32)
        lep1_eta = np.asarray(raw["lep1_eta"], dtype=np.float32) if "lep1_eta" in raw.fields else np.full(n, -999, dtype=np.float32)
        lep1_phi = np.asarray(raw["lep1_phi"], dtype=np.float32) if "lep1_phi" in raw.fields else np.full(n, -999, dtype=np.float32)
        lep1_mass = np.asarray(raw["lep1_mass"], dtype=np.float32) if "lep1_mass" in raw.fields else np.full(n, -999, dtype=np.float32)
        lep1_pdgId = np.asarray(raw["lep1_pdgId"], dtype=np.float32) if "lep1_pdgId" in raw.fields else np.full(n, -999, dtype=np.float32)

        # ------------------------------------------------------------------
        # Jagged jet branches. Store raw jagged arrays for flatten plots.
        # If missing, use empty jagged arrays.
        # ------------------------------------------------------------------
        empty_jagged = ak.Array([[] for _ in range(n)])

        ak8_pt = raw["ak8_pt"] if "ak8_pt" in raw.fields else empty_jagged
        ak8_eta = raw["ak8_eta"] if "ak8_eta" in raw.fields else empty_jagged
        ak8_phi = raw["ak8_phi"] if "ak8_phi" in raw.fields else empty_jagged
        ak8_sdmass = raw["ak8_sdmass"] if "ak8_sdmass" in raw.fields else empty_jagged

        ak4_pt = raw["ak4_pt"] if "ak4_pt" in raw.fields else empty_jagged
        ak4_eta = raw["ak4_eta"] if "ak4_eta" in raw.fields else empty_jagged
        ak4_phi = raw["ak4_phi"] if "ak4_phi" in raw.fields else empty_jagged

        # ------------------------------------------------------------------
        # Leading variables
        # ------------------------------------------------------------------
        ak8_pt_0 = np.asarray(get_leading(ak8_pt, -999), dtype=np.float32)
        ak8_eta_0 = np.asarray(get_leading(ak8_eta, -999), dtype=np.float32)
        ak8_phi_0 = np.asarray(get_leading(ak8_phi, -999), dtype=np.float32)
        ak8_sdmass_0 = np.asarray(get_leading(ak8_sdmass, -999), dtype=np.float32)

        ak4_pt_0 = np.asarray(get_leading(ak4_pt, -999), dtype=np.float32)
        ak4_eta_0 = np.asarray(get_leading(ak4_eta, -999), dtype=np.float32)
        ak4_phi_0 = np.asarray(get_leading(ak4_phi, -999), dtype=np.float32)

        # ------------------------------------------------------------------
        # MET
        # ------------------------------------------------------------------
        if met_pt_branch is not None and met_pt_branch in raw.fields:
            met = np.asarray(raw[met_pt_branch], dtype=np.float32)
        else:
            met = np.full(n, -999, dtype=np.float32)

        if met_phi_branch is not None and met_phi_branch in raw.fields:
            met_phi = np.asarray(raw[met_phi_branch], dtype=np.float32)
        else:
            met_phi = np.full(n, -999, dtype=np.float32)

        # ------------------------------------------------------------------
        # Score helper
        # ------------------------------------------------------------------
        def leading_or_zero(branch):
            if branch in raw.fields:
                return np.asarray(get_leading(raw[branch], 0), dtype=np.float32)
            return np.zeros(n, dtype=np.float32)

        # ------------------------------------------------------------------
        # EventClassifier scores
        # ------------------------------------------------------------------
        s_w_qq = leading_or_zero("score_cata_w_qq")
        s_qcd = leading_or_zero("score_cata_qcd")
        s_top_bqq = leading_or_zero("score_cata_top_bqq")
        s_top_bc = leading_or_zero("score_cata_top_bc")
        s_top_bq = leading_or_zero("score_cata_top_bq")
        s_non = leading_or_zero("score_cata_non")

        sum_ec = (
            s_w_qq
            + s_qcd
            + s_top_bqq
            + s_top_bc
            + s_top_bq
            + s_non
            + 1e-10
        )

        score_cata_w_qq_norm = s_w_qq / sum_ec
        score_cata_qcd_norm = s_qcd / sum_ec
        score_cata_top_bqq_norm = s_top_bqq / sum_ec
        score_cata_top_bc_norm = s_top_bc / sum_ec
        score_cata_top_bq_norm = s_top_bq / sum_ec
        score_cata_non_norm = s_non / sum_ec

        score_SC = score_cata_w_qq_norm

        # ------------------------------------------------------------------
        # Dbc BDT score
        # ------------------------------------------------------------------
        g_bc = leading_or_zero("ak8_gpt_bc")
        g_bb = leading_or_zero("ak8_gpt_bb")
        g_cc = leading_or_zero("ak8_gpt_cc")
        g_qcd = leading_or_zero("ak8_gpt_qcd")
        g_bs = leading_or_zero("ak8_gpt_bs")
        g_qq = leading_or_zero("ak8_gpt_qq")
        g_cs = leading_or_zero("ak8_gpt_cs")
        g_topbw = leading_or_zero("ak8_gpt_topbw")

        score_Dbc = self.dbc_eval.get_Dbc(
            g_bc,
            g_bb,
            g_cc,
            g_qcd,
            g_bs,
            g_qq,
            g_cs,
            g_topbw,
        )
        score_Dbc = np.asarray(score_Dbc, dtype=np.float32)

        # ------------------------------------------------------------------
        # True category as int8 code
        # ------------------------------------------------------------------
        true_cat = self.make_true_category(ak8_type, n_c, is_qcd)
        is_signal = np.full(n, sample_group in self.cfg.signal_groups, dtype=bool)

        # ------------------------------------------------------------------
        # MAT-overlay category of the leading AK8 jet (truth-matched):
        #   0 = W->cb        (ak8_is_wbc, signal sample only)
        #   1 = W->cq        (ak8_match_wcq, any MC)
        #   2 = t->(cb)      (Cat_Top_bc, any MC)
        #   3 = rest
        #   4 = W->qq' light (ak8_match_wqq, any MC)
        # Priority rest < t(bc) < W->qq' < W->cq < W->cb  (the ak8_match_*
        # flags are mutually exclusive, so priority only matters vs. t(bc)).
        # ------------------------------------------------------------------
        def _lead_flag(branch):
            return np.asarray(
                get_leading(raw[branch], 0) if branch in raw.fields
                else np.zeros(n), dtype=np.int8,
            )

        is_wbc_lead = _lead_flag("ak8_is_wbc")
        match_wcq_lead = _lead_flag("ak8_match_wcq")
        match_wqq_lead = _lead_flag("ak8_match_wqq")

        _code_topbc = self.cfg.true_cat_codes["Cat_Top_bc"]
        mat_cat = np.full(n, 3, dtype=np.int8)
        mat_cat[true_cat == _code_topbc] = 2
        mat_cat[match_wqq_lead == 1] = 4
        mat_cat[match_wcq_lead == 1] = 1
        mat_cat[(is_wbc_lead == 1) & is_signal] = 0

        # ------------------------------------------------------------------
        # Derived: W transverse mass and W mass from lepton + MET
        # ------------------------------------------------------------------
        valid = (lep1_pt > 0) & (met > 0)
        dphi = np.abs(lep1_phi - met_phi)
        dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
        mTW = np.where(valid, np.sqrt(2 * lep1_pt * met * (1 - np.cos(dphi))), -999).astype(np.float32)

        # Neutrino 4-vector: (E=met, px, py, pz=0) assuming eta=0, m=0
        lep_px = lep1_pt * np.cos(lep1_phi)
        lep_py = lep1_pt * np.sin(lep1_phi)
        lep_pz = lep1_pt * np.sinh(lep1_eta)
        lep_E = np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep1_mass**2)
        nu_px = met * np.cos(met_phi)
        nu_py = met * np.sin(met_phi)
        w_mass = np.where(valid,
            np.sqrt(np.maximum((lep_E + met)**2 - (lep_px + nu_px)**2 - (lep_py + nu_py)**2 - lep_pz**2, 0)),
            -999).astype(np.float32)

        # ------------------------------------------------------------------
        # MAT jet-kinematics study: extra pass-through + derived observables
        # for exploring Wcb (Cat_W_qq) vs Top(bc) (Cat_Top_bc) leading-AK8
        # jets. Everything below is leading-AK8 / event-level, so the same
        # `true_cat` (defined from the leading AK8 jet) applies.
        # ------------------------------------------------------------------
        def ev(branch, default=-999.0):
            """Event-level scalar branch, or a `default`-filled column if absent."""
            if branch in raw.fields:
                return np.asarray(raw[branch], dtype=np.float32)
            return np.full(n, default, dtype=np.float32)

        def lead(branch, default=-999.0):
            """Leading element of a jagged branch, or `default` if absent/empty."""
            if branch in raw.fields:
                return np.asarray(get_leading(raw[branch], default), dtype=np.float32)
            return np.full(n, default, dtype=np.float32)

        def dphi_pipi(a, b):
            d = np.abs(a - b)
            return np.where(d > np.pi, 2.0 * np.pi - d, d)

        def safe_ratio(num, den):
            good = den > 0
            return np.where(good, num / np.where(good, den, 1.0), -999.0).astype(np.float32)

        # -- leading-AK8 substructure --------------------------------------
        ak8_rawFactor_0 = lead("ak8_rawFactor")
        ak8_tau21_0     = lead("ak8_tau21")
        ak8_tau32_0     = lead("ak8_tau32")
        ak8_nConst_0    = lead("ak8_nConstituents")
        ak4_mass_0      = lead("ak4_mass")

        # -- leading-AK8 GloParT raw scores. g_bc/g_bb/... (the Dbc inputs)
        #    are already leading values from leading_or_zero() above.
        g_bqq  = leading_or_zero("ak8_gpt_bqq")
        g_topw = leading_or_zero("ak8_gpt_topw")

        gpt_sum = (g_bc + g_bb + g_cc + g_qcd + g_bs + g_qq + g_cs
                   + g_topbw + g_bqq + g_topw + 1e-10)

        # discriminant-style GloParT ratios (W(cb) vs Top(bc) separation)
        gpt_bc_frac  = (g_bc / gpt_sum).astype(np.float32)
        gpt_bc_vs_cc = safe_ratio(g_bc, g_bc + g_cc)
        gpt_bc_vs_bb = safe_ratio(g_bc, g_bc + g_bb)
        gpt_bc_vs_qq = safe_ratio(g_bc, g_bc + g_qq)
        gpt_cs_vs_qq = safe_ratio(g_cs, g_cs + g_qq)
        gpt_hf_w     = ((g_bc + g_cs + g_bs) / gpt_sum).astype(np.float32)
        gpt_top_frac = ((g_topbw + g_topw + g_bqq) / gpt_sum).astype(np.float32)
        gpt_w_vs_top = safe_ratio(g_bc + g_cs + g_qq,
                                  g_bc + g_cs + g_qq + g_topbw + g_topw + g_bqq)

        # -- derived angular variables (leading AK8 vs lepton / MET / AK4) --
        ak8_ok = ak8_pt_0 > 0
        lep_ok = lep1_pt > 0
        ak4_ok = ak4_pt_0 > 0
        met_ok = met > 0

        dphi_lep_ak8 = np.where(ak8_ok & lep_ok, dphi_pipi(lep1_phi, ak8_phi_0), -999).astype(np.float32)
        deta_lep_ak8 = np.where(ak8_ok & lep_ok, np.abs(lep1_eta - ak8_eta_0), -999).astype(np.float32)
        dR_lep_ak8 = np.where(ak8_ok & lep_ok,
                              np.sqrt(np.maximum(deta_lep_ak8, 0)**2 + dphi_lep_ak8**2), -999).astype(np.float32)
        dphi_met_ak8 = np.where(ak8_ok & met_ok, dphi_pipi(met_phi, ak8_phi_0), -999).astype(np.float32)
        dphi_lep_met = np.where(lep_ok & met_ok, dphi_pipi(lep1_phi, met_phi), -999).astype(np.float32)
        dphi_ak4_ak8 = np.where(ak8_ok & ak4_ok, dphi_pipi(ak4_phi_0, ak8_phi_0), -999).astype(np.float32)
        deta_ak4_ak8 = np.where(ak8_ok & ak4_ok, np.abs(ak4_eta_0 - ak8_eta_0), -999).astype(np.float32)
        dR_ak4_ak8 = np.where(ak8_ok & ak4_ok,
                              np.sqrt(np.maximum(deta_ak4_ak8, 0)**2 + dphi_ak4_ak8**2), -999).astype(np.float32)
        ptrel_lep_ak8 = np.where(ak8_ok & lep_ok, lep1_pt / np.where(ak8_ok, ak8_pt_0, 1.0), -999).astype(np.float32)

        # ------------------------------------------------------------------
        # Return awkward array.
        # Important: no object dtype.
        # ------------------------------------------------------------------
        return ak.Array({
            "weights": weights,

            # event variables
            "n_ak8": n_ak8,
            "n_ak4": n_ak4,
            "lep1_pt": lep1_pt,
            "lep1_eta": lep1_eta,
            "lep1_phi": lep1_phi,
            "lep1_pdgId": lep1_pdgId,
            "met": met,
            "met_phi": met_phi,
            "mTW": mTW,
            "w_mass": w_mass,

            # jagged variables for flatten plots
            "ak8_pt": ak8_pt,
            "ak8_eta": ak8_eta,
            "ak8_phi": ak8_phi,
            "ak8_sdmass": ak8_sdmass,

            "ak4_pt": ak4_pt,
            "ak4_eta": ak4_eta,
            "ak4_phi": ak4_phi,

            # leading variables
            "ak8_pt_0": ak8_pt_0,
            "ak8_eta_0": ak8_eta_0,
            "ak8_phi_0": ak8_phi_0,
            "ak8_sdmass_0": ak8_sdmass_0,

            "ak4_pt_0": ak4_pt_0,
            "ak4_eta_0": ak4_eta_0,
            "ak4_phi_0": ak4_phi_0,

            # scores
            "score_Dbc": score_Dbc,
            "score_SC": score_SC,

            "score_cata_w_qq_norm": score_cata_w_qq_norm,
            "score_cata_qcd_norm": score_cata_qcd_norm,
            "score_cata_top_bqq_norm": score_cata_top_bqq_norm,
            "score_cata_top_bc_norm": score_cata_top_bc_norm,
            "score_cata_top_bq_norm": score_cata_top_bq_norm,
            "score_cata_non_norm": score_cata_non_norm,

            # ---- MAT jet-kinematics study ---------------------------------
            # leading-AK8 substructure
            "ak8_rawFactor_0": ak8_rawFactor_0,
            "ak8_tau21_0": ak8_tau21_0,
            "ak8_tau32_0": ak8_tau32_0,
            "ak8_nConst_0": ak8_nConst_0,
            "ak4_mass_0": ak4_mass_0,

            # leading-AK8 GloParT raw scores
            "ak8_gpt_bc_0": g_bc,
            "ak8_gpt_bb_0": g_bb,
            "ak8_gpt_cc_0": g_cc,
            "ak8_gpt_qcd_0": g_qcd,
            "ak8_gpt_bs_0": g_bs,
            "ak8_gpt_cs_0": g_cs,
            "ak8_gpt_qq_0": g_qq,
            "ak8_gpt_bqq_0": g_bqq,
            "ak8_gpt_topbw_0": g_topbw,
            "ak8_gpt_topw_0": g_topw,

            # GloParT discriminant ratios
            "gpt_bc_frac": gpt_bc_frac,
            "gpt_bc_vs_cc": gpt_bc_vs_cc,
            "gpt_bc_vs_bb": gpt_bc_vs_bb,
            "gpt_bc_vs_qq": gpt_bc_vs_qq,
            "gpt_cs_vs_qq": gpt_cs_vs_qq,
            "gpt_hf_w": gpt_hf_w,
            "gpt_top_frac": gpt_top_frac,
            "gpt_w_vs_top": gpt_w_vs_top,

            # reconstructed leptonic-W / vector boson
            "v_pt": ev("v_pt"),
            "v_eta": ev("v_eta"),
            "v_phi": ev("v_phi"),
            "v_mass": ev("v_mass"),

            # event activity / flavour multiplicities
            "ht": ev("ht"),
            "ht_b": ev("ht_b"),
            "ht_c": ev("ht_c"),
            "ht_bc": ev("ht_bc"),
            "n_btag": ev("n_btag"),
            "n_ctag": ev("n_ctag"),
            "n_btagM": ev("n_btagM"),
            "n_btagT": ev("n_btagT"),
            "n_ctagM": ev("n_ctagM"),
            "n_ctagT": ev("n_ctagT"),

            # dR / invariant-mass observables from tagged AK4 jets
            "minDR_b": ev("minDR_b"),
            "mass_minDR_b": ev("mass_minDR_b"),
            "maxMass_b": ev("maxMass_b"),
            "minDR_c": ev("minDR_c"),
            "mass_minDR_c": ev("mass_minDR_c"),
            "maxMass_c": ev("maxMass_c"),
            "minDR_bc": ev("minDR_bc"),
            "mass_minDR_bc": ev("mass_minDR_bc"),
            "maxMass_bc": ev("maxMass_bc"),

            # derived angular variables (leading AK8 vs lepton / MET / AK4)
            "dphi_lep_ak8": dphi_lep_ak8,
            "deta_lep_ak8": deta_lep_ak8,
            "dR_lep_ak8": dR_lep_ak8,
            "dphi_met_ak8": dphi_met_ak8,
            "dphi_lep_met": dphi_lep_met,
            "dphi_ak4_ak8": dphi_ak4_ak8,
            "deta_ak4_ak8": deta_ak4_ak8,
            "dR_ak4_ak8": dR_ak4_ak8,
            "ptrel_lep_ak8": ptrel_lep_ak8,

            # truth and misc
            "ak8_type": ak8_type,
            "ak8_n_c_in_jet": n_c,
            "ak8_n_b_in_jet": n_b,
            "ak8_n_in_jet": n_in_jet,
            "ak8_is_wbc_0": is_wbc_lead,
            "ak8_match_wcq_0": match_wcq_lead,
            "ak8_match_wqq_0": match_wqq_lead,
            "mat_cat": mat_cat,
            "is_qcd": is_qcd,
            "true_cat": true_cat,
            "is_signal": is_signal,
        })

    def make_true_category(self, ak8_type, n_c, is_qcd):
        code = self.cfg.true_cat_codes

        out = np.full(
            len(ak8_type),
            code["Cat_Other"],
            dtype=np.int8,
        )

        mask_qcd = is_qcd
        mask_top_bqq = (ak8_type == 4) & (~mask_qcd)
        mask_top_bc = (ak8_type == 2) & (n_c == 1) & (~mask_qcd)
        mask_top_bq = (ak8_type == 2) & (n_c == 0) & (~mask_qcd)
        mask_w_qq = (ak8_type == 1) & (~mask_qcd)

        out[mask_qcd] = code["Cat_QCD"]
        out[mask_top_bqq] = code["Cat_Top_bqq"]
        out[mask_top_bc] = code["Cat_Top_bc"]
        out[mask_top_bq] = code["Cat_Top_bq"]
        out[mask_w_qq] = code["Cat_W_qq"]

        return out


# ======================================================================================
# 4. Histogrammer
# ======================================================================================

class Histogrammer:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_values_weights_for_plot(self, arr, plot_cfg, is_data):
        var = plot_cfg["var"]
        var_mode = plot_cfg.get("var_mode", "event")
        cut = plot_cfg.get("cut", "1")

        if var not in arr.fields:
            return None, None, None, None

        event_mask = eval_cut(cut, arr)
        event_weights = ak.to_numpy(arr["weights"])[event_mask]

        def _field_event(field):
            """Event-level column sliced by event_mask, or None if absent."""
            if field not in arr.fields:
                return None
            return ak.to_numpy(arr[field])[event_mask]

        if is_data:
            event_weights = np.ones_like(event_weights, dtype=np.float32)

        if var_mode in ["event", "leading"]:
            values = ak.to_numpy(arr[var])[event_mask]
            weights = event_weights

            finite_mask = np.isfinite(values) & (values > SENTINEL_CUT)
            values = values[finite_mask]
            weights = weights[finite_mask]

            true_cat = ak.to_numpy(arr["true_cat"])[event_mask][finite_mask]
            mc = _field_event("mat_cat")
            mat_cat = mc[finite_mask] if mc is not None else None

            return values, weights, true_cat, mat_cat

        elif var_mode == "flatten":
            jagged = arr[var][event_mask]
            values = flatten_jagged(jagged)
            weights = repeat_event_weights_for_jagged(jagged, event_weights)

            counts = ak.to_numpy(ak.num(jagged, axis=1))
            true_cat_event = ak.to_numpy(arr["true_cat"])[event_mask]
            true_cat = np.repeat(true_cat_event, counts)
            mc = _field_event("mat_cat")
            mat_cat = np.repeat(mc, counts) if mc is not None else None

            finite_mask = np.isfinite(values) & (values > SENTINEL_CUT)
            values = values[finite_mask]
            weights = weights[finite_mask]
            true_cat = true_cat[finite_mask]
            if mat_cat is not None:
                mat_cat = mat_cat[finite_mask]

            return values, weights, true_cat, mat_cat

        else:
            raise ValueError(f"Unknown var_mode: {var_mode}")

    def make_histograms(self, samples, plot_cfg):
        """Return ``(hist_data, hist_var)`` -- two dicts keyed the same way.
        ``hist_data[k]`` is the summed bin content, ``hist_var[k]`` the summed
        per-bin variance (sum of w^2), both with under/overflow folded into the
        first/last bin by ``hist_with_flow``. ``sqrt(hist_var[k])`` is the
        stat uncertainty of ``hist_data[k]``.
        """
        mode = plot_cfg.get("mode", "sample")
        bins = plot_cfg["bins"]

        hist_data = {}
        hist_var = {}

        def add(key, values_sel, weights_sel):
            c, v = hist_with_flow(values_sel, bins, weights_sel)
            hist_data[key] = hist_data.get(key, 0) + c
            hist_var[key] = hist_var.get(key, 0) + v

        for sample in samples:
            arr = sample["array"]
            group = sample["group"]
            is_data = sample["is_data"]

            values, weights, true_cat, mat_cat = self.get_values_weights_for_plot(
                arr=arr,
                plot_cfg=plot_cfg,
                is_data=is_data,
            )

            if values is None or len(values) == 0:
                continue

            if is_data:
                add("Data", values, weights)
                continue

            if mode == "sample":
                add(group, values, weights)

            elif mode == "true":
                for key in self.cfg.true_cat_order:
                    m = true_cat == self.cfg.true_cat_codes[key]
                    if np.sum(m) == 0:
                        continue
                    add(key, values[m], weights[m])

            elif mode == "mat":
                # Truth-matched leading-AK8 overlay, driven by `mat_cat`:
                #   0 W->cb (signal)  1 W->cq  2 t->(cb)  3 rest  4 W->qq' light
                if mat_cat is None:
                    continue

                for code, key in ((0, "Wcb"), (1, "Wcq"), (2, "Cat_Top_bc"),
                                  (3, "Rest"), (4, "Wqq_light")):
                    m = mat_cat == code
                    if np.sum(m) == 0:
                        continue
                    add(key, values[m], weights[m])

            elif mode == "mixed":
                if group in self.cfg.signal_groups:
                    add(group, values, weights)
                else:
                    for key in self.cfg.true_cat_order:
                        m = true_cat == self.cfg.true_cat_codes[key]
                        if np.sum(m) == 0:
                            continue
                        add(key, values[m], weights[m])

            else:
                raise ValueError(f"Unknown histogram mode: {mode}")

        return hist_data, hist_var


# ======================================================================================
# 5. Plotter
# ======================================================================================

# --------------------------------------------------------------------------------------
# Cosmetics for a 600x600 canvas.
# Text sizes / margins follow the ROOT conventions in ../Hgg/Root_plot.py
# (Root_plot.py in this dir): 490-600 px square canvas, helvetica, axis titles
# ~0.066 and labels ~0.057 of the pad height, CMS label bold ~0.074, thin frame,
# inward ticks on all four sides.
# --------------------------------------------------------------------------------------
PLOT_STYLE = {
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "font.size": 11,
    "axes.labelsize": 15,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.major.size": 6,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.3,
}

CMS_LABEL_FONTSIZE = 15


class Plotter:
    def __init__(self, cfg):
        self.cfg = cfg
        hep.style.use("CMS")
        plt.rcParams.update(PLOT_STYLE)
        ensure_dir(cfg.figure_path)
        self.saved_pngs = []
        self._viewer = self._resolve_viewer()

    # ----------------------------------------------------------------------
    # Pop each PNG up as soon as it is written (one window per plot), using
    # ImageMagick "display" (override with PNG_VIEWER=<cmd>, disable with
    # SHOW=0). Needs an X server: ssh -Y, or run on a machine with a display.
    # ----------------------------------------------------------------------
    @staticmethod
    def _resolve_viewer():
        if os.environ.get("SHOW", "1") == "0":
            return None
        viewer = os.environ.get("PNG_VIEWER", "display")
        if not os.environ.get("DISPLAY"):
            print("[SHOW] $DISPLAY not set - no X server; plots will not pop up. "
                  "Reconnect with 'ssh -Y lxplus...' or set SHOW=0.")
            return None
        if shutil.which(viewer) is None:
            print(f"[SHOW] '{viewer}' not found in PATH; set PNG_VIEWER=<cmd> or SHOW=0.")
            return None
        return viewer

    def _save_and_show(self, fig, outbase, tag="PLOT"):
        png = os.path.abspath(outbase + ".png")
        fig.savefig(png, dpi=100)
        plt.close(fig)
        self.saved_pngs.append(png)
        print(f"[{tag}] Saved {os.path.basename(png)}")
        if self._viewer:
            # pop-up viewer disabled -- just write the PNG
            # os.system(f'{self._viewer} "{png}" &')
            # print(f"[SHOW] {self._viewer} {png} &")
            pass

    def label(self, key):
        return self.cfg.labels.get(key, key)

    def color(self, key, mode):
        if mode == "sample":
            return self.cfg.colors_sample.get(key, "#999999")
        return self.cfg.colors_true.get(key, self.cfg.colors_sample.get(key, "#999999"))

    def draw_datamc(self, hist_data, plot_cfg):
        cfg = dict(self.cfg.default_plot)
        cfg.update(plot_cfg)

        bins = cfg["bins"]
        mode = cfg.get("mode", "sample")
        signal_scale = cfg.get("signal_scale", 1)
        name = cfg["name"]

        # 64 / 36 upper:lower split, ~no gap (Root_plot.py pad layout)
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            gridspec_kw={"height_ratios": [64, 36], "hspace": 0.06},
            sharex=True,
        )

        ax = axs[0]
        ax_ratio = axs[1]

        bkg_hists = []
        bkg_labels = []
        bkg_colors = []
        sig_items = []
        data_hist = None

        if mode == "sample":
            key_order = ["QCD", "Diboson", "Rare", "single-top", "tt", "Other", "Wcb", "Data"]
        else:
            key_order = [
                "Cat_QCD",
                "Cat_Top_bqq",
                "Cat_Top_bc",
                "Cat_Top_bq",
                "Cat_W_qq",
                "Cat_Other",
                "Wcb",
                "Data",
            ]

        for key in key_order:
            if key not in hist_data:
                continue

            counts = hist_data[key]

            if key == "Data":
                data_hist = counts
            elif key in self.cfg.signal_groups:
                sig_items.append((key, counts))
            else:
                bkg_hists.append(counts)
                bkg_labels.append(self.label(key))
                bkg_colors.append(self.color(key, mode))

        for key in hist_data:
            if key in key_order:
                continue

            counts = hist_data[key]

            if key == "Data":
                data_hist = counts
            elif key in self.cfg.signal_groups:
                sig_items.append((key, counts))
            else:
                bkg_hists.append(counts)
                bkg_labels.append(self.label(key))
                bkg_colors.append(self.color(key, mode))

        if bkg_hists:
            hep.histplot(
                bkg_hists,
                bins=bins,
                stack=True,
                histtype="fill",
                label=bkg_labels,
                color=bkg_colors,
                ax=ax,
            )

        for key, counts in sig_items:
            hep.histplot(
                counts * signal_scale,
                bins=bins,
                histtype="step",
                linewidth=2,
                color=self.color(key, mode),
                label=f"{self.label(key)} $\\times$ {signal_scale}",
                ax=ax,
            )

        total_mc = np.zeros(len(bins) - 1, dtype=np.float64)

        if bkg_hists:
            total_mc += np.sum(np.vstack(bkg_hists), axis=0)

        for _, counts in sig_items:
            total_mc += counts

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        if self.cfg.enable_data and data_hist is not None:
            yerr = np.sqrt(np.maximum(data_hist, 0))

            hep.histplot(
                data_hist,
                bins=bins,
                yerr=yerr,
                histtype="errorbar",
                color="black",
                marker="o",
                label=self.label("Data"),
                ax=ax,
            )

        # ------------------------------------------------------------------
        # Ratio panel
        # Negative MC bins can appear because of negative MC weights.
        # For Data/MC ratio, only draw bins with positive MC prediction.
        # ------------------------------------------------------------------
        valid_ratio = total_mc > 0

        ratio = np.full_like(data_hist, np.nan, dtype=np.float64)
        ratio_err = np.full_like(data_hist, np.nan, dtype=np.float64)

        ratio[valid_ratio] = data_hist[valid_ratio] / total_mc[valid_ratio]
        ratio_err[valid_ratio] = yerr[valid_ratio] / total_mc[valid_ratio]

        # Safety: matplotlib requires non-negative yerr.
        ratio_err = np.abs(ratio_err)

        ax_ratio.errorbar(
            bin_centers[valid_ratio],
            ratio[valid_ratio],
            yerr=ratio_err[valid_ratio],
            fmt="o",
            color="black",
            markersize=4,
        )


        ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax_ratio.set_ylim(0.4, 1.6)
        ax_ratio.set_ylabel("Data/MC", fontsize=13)
        ax_ratio.set_xlabel(cfg.get("xlabel", cfg.get("var", "")))
        ax_ratio.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
        ax_ratio.tick_params(labelsize=11)

        ax.set_ylabel(cfg.get("ylabel", "Events"))
        ax.tick_params(labelsize=11)

        if cfg.get("logy", False):
            ax.set_yscale("log")
            ax.set_ylim(bottom=cfg.get("ylim_bottom", 0.1))
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 3)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.4)

        ax.legend(ncol=2, fontsize=8.5, frameon=False, loc="upper right",
                  handlelength=1.4, columnspacing=1.0, labelspacing=0.3,
                  borderaxespad=0.4)

        hep.cms.label(
            "Preliminary",
            data=self.cfg.enable_data,
            lumi=self.cfg.lumi,
            ax=ax,
            fontsize=CMS_LABEL_FONTSIZE,
        )

        ax.set_xlim(cfg.get("xlim", (bins[0], bins[-1])))

        fig.subplots_adjust(left=0.135, right=0.96, top=0.91, bottom=0.12)

        outbase = os.path.join(self.cfg.figure_path, name)
        self._save_and_show(fig, outbase, tag="PLOT")

    def draw_matching(self, hist_data, plot_cfg, hist_var=None):
        """
        Truth-matched leading-AK8 overlay (--MAT mode).

        Shape-normalized (sum-of-weights = 1) step-line histograms:
          W->cb jets (signal sample), W->cq jets, t->(cb) jet, rest jets.
        Lower panel: (W->cb) / (t->(cb)) ratio.

        `hist_var` (per-bin sum of w^2, under/overflow already folded in) gives
        the shaded stat-uncertainty band: err(shape_i) = sqrt(var_i) / N_tot.
        """
        cfg = dict(self.cfg.default_plot)
        cfg.update(plot_cfg)

        bins = cfg["bins"]
        name = cfg["name"]
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        key_order = ["Wcb", "Wcq", "Wqq_light", "Cat_Top_bc", "Rest"]

        mat_labels = {
            "Wcb": r"$W\to cb$ jets",
            "Wcq": r"$W\to cq$ jets",
            "Wqq_light": r"$W\to qq'$ (light) jets",
            "Cat_Top_bc": r"$t\to(cb)$ jet",
            "Rest": "rest jets",
        }

        hist_var = hist_var or {}
        norm_hists = {}
        norm_errs = {}
        for key in key_order:
            counts = hist_data.get(key)
            if counts is None:
                continue

            total = np.sum(counts)
            if total > 0:
                norm_hists[key] = counts / total
                var = hist_var.get(key)
                # shape uncertainty: scale the per-bin stat error by 1/N_tot
                # (the normalisation's own uncertainty is neglected, as usual
                # for shape-only comparisons)
                norm_errs[key] = (np.sqrt(var) / total
                                  if var is not None else np.zeros_like(counts))
            else:
                norm_hists[key] = counts
                norm_errs[key] = np.zeros_like(counts)

        if "Wcb" not in norm_hists or "Cat_Top_bc" not in norm_hists:
            print(f"[WARN] Missing Wcb or Cat_Top_bc for matching plot {name}, skip.")
            return

        def _stepify(y):
            return np.repeat(y, 2)

        step_x = np.repeat(bins, 2)[1:-1]

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            gridspec_kw={"height_ratios": [64, 36], "hspace": 0.06},
            sharex=True,
        )
        ax = axs[0]
        ax_ratio = axs[1]

        # MAT-overlay styling:
        #   W->cb          wine-red, solid
        #   W->cq          light yellow, dashed (ROOT line style 2)
        #   W->qq' light   light green, dashed
        #   t->(cb)        blue, solid
        #   rest           light grey, dashed (ROOT line style 7)
        style_overrides = {
            "Wcb":        {"color": "#722F37", "linestyle": "-",         "linewidth": 2},
            "Wcq":        {"color": "#EDD24E", "linestyle": (0, (4, 2)), "linewidth": 1.6},
            "Wqq_light":  {"color": "#8ED081", "linestyle": (0, (5, 2)), "linewidth": 1.6},
            "Cat_Top_bc": {"color": "#2C5FAA", "linestyle": "-",         "linewidth": 2},
            "Rest":       {"color": "lightgrey", "linestyle": (0, (6, 3)), "linewidth": 1.5},
        }

        for key in key_order:
            if key not in norm_hists:
                continue

            style = {"color": self.color(key, "true"), "linestyle": "-", "linewidth": 2}
            style.update(style_overrides.get(key, {}))

            hep.histplot(
                norm_hists[key],
                bins=bins,
                histtype="step",
                label=mat_labels.get(key, self.label(key)),
                ax=ax,
                **style,
            )

            # shaded per-bin stat-uncertainty band
            e = norm_errs.get(key)
            if e is not None and np.any(e > 0):
                y = norm_hists[key]
                ax.fill_between(
                    step_x, _stepify(y - e), _stepify(y + e),
                    color=style["color"], alpha=0.18, linewidth=0, zorder=1,
                )

        wcb, top = norm_hists["Wcb"], norm_hists["Cat_Top_bc"]
        e_wcb = norm_errs.get("Wcb", np.zeros_like(wcb))
        e_top = norm_errs.get("Cat_Top_bc", np.zeros_like(top))

        valid = top > 0
        ratio = np.full_like(wcb, np.nan, dtype=np.float64)
        ratio[valid] = wcb[valid] / top[valid]

        # ratio stat uncertainty: quadrature of the two relative errors
        rerr = np.full_like(wcb, np.nan, dtype=np.float64)
        good = valid & (wcb > 0)
        rerr[good] = ratio[good] * np.sqrt(
            (e_wcb[good] / wcb[good]) ** 2 + (e_top[good] / top[good]) ** 2
        )
        band = good & np.isfinite(rerr)
        if np.any(band):
            ax_ratio.fill_between(
                step_x, _stepify(np.where(band, ratio - rerr, np.nan)),
                _stepify(np.where(band, ratio + rerr, np.nan)),
                color="0.6", alpha=0.35, linewidth=0,
            )

        ax_ratio.plot(
            bin_centers[valid],
            ratio[valid],
            drawstyle="steps-mid",
            color="black",
            linewidth=1.3,
        )
        ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax_ratio.set_ylim(cfg.get("ratio_ylim", (0.0, 4.0)))
        ax_ratio.set_ylabel(r"$W\to cb$ / $t\to(cb)$", fontsize=11)
        ax_ratio.set_xlabel(cfg.get("xlabel", cfg.get("var", "")))
        ax_ratio.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
        ax_ratio.tick_params(labelsize=11)

        ax.set_ylabel("Normalized to unity")
        ax.tick_params(labelsize=11)

        if cfg.get("logy", False):
            ax.set_yscale("log")
            ax.set_ylim(cfg.get("ylim_bottom", 1e-4), cfg.get("ylim_top", 1.0))
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.4)

        ax.legend(fontsize=12, frameon=False, loc="upper right")

        hep.cms.label(
            "",
            data=False,
            lumi=None,
            ax=ax,
            fontsize=CMS_LABEL_FONTSIZE,
        )

        ax.set_xlim(cfg.get("xlim", (bins[0], bins[-1])))

        fig.subplots_adjust(left=0.135, right=0.96, top=0.91, bottom=0.12)

        outbase = os.path.join(self.cfg.figure_path, name)
        self._save_and_show(fig, outbase, tag="MAT")

    def draw_eventclassifier_multiclass_roc_binned(self, samples, roc_cfg):
        print(f"[INFO] Drawing EventClassifier ROC: {roc_cfg['name']}")

        score_keys = self.cfg.ec_score_branches

        score_norm_keys = {
            "score_cata_w_qq": "score_cata_w_qq_norm",
            "score_cata_qcd": "score_cata_qcd_norm",
            "score_cata_top_bqq": "score_cata_top_bqq_norm",
            "score_cata_top_bc": "score_cata_top_bc_norm",
            "score_cata_top_bq": "score_cata_top_bq_norm",
            "score_cata_non": "score_cata_non_norm",
        }

        label_map = self.cfg.ec_label_map
        truth_map = self.cfg.ec_truth_map

        linestyles = [
            "-",
            "--",
            "-.",
            ":",
            (0, (5, 5)),
            (0, (3, 5, 1, 5, 1, 5)),
        ]

        bin_var = roc_cfg["bin_var"]
        bin_edges = roc_cfg["bin_edges"]
        base_cut = roc_cfg.get("cut", "1")
        name = roc_cfg["name"]
        ymin = roc_cfg.get("ymin", 1e-3)

        for ibin in range(len(bin_edges) - 1):
            low = bin_edges[ibin]
            high = bin_edges[ibin + 1]

            if np.isinf(high):
                bin_label = f"{bin_var}: [{low}, inf)"
                bin_tag = f"{bin_var}_{low}_inf"
            else:
                bin_label = f"{bin_var}: [{low}, {high})"
                bin_tag = f"{bin_var}_{low}_{high}"

            print(f"  [BIN] {bin_label}")

            merged = {
                "weights": [],
                "true_cat": [],
            }

            for norm_key in score_norm_keys.values():
                merged[norm_key] = []

            for sample in samples:
                if sample["is_data"]:
                    continue

                arr = sample["array"]

                if bin_var not in arr.fields:
                    print(f"    [WARN] bin_var {bin_var} not found in {sample['name']}, skip.")
                    continue

                mask_base = eval_cut(base_cut, arr)

                x = ak.to_numpy(arr[bin_var])

                if np.isinf(high):
                    mask_bin = x >= low
                else:
                    mask_bin = (x >= low) & (x < high)

                mask = mask_base & mask_bin

                if np.sum(mask) == 0:
                    continue

                w = ak.to_numpy(arr["weights"])[mask]
                pos_w_mask = w > 0

                if np.sum(pos_w_mask) == 0:
                    continue

                merged["weights"].append(w[pos_w_mask])
                merged["true_cat"].append(ak.to_numpy(arr["true_cat"])[mask][pos_w_mask])

                for raw_key, norm_key in score_norm_keys.items():
                    merged[norm_key].append(ak.to_numpy(arr[norm_key])[mask][pos_w_mask])

            if len(merged["weights"]) == 0:
                print(f"    [WARN] Empty bin: {bin_label}")
                continue

            weights = np.concatenate(merged["weights"])
            true_cat = np.concatenate(merged["true_cat"])

            score_values = {}
            for raw_key, norm_key in score_norm_keys.items():
                score_values[raw_key] = np.concatenate(merged[norm_key])

            for sig_key in score_keys:
                sig_label = label_map[sig_key]
                sig_truth_name = truth_map[sig_key]
                sig_truth_code = self.cfg.true_cat_codes[sig_truth_name]

                print(f"    [PLOT] Signal = {sig_label}")

                hep.style.use("CMS")
                fig, ax = plt.subplots(figsize=(6, 6))

                sig_score_val = score_values[sig_key]
                mask_sig = true_cat == sig_truth_code

                style_idx = 0

                # Inclusive
                y_true = mask_sig.astype(int)
                y_score = sig_score_val
                w_curr = weights

                if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
                    fpr, tpr, _ = roc_curve(
                        y_true,
                        y_score,
                        sample_weight=w_curr,
                    )
                    roc_auc = auc(fpr, tpr)

                    ax.plot(
                        tpr,
                        fpr,
                        color="black",
                        linestyle=linestyles[style_idx],
                        lw=2,
                        label=f"{sig_label} vs All (AUC={roc_auc:.2f})",
                    )

                style_idx += 1

                # Pairwise
                for bkg_key in score_keys:
                    if sig_key == bkg_key:
                        continue

                    bkg_label = label_map[bkg_key]
                    bkg_truth_name = truth_map[bkg_key]
                    bkg_truth_code = self.cfg.true_cat_codes[bkg_truth_name]

                    mask_bkg = true_cat == bkg_truth_code
                    pair_mask = mask_sig | mask_bkg

                    if np.sum(pair_mask) < 10:
                        continue

                    y_true_pair = mask_sig[pair_mask].astype(int)
                    y_score_pair = sig_score_val[pair_mask]
                    w_pair = weights[pair_mask]

                    if len(np.unique(y_true_pair)) < 2:
                        continue

                    fpr, tpr, _ = roc_curve(
                        y_true_pair,
                        y_score_pair,
                        sample_weight=w_pair,
                    )
                    roc_auc = auc(fpr, tpr)

                    current_ls = linestyles[style_idx % len(linestyles)]

                    ax.plot(
                        tpr,
                        fpr,
                        color="black",
                        linestyle=current_ls,
                        lw=2,
                        label=f"{sig_label} vs {bkg_label} (AUC={roc_auc:.2f})",
                    )

                    style_idx += 1

                ax.set_yscale("log")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([ymin, 1.0])

                ax.set_xlabel("Signal efficiency", fontsize=14)
                ax.set_ylabel("Background efficiency", fontsize=14)

                # bin label inside plot top-left
                ax.text(0.02, 0.97, bin_label, transform=ax.transAxes,
                        fontsize=10, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                hep.cms.label(
                    "Simulation",
                    data=False,
                    lumi=None,
                    ax=ax,
                    loc=0,
                    fontsize=CMS_LABEL_FONTSIZE,
                )

                n_entries = sum(1 for _ in ax.get_legend_handles_labels()[1])
                if n_entries >= 6:
                    ax.legend(loc="lower right", fontsize=8, frameon=False)
                else:
                    ax.legend(loc="lower right", fontsize=9, frameon=False)
                ax.grid(True, which="major", ls="-", alpha=0.2)

                outbase = os.path.join(
                    self.cfg.figure_path,
                    f"{name}__Signal_{safe_name(sig_label)}__{safe_name(bin_tag)}",
                )
                self._save_and_show(fig, outbase, tag="SAVE")
    def draw_dbc_roc_binned(self, samples, roc_cfg):
        """
        Boosted Dbc binary ROC.

        Signal definition:
        ak8_n_b_in_jet == 1 and ak8_n_c_in_jet == 1

        Background:
        all other MC events

        X-axis: Signal efficiency
        Y-axis: Background efficiency
        """

        print(f"[INFO] Drawing Dbc ROC: {roc_cfg['name']}")

        score_name = roc_cfg.get("score", "score_Dbc")
        bin_var = roc_cfg["bin_var"]
        bin_edges = roc_cfg["bin_edges"]
        base_cut = roc_cfg.get("cut", "1")
        name = roc_cfg["name"]
        ymin = roc_cfg.get("ymin", 1e-4)

        linestyles = [
            "-",
            "--",
            "-.",
            ":",
            (0, (5, 5)),
            (0, (3, 5, 1, 5, 1, 5)),
        ]

        fig, ax = plt.subplots(figsize=(6, 6))

        for ibin in range(len(bin_edges) - 1):
            low = bin_edges[ibin]
            high = bin_edges[ibin + 1]

            if np.isinf(high):
                bin_label = f"{bin_var}: [{low}, inf)"
            else:
                bin_label = f"{bin_var}: [{low}, {high})"

            print(f"  [BIN] {bin_label}")

            all_scores = []
            all_labels = []
            all_weights = []

            for sample in samples:
                if sample["is_data"]:
                    continue

                arr = sample["array"]

                required = [
                    score_name,
                    bin_var,
                    "ak8_n_b_in_jet",
                    "ak8_n_c_in_jet",
                    "weights",
                ]

                missing = [x for x in required if x not in arr.fields]
                if len(missing) > 0:
                    print(f"    [WARN] Missing fields {missing} in {sample['name']}, skip.")
                    continue

                mask_base = eval_cut(base_cut, arr)

                x = ak.to_numpy(arr[bin_var])

                if np.isinf(high):
                    mask_bin = x >= low
                else:
                    mask_bin = (x >= low) & (x < high)

                mask = mask_base & mask_bin

                if np.sum(mask) == 0:
                    continue

                score = ak.to_numpy(arr[score_name])[mask]
                weight = ak.to_numpy(arr["weights"])[mask]

                n_b = ak.to_numpy(arr["ak8_n_b_in_jet"])[mask]
                n_c = ak.to_numpy(arr["ak8_n_c_in_jet"])[mask]

                # Positive weights only for sklearn weighted ROC stability
                pos_w = weight > 0

                score = score[pos_w]
                weight = weight[pos_w]
                n_b = n_b[pos_w]
                n_c = n_c[pos_w]
                is_qcd = ak.to_numpy(arr["is_qcd"])[mask][pos_w]
                
                if len(score) == 0:
                    continue

                y_true = ((n_b == 1) & (n_c == 1) & (~is_qcd)).astype(int)

                all_scores.append(score)
                all_labels.append(y_true)
                all_weights.append(weight)

            if len(all_scores) == 0:
                print(f"    [WARN] Empty bin: {bin_label}")
                continue

            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            weights = np.concatenate(all_weights)

            n_sig = np.sum(labels == 1)
            n_bkg = np.sum(labels == 0)

            if n_sig == 0 or n_bkg == 0:
                print(
                    f"    [WARN] Skip {bin_label}: "
                    f"n_sig={n_sig}, n_bkg={n_bkg}"
                )
                continue

            fpr, tpr, _ = roc_curve(
                labels,
                scores,
                sample_weight=weights,
            )
            roc_auc = auc(fpr, tpr)

            ls = linestyles[ibin % len(linestyles)]

            ax.plot(
                tpr,
                fpr,
                color="black",
                linestyle=ls,
                lw=2,
                label=f"{bin_label} (AUC={roc_auc:.3f})",
            )

        ax.set_yscale("log")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([ymin, 1.0])

        ax.set_xlabel("Signal efficiency", fontsize=14)
        ax.set_ylabel("Background efficiency", fontsize=14)

        title = roc_cfg.get("title", r"Boosted $D_{bc}$ ROC")
        ax.text(0.02, 0.97, title, transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        hep.cms.label(
            "Simulation",
            data=False,
            lumi=None,
            ax=ax,
            loc=0,
            fontsize=CMS_LABEL_FONTSIZE,
        )

        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.grid(True, which="major", ls="-", alpha=0.2)

        outbase = os.path.join(
            self.cfg.figure_path,
            safe_name(name),
        )
        self._save_and_show(fig, outbase, tag="ROC")


# ======================================================================================
# 6. Plot settings
# ======================================================================================

def build_plot_settings():
    plot_settings = []

    # ------------------------------------------------------------------
    # Event-level plots
    # ------------------------------------------------------------------
    event_vars = [
        {
            "var": "lep1_pt",
            "bins": np.linspace(30, 300, 55),
            "xlabel": "Lepton $p_{T}$ [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "lep1_pdgId",
            "bins": np.linspace(-15, 15, 31),
            "xlabel": "Lepton PDG ID",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "n_ak8",
            "bins": np.linspace(0, 6, 7),
            "xlabel": "Number of AK8 jets",
            "cut": "1",
        },
        {
            "var": "n_ak4",
            "bins": np.linspace(0, 12, 13),
            "xlabel": "Number of AK4 jets",
            "cut": "1",
        },
        {
            "var": "met",
            "bins": np.linspace(0, 500, 51),
            "xlabel": "MET [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "met_phi",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "MET $\\phi$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "lep1_eta",
            "bins": np.linspace(-2.5, 2.5, 51),
            "xlabel": "Lepton $\\eta$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "lep1_phi",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "Lepton $\\phi$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "mTW",
            "bins": np.linspace(0, 300, 61),
            "xlabel": "$m_T(W)$ from lepton + MET [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "w_mass",
            "bins": np.linspace(0, 300, 61),
            "xlabel": "$m(W)$ from lepton + MET ($\\eta_{\\nu}=0$) [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
    ]

    for item in event_vars:
        var = item["var"]

        plot_settings.append({
            "name": f"{var}_sample",
            "var": var,
            "var_mode": "event",
            "bins": item["bins"],
            "xlabel": item["xlabel"],
            "mode": "sample",
            "cut": item["cut"],
            "logy": False,
            "signal_scale": 1000,
        })

        if var in ["met", "met_phi", "lep1_eta", "lep1_phi", "mTW", "w_mass"]:
            plot_settings.append({
                "name": f"{var}_true",
                "var": var,
                "var_mode": "event",
                "bins": item["bins"],
                "xlabel": item["xlabel"],
                "mode": "true",
                "cut": item["cut"],
                "logy": False,
                "signal_scale": 1000,
            })

    # ------------------------------------------------------------------
    # Leading jet plots
    # ------------------------------------------------------------------
    leading_vars = [
        {
            "var": "ak8_pt_0",
            "bins": np.linspace(200, 800, 31),
            "xlabel": "Leading AK8 jet $p_{T}$ [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_eta_0",
            "bins": np.linspace(-2.5, 2.5, 31),
            "xlabel": "Leading AK8 jet $\\eta$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_phi_0",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "Leading AK8 jet $\\phi$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_sdmass_0",
            "bins": np.linspace(0, 250, 26),
            "xlabel": "Leading AK8 soft-drop mass [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak4_pt_0",
            "bins": np.linspace(30, 500, 48),
            "xlabel": "Leading AK4 jet $p_{T}$ [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak4_eta_0",
            "bins": np.linspace(-2.5, 2.5, 31),
            "xlabel": "Leading AK4 jet $\\eta$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak4_phi_0",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "Leading AK4 jet $\\phi$",
            "cut": "ak8_pt[0] > 200",
        },
    ]

    for item in leading_vars:
        var = item["var"]

        plot_settings.append({
            "name": f"{var}_sample",
            "var": var,
            "var_mode": "leading",
            "bins": item["bins"],
            "xlabel": item["xlabel"],
            "mode": "sample",
            "cut": item["cut"],
            "logy": False,
            "signal_scale": 1000,
        })

        if var.startswith("ak8_"):
            plot_settings.append({
                "name": f"{var}_true",
                "var": var,
                "var_mode": "leading",
                "bins": item["bins"],
                "xlabel": item["xlabel"],
                "mode": "true",
                "cut": item["cut"],
                "logy": False,
                "signal_scale": 1000,
            })

    # ------------------------------------------------------------------
    # Flatten jet plots
    # ------------------------------------------------------------------
    flatten_vars = [
        {
            "var": "ak8_pt",
            "bins": np.linspace(200, 800, 31),
            "xlabel": "AK8 jet $p_{T}$ [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_eta",
            "bins": np.linspace(-2.5, 2.5, 31),
            "xlabel": "AK8 jet $\\eta$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_phi",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "AK8 jet $\\phi$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_sdmass",
            "bins": np.linspace(0, 250, 26),
            "xlabel": "AK8 jet soft-drop mass [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak4_pt",
            "bins": np.linspace(30, 500, 48),
            "xlabel": "AK4 jet $p_{T}$ [GeV]",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak4_eta",
            "bins": np.linspace(-2.5, 2.5, 31),
            "xlabel": "AK4 jet $\\eta$",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak4_phi",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "AK4 jet $\\phi$",
            "cut": "ak8_pt[0] > 200",
        },
    ]

    for item in flatten_vars:
        var = item["var"]

        plot_settings.append({
            "name": f"{var}_flatten_sample",
            "var": var,
            "var_mode": "flatten",
            "bins": item["bins"],
            "xlabel": item["xlabel"],
            "ylabel": "Jets",
            "mode": "sample",
            "cut": item["cut"],
            "logy": False,
            "signal_scale": 1000,
        })

    # ------------------------------------------------------------------
    # Score plots
    # ------------------------------------------------------------------
    score_vars = [
        {
            "var": "score_Dbc",
            "xlabel": "$D_{bc}$",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_SC",
            "xlabel": r"$S_{\mathrm{EVT}}$",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_cata_w_qq_norm",
            "xlabel": r"EventClassifier $W \to qq'$ score",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_cata_qcd_norm",
            "xlabel": "EventClassifier QCD score",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_cata_top_bqq_norm",
            "xlabel": r"EventClassifier Top $bqq'$ score",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_cata_top_bc_norm",
            "xlabel": r"EventClassifier Top $bc$ score",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_cata_top_bq_norm",
            "xlabel": r"EventClassifier Top $bq$ score",
            "bins": np.linspace(0, 1, 51),
        },
        {
            "var": "score_cata_non_norm",
            "xlabel": "EventClassifier none-of-the-above score",
            "bins": np.linspace(0, 1, 51),
        },
    ]

    for item in score_vars:
        var = item["var"]
        ylim_btm = 1e3 if var == "score_Dbc" else 0.1

        plot_settings.append({
            "name": f"{var}_sample",
            "var": var,
            "var_mode": "event",
            "bins": item["bins"],
            "xlabel": item["xlabel"],
            "xlim": (0.0, 1.0),
            "mode": "sample",
            "cut": "ak8_pt[0] > 200",
            "logy": True,
            "signal_scale": 1000,
            "ylim_bottom": ylim_btm,
        })

        plot_settings.append({
            "name": f"{var}_true",
            "var": var,
            "var_mode": "event",
            "bins": item["bins"],
            "xlabel": item["xlabel"],
            "xlim": (0.0, 1.0),
            "mode": "true",
            "cut": "ak8_pt[0] > 200",
            "logy": True,
            "signal_scale": 1000,
            "ylim_bottom": ylim_btm,
        })

    # ------------------------------------------------------------------
    # New useful cut plots
    # ------------------------------------------------------------------
    cut_plots = [
        {
            "name": "ak8_sdmass_Dbc_gt_0985_sample",
            "var": "ak8_sdmass_0",
            "var_mode": "leading",
            "bins": np.linspace(0, 250, 26),
            "xlabel": "Leading AK8 soft-drop mass [GeV], $D_{bc} > 0.985$",
            "mode": "sample",
            "cut": "(ak8_pt[0] > 200) and (score_Dbc > 0.985)",
            "logy": False,
            "signal_scale": 1,
        },
        {
            "name": "score_SC_mixed_Dbc_gt_0985",
            "var": "score_SC",
            "var_mode": "event",
            "bins": np.linspace(0, 1, 51),
            "xlabel": r"$S_{\mathrm{EVT}}$, $D_{bc} > 0.985$",
            "xlim": (0.0, 1.0),
            "mode": "mixed",
            "cut": "(ak8_pt[0] > 200) and (score_Dbc > 0.985)",
            "logy": False,
            "signal_scale": 1,
        },
        {
            "name": "score_Dbc_mass_window_sample",
            "var": "score_Dbc",
            "var_mode": "event",
            "bins": np.linspace(0, 1, 51),
            "xlabel": "$D_{bc}$, $50 < m_{SD} < 120$ GeV",
            "xlim": (0.0, 1.0),
            "mode": "sample",
            "cut": "(ak8_pt[0] > 200) and (ak8_sdmass[0] > 50) and (ak8_sdmass[0] < 120)",
            "logy": True,
            "signal_scale": 1000,
            "ylim_bottom": 1000,
        },
        {
            "name": "score_SC_mass_window_sample",
            "var": "score_SC",
            "var_mode": "event",
            "bins": np.linspace(0, 1, 51),
            "xlabel": r"$S_{\mathrm{EVT}}$, $50 < m_{SD} < 120$ GeV",
            "xlim": (0.0, 1.0),
            "mode": "sample",
            "cut": "(ak8_pt[0] > 200) and (ak8_sdmass[0] > 50) and (ak8_sdmass[0] < 120)",
            "logy": True,
            "signal_scale": 1000,
            "ylim_bottom": 0.1,
        },
    ]

    plot_settings.extend(cut_plots)

    # Only keep the leading AK8 soft-drop mass and the Dbc BDT score.
    keep_names = {
        "ak8_sdmass_0_sample",
        "ak8_sdmass_0_true",
        "score_Dbc_sample",
        "score_Dbc_true",
    }
    plot_settings = [p for p in plot_settings if p["name"] in keep_names]

    return plot_settings


def build_mat_plot_settings():
    """
    Truth-matched leading-AK8 overlay (--MAT mode). Five shape-normalized
    step histograms, driven by `mat_cat` (built in build_derived_array):
      W->cb jets       (ak8_is_wbc, signal sample only),
      W->cq jets       (ak8_match_wcq, any MC),
      W->qq' light jets (ak8_match_wqq, any MC),
      t->(cb) jet      (Cat_Top_bc, any MC),
      rest jets        (none of the above).
    Lower panel: (W->cb)/(t->(cb)) ratio.

    All entries below describe the *leading AK8 jet* or the event it sits in.
    The list is a broad kinematic scan (~65 observables) meant to expose where
    a genuine W->cb jet differs from a W->cq or a Top->bc jet: GloParT tagger
    outputs, jet substructure, angular/dR variables and event activity. Same
    baseline phase space (leading AK8 pt > 200 GeV) as the rest of the
    framework.

    Each spec row: (short_name, var, lo, hi, nbins, xlabel, logy).
    SENTINEL (-999) fills are removed before histogramming; genuine
    out-of-range entries are folded into the first/last bin by
    hist_with_flow (ROOT UnderOverFlow1D style).
    """
    CUT = "ak8_pt[0] > 200"
    PI = float(np.pi)

    def m(name, var, lo, hi, nb, xlabel, logy):
        return {
            "name": f"MAT_{name}",
            "var": var,
            "var_mode": "event",
            "mode": "mat",
            "bins": np.linspace(lo, hi, nb + 1),
            "xlabel": xlabel,
            "xlim": (lo, hi),
            "cut": CUT,
            "logy": logy,
            "ylim_bottom": 1e-4,
            "ylim_top": 1.0,
            "ratio_ylim": (0.0, 4.0),
        }

    specs = [
        # ---- classifier / discriminant outputs ------------------------------
        ("score_Dbc",              "score_Dbc",              0.0, 1.0, 50, r"$D_{bc}$",                      True),
        ("score_SC",               "score_SC",               0.0, 1.0, 50, r"$S_{\mathrm{EVT}}$",            True),
        ("evtcls_w_qq",            "score_cata_w_qq_norm",    0.0, 1.0, 50, r"EventClassifier $W\to qq'$ score",   True),
        ("evtcls_top_bc",          "score_cata_top_bc_norm",  0.0, 1.0, 50, r"EventClassifier Top($bc$) score",    True),
        ("evtcls_top_bqq",         "score_cata_top_bqq_norm", 0.0, 1.0, 50, r"EventClassifier Top($bqq'$) score",  True),
        ("evtcls_top_bq",          "score_cata_top_bq_norm",  0.0, 1.0, 50, r"EventClassifier Top($bq$) score",    True),
        ("evtcls_qcd",             "score_cata_qcd_norm",     0.0, 1.0, 50, r"EventClassifier QCD score",          True),
        ("evtcls_none",            "score_cata_non_norm",     0.0, 1.0, 50, r"EventClassifier none-of-the-above score", True),

        # ---- leading-AK8 GloParT raw scores --------------------------------
        ("gpt_bc",                 "ak8_gpt_bc_0",            0.0, 1.0, 50, r"GloParT $bc$",                  True),
        ("gpt_bb",                 "ak8_gpt_bb_0",            0.0, 1.0, 50, r"GloParT $bb$",                  True),
        ("gpt_cc",                 "ak8_gpt_cc_0",            0.0, 1.0, 50, r"GloParT $cc$",                  True),
        ("gpt_cs",                 "ak8_gpt_cs_0",            0.0, 1.0, 50, r"GloParT $cs$",                  True),
        ("gpt_bs",                 "ak8_gpt_bs_0",            0.0, 1.0, 50, r"GloParT $bs$",                  True),
        ("gpt_qq",                 "ak8_gpt_qq_0",            0.0, 1.0, 50, r"GloParT $qq$",                  True),
        ("gpt_qcd",                "ak8_gpt_qcd_0",           0.0, 1.0, 50, r"GloParT QCD",                   True),
        ("gpt_topbw",              "ak8_gpt_topbw_0",         0.0, 1.0, 50, r"GloParT top score ($b{+}W$ in jet)", True),
        ("gpt_topw",               "ak8_gpt_topw_0",          0.0, 1.0, 50, r"GloParT top-$W$ score ($b$ outside jet)", True),

        # ---- GloParT discriminant ratios ----------------------------------
        ("gpt_bc_frac",            "gpt_bc_frac",             0.0, 1.0, 50, r"$bc/\sum$ GloParT",             True),
        ("gpt_bc_vs_cc",           "gpt_bc_vs_cc",            0.0, 1.0, 50, r"$bc/(bc+cc)$",                  False),
        ("gpt_bc_vs_bb",           "gpt_bc_vs_bb",            0.0, 1.0, 50, r"$bc/(bc+bb)$",                  False),
        ("gpt_bc_vs_qq",           "gpt_bc_vs_qq",            0.0, 1.0, 50, r"$bc/(bc+qq)$",                  False),
        ("gpt_cs_vs_qq",           "gpt_cs_vs_qq",            0.0, 1.0, 50, r"$cs/(cs+qq)$",                  False),
        ("gpt_hf_w",               "gpt_hf_w",                0.0, 1.0, 50, r"$(bc+cs+bs)/\sum$ GloParT",     True),
        ("gpt_top_frac",           "gpt_top_frac",            0.0, 1.0, 50, r"top-like fraction GloParT",     True),
        ("gpt_w_vs_top",           "gpt_w_vs_top",            0.0, 1.0, 50, r"W-like $/$ (W-like $+$ top-like)", False),

        # ---- leading-AK8 kinematics & substructure ------------------------
        ("ak8_pt_0",               "ak8_pt_0",              200.0, 1000.0, 40, r"Leading AK8 $p_T$ [GeV]",   True),
        ("ak8_eta_0",              "ak8_eta_0",             -2.5,  2.5,   50, r"Leading AK8 $\eta$",          False),
        ("ak8_phi_0",              "ak8_phi_0",             -PI,   PI,    50, r"Leading AK8 $\phi$",          False),
        ("ak8_sdmass_0",           "ak8_sdmass_0",            0.0, 250.0, 50, r"Leading AK8 $m_{\mathrm{SD}}$ [GeV]", False),
        ("ak8_rawFactor_0",        "ak8_rawFactor_0",         0.0, 1.0,   50, r"Leading AK8 raw factor",      False),
        ("ak8_tau21_0",            "ak8_tau21_0",             0.0, 1.0,   50, r"Leading AK8 $\tau_{21}$",     False),
        ("ak8_tau32_0",            "ak8_tau32_0",             0.0, 1.0,   50, r"Leading AK8 $\tau_{32}$",     False),
        ("ak8_nConst_0",           "ak8_nConst_0",            0.0, 120.0, 60, r"Leading AK8 $N_{\mathrm{const}}$", False),
        ("ak8_n_b_in_jet",         "ak8_n_b_in_jet",         -0.5, 5.5,    6, r"# b in leading AK8 (gen)",    False),
        ("ak8_n_c_in_jet",         "ak8_n_c_in_jet",         -0.5, 5.5,    6, r"# c in leading AK8 (gen)",    False),
        ("ak8_n_in_jet",           "ak8_n_in_jet",           -0.5, 10.5,  11, r"# partons in leading AK8 (gen)", False),

        # ---- leading-AK4 -------------------------------------------------
        ("ak4_pt_0",               "ak4_pt_0",                0.0, 600.0, 50, r"Leading AK4 $p_T$ [GeV]",     True),
        ("ak4_eta_0",              "ak4_eta_0",              -2.5, 2.5,   50, r"Leading AK4 $\eta$",          False),
        ("ak4_mass_0",             "ak4_mass_0",              0.0, 60.0,  40, r"Leading AK4 mass [GeV]",      False),

        # ---- angular / dR variables ------------------------------------
        ("dR_lep_ak8",             "dR_lep_ak8",              0.0, 5.0,   50, r"$\Delta R(\ell,\,J)$",        False),
        ("dphi_lep_ak8",           "dphi_lep_ak8",            0.0, PI,    40, r"$\Delta\phi(\ell,\,J)$",      False),
        ("deta_lep_ak8",           "deta_lep_ak8",            0.0, 5.0,   50, r"$\Delta\eta(\ell,\,J)$",      False),
        ("dphi_met_ak8",           "dphi_met_ak8",            0.0, PI,    40, r"$\Delta\phi(p_T^{\mathrm{miss}},\,J)$", False),
        ("dphi_lep_met",           "dphi_lep_met",            0.0, PI,    40, r"$\Delta\phi(\ell,\,p_T^{\mathrm{miss}})$", False),
        ("dR_ak4_ak8",             "dR_ak4_ak8",              0.0, 5.0,   50, r"$\Delta R(j_{\mathrm{AK4}},\,J)$", False),
        ("dphi_ak4_ak8",           "dphi_ak4_ak8",            0.0, PI,    40, r"$\Delta\phi(j_{\mathrm{AK4}},\,J)$", False),
        ("deta_ak4_ak8",           "deta_ak4_ak8",            0.0, 5.0,   50, r"$\Delta\eta(j_{\mathrm{AK4}},\,J)$", False),
        ("ptrel_lep_ak8",          "ptrel_lep_ak8",           0.0, 2.0,   40, r"$p_T^{\ell}/p_T^{J}$",        False),

        # ---- dR / invariant-mass observables from tagged AK4 jets --------
        ("minDR_b",                "minDR_b",                 0.0, 5.0,   50, r"$\Delta R_{\mathrm{min}}$ between b-tagged AK4 jets", False),
        ("minDR_c",                "minDR_c",                 0.0, 5.0,   50, r"$\Delta R_{\mathrm{min}}$ between c-tagged AK4 jets", False),
        ("minDR_bc",               "minDR_bc",                0.0, 5.0,   50, r"$\Delta R_{\mathrm{min}}$ ($b$-tag, $c$-tag AK4)",   False),
        ("mass_minDR_b",           "mass_minDR_b",            0.0, 300.0, 50, r"$m$(closest $b$-tagged AK4 pair) [GeV]",             True),
        ("mass_minDR_c",           "mass_minDR_c",            0.0, 300.0, 50, r"$m$(closest $c$-tagged AK4 pair) [GeV]",             False),
        ("mass_minDR_bc",          "mass_minDR_bc",           0.0, 300.0, 50, r"$m$(closest $b$-tag$+$$c$-tag AK4 pair) [GeV]",      False),
        ("maxMass_bc",             "maxMass_bc",              0.0, 500.0, 50, r"$m_{\mathrm{max}}$($b$-tag$+$$c$-tag AK4 pair) [GeV]", False),

        # ---- event activity -------------------------------------------
        ("ht",                     "ht",                      0.0, 2000.0, 50, r"$H_T$ [GeV]",               True),
        ("ht_bc",                  "ht_bc",                   0.0, 1500.0, 50, r"$\sum p_T$ of $b$- and $c$-tagged AK4 jets [GeV]", True),
        ("n_ak4",                  "n_ak4",                  -0.5, 14.5,   15, r"# AK4 jets",                False),
        ("n_ak8",                  "n_ak8",                  -0.5, 6.5,     7, r"# AK8 jets",                False),
        ("n_btag",                 "n_btag",                 -0.5, 8.5,     9, r"# b-tags",                  False),
        ("n_ctag",                 "n_ctag",                 -0.5, 8.5,     9, r"# c-tags",                  False),
        ("met",                    "met",                     0.0, 400.0,  40, r"$p_T^{\mathrm{miss}}$ [GeV]", True),
        ("mTW",                    "mTW",                     0.0, 250.0,  50, r"$m_T^W$ [GeV]",             False),
        ("w_mass",                 "w_mass",                  0.0, 400.0,  50, r"$m(\ell\nu)$ [GeV]",        False),
        ("v_pt",                   "v_pt",                    0.0, 600.0,  50, r"$p_T(\ell\nu)$ [GeV]",      True),
        ("v_mass",                 "v_mass",                  0.0, 200.0,  50, r"$m_V$ [GeV]",               False),
        ("lep1_pt",                "lep1_pt",                 0.0, 400.0,  40, r"Lepton $p_T$ [GeV]",        True),
        ("lep1_eta",               "lep1_eta",               -2.5, 2.5,   50, r"Lepton $\eta$",             False),
    ]

    return [m(*s) for s in specs]


def build_roc_settings():
    return [
        {
            "name": "ROC_EventClassifier_mass_bins",
            "bin_var": "ak8_sdmass_0",
            "bin_edges": [0, 50, 120, np.inf],
            "cut": "ak8_pt[0] > 200",
            "ymin": 1e-3,
            "title": "EventClassifier ROC",
        },
        {
            "name": "ROC_EventClassifier_pt_bins",
            "bin_var": "ak8_pt_0",
            "bin_edges": [200, 300, 450, np.inf],
            "cut": "ak8_pt[0] > 200",
            "ymin": 1e-3,
            "title": "EventClassifier ROC",
        },
    ]

def build_dbc_roc_settings():
    return [
        {
            "name": "ROC_Dbc_mass_bins_nb1_nc1",
            "score": "score_Dbc",
            "bin_var": "ak8_sdmass_0",
            "bin_edges": [0, 50, 120, np.inf],
            "cut": "ak8_pt[0] > 200",
            "ymin": 1e-4,
            "title": r"Boosted $D_{bc}$ ROC, signal: $n_b=1,n_c=1$",
        },
        {
            "name": "ROC_Dbc_pt_bins_nb1_nc1",
            "score": "score_Dbc",
            "bin_var": "ak8_pt_0",
            "bin_edges": [200, 300, 450, np.inf],
            "cut": "ak8_pt[0] > 200",
            "ymin": 1e-4,
            "title": r"Boosted $D_{bc}$ ROC, signal: $n_b=1,n_c=1$",
        },
    ]

# ======================================================================================
# 7. Main
# ======================================================================================

def fmt_hms(seconds):
    """'Xm Ys' like the timers in Root_plot.py (divmod on 60)."""
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Data/MC plotting framework")
    parser.add_argument(
        "mode",
        nargs="?",
        default="ALL",
        type=str.upper,
        choices=["ALL", "MAT"],
        help="ALL (default): run the normal Data/MC batch plots. "
             "MAT: matching-truth overlay only (Wcb vs Cat_Top_bc vs Rest, "
             "normalized, with a ratio panel).",
    )
    return parser.parse_args()


def main():
    t_total0 = time.time()

    args = parse_args()
    cfg = Config()
    ensure_dir(cfg.figure_path)
    ensure_dir(cfg.cache_path)

    print("=" * 100)
    print("[INFO] Batch Data/MC plotting framework")
    print(f"[INFO] MC path      : {cfg.mc_path}")
    print(f"[INFO] Data path    : {cfg.data_path}")
    print(f"[INFO] Figure path  : {cfg.figure_path}")
    print(f"[INFO] Cache path   : {cfg.cache_path}")
    print(f"[INFO] Cache tag    : {cfg.cache_tag}")
    print(f"[INFO] Use cache    : {cfg.use_cache}")
    print(f"[INFO] Force reload : {cfg.force_reload}")
    print("=" * 100)

    # ------- Load phase ----------
    t_load0 = time.time()
    manager = DataManager(cfg)
    samples = manager.load_all()
    t_load = time.time() - t_load0

    print(f"[INFO] Loaded {len(samples)} samples.")
    print(f"[INFO] Memory after loading: {get_memory_mb():.1f} MB")
    print(f"[TIME] Data loaded in {fmt_hms(t_load)}")

    hist_maker = Histogrammer(cfg)
    plotter = Plotter(cfg)

    # ------- MAT mode: matching-truth overlay only ----------
    if args.mode == "MAT":
        mat_settings = build_mat_plot_settings()

        t_draw0 = time.time()
        plot_times = []

        print("=" * 100)
        print(f"[INFO] Start drawing MAT (matching-truth) plots: {len(mat_settings)} plots")
        print("=" * 100)

        for i, p in enumerate(mat_settings):
            print(f"[RUN] Plot {i + 1}/{len(mat_settings)}: {p['name']}")
            t_p0 = time.time()
            hist_data, hist_var = hist_maker.make_histograms(samples, p)

            if len(hist_data) == 0:
                print(f"[WARN] Empty histogram for {p['name']}, skip.")
                continue

            plotter.draw_matching(hist_data, p, hist_var)
            dt = time.time() - t_p0
            plot_times.append(dt)
            print(f"[TIME] Plot {i + 1}/{len(mat_settings)} done in {dt:.1f}s")

        t_draw = time.time() - t_draw0
        t_total = time.time() - t_total0
        n = len(plotter.saved_pngs)

        print("=" * 100)
        print(f"[DONE] {n} plots in {fmt_hms(t_total)}")
        print(f"[TIME]   load : {fmt_hms(t_load)}  ({t_load:.1f}s)")
        print(f"[TIME]   draw : {fmt_hms(t_draw)}  ({t_draw:.1f}s"
              + (f", {sum(plot_times) / len(plot_times):.1f}s/plot avg" if plot_times else "")
              + ")")
        print(f"[TIME]   total: {fmt_hms(t_total)}  ({t_total:.1f}s)")
        print(f"[DONE] Figures saved to: {cfg.figure_path}")
        for p in plotter.saved_pngs:
            print(f"       {os.path.basename(p)}")
        print(f"[DONE] Run duration: {fmt_hms(t_total)}  ({t_total:.1f}s)")
        print("=" * 100)
        return

    plot_settings = build_plot_settings()
    roc_settings = []          # ROC plots disabled: only SD mass + Dbc score
    dbc_roc_settings = []

    # ------- Draw phase ----------
    t_draw0 = time.time()
    plot_times = []

    print("=" * 100)
    print(f"[INFO] Start drawing Data/MC plots: {len(plot_settings)} plots")
    print("=" * 100)

    for i, p in enumerate(plot_settings):
        print(f"[RUN] Plot {i + 1}/{len(plot_settings)}: {p['name']}")
        t_p0 = time.time()
        hist_data, hist_var = hist_maker.make_histograms(samples, p)

        if len(hist_data) == 0:
            print(f"[WARN] Empty histogram for {p['name']}, skip.")
            continue

        plotter.draw_datamc(hist_data, p)
        dt = time.time() - t_p0
        plot_times.append(dt)
        print(f"[TIME] Plot {i + 1}/{len(plot_settings)} done in {dt:.1f}s")

    if roc_settings or dbc_roc_settings:
        print("=" * 100)
        print(f"[INFO] Start drawing ROC plots: "
              f"{len(roc_settings) + len(dbc_roc_settings)} settings")
        print("=" * 100)
        for r in roc_settings:
            t_p0 = time.time()
            plotter.draw_eventclassifier_multiclass_roc_binned(samples, r)
            print(f"[RUN] ROC {r['name']}  ({time.time() - t_p0:.1f}s)")
        for r in dbc_roc_settings:
            t_p0 = time.time()
            plotter.draw_dbc_roc_binned(samples, r)
            print(f"[RUN] ROC {r['name']}  ({time.time() - t_p0:.1f}s)")

    t_draw = time.time() - t_draw0
    t_total = time.time() - t_total0
    n = len(plotter.saved_pngs)

    # ------- Runtime summary (Root_plot.py style) ----------
    print("=" * 100)
    print(f"[DONE] {n} plots in {fmt_hms(t_total)}")
    print(f"[TIME]   load : {fmt_hms(t_load)}  ({t_load:.1f}s)")
    print(f"[TIME]   draw : {fmt_hms(t_draw)}  ({t_draw:.1f}s"
          + (f", {sum(plot_times) / len(plot_times):.1f}s/plot avg" if plot_times else "")
          + ")")
    print(f"[TIME]   total: {fmt_hms(t_total)}  ({t_total:.1f}s)")
    print(f"[DONE] Figures saved to: {cfg.figure_path}")
    for p in plotter.saved_pngs:
        print(f"       {os.path.basename(p)}")
    print(f"[DONE] Run duration: {fmt_hms(t_total)}  ({t_total:.1f}s)")
    print("=" * 100)

if __name__ == "__main__":
    main()