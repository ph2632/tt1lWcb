#!/eos/home-y/youpeng/miniforge3/envs/mlenv/bin/python

import os
import re
import gc
import time
import shutil
import argparse
import subprocess
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


def _dir_readable(path):
    """True if `path` is a directory we can actually list (EOS/AFS ACLs can
    deny listing even when os.path.isdir() succeeds)."""
    try:
        os.listdir(path)
        return True
    except OSError:
        return False


# ======================================================================================
# 1. Configuration
# ======================================================================================

class Config:
    def __init__(self):
        # ------------------------------------------------------------------
        # Paths
        # ------------------------------------------------------------------
        self.base_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/"
        # MC sample directory.  youpeng reproduces the scored ntuples every
        # few weeks under a new sub-directory (and has flip-flopped MC/ <->
        # mc/); pick the first readable candidate, newest first.  Override
        # with $WCB_MC_DIR (absolute path to the dir of *_merged[_Skim].root).
        _mc_candidates = [
            os.environ.get("WCB_MC_DIR", ""),
            # 2026-09-02 "2final" rerun (shared 2026-09-04): +24 GloParT
            # sub-nodes, +genZ_eta/phi, +is_qcd; -ak8_gpt_bqq (was a dead
            # all-zero branch anyway).  Files are now *_merged_Skim.root.
            # Content verified 1:1 with the previous production for every
            # branch the analysis reads (xcheck_new_samples.py).
            self.base_path + "MC/scored_samples_2final_v20260902_rerun/",
            self.base_path + "mc/scored_samples_2final_v20260902_rerun/",
            # previous production
            self.base_path + "mc/scored_samples_1merged_/",
            self.base_path + "MC/scored_samples_1merged_/",
        ]
        self.mc_path = next(
            (p for p in _mc_candidates if p and os.path.isdir(p)
             and _dir_readable(p)),
            self.base_path + "mc/scored_samples_1merged_/")
        self.data_path = os.path.join(self.base_path, "data/scored_samples_1merged_/")

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
        # v7: + GloParT combination ratios (bc vs top(bW), top(bW) vs all
        #     top-like, charm-/b-tagged W vs light W) -> full re-derive.
        # v8: split the MAT Wcb line into merged (ak8_match_wqq_wcb, mat_cat=0)
        #     vs not-merged/other (mat_cat=5) -> full re-derive.
        # v9: 3-way Wcb split -- merged+leading (0) / merged+sub-leading (5) /
        #     not-merged-anywhere (6), using event-level any(ak8_match_wqq_wcb).
        # v10: MAT top line split into 3 truth classes -- t->bc (2, the proxy),
        #      t->bq (7), t->bqq (8). Previously bq/bqq fell into Rest.
        # v11: + ak8_tau31_0 (= tau32*tau21).
        # v12: + leading/sub-leading AK8 mSD under three jet orderings
        #      (pT, mSD, GloParT bc score): ak8_sdmass_{maxmass,lead_bc,
        #      sub_pt,sub_mass,sub_bc}_0  (j^1 / J / j_2 / j^2 / j_b).
        # v13: + w_decay; 3-way split of the MAT W->cq and W->qq' lines
        #      (=j_1 / !=j_1 / resolved) mirroring the W->cb split. Top-tagged
        #      leading jets keep their top category (resolved W-classes never
        #      override t->bc/bq/bqq).  Also + gpt_bc_vs_2prong =
        #      bc/(bc+cc+bb+qq+bs+cs).
        # v14: + gpt_bc_vs_nontop = bc/(bc+cc+bb+qq+bs+cs+QCD), i.e. the same
        #      as gpt_bc_vs_2prong but with the QCD hypothesis added to the
        #      denominator -- W(cb) vs everything that is not top-like (all
        #      2-prong W flavours + QCD).
        # v15: CANDIDATE-JET REDEFINITION -- the cb candidate is now J = the
        #      AK8 with the highest GloParT bc score, not j_1 (pT-leading).
        #      `_cand_ak8()` in build_derived_array is the single choke point;
        #      it re-anchors true_cat / mat_cat / score_Dbc / every gpt_* ratio
        #      / all ak8_*_0 kinematics+substructure / the ak8_match_* flags on
        #      J.  Valid because every AK8 in these ntuples has pT > 200 GeV
        #      (min 200.1), so the phase space is unchanged.  AK4 stays
        #      pT-leading.  ak8_sdmass_lead_bc_0 -> ak8_sdmass_lead_pt_0 (J
        #      now == ak8_sdmass_0, so the diagnostic slot holds j_1 instead).
        #      Whole framework (ALL + MAT).  Pre-v15 code in BU_29_8_26/.
        # v15b: + n_{c,b}_nearcb_{L,M} -- reco b/c-tagged AK4 (ParticleNetAK4
        #       `ak4_tag`) multiplicity within dR<1 of the cb candidate J.
        # v15c: + dphi_w_ak8 = dphi(leptonic W [v_phi], J).
        # v15f: + minDphi_bc = min |dphi| over loose b-tag x loose c-tag AK4
        #       pairs; mSD_minus_mbc = mSD(J) - m(b^L c^L)_dR-closest.
        # v15d: + dR_w_ak8 (W(lv) vs J) and mass_cL_nearb = m(highest-score
        #       loose c-tag AK4 + its nearest-dR loose b-tag AK4).  ak4_pn_c
        #       added to the branch load.
        # v15g: + gpt_bc_vs_top = bc/(bc+topbw+topw+bqq') -- the `bc` GloParT
        #       node against ALL top-like nodes.  Dedicated W(cb) vs t->bc
        #       ("topWcb") ratio: a prompt W->cb jet has no extra b in cone and
        #       fires only `bc`, a t->bc jet leaks onto the full top hypothesis.
        #       (gpt_bc_vs_topbw only used the single topbw node.)
        # v15h: + minDR_bc_LM / minDR_bc_MM -- dR of the closest b/c-tagged AK4
        #       pair at tighter ParticleNetAK4 WP combinations (LM = both loose
        #       & at least one medium ; MM = both medium).  ev() branch
        #       minDR_bc is the loose x loose version.
        # v15i: + score_cata_top_sum_norm ("top-like score") = summed EC
        #       Top(bc)+Top(bq)+Top(bqq') normalised probability.
        # v15j: + z_decay / genZ_pt loaded -> 3 merged hadronic-Z flavour
        #       lines carved out of 'rest' (mat_cat 13/14/15 = Z->qq' light /
        #       Z->cc / Z->bb), "merged" = loose pT proxy pT(J)/pT(Z_gen)>0.7;
        #       failing that -> "Z resolved" stays in 'rest'.
        # v15k: + gpt_bc_vs_qcd = bc / (bc + QCD)  (GloParT W(cb) vs QCD).
        # v15l: + ak8_tau21_sub_bc_0 = tau21 of j_b (2nd-highest bc-score AK8),
        #       for the jb-Region selection.
        # v15m: + gpt_2prong_sum = bc+bb+cc+bs+cs+qq (summed V->2-quark GloParT
        #       probability of J).
        # v16  (derived_v16_ttReco_v1): + ak4_pn_b; + btt{1,2}_* seeds (2
        #       highest-pn_b AK4 outside J) for the 2-top pairing.
        # v16b (derived_v16b_ttReco_v1, 2026-09-01): + the t_a-ONLY seeds --
        #       bdr{1,2}_{pt,eta,phi,m,pnb,tag}  (2 dR-closest AK4 outside J,
        #       any flavour) and bL{1,2,3}_{pt,eta,phi,m} + n_bL_out_ja (3
        #       dR-closest LOOSE-b AK4 outside J).  Plot-time attempts
        #       m_ta_dr / m_ta_drb / m_ta_bestm built in _attach_runtime_fields
        #       (env: WCB_TTRECO_MTOP target 172.5 ; WCB_TTRECO_AMB / _BWP for
        #       the parked 2-top vars).
        # 2026-09-01: MC ROOT area came back (youpeng renamed MC/ -> mc/, see
        # the path block above).  The v16 parquets on disk are STALE (btt only,
        # derived before the bdr/bL seeds existed) -> tag bumped to v16b, so
        # the next `python Make_plots.py ...` does a full ~18-min re-derive
        # into *__derived_v16b_ttReco_v1.parquet, then ~3 min/run.  (v15m is
        # folded in.)  Old v16 parquets can be deleted.  To go back to the
        # pre-tt-reco cache with no re-derive: "derived_v15k_cbCandByScore_v1".
        # v16c (derived_v16c_ttReco_v1): + 3 new AK4-combinatorics masses --
        #   mass_bc_bestmW  = m(b^L c^L) of the pair with mass closest to m_W
        #                     (80 GeV) -- "best-m_W" analogue of mass_minDR_bc.
        #   mass_bbc_minDR  = m(b^L b^L c^L) of the most collimated triplet
        #                     (min summed pairwise dR).
        #   mass_bbc_bestmt = m(b^L b^L c^L) of the triplet with mass closest
        #                     to m_t (172.5 GeV).
        #   New cached fields -> full re-derive (~18 min first run); v16b
        #   vars fold in.
        #   ALSO v16c: + ak8_match_tbqq_wcb loaded, and the t^3 MAT class is
        #   split -- mat_cat 8 = t->bqq' (W->cs/cd/ud/us), NEW code 20 =
        #   t->bbc (fully-merged top whose W->cb).  Fixes the fully-merged
        #   W->cb top being buried in Rest by the sig_wbc override.
        # v16d (derived_v16d_ttReco_v1, 2026-09-02): + score_Dbc_{jb,jsup1,
        #   jsup2} -- the D_bc BDT re-evaluated on j_b (2nd-highest bc-score
        #   AK8), j^1 (highest-mSD) and j^2 (2nd-highest-mSD), via the same
        #   nth_by node picks that feed gpt_w_vs_top_jb.  SENTINEL where the
        #   jet is absent (n_ak8 < 2 for j_b / j^2).
        # v16g (derived_v16g_ttReco_v1, 2026-09-02): TOP-MASS RECO COLLAPSED
        #   to ONE spec `mt` = m(J + AK4), AK4 = first of bmk1..5 (top-5
        #   best-|m(J+AK4)-172.5| jets outside J) with top-consistency
        #   x = dR*pT(J+AK4)/(2*172.5) < WCB_TTRECO_XMT (=2.0).  No-candidate
        #   -> m = -1 (visible low bin, NOT SENTINEL).  bmk seeds cache
        #   {pt,eta,phi,m,x,pnb,pnuds,pnc}; bvuds_bestm = PNet b/(b+uds) of the
        #   EXACT winning AK4 (-1 if none).  mt1-10/dmt6/mt7/ptasym + hfuds_*
        #   specs all removed.  (v16f: + `ak4_pn_uds` + bmk{pt,eta,phi,m,x}.)
        #   Full re-derive (~25 min).
        # v16e (derived_v16e_ttReco_v1, 2026-09-02): + 2 "light-AK4" seeds
        #   (nearest AK4 outside J, dR-ranked; SENTINEL if none outside J):
        #   blite1_* = ak4_tag < 40 (neither b_L>=50 nor c_L 40-49); fallback
        #              = out-of-cone AK4 with lowest (ak4_pn_b + ak4_pn_c).
        #   bnb1_*   = ak4_tag < 50 (not b_L; c_L / untagged OK);   fallback
        #              = out-of-cone AK4 with lowest ak4_pn_b.
        #   Feed the plot-time m_ta_dr_lite / m_ta_dr_notbL.
        #   New cached fields -> full re-derive (~20 min); v16d vars fold in.
        # v16h (derived_v16h_ttReco_v1, 2026-09-03): + bmk{1..5}_tag (the
        #   ParticleNetAK4 WP integer of each best-mass AK4) so the flavour
        #   categorical of j* (the windowed pick, aka "the AK4 used by mt")
        #   can be built at plot time -> new spec `ttreco_jstar_flav`.
        #   Only new cached field; everything else folds in.  Full re-derive
        #   (~25 min).
        # v17 (derived_v17_2final_v1, 2026-09-04): NEW MC production
        #   `MC/scored_samples_2final_v20260902_rerun/` (*_merged_Skim.root).
        #   No derived-code change -- the tag bump is only to force a fresh
        #   derive off the new ntuples.  Branch content verified identical to
        #   the old production for everything the analysis reads; the only
        #   removed branch is `ak8_gpt_bqq` (was all-zero -> _cand_ak8 returns
        #   0 for it anyway, no behaviour change).  Full re-derive (~25 min).
        # v18 (2026-09-04): + the 25 finer GloParT sub-nodes from the 2final
        #   ntuples (_GPT_EXTRA_NODES -> ak8_gpt_<node>_0).  New cached
        #   fields -> full re-derive.
        self.cache_tag = "derived_v18_gptnodes_v1"

        # ------------------------------------------------------------------
        # Switches
        # ------------------------------------------------------------------
        self.enable_data = True
        self.use_cache = True
        self.force_reload = False

        # Entry-chunk size for the ROOT read/derive loop (uproot accepts a
        # memory-size string or an int number of entries). Keeps peak RSS
        # bounded so the re-derive fits the lxplus interactive cgroup;
        # override with $WCB_READ_STEP.
        self.read_step_size = os.environ.get("WCB_READ_STEP", "250 MB")

        self.lumi = 41.479

        # ------------------------------------------------------------------
        # BDT Dbc
        # ------------------------------------------------------------------
        self.dbc_model_path = "./train_bdt/dbc_bdt_output/bdt_dbc_model.pkl"

        # ------------------------------------------------------------------
        # Sample catalog
        # ------------------------------------------------------------------
        self.signal_groups = ["Wcb"]
        # MAT-SIGNAL mode: load only the signal_groups samples, draw only the
        # signal MAT lines.  Set from the CLI mode token in main().
        self.signal_only = False

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
            "score_cata_top_sum_norm": r"top-like score",
            "score_cata_top_sum2_norm": r"top-like-2 score (no Top $bc$)",
            # "Other"/"non"/"none-of-the-above" are all the same 6th EC output:
            # leading AK8 matched to no W/top/QCD category (gen flag ak8_match_non).
            "score_cata_non_norm": r"EC no-match score (not $W$/top/QCD)",
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
            "Cat_Top_bbc": "#C21FA8",
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
            "ak4_tag",
            "ak4_pn_c",
            "ak4_pn_b",
            "ak4_pn_uds",

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
        ] + ["ak8_gpt_" + _n for _n in _GPT_EXTRA_NODES] + [
            # (the 2final finer GloParT sub-nodes -- see _GPT_EXTRA_NODES)

            # leading-AK8 substructure (MAT jet-kinematics study)
            "ak8_rawFactor",
            "ak8_tau21",
            "ak8_tau32",
            "ak8_tau31",
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
            # contained/merged W->cb jet (2-prong c+b in the R=0.8 cone); the
            # W->cb analog of ak8_match_wcq. Used to split the Wcb line into
            # (a) merged-W->cb vs (b) everything else.
            "ak8_match_wqq_wcb",
            # fully-merged 3-prong top whose W->cb: the top-b AND both W->cb
            # daughters (c + b) are all inside J  (<=> ak8_type==4 & is_wbc).
            # Splits the t^3 MAT class into t->bbc vs t->bqq' (q'=u,d,s,c).
            "ak8_match_tbqq_wcb",
            # event-level hadronic-W decay code: 0=W->lv, 2=W->ud, 3=W->us,
            # 4=W->cq (cs/cd), 5=W->cb. Drives the "resolved" W->cq / W->qq'
            # MAT classes (decay happened, but no AK8 merged it).
            "w_decay",
            # event-level hadronic-Z decay code (|pdgId| of the Z daughter):
            # 0 = Z->ll / Z->vv / no gen Z ; 1/2/3 = Z->dd/uu/ss (light) ;
            # 4 = Z->cc ; 5 = Z->bb.  With genZ_pt this drives the merged
            # hadronic-Z flavour split of 'rest' (loose pT proxy for "merged":
            # J carries most of the Z momentum; no gen-Z direction stored).
            "z_decay",
            "genZ_pt",
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
    shortname = shortname.replace("_merged_Skim.root", "").replace("_merged.root", "")

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

    # Only the identifiers that actually appear in the expression are pulled
    # from the array. Converting every event-level field to numpy (~150 of
    # them) on every call was the dominant cost of the draw phase.
    referenced = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))

    local_dict = {}

    # ------------------------------------------------------------------
    # 1. Add the referenced event-level flat fields
    # ------------------------------------------------------------------
    for key in arrays.fields:
        if key not in referenced:
            continue
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

        # Only build the alias if the expression uses it.
        if alias not in referenced:
            continue

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


def scatter_hist_by_category(values, weights, category, bins, codes):
    """One pass over the events: for every code in ``codes`` return the
    flow-folded ``(counts, sum_w2)`` histogram of the subset ``category == code``.

    Equivalent to calling ``hist_with_flow`` once per code on a boolean-masked
    slice, but ~5x faster: a single arithmetic bin index (uniform grid) or one
    ``np.digitize`` (non-uniform), then one ``np.bincount`` into a
    ``(len(codes), nbins)`` grid instead of ``2 * len(codes)`` ``np.histogram``
    calls. Under/overflow are folded into the first/last visible bin (same
    convention as ``hist_with_flow``); values are assumed already cleaned of
    NaN/inf and the SENTINEL fill.
    """
    bins = np.asarray(bins, dtype=np.float64)
    nb = len(bins) - 1
    codes = list(codes)
    ncat = len(codes)

    # bin index, identical to np.histogram; under/overflow folded to edge bins
    idx = np.searchsorted(bins, values, side="right") - 1
    np.clip(idx, 0, nb - 1, out=idx)

    maxc = int(max(codes))
    row_of = np.full(maxc + 1, -1, dtype=np.int64)
    for i, c in enumerate(codes):
        row_of[c] = i
    cat = np.asarray(category)
    row = row_of[np.clip(cat, 0, maxc)]

    w = np.asarray(weights, dtype=np.float64)
    bad = row < 0                                    # category not in `codes`
    if bad.any():
        row = np.where(bad, 0, row)
        w = np.where(bad, 0.0, w)

    flat = row * nb + idx
    size = ncat * nb
    counts = np.bincount(flat, weights=w, minlength=size)[:size].reshape(ncat, nb)
    sumw2 = np.bincount(flat, weights=w * w, minlength=size)[:size].reshape(ncat, nb)
    return {c: (counts[i], sumw2[i]) for i, c in enumerate(codes)}


# ======================================================================================
# 3. Data manager
# ======================================================================================

# Event-level parquet columns that must ALWAYS be read from the cache,
# independent of which plot specs are active:
#   * infra   -- weights / truth codes consumed by event_context, mat_report,
#                the true/mixed draw paths and the MAT significance split.
#   * runtime -- the inputs DataManager._attach_runtime_fields reads to rebuild
#                the plot-time-only fields (m_ta*, m_ta_dr/drb/bestm, the three
#                gpt_bc_vs_* ratios, sdwin_tag_nj2, ...).  Each block there is
#                guarded by `<inputs> <= set(arr.fields)`, so a pruned input
#                would SILENTLY drop the derived field and its plot.
_CACHE_ALWAYS_COLUMNS = frozenset({
    # --- infra ---
    "weights", "true_cat", "mat_cat", "is_signal", "is_qcd", "n_ak8",
    "ak8_type", "ak8_n_b_in_jet", "ak8_n_c_in_jet",
    # --- _attach_runtime_fields inputs: SD masses / EC & GloParT scores ---
    "ak8_sdmass_0", "ak8_sdmass_sub_bc_0",
    "ak8_sdmass_maxmass_0", "ak8_sdmass_sub_mass_0",
    "score_cata_top_bq_norm", "score_cata_top_bqq_norm",
    "ak8_gpt_bc_0", "ak8_gpt_cc_0", "ak8_gpt_bb_0",
    "ak8_gpt_bs_0", "ak8_gpt_cs_0", "ak8_gpt_qq_0",
    # --- _attach_runtime_fields inputs: t_a top-mass reco (mt) + AK4 diags ---
    #   bdr1    = dR-closest AK4 to J       -> bdr1_flav, bvuds_near
    #   bbestm1 = best-|m_t| AK4              -> bbestm1_flav
    #   bmk1..5 = top-5 best-|m_t| AK4 + x/pnb/pnc/pnuds -> mt (x<2 window,
    #             plot time, env WCB_TTRECO_XMT), {b,c,bc}vuds_bestm, jstar_flav
    #   (btt*/bL*/bdr2/blite1/bnb1 seeds stay in the parquet but are no longer
    #    read -- their 2-top / ptasym / mt1-6 specs are all removed.)
    "ak8_pt_0", "ak8_eta_0", "ak8_phi_0",
    "v_pt", "v_eta", "v_phi", "v_mass",
    "bdr1_pt", "bdr1_eta", "bdr1_phi", "bdr1_m", "bdr1_tag",
    "bbestm1_pt", "bbestm1_eta", "bbestm1_phi", "bbestm1_m", "bbestm1_tag",
    "bdr1_pnuds", "bdr1_pnb",
    # dR_ja_bL12 (2026-09-04): dR of the 2 loose-b AK4 nearest J
    "bL1_pt", "bL1_eta", "bL1_phi", "bL2_pt", "bL2_eta", "bL2_phi",
    "bmk1_pt", "bmk1_eta", "bmk1_phi", "bmk1_m", "bmk1_x", "bmk1_tag", "bmk1_pnb", "bmk1_pnc", "bmk1_pnuds",
    "bmk2_pt", "bmk2_eta", "bmk2_phi", "bmk2_m", "bmk2_x", "bmk2_tag", "bmk2_pnb", "bmk2_pnc", "bmk2_pnuds",
    "bmk3_pt", "bmk3_eta", "bmk3_phi", "bmk3_m", "bmk3_x", "bmk3_tag", "bmk3_pnb", "bmk3_pnc", "bmk3_pnuds",
    "bmk4_pt", "bmk4_eta", "bmk4_phi", "bmk4_m", "bmk4_x", "bmk4_tag", "bmk4_pnb", "bmk4_pnc", "bmk4_pnuds",
    "bmk5_pt", "bmk5_eta", "bmk5_phi", "bmk5_m", "bmk5_x", "bmk5_tag", "bmk5_pnb", "bmk5_pnc", "bmk5_pnuds",
})


def columns_for_cache(cfg):
    """The set of event-level parquet columns this run will actually touch.

    The derived cache carries ~160 flat columns, most of them feeding plot
    specs that are currently commented out.  Reading the full set is what
    pushes the load-phase RSS over the lxplus memory cgroup (tt-semi alone is
    ~6 GB on disk / ~13 GB unpacked).  Restricting `ak.from_parquet` to the
    columns the active specs + runtime fields need roughly halves it.

    Returns a set of names, or None to load everything (safe fallback).
    Set env WCB_LOAD_ALL_COLS=1 to force the old full-load behaviour.
    """
    if os.environ.get("WCB_LOAD_ALL_COLS"):
        return None

    keep = set(_CACHE_ALWAYS_COLUMNS)
    try:
        mode = getattr(cfg, "plot_mode", "MAT")
        if mode.startswith("MAT"):
            specs = build_mat_plot_settings(getattr(cfg, "mat_sel", "PRE"),
                                            getattr(cfg, "signal_only", False))
        else:
            specs = build_plot_settings()
        # ROC settings share the cache; include their fields even though the
        # ROC plots are usually disabled, so flipping them back on Just Works.
        try:
            specs = list(specs) + build_roc_settings() + build_dbc_roc_settings()
        except Exception:
            pass
        for p in specs:
            for k in ("var", "bin_var", "score"):
                v = p.get(k)
                if v:
                    keep.add(v)
            keep |= set(re.findall(
                r"[A-Za-z_][A-Za-z0-9_]*",
                normalize_cut_expr(p.get("cut", "1"))))
    except Exception as e:
        print(f"[WARN] cache-column introspection failed ({e}); "
              f"loading all columns")
        return None
    return keep


class DataManager:
    def __init__(self, cfg):
        self.cfg = cfg
        ensure_dir(cfg.cache_path)

        self.dbc_eval = DbcEvaluator(
            mode="bdt",
            model_path=self.cfg.dbc_model_path,
        )

        # Columns to pull from the derived parquet cache (see columns_for_cache).
        # None -> read every column (old behaviour / WCB_LOAD_ALL_COLS=1).
        self._keep_cols = columns_for_cache(cfg)
        self._cache_cols_logged = False

    def _read_cache_array(self, cache_file):
        """ak.from_parquet(cache_file), restricted to self._keep_cols when set.

        The request is intersected with the file's actual schema, so a
        runtime-only field name in the keep set (m_ta_bestm, gpt_bc_vs_ccbb,
        ...) or a spec referencing a column absent from an older cache tag is
        simply not requested rather than raising.
        """
        if not self._keep_cols:
            return ak.from_parquet(cache_file)
        try:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(cache_file).names)
        except Exception:
            return ak.from_parquet(cache_file)
        want = sorted(self._keep_cols & have)
        if not want or len(want) >= len(have):
            return ak.from_parquet(cache_file)
        if not self._cache_cols_logged:
            print(f"[INFO] Cache column prune: reading {len(want)}/{len(have)} "
                  f"event-level columns (set WCB_LOAD_ALL_COLS=1 to disable)")
            self._cache_cols_logged = True
        return ak.from_parquet(cache_file, columns=want)

    def load_all(self):
        samples = []

        try:
            mc_files = [
                os.path.join(self.cfg.mc_path, f)
                for f in os.listdir(self.cfg.mc_path)
                if f.endswith(".root")
            ]
        except OSError as e:
            # MC ROOT area unreadable (EOS down / permissions). Fall back to
            # whatever parquet caches exist for the current cache_tag -- the
            # synthetic ".root" paths below never get opened because
            # load_one_file hits the cache file first.
            print(f"[WARN] MC path not readable ({e}); "
                  f"falling back to cached parquets for tag "
                  f"'{self.cfg.cache_tag}'.")
            _sfx = f"__{self.cfg.cache_tag}.parquet"
            try:
                _cached = sorted(f for f in os.listdir(self.cfg.cache_path)
                                 if f.endswith(_sfx))
            except OSError:
                _cached = []
            mc_files = [
                os.path.join(self.cfg.mc_path, f[:-len(_sfx)] + ".root")
                for f in _cached
            ]
            if not mc_files:
                raise RuntimeError(
                    f"MC path unreadable and no '{_sfx}' caches in "
                    f"{self.cfg.cache_path} -- cannot run.")

        print(f"[INFO] Found {len(mc_files)} MC files in {self.cfg.mc_path}")

        for i, path in enumerate(mc_files):
            group = get_sample_group(path, self.cfg.catalog_sample)

            # MAT-SIGNAL: skip every non-signal sample at load time.
            if self.cfg.signal_only and group not in self.cfg.signal_groups:
                continue

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
                return self._attach_runtime_fields(
                    self._read_cache_array(cache_file), self._keep_cols)
            except Exception as e:
                print(f"\n[WARN] Broken cache: {cache_file}")
                print(f"[WARN] {e}")
                print("[WARN] Reloading from ROOT.")

        # Read + derive in entry chunks: a single tree.arrays() over a big
        # tree (tt-semi is ~4 GB) blows the lxplus per-job memory cgroup.
        # build_derived_array is fully per-event vectorised, so deriving each
        # chunk and concatenating the (much smaller) derived output is
        # equivalent and keeps the peak RSS bounded by one chunk of raw.
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

                parts = []
                for raw in tree.iterate(
                    branches,
                    library="ak",
                    step_size=self.cfg.read_step_size,
                ):
                    if len(raw) == 0:
                        continue
                    parts.append(self.build_derived_array(
                        raw=raw,
                        is_data=is_data,
                        sample_group=sample_group,
                        filename=basename,
                        met_pt_branch=met_pt_branch,
                        met_phi_branch=met_phi_branch,
                    ))
                    del raw
                    gc.collect()

        except Exception as e:
            print(f"\n[WARN] Cannot read {root_path}: {e}")
            return None

        if not parts:
            return None

        arr = parts[0] if len(parts) == 1 else ak.concatenate(parts)
        del parts
        gc.collect()

        if self.cfg.use_cache:
            try:
                ak.to_parquet(arr, cache_file)
            except (OSError, PermissionError) as e:
                print(f"\n[WARN] Could not write cache {cache_file}: {e}")
                print("[WARN] Continuing without caching this file "
                      "(set $WCB_CACHE_DIR to a writable path).")

        return self._attach_runtime_fields(arr, self._keep_cols)

    @staticmethod
    def _attach_runtime_fields(arr, need=None):
        """Attach plot-time-only derived fields that are deliberately NOT
        written to the parquet cache, so they work under the pinned-cache /
        MC-ROOT-EOS-down regime with no re-derive.

        All fields here are restricted to events with >= 2 AK8 jets: for
        n_ak8 < 2 the value is set to SENTINEL so those events drop out of
        the histogram cleanly (multiplying by the 0/1 mask instead would
        pile them into the first bin, and would also turn the already-
        SENTINEL 1-jet fills of the sub-leading slots into 0).

          - ak8_sdmass_{ja,jb,jsup1,jsup2}_nj2 : the four soft-drop masses
            (J / j_b = highest / 2nd-highest GloParT bc score;
             j^1 / j^2 = highest / 2nd-highest mSD).
          - sdwin_tag_nj2 : 2-bin categorical, value = [mSD(j_b) in W] with
            W = [65,105] GeV  (0 = j_b out of the W-window, 1 = j_b in).
            Filled only for n_ak8>=2 events with mSD(J) ALSO in W (that gate
            is implicit -- SENTINEL otherwise), all truth classes incl. signal.
        """
        if arr is None or "n_ak8" not in arr.fields:
            return arr

        # `need` = the set of field names any active plot spec / cut references
        # (== DataManager._keep_cols).  A heavy block is SKIPPED when none of
        # its outputs are needed -- keeps the load-phase RSS bounded (the tt /
        # t_a reco + ptasym blocks each allocate several GB of temporaries per
        # file).  need is None => compute everything (WCB_LOAD_ALL_COLS).
        def _want(*names):
            return need is None or not set(names).isdisjoint(need)

        # ja_truth_cat : 14-bin categorical of what J / the event is at gen
        # level, from the RAW cached mat_cat + true_cat + ak8_n_c_in_jet (the
        # code-6 -> 20/21/22 and code-8 -> 23 splits event_context does at plot
        # time are replicated here).
        # Drives the standalone MAT_ttreco_ja_truth composition plot.
        # EACH BIN == EXACTLY ONE DRAWN OVERLAY LINE (2026-09-04, user):
        # the 4 top-proxy categories each get their own bin, and every
        # mat_cat that draws as the grey "Rest" line (3, 9, 10, 11, 12 --
        # W !=j_a + resolved W + rest) is pooled into the single "rest" bin
        # (was split "resolved bkg" | "rest").
        #   SIGNAL (all W->cb):
        #     0  W(cb)          2-prong merged W->cb jet         mat_cat 0/5
        #     1  t^2(b'c)       top-b + W's c merged             mat_cat 6 & Top_bc
        #     2  t^2(b'b)       top-b + W's b merged             mat_cat 6 & Top_bq
        #     3  t^3(b'bc)      fully-merged top t->(bbc)   mat_cat 20 | 6&Top_bqq
        #     4  c,b,b' res     W->cb with c,b NOT merged in J   mat_cat 6 (else)
        #   BACKGROUND -- hadronic-top proxy (one bin each):
        #     5  t^2(b'c) proxy real top, W's c prong merged     mat_cat 2
        #     6  t^2(b'q)       real top, W's light prong merged mat_cat 7
        #     7  t^3(b'cq)      full 3-merge, c prong in J       mat_cat 8 & n_c>=1
        #     8  t^3(b'qq)      full 3-merge, no c prong         mat_cat 8 & n_c<1
        #   BACKGROUND -- hadronic V (=j_a only, own line):
        #     9  W(cq)          merged W->cq jet, IS j_a         mat_cat 1
        #    10  W(qq)          merged light-W jet, IS j_a       mat_cat 4
        #    11  Z(cc/bb)       merged heavy-flavour Z jet       mat_cat 14/15
        #    12  Z(qq)          merged light-Z jet               mat_cat 13
        #    13  rest           the grey "Rest" line: W !=j_a (9/11),
        #                       resolved W (10/12), everything else (3)
        # true_cat codes (cfg.true_cat_codes): Top_bqq 1, Top_bc 2, Top_bq 3.
        if _want("ja_truth_cat") and {"mat_cat", "true_cat"} <= set(arr.fields):
            _mc = ak.to_numpy(arr["mat_cat"])
            _tc = ak.to_numpy(arr["true_cat"])
            _nc = (ak.to_numpy(arr["ak8_n_c_in_jet"])
                   if "ak8_n_c_in_jet" in arr.fields else np.zeros(len(_mc)))
            _jt = np.full(len(_mc), 13.0, dtype=np.float32)     # rest (Rest line)
            _jt[_mc == 13] = 12.0                               # Z->(qq) merged
            _jt[np.isin(_mc, (14, 15))] = 11.0                  # Z->(cc/bb) merged
            _jt[_mc == 4] = 10.0                                # W->(qq) merged (=j_a)
            _jt[_mc == 1] = 9.0                                 # W->(cq) merged (=j_a)
            _jt[(_mc == 8) & (_nc < 1)] = 8.0                   # t^3(b'qq) full merge
            _jt[(_mc == 8) & (_nc >= 1)] = 7.0                  # t^3(b'cq) full merge
            _jt[_mc == 7] = 6.0                                 # t^2(b'q) proxy
            _jt[_mc == 2] = 5.0                                 # t^2(b'c) proxy
            _c6 = _mc == 6
            _jt[_c6] = 4.0                                      # (cb) resolved (SIGNAL)
            _jt[_c6 & (_tc == 2)] = 1.0                         # t^2(b'c) partial
            _jt[_c6 & (_tc == 3)] = 2.0                         # t^2(b'b) partial
            _jt[_c6 & (_tc == 1)] = 3.0                         # full merge
            _jt[_mc == 20] = 3.0                                # full merge t->(bbc)
            _jt[np.isin(_mc, (0, 5))] = 0.0                     # 2-prong W->cb merge
            arr = ak.with_field(arr, _jt, "ja_truth_cat")

        two = ak.to_numpy(arr["n_ak8"]) >= 2
        if _want("ak8_sdmass_ja_nj2", "ak8_sdmass_jb_nj2",
                 "ak8_sdmass_jsup1_nj2", "ak8_sdmass_jsup2_nj2"):
            for src, dst in (
                ("ak8_sdmass_0",           "ak8_sdmass_ja_nj2"),
                ("ak8_sdmass_sub_bc_0",    "ak8_sdmass_jb_nj2"),
                ("ak8_sdmass_maxmass_0",   "ak8_sdmass_jsup1_nj2"),
                ("ak8_sdmass_sub_mass_0",  "ak8_sdmass_jsup2_nj2"),
            ):
                if src in arr.fields:
                    v = ak.to_numpy(arr[src]).astype(np.float32)
                    arr = ak.with_field(
                        arr, np.where(two, v, SENTINEL).astype(np.float32), dst)

        if _want("sdwin_tag_nj2") and \
           {"ak8_sdmass_0", "ak8_sdmass_sub_bc_0"} <= set(arr.fields):
            _WLO, _WHI = 65.0, 105.0
            _msa = ak.to_numpy(arr["ak8_sdmass_0"]).astype(np.float32)
            _msb = ak.to_numpy(arr["ak8_sdmass_sub_bc_0"]).astype(np.float32)
            _ja_in = (_msa >= _WLO) & (_msa <= _WHI)   # gate: mSD(J) in W-window
            _jb_in = (_msb >= _WLO) & (_msb <= _WHI)   # bin:  mSD(j_b) in W-window
            _keep = two & _ja_in                      # n_ak8>=2 AND J in window
            arr = ak.with_field(
                arr, np.where(_keep, _jb_in.astype(np.float32), SENTINEL
                              ).astype(np.float32),
                "sdwin_tag_nj2")

        # ---- "top-like-2" score (PLOT-TIME; no re-derive) ------------------
        # Same summed-EC top-like probability as score_cata_top_sum_norm but
        # WITHOUT the Top(bc) term: Top(bq) + Top(bqq') only.  Built from the
        # two cached normalised EC scores, so it works under the pinned cache.
        if _want("score_cata_top_sum2_norm") and {"score_cata_top_bq_norm",
            "score_cata_top_bqq_norm"} <= set(arr.fields):
            _tbq = ak.to_numpy(arr["score_cata_top_bq_norm"]).astype(np.float32)
            _tbqq = ak.to_numpy(arr["score_cata_top_bqq_norm"]).astype(np.float32)
            arr = ak.with_field(arr, (_tbq + _tbqq).astype(np.float32),
                                "score_cata_top_sum2_norm")

        # ---- bc / (bc + cc + bb) GloParT ratio (PLOT-TIME; no re-derive) ---
        # heavy-flavour 2-prong discriminant: is the merged jet a genuine
        # W(cb) or a cc / bb resonance?  Built from the cached raw GloParT
        # leading-jet scores.
        if _want("gpt_bc_vs_ccbb") and {"ak8_gpt_bc_0", "ak8_gpt_cc_0",
            "ak8_gpt_bb_0"} <= set(arr.fields):
            _gbc = ak.to_numpy(arr["ak8_gpt_bc_0"]).astype(np.float64)
            _gcc = ak.to_numpy(arr["ak8_gpt_cc_0"]).astype(np.float64)
            _gbb = ak.to_numpy(arr["ak8_gpt_bb_0"]).astype(np.float64)
            _den = _gbc + _gcc + _gbb
            _rat = np.where(_den > 0.0, _gbc / np.maximum(_den, 1e-12), 0.0)
            arr = ak.with_field(arr, _rat.astype(np.float32),
                                "gpt_bc_vs_ccbb")
            del _gbc, _gcc, _gbb, _den, _rat

        # ---- (bc+bs+cs) / (bc+bs+cs+cc+bb+qq)  (PLOT-TIME; no re-derive) ---
        # mixed heavy-flavour 2-prong (bc/bs/cs) vs all six 2-prong GloParT
        # nodes: fraction of the resonance-like probability carried by the
        # b/c-mixed decay modes rather than cc / bb / light qq'.
        if _want("gpt_hfmix_vs_2prong") and {"ak8_gpt_bc_0", "ak8_gpt_bs_0",
            "ak8_gpt_cs_0", "ak8_gpt_cc_0", "ak8_gpt_bb_0",
            "ak8_gpt_qq_0"} <= set(arr.fields):
            _num6 = (ak.to_numpy(arr["ak8_gpt_bc_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_bs_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_cs_0"]).astype(np.float64))
            _den6 = (_num6
                     + ak.to_numpy(arr["ak8_gpt_cc_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_bb_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_qq_0"]).astype(np.float64))
            _rat6 = np.where(_den6 > 0.0,
                             _num6 / np.maximum(_den6, 1e-12), 0.0)
            arr = ak.with_field(arr, _rat6.astype(np.float32),
                                "gpt_hfmix_vs_2prong")
            del _num6, _den6, _rat6

        # ---- bc / (bc+bs+cs+cc+bb)  (PLOT-TIME; no re-derive) -------------
        # bc vs the five heavy-flavour 2-prong GloParT nodes (light qq'
        # excluded): purity of the bc assignment among HF resonance modes.
        if _want("gpt_bc_vs_hf2prong") and {"ak8_gpt_bc_0", "ak8_gpt_bs_0",
            "ak8_gpt_cs_0", "ak8_gpt_cc_0", "ak8_gpt_bb_0"} <= set(arr.fields):
            _gbc5 = ak.to_numpy(arr["ak8_gpt_bc_0"]).astype(np.float64)
            _den5 = (_gbc5
                     + ak.to_numpy(arr["ak8_gpt_bs_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_cs_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_cc_0"]).astype(np.float64)
                     + ak.to_numpy(arr["ak8_gpt_bb_0"]).astype(np.float64))
            _rat5 = np.where(_den5 > 0.0,
                             _gbc5 / np.maximum(_den5, 1e-12), 0.0)
            arr = ak.with_field(arr, _rat5.astype(np.float32),
                                "gpt_bc_vs_hf2prong")
            del _gbc5, _den5, _rat5
        gc.collect()

        # ---- tt reconstruction (PLOT-TIME; retunable with NO re-derive) ----
        #   t_a = J + b_h      hadronic top      (expect m ~ 172 +- 20%)
        #   t_b = W(lv) + b_l    leptonic top      (m biased low: nu p_z = 0)
        # b_h / b_l are drawn from the two cached seeds (btt1/btt2 = the two
        # highest-b-score AK4 jets outside J) by:
        #   Rule 1 (dR)      : the seed nearer J is b_h, the other is b_l.
        #   Rule 2 (pT bal.) : if the seeds are ~equidistant from J
        #                      (|d(dR)| < WCB_TTRECO_AMB, default 0.4), take
        #                      instead the pairing with the smaller top
        #                      pT-asymmetry |pt_ta-pt_tb|/(pt_ta+pt_tb).
        # Single-seed events assign that one b by dR (J vs W(lv)); the
        # other top (and hence dphi_tatb / ptasym_tt) stays SENTINEL.
        # Env knobs (no re-derive): WCB_TTRECO_AMB, WCB_TTRECO_BWP (min
        # ak4_tag WP required of a seed, default -1 = accept the b-score pick).
        _N = len(arr) if arr is not None else 0
        _tt_in = {"ak8_pt_0", "ak8_eta_0", "ak8_phi_0", "ak8_sdmass_0",
                  "v_pt", "v_eta", "v_phi", "v_mass",
                  "btt1_pt", "btt1_eta", "btt1_phi", "btt1_m",
                  "btt2_pt", "btt2_eta", "btt2_phi", "btt2_m"}
        # mt7 = J + b_h (b_h = one of the 2 highest-PNet-b AK4 outside J,
        #          split from the other by dR + a pT-balance tie-break).
        # ptasym_tt still needs the leptonic-side pairing internally, but it is
        # not exposed as a mass plot.
        _tt_out = ("mt7", "pt_ta", "pt_tb", "ptasym_tt",
                   "dphi_tatb", "dR_ja_bh", "dR_wlv_bl")
        if _want(*_tt_out) and arr is not None and _tt_in <= set(arr.fields):
            _amb = float(os.environ.get("WCB_TTRECO_AMB", "0.4"))
            _bwp = float(os.environ.get("WCB_TTRECO_BWP", "-1"))

            # float64 for the 4-vector / invariant-mass math: m^2 = E^2 - p^2
            # is a difference of large numbers and loses ~O(50 GeV) in float32.
            def _f(name):
                return ak.to_numpy(arr[name]).astype(np.float64)

            def _p4(pt, eta, phi, m):
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                pz = pt * np.sinh(np.clip(eta, -10.0, 10.0))
                E = np.sqrt(px * px + py * py + pz * pz
                            + np.maximum(m, 0.0) ** 2)
                return np.stack([px, py, pz, E])

            def _m1(a):
                return np.sqrt(np.maximum(
                    a[3] ** 2 - a[0] ** 2 - a[1] ** 2 - a[2] ** 2, 0.0))

            def _ptv(a):
                return np.hypot(a[0], a[1])

            def _phi_of(a):
                return np.arctan2(a[1], a[0])

            def _dphi(p1, p2):
                d = np.abs(p1 - p2)
                return np.where(d > np.pi, 2.0 * np.pi - d, d)

            def _dR(e1, p1, e2, p2):
                return np.hypot(e1 - e2, _dphi(p1, p2))

            def _asym(x, y):
                s = x + y
                return np.where(s > 0.0, np.abs(x - y) / s, 9.9)

            _ja = _p4(_f("ak8_pt_0"), _f("ak8_eta_0"),
                      _f("ak8_phi_0"), _f("ak8_sdmass_0"))
            _wlv = _p4(_f("v_pt"), _f("v_eta"), _f("v_phi"), _f("v_mass"))
            _b1 = _p4(_f("btt1_pt"), _f("btt1_eta"), _f("btt1_phi"), _f("btt1_m"))
            _b2 = _p4(_f("btt2_pt"), _f("btt2_eta"), _f("btt2_phi"), _f("btt2_m"))

            _jae, _jap = _f("ak8_eta_0"), _f("ak8_phi_0")
            _we, _wp = _f("v_eta"), _f("v_phi")
            _b1e, _b1p = _f("btt1_eta"), _f("btt1_phi")
            _b2e, _b2p = _f("btt2_eta"), _f("btt2_phi")
            _t1 = _f("btt1_tag") if "btt1_tag" in arr.fields else np.full(_N, 999.0)
            _t2 = _f("btt2_tag") if "btt2_tag" in arr.fields else np.full(_N, 999.0)
            _b1ok = (_f("btt1_pt") > 0.0) & (_t1 >= _bwp)
            _b2ok = (_f("btt2_pt") > 0.0) & (_t2 >= _bwp)

            _dR1a = _dR(_b1e, _b1p, _jae, _jap)
            _dR2a = _dR(_b2e, _b2p, _jae, _jap)
            _dR1w = _dR(_b1e, _b1p, _we, _wp)
            _dR2w = _dR(_b2e, _b2p, _we, _wp)

            _taA, _tbA = _ja + _b1, _wlv + _b2      # hypo A: b_h = seed 1
            _taB, _tbB = _ja + _b2, _wlv + _b1      # hypo B: b_h = seed 2
            _asymA = _asym(_ptv(_taA), _ptv(_tbA))
            _asymB = _asym(_ptv(_taB), _ptv(_tbB))
            _both = _b1ok & _b2ok
            _amask = np.abs(_dR1a - _dR2a) < _amb
            _useA = np.where(_amask, _asymA <= _asymB, _dR1a <= _dR2a)

            _m_ta = np.full(_N, SENTINEL); _m_tb = np.full(_N, SENTINEL)
            _pt_ta = np.full(_N, SENTINEL); _pt_tb = np.full(_N, SENTINEL)
            _asym_tt = np.full(_N, SENTINEL); _dphi_tt = np.full(_N, SENTINEL)
            _dRh = np.full(_N, SENTINEL); _dRl = np.full(_N, SENTINEL)

            for _sel, _ta, _tb, _rh, _rl in (
                (_both & _useA, _taA, _tbA, _dR1a, _dR2w),
                (_both & ~_useA, _taB, _tbB, _dR2a, _dR1w),
            ):
                _m_ta[_sel] = _m1(_ta)[_sel]
                _m_tb[_sel] = _m1(_tb)[_sel]
                _pt_ta[_sel] = _ptv(_ta)[_sel]
                _pt_tb[_sel] = _ptv(_tb)[_sel]
                _asym_tt[_sel] = _asym(_ptv(_ta), _ptv(_tb))[_sel]
                _dphi_tt[_sel] = _dphi(_phi_of(_ta), _phi_of(_tb))[_sel]
                _dRh[_sel] = _rh[_sel]
                _dRl[_sel] = _rl[_sel]

            # single-seed events: assign the one b by dR, other top SENTINEL
            for _ok, _other, _be, _bp, _bp4, _dRa, _dRwv in (
                (_b1ok & ~_b2ok, _b2ok, _b1e, _b1p, _b1, _dR1a, _dR1w),
                (_b2ok & ~_b1ok, _b1ok, _b2e, _b2p, _b2, _dR2a, _dR2w),
            ):
                _h = _ok & (_dRa <= _dRwv)
                _l = _ok & (_dRa > _dRwv)
                _ta1, _tb1 = _ja + _bp4, _wlv + _bp4
                _m_ta[_h] = _m1(_ta1)[_h]
                _pt_ta[_h] = _ptv(_ta1)[_h]
                _dRh[_h] = _dRa[_h]
                _m_tb[_l] = _m1(_tb1)[_l]
                _pt_tb[_l] = _ptv(_tb1)[_l]
                _dRl[_l] = _dRwv[_l]

            for _nm, _v in (("mt7", _m_ta),
                            ("pt_ta", _pt_ta), ("pt_tb", _pt_tb),
                            ("ptasym_tt", _asym_tt), ("dphi_tatb", _dphi_tt),
                            ("dR_ja_bh", _dRh), ("dR_wlv_bl", _dRl)):
                arr = ak.with_field(arr, _v.astype(np.float32), _nm)
            del (_ja, _wlv, _b1, _b2, _taA, _tbA, _taB, _tbB,
                 _asymA, _asymB, _dR1a, _dR2a, _dR1w, _dR2w, _useA,
                 _m_ta, _m_tb, _pt_ta, _pt_tb, _asym_tt, _dphi_tt, _dRh, _dRl)
            gc.collect()
        elif _want(*_tt_out) and arr is not None:
            for _nm in _tt_out:
                if _nm not in arr.fields:
                    arr = ak.with_field(
                        arr, np.full(_N, SENTINEL, dtype=np.float32), _nm)

        # ---- top pT-asymmetry variants (PLOT-TIME) ------------------------
        # t_a = J + b_h  (hadronic) ; t_b = W(lv) + b_l  (leptonic).
        # b_h and b_l are picked from the loose-b AK4 seeds bL1..bL3 (already
        # ranked by dR to J).  Only the b_h RULE changes between variants;
        # b_l is always "the OTHER present loose-b seed nearest W(lv) in dR",
        # so each plot isolates one b_h criterion:
        #   ptasym_dr    : b_h = loose-b AK4 nearest J in dR   (= bL1)
        #   ptasym_bestm : b_h = loose-b AK4 giving m(J+b_h) closest to M_top
        #                  (env WCB_TTRECO_MTOP, default 172.5)
        # Needs >= 2 present loose-b seeds; SENTINEL otherwise.
        _ps_in = {"ak8_pt_0", "ak8_eta_0", "ak8_phi_0", "ak8_sdmass_0",
                  "v_pt", "v_eta", "v_phi", "v_mass", "bL1_pt", "bL2_pt"}
        _ps_out = ("ptasym_dr",)
        if _want(*_ps_out) and arr is not None and _ps_in <= set(arr.fields):
            _mtop_ps = float(os.environ.get("WCB_TTRECO_MTOP", "172.5"))
            _NN = len(arr)

            def _h(name):   # float64: invariant-mass precision (see _f above)
                return ak.to_numpy(arr[name]).astype(np.float64)

            def _p4h(pt, eta, phi, m):
                px = pt * np.cos(phi); py = pt * np.sin(phi)
                pz = pt * np.sinh(np.clip(eta, -10.0, 10.0))
                E = np.sqrt(px * px + py * py + pz * pz
                            + np.maximum(m, 0.0) ** 2)
                return np.stack([px, py, pz, E])

            def _pth(a):
                return np.hypot(a[0], a[1])

            def _mh(a):
                return np.sqrt(np.maximum(
                    a[3] ** 2 - a[0] ** 2 - a[1] ** 2 - a[2] ** 2, 0.0))

            def _dR(e1, p1, e2, p2):
                d = np.abs(p1 - p2)
                d = np.where(d > np.pi, 2.0 * np.pi - d, d)
                return np.hypot(e1 - e2, d)

            _jaN = _p4h(_h("ak8_pt_0"), _h("ak8_eta_0"),
                        _h("ak8_phi_0"), _h("ak8_sdmass_0"))
            _wN = _p4h(_h("v_pt"), _h("v_eta"), _h("v_phi"), _h("v_mass"))
            _jae, _jap = _h("ak8_eta_0"), _h("ak8_phi_0")
            _we, _wp = _h("v_eta"), _h("v_phi")

            # per loose-b seed: (present, m(J+bL), dR(bL,J), dR(bL,W),
            #                    pt(J+bL), pt(W+bL)) -- stacked (K, N) float32
            # (float32 storage is fine here: pT and dR keep full precision, and
            #  m is only used to *rank* the seeds, not as a plotted value).
            _ok_l, _mta_l, _dRj_l, _dRw_l, _pta_l, _ptb_l = [], [], [], [], [], []
            for _s in ("bL1", "bL2", "bL3"):
                if _s + "_pt" not in arr.fields:
                    continue
                _e, _p = _h(_s + "_eta"), _h(_s + "_phi")
                _pf = _p4h(_h(_s + "_pt"), _e, _p, _h(_s + "_m"))
                _ta4 = _jaN + _pf
                _tb4 = _wN + _pf
                _ok_l.append(ak.to_numpy(arr[_s + "_pt"]) > 0.0)
                _mta_l.append(_mh(_ta4).astype(np.float32))
                _dRj_l.append(_dR(_e, _p, _jae, _jap).astype(np.float32))
                _dRw_l.append(_dR(_e, _p, _we, _wp).astype(np.float32))
                _pta_l.append(_pth(_ta4).astype(np.float32))
                _ptb_l.append(_pth(_tb4).astype(np.float32))
                del _e, _p, _pf, _ta4, _tb4
            del _jaN, _wN, _jae, _jap, _we, _wp

            _ok = np.stack(_ok_l);   del _ok_l
            _mta = np.stack(_mta_l); del _mta_l
            _dRj = np.stack(_dRj_l); del _dRj_l
            _dRw = np.stack(_dRw_l); del _dRw_l
            _pta = np.stack(_pta_l); del _pta_l
            _ptb = np.stack(_ptb_l); del _ptb_l
            _idxN = np.arange(_NN)
            _have2 = _ok.sum(axis=0) >= 2

            def _ptasym_variant(_bh_metric):
                _bh = np.argmin(np.where(_ok, _bh_metric, np.inf), axis=0)
                _blm = np.where(_ok, _dRw, np.inf)
                _blm[_bh, _idxN] = np.inf               # exclude the b_h seed
                _bl = np.argmin(_blm, axis=0)
                _pa = _pta[_bh, _idxN]
                _pb = _ptb[_bl, _idxN]
                _s = _pa + _pb
                _v = np.where(_s > 0.0, np.abs(_pa - _pb) / np.maximum(_s, 1e-9),
                              SENTINEL)
                return np.where(_have2, _v, SENTINEL).astype(np.float32)

            for _nm, _v in (
                ("ptasym_dr",    _ptasym_variant(_dRj)),
            ):
                arr = ak.with_field(arr, _v, _nm)
            del _ok, _mta, _dRj, _dRw, _pta, _ptb, _have2
            gc.collect()
        elif _want(*_ps_out) and arr is not None:
            for _nm in _ps_out:
                if _nm not in arr.fields:
                    arr = ak.with_field(
                        arr, np.full(_N, SENTINEL, dtype=np.float32), _nm)

        # ---- t_a-ONLY top-mass reco  (t_a = J + one AK4 OUTSIDE the cone) --
        #   mt  = m(J + AK4) for the FIRST of the top-5 best-|m - M_top| AK4
        #         (seeds bmk1..bmk5) with top-consistency
        #         x = dR(AK4,J)*pT(J+AK4)/(2*172.5) < WCB_TTRECO_XMT (=2.0).
        #         M_top = WCB_TTRECO_MTOP env, default 172.5 GeV.
        #
        # FULLY-MERGED-TOP EXCEPTION (2026-09-04, user):
        #   mt = mSD(J)  whenever  mSD(J) > WCB_TTRECO_MRG_MSD (=130 GeV)
        #                          OR  no compatible j* was found.
        #   i.e. J alone is taken as the top when it already carries a
        #   top-like mass (>130 -- the crossover where mSD(J) beats
        #   m(J+j*) for EVERY W->cb topology in the ttbar-powheg study:
        #   fully-merged tops recall 6% -> 64%, partly-merged reco
        #   essentially untouched), or when there is nothing to add.
        #   Only when mSD(J) <= 130 AND a j* exists do we combine.
        #   The bmk1..5 search still runs (its m(J+j*) and the j*
        #   diagnostics below are only used on that mSD(J)<=130 & has-j*
        #   remainder).  Set WCB_TTRECO_MRG_MSD huge to disable.
        #   Events with mSD(J)<=130 and NO j* -> mt = -1 (underflow bin,
        #   counted, not dropped like SENTINEL).
        # Plus the AK4-flavour / PNet-discriminant diagnostics of the chosen
        # jets (bdr1 = dR-closest AK4; bbestm1 = best-m_t AK4).
        _tadr_out = ("mt", "mt_xself", "mt_nojs",
                     "dR_ja_mt", "dR_ja_bdr1", "dR_ja_bL12",
                     "xself_jstar", "dRpt_jstar", "pt_jstar", "xself_bmk1",
                     "jstar_flav",
                     "bdr1_flav", "bbestm1_flav",
                     "bvuds_near", "bvuds_bestm", "cvuds_bestm",
                     "bcvuds_bestm", "cvsb_bestm",
                     "mt_bjstar", "mt_cjstar", "mt_udsjstar", "mt_bcjstar")
        if _want(*_tadr_out) and arr is not None and {"ak8_pt_0", "ak8_eta_0",
                "ak8_phi_0", "ak8_sdmass_0", "bdr1_pt"} <= set(arr.fields):

            def _g(name):   # float64: invariant-mass precision (see _f above)
                return ak.to_numpy(arr[name]).astype(np.float64)

            def _p4b(pt, eta, phi, m):
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)
                pz = pt * np.sinh(np.clip(eta, -10.0, 10.0))
                E = np.sqrt(px * px + py * py + pz * pz
                            + np.maximum(m, 0.0) ** 2)
                return np.stack([px, py, pz, E])

            def _mm(a):
                return np.sqrt(np.maximum(
                    a[3] ** 2 - a[0] ** 2 - a[1] ** 2 - a[2] ** 2, 0.0))

            _jap4 = _p4b(_g("ak8_pt_0"), _g("ak8_eta_0"),
                         _g("ak8_phi_0"), _g("ak8_sdmass_0"))

            def _ta_with(seed):
                """(m(J+seed), pt(J+seed), dR(J,seed), present-mask)."""
                ok = _g(seed + "_pt") > 0.0
                s = _jap4 + _p4b(_g(seed + "_pt"), _g(seed + "_eta"),
                                 _g(seed + "_phi"), _g(seed + "_m"))
                dp = np.abs(_g(seed + "_phi") - _g("ak8_phi_0"))
                dp = np.where(dp > np.pi, 2.0 * np.pi - dp, dp)
                dR = np.hypot(_g(seed + "_eta") - _g("ak8_eta_0"), dp)
                return _mm(s), np.hypot(s[0], s[1]), dR, ok

            _m_dr, _pt_dr, _dR_dr, _ok_dr = _ta_with("bdr1")
            _bdr1_tag = (_g("bdr1_tag") if "bdr1_tag" in arr.fields
                         else np.full(_N, -1.0))
            # 4-bin categorical ParticleNetAK4 flavour of a chosen AK4
            # (0 = light/untag. tag<40, 1 = c_L 40-49, 2 = b_L-not-b_M (==50),
            #  3 = b_M >=51; SENTINEL if the seed jet is absent).  ak4_tag WP
            # ladder: c_L/M/T = 40/41/42, b_L/M/T = 50/51/52.
            #   bdr1_flav    -- the dR-closest AK4 to J
            #   bbestm1_flav -- the best-m_t AK4
            def _flav4(_tag, _ok):
                _f = np.where(_ok, 0.0, SENTINEL)
                _f = np.where(_ok & (_tag >= 40.0) & (_tag < 50.0), 1.0, _f)
                _f = np.where(_ok & (_tag >= 50.0) & (_tag < 51.0), 2.0, _f)
                _f = np.where(_ok & (_tag >= 51.0), 3.0, _f)
                return _f

            _out = {
                "bdr1_flav":  _flav4(_bdr1_tag, _ok_dr),
                # dR(J, nearest AK4 outside the cone) -- strongest single tag
                # of the "isolated boosted W-jet" (w/o-j*) population:
                # ~2.8 there vs ~1.5 for a normal boosted W->cb top
                # (2026-09-04, user).  -1 when there is no such AK4.
                "dR_ja_bdr1": np.where(_ok_dr, _dR_dr, -1.0),
            }
            # dR between the TWO loose-b AK4 nearest J (bL1, bL2 seeds) --
            # "dR of the nearest 2 b^L jets", restricted to outside the J cone
            # (2026-09-04, user).  -1 when < 2 such loose-b jets.
            if {"bL1_pt", "bL2_pt", "bL1_eta", "bL2_eta"} <= set(arr.fields):
                _b1ok = _g("bL1_pt") > 0.0
                _b2ok = _g("bL2_pt") > 0.0
                _dpb = np.abs(_g("bL1_phi") - _g("bL2_phi"))
                _dpb = np.where(_dpb > np.pi, 2.0 * np.pi - _dpb, _dpb)
                _drb12 = np.hypot(_g("bL1_eta") - _g("bL2_eta"), _dpb)
                _out["dR_ja_bL12"] = np.where(_b1ok & _b2ok, _drb12, -1.0)
            if "bbestm1_tag" in arr.fields:
                _out["bbestm1_flav"] = _flav4(
                    _g("bbestm1_tag"), _g("bbestm1_pt") > 0.0)

            # PNetAK4  b/(b+uds)  of a chosen AK4 (SENTINEL when the seed jet
            # is absent).  "_near" = dR-closest AK4 to J (bdr1);
            # "_bestm" = the AK4 the final mt reco ends up using.
            def _bvuds(_pb, _pu, _ok):
                _pb = np.maximum(_pb, 0.0); _pu = np.maximum(_pu, 0.0)
                return np.where(_ok & (_pb + _pu > 0.0),
                                _pb / np.maximum(_pb + _pu, 1e-12), SENTINEL)

            if "bdr1_pnuds" in arr.fields:
                _out["bvuds_near"] = _bvuds(
                    _g("bdr1_pnb"), _g("bdr1_pnuds"), _ok_dr)
            # PNetAK4 score RATIO of a chosen AK4 (SENTINEL when absent):
            #   _ratio_pn(num, den, ok) = num / den, clamped, guarded.
            def _ratio_pn(_num, _den, _ok):
                _num = np.maximum(_num, 0.0); _den = np.maximum(_den, 0.0)
                return np.where(_ok & (_den > 0.0),
                                _num / np.maximum(_den, 1e-12), SENTINEL)

            # m_t reconstruction -- see the big block above build_mat_plot_
            # settings for the full rationale.  m_t = m(J+j*) only when
            # 40 < m(J) < 130 AND a j* is found, else m_t = m(J).  Two j*
            # gates run in parallel:
            #   PRIMARY  ttreco_mt        dR*pT(J+j*)/2 < 2*172.5   (env XMT)
            #   ALT      ttreco_mt_xself  dR*pT(J+j*)/2 < 2*m(J+j*)  (env XSELF)
            # j* = first of bmk1..bmk5 (5 best-|m(J+AK4)-172.5| AK4) to pass.
            if "bmk1_pt" in arr.fields:
                _msd_j = _g("ak8_sdmass_0")
                _msd_ok = _msd_j > 0.0
                # top reco (J+j*) is only done when J's soft-drop mass says J
                # is a W / merged-top jet -- i.e. m(J) in [MRG_LO, MRG_HI]
                # (default 40..130 GeV, 2026-09-04 user).  Below 40 GeV J is a
                # light/QCD "Rest" jet with no W or t^2 candidate; above 130 J
                # already IS the (merged) top.  Outside the window (and when
                # no j* is found) -> m_t = m(J).
                _mrg_lo = float(os.environ.get("WCB_TTRECO_MRG_MSD_LO", "40"))
                _mrg_hi = float(os.environ.get("WCB_TTRECO_MRG_MSD",    "130"))
                _win = _msd_ok & (_msd_j >= _mrg_lo) & (_msd_j <= _mrg_hi)

                # PRIMARY j* gate:      dR*pT(J+j*)/2 < K * 172.5
                # ALTERNATIVE j* gate:  dR*pT(J+j*)/2 < K * m(J+j*)
                # K = env WCB_TTRECO_XMT / WCB_TTRECO_XSELF, both default 2.0
                # (the "2" is optimal for the primary).  Primary = a top-mass
                # hypothesis test (best for a tt sample, recovers the t^2
                # partial merges); alternative = mass-agnostic (best for
                # shape-based template fits).  Both == "dR*pT < 4 * M".
                # defaults = the Youden-optimal coefficients from the
                # true-b (topb_jetidx) study: 1.35*172.5 for the fixed ref,
                # 1.10*m(J+j*) for the self ref (2026-09-04, user).
                _kfix = float(os.environ.get("WCB_TTRECO_XMT",   "1.35"))
                _kslf = float(os.environ.get("WCB_TTRECO_XSELF", "1.10"))
                _MT_REF = 172.5

                _mjjs   = np.full(_N, -1.0)   # m(J+j*), primary gate
                _mjjs2  = np.full(_N, -1.0)   # m(J+j*), x_self gate
                _dr_sel = np.full(_N, -1.0)
                _xs_sel = np.full(_N, -1.0)
                _drpt_sel = np.full(_N, -1.0)
                _pt_sel = np.full(_N, -1.0)
                _xs_bmk1 = np.full(_N, -1.0)
                _tag_sel = np.full(_N, -1.0)
                _pnb_sel = np.full(_N, -1.0)
                _pnc_sel = np.full(_N, -1.0)
                _pnu_sel = np.full(_N, -1.0)
                _done  = np.zeros(_N, dtype=bool)   # found a j* (primary gate)
                _done2 = np.zeros(_N, dtype=bool)   # found a j* (x_self gate)
                for _k in range(1, 6):
                    _kb = "bmk%d" % _k
                    if _kb + "_pt" not in arr.fields:
                        continue
                    _mk, _ptk, _dRk, _okk = _ta_with(_kb)
                    _drpt_k = _dRk * _ptk
                    _xself_k = np.where(_okk & (_mk > 0.0),
                                        _drpt_k / (2.0 * np.maximum(_mk, 1e-6)),
                                        1.0e9)
                    if _k == 1:
                        _xs_bmk1 = np.where(_okk, _xself_k, -1.0)
                    # primary gate:  dR*pT(J+j*)/2 < K * 172.5
                    _take = (~_done) & _okk & (
                        0.5 * _drpt_k < _kfix * _MT_REF)
                    _mjjs = np.where(_take, _mk, _mjjs)
                    _dr_sel = np.where(_take, _dRk, _dr_sel)
                    _xs_sel = np.where(_take, _xself_k, _xs_sel)
                    _drpt_sel = np.where(_take, 0.5 * _drpt_k, _drpt_sel)
                    _pt_sel = np.where(_take, _ptk, _pt_sel)
                    if _kb + "_tag" in arr.fields:
                        _tag_sel = np.where(_take, _g(_kb + "_tag"), _tag_sel)
                    if _kb + "_pnb" in arr.fields:
                        _pnb_sel = np.where(_take, _g(_kb + "_pnb"), _pnb_sel)
                        _pnu_sel = np.where(_take, _g(_kb + "_pnuds"), _pnu_sel)
                    if _kb + "_pnc" in arr.fields:
                        _pnc_sel = np.where(_take, _g(_kb + "_pnc"), _pnc_sel)
                    _done |= _take
                    # x_self gate (for the alternative m_t)
                    # alternative gate:  dR*pT(J+j*)/2 < K * m(J+j*)
                    _take2 = (~_done2) & _okk & (
                        0.5 * _drpt_k < _kslf * np.maximum(_mk, 1e-6))
                    _mjjs2 = np.where(_take2, _mk, _mjjs2)
                    _done2 |= _take2

                # m_t = m(J+j*) only inside the m(J) window with a j*;
                # else m_t = m(J).  m_t = -1 never happens now (m(J) is always
                # a valid low value) -- unless J itself is missing (mSD<=0).
                _mres  = np.where(_win & _done,  _mjjs,
                                  np.where(_msd_ok, _msd_j, -1.0))
                _mres2 = np.where(_win & _done2, _mjjs2,
                                  np.where(_msd_ok, _msd_j, -1.0))
                _out["mt"]       = _mres
                _out["mt_xself"] = _mres2
                # j* actually used to build the PRIMARY m_t
                _has_js = _win & _done
                # "no j* candidate" flag (primary gate) -> the dotted
                # "J w/o-j*" overlay (Wcb_nomt / Cat_Top_bc_nomt).
                _out["mt_nojs"] = (~_done).astype(np.float64)
                # dR / x_self / dR*pT / pT of the j* CANDIDATE -- reported
                # whenever a candidate was found (`_done`), i.e. the -1 /
                # underflow bin means strictly "no j* candidate" and lines up
                # with the "J w/o-j*" (mt_nojs) overlay.  (For the merged-top
                # exception, mSD(J)>130, mt does NOT use this candidate, but
                # its ΔR is still informative -- it is usually the wide-angle
                # b_l / ISR jet we chose to ignore.)
                _out["dR_ja_mt"]   = np.where(_done, _dr_sel, -1.0)
                _out["xself_jstar"] = np.where(_done, _xs_sel, -1.0)
                _out["dRpt_jstar"]  = np.where(_done, _drpt_sel, -1.0)
                # no j* -> the "top pT" is just pT(J)
                _out["pt_jstar"]    = np.where(_done, _pt_sel, _g("ak8_pt_0"))
                _out["xself_bmk1"]  = _xs_bmk1
                # jstar_flav: 5-bin ParticleNetAK4-tag categorical of j* --
                # bin -1 "w/o j*" (in the m(J) window but no j* found),
                # then 0 light/untag. | 1 c_L | 2 b_L | 3 b_M.  SENTINEL
                # (dropped) only OUTSIDE the m(J) window (J is not a top
                # candidate there).
                if "bmk1_tag" in arr.fields:
                    _jf = _flav4(_tag_sel, _has_js)
                    # merge b_L-not-b_M (2) and b_M (3) into one "b_L" bin
                    # (2026-09-04, user) -- any b-tag, don't split by WP.
                    _jf = np.where(_jf == 3.0, 2.0, _jf)
                    _jf = np.where(_win & (~_done), -1.0, _jf)
                    _out["jstar_flav"] = _jf
                if "bmk1_pnb" in arr.fields:
                    _out["bvuds_bestm"] = np.where(
                        _has_js, _ratio_pn(_pnb_sel, _pnb_sel + _pnu_sel, _has_js),
                        -1.0)
                if "bmk1_pnc" in arr.fields:
                    _out["cvuds_bestm"] = np.where(
                        _has_js, _ratio_pn(_pnc_sel, _pnc_sel + _pnu_sel, _has_js),
                        -1.0)
                if "bmk1_pnb" in arr.fields and "bmk1_pnc" in arr.fields:
                    _out["bcvuds_bestm"] = np.where(
                        _has_js,
                        _ratio_pn(_pnb_sel + _pnc_sel,
                                  _pnb_sel + _pnc_sel + _pnu_sel, _has_js),
                        -1.0)
                    # c-vs-b PNetAK4 discriminant of j*:  c / (c + b).
                    # ~0 for a b jet (real top-b), ~1 for a c jet.
                    _out["cvsb_bestm"] = np.where(
                        _has_js,
                        _ratio_pn(_pnc_sel, _pnc_sel + _pnb_sel, _has_js),
                        -1.0)
                # m_t re-drawn under a ParticleNetAK4-TAG cut on j*
                # (2026-09-04, user; ak4_tag ladder c_L/M/T = 40/41/42,
                # b_L/M/T = 50/51/52):
                #   mt_bjstar   : j* is b_L        (tag >= 50)
                #   mt_cjstar   : j* is c_L        (40 <= tag < 50)
                #   mt_bcjstar  : j* is b_L or c_L (tag >= 40)
                #   mt_udsjstar : j* light/untag.  (tag < 40)
                # -1 (same sentinel as mt) when no j* or j* fails the cut.
                if "bmk1_tag" in arr.fields:
                    _jt = _tag_sel
                    _out["mt_bjstar"]   = np.where(_has_js & (_jt >= 50.0), _mres, -1.0)
                    _out["mt_cjstar"]   = np.where(_has_js & (_jt >= 40.0) & (_jt < 50.0), _mres, -1.0)
                    _out["mt_bcjstar"]  = np.where(_has_js & (_jt >= 40.0), _mres, -1.0)
                    _out["mt_udsjstar"] = np.where(_has_js & (_jt >= 0.0) & (_jt < 40.0), _mres, -1.0)

            for _nm, _v in _out.items():
                arr = ak.with_field(arr, np.asarray(_v, np.float32), _nm)
            del _jap4, _out, _m_dr, _pt_dr, _dR_dr, _ok_dr, _bdr1_tag
            gc.collect()
        elif _want(*_tadr_out) and arr is not None:
            for _nm in _tadr_out:
                if _nm not in arr.fields:
                    if _nm == "mt_nojs":
                        _dflt = 0.0            # flag: "no j*" -> False
                    elif _nm in ("mt", "mt_xself", "dR_ja_mt", "dR_ja_bdr1",
                                 "dR_ja_bL12",
                                 "xself_jstar", "dRpt_jstar", "pt_jstar",
                                 "xself_bmk1",
                                 "bvuds_bestm",
                                 "cvuds_bestm", "bcvuds_bestm", "cvsb_bestm",
                                 "mt_bjstar", "mt_cjstar", "mt_udsjstar",
                                 "mt_bcjstar"):
                        _dflt = -1.0
                    else:
                        _dflt = SENTINEL
                    arr = ak.with_field(
                        arr, np.full(_N, _dflt, dtype=np.float32), _nm)
        gc.collect()
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
        # cb-candidate AK8 jet  ==  J = the AK8 jet with the highest GloParT
        # bc score  (was: j_1, the pT-leading AK8).  Every AK8 jet in these
        # ntuples is already produced with pT > 200 GeV (verified 2026-08-29:
        # min 200.1, 0% below 200 across all samples), so ranking by bc score
        # instead of pT never drops a jet or moves the phase space -- it only
        # changes *which* of the >=1 accepted jets is treated as the cb
        # candidate.  `_cand_ak8(branch, default)` returns that jet's value of
        # `branch` (event-level scalar branches pass straight through) and is
        # the single choke point for the candidate definition: truth match
        # (ak8_type / n_{b,c} -> true_cat / mat_cat), GloParT scores
        # (-> score_Dbc and every gpt_* ratio), the ak8_match_* / ak8_is_wbc
        # flags, and the candidate-jet substructure / kinematics
        # (ak8_{pt,eta,phi,sdmass,tau*,rawFactor,nConstituents}_0).
        # AK4 quantities and the "sub_*" / "maxmass" alternative orderings stay
        # as they were.
        # ------------------------------------------------------------------
        if (not is_data) and ("ak8_gpt_bc" in raw.fields):
            _cand_idx = ak.argmax(raw["ak8_gpt_bc"], axis=1, keepdims=True)
        else:
            _cand_idx = None

        def _cand_ak8(branch, default):
            """Value of `branch` on the cb-candidate AK8 jet (J)."""
            if branch not in raw.fields:
                return np.full(n, default)
            v = raw[branch]
            try:
                is_jagged = v.ndim > 1
            except Exception:
                is_jagged = False
            if not is_jagged:
                return ak.to_numpy(ak.fill_none(v, default))
            picked = v if _cand_idx is None else v[_cand_idx]
            return ak.to_numpy(ak.fill_none(ak.firsts(picked, axis=1), default))

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

            ak8_type = np.asarray(_cand_ak8("ak8_type", -1), dtype=np.int32)
            n_c = np.asarray(_cand_ak8("ak8_n_c_in_jet", -1), dtype=np.int32)
            n_b = np.asarray(_cand_ak8("ak8_n_b_in_jet", -1), dtype=np.int32)
            n_in_jet = np.asarray(_cand_ak8("ak8_n_in_jet", -1), dtype=np.int32)

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
        # Candidate-jet variables  (J = highest GloParT bc score; see
        # `_cand_ak8` above).  AK4 stays pT-leading.
        # ------------------------------------------------------------------
        ak8_pt_0 = np.asarray(_cand_ak8("ak8_pt", -999), dtype=np.float32)
        ak8_eta_0 = np.asarray(_cand_ak8("ak8_eta", -999), dtype=np.float32)
        ak8_phi_0 = np.asarray(_cand_ak8("ak8_phi", -999), dtype=np.float32)
        ak8_sdmass_0 = np.asarray(_cand_ak8("ak8_sdmass", -999), dtype=np.float32)

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
            # cb-candidate jet for jagged ak8_* branches (GloParT scores),
            # pass-through for event-level scalars (score_cata_*).
            if branch in raw.fields:
                return np.asarray(_cand_ak8(branch, 0), dtype=np.float32)
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

        # summed EventClassifier top-like probability: Top(bc)+Top(bq)+Top(bqq')
        score_cata_top_sum_norm = (score_cata_top_bc_norm
                                   + score_cata_top_bq_norm
                                   + score_cata_top_bqq_norm)

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
        # MAT-overlay category of the cb-CANDIDATE AK8 jet (J = highest
        # GloParT bc score; truth-matched).
        #
        # Each hadronic-W flavour (cb / cq / qq') is split three ways to expose
        # how much of the low-mass tail is a "the bc-tagger picked the wrong
        # jet" effect vs. a genuine "the W never merged" effect:
        #     =J      the merged 2-prong W jet IS the candidate (bc-tagged)
        #     !=J     that merged jet exists but is NOT the candidate -- the
        #               highest-bc-score jet is some other jet (lept-side b,
        #               merged top, ISR, ...)
        #     resolved  the W decayed this way but no AK8 merged it
        #
        #   W->cb (signal only): 0 = =J, 5 = !=J, 6 = resolved
        #                        (gate: ak8_is_wbc, i.e. w_decay==5 & is_signal)
        #   W->cq              : 1 = =J, 9 = !=J, 10 = resolved
        #   W->qq' (light)     : 4 = =J, 11 = !=J, 12 = resolved
        #   2 = t->bc  (proxy) , 7 = t->bq , 8 = t->bqq'   (Cat_Top_*, any MC)
        #   3 = rest
        #
        # `ak8_match_wcq`/`ak8_match_wqq` require >=2 W daughters in one cone
        # (merged); `w_decay` is the event-level hadronic-W decay code. The
        # "resolved" and "!=J" W-classes are assigned FIRST, so the top
        # categories (2/7/8) always win -- a top-tagged candidate jet keeps its
        # top label and is never pulled into a resolved/!=J W-class. QCD /
        # V+jets (is_qcd) are excluded from every W-class -> stay in 'rest'.
        # ------------------------------------------------------------------
        def _lead_flag(branch):
            # flag value on the cb-candidate AK8 jet (J)
            return np.asarray(_cand_ak8(branch, 0), dtype=np.int8)

        def _any_flag(branch):
            if branch in raw.fields:
                return np.asarray(
                    ak.to_numpy(ak.any(raw[branch] == 1, axis=1)), dtype=bool)
            return np.zeros(n, dtype=bool)

        is_wbc_lead = _lead_flag("ak8_is_wbc")
        match_wcq_lead = _lead_flag("ak8_match_wcq")
        match_wqq_lead = _lead_flag("ak8_match_wqq")
        match_wqqwcb_lead = _lead_flag("ak8_match_wqq_wcb")
        match_tbqqwcb_lead = _lead_flag("ak8_match_tbqq_wcb")

        any_wqqwcb = _any_flag("ak8_match_wqq_wcb")
        any_wcq = _any_flag("ak8_match_wcq")
        any_wqq = _any_flag("ak8_match_wqq")

        w_decay = np.asarray(
            raw["w_decay"] if "w_decay" in raw.fields else np.full(n, -1),
            dtype=np.int32,
        )
        not_qcd = ~is_qcd
        wcq_decay = (w_decay == 4) & not_qcd                  # W->cs / W->cd
        wqq_decay = np.isin(w_decay, (2, 3)) & not_qcd        # W->ud / W->us

        _code_topbc = self.cfg.true_cat_codes["Cat_Top_bc"]
        _code_topbq = self.cfg.true_cat_codes["Cat_Top_bq"]
        _code_topbqq = self.cfg.true_cat_codes["Cat_Top_bqq"]
        mat_cat = np.full(n, 3, dtype=np.int8)

        # --- resolved / sub-leading W->cq, W->qq' (tops + =j_1 override below) --
        mat_cat[wqq_decay & ~any_wqq] = 12                    # W->qq' resolved
        mat_cat[wcq_decay & ~any_wcq] = 10                    # W->cq  resolved
        mat_cat[any_wqq & (match_wqq_lead != 1) & not_qcd] = 11   # W->qq' !=j_1
        mat_cat[any_wcq & (match_wcq_lead != 1) & not_qcd] = 9    # W->cq  !=j_1

        # --- top categories: kept exactly as before (win over the W-classes) ---
        # t^3 = fully-merged 3-prong top (true_cat == Cat_Top_bqq, ak8_type==4).
        # Split by the hadronic-W flavour that is merged inside the top:
        #    8  = t -> b q q'   (W -> cs / cd / ud / us ;  q,q' in u,d,s,c)
        #   20  = t -> b b c    (W -> cb ;  top-b + c + b_W all inside J --
        #                        gen flag ak8_match_tbqq_wcb on J)
        mat_cat[true_cat == _code_topbqq] = 8
        mat_cat[(true_cat == _code_topbqq) & (match_tbqqwcb_lead == 1)] = 20
        mat_cat[true_cat == _code_topbc] = 2
        mat_cat[true_cat == _code_topbq] = 7

        # --- merged & leading W->qq' / W->cq (leading jet type==1, not a top) ---
        mat_cat[match_wqq_lead == 1] = 4
        mat_cat[match_wcq_lead == 1] = 1

        # --- W->cb signal: 0/5/6 partition sig_wbc, override everything else ---
        sig_wbc = (is_wbc_lead == 1) & is_signal
        mat_cat[sig_wbc] = 6                                  # resolved
        mat_cat[sig_wbc & any_wqqwcb] = 5                     # merged, sub-leading
        mat_cat[sig_wbc & (match_wqqwcb_lead == 1)] = 0       # merged, leading

        # A fully-merged 3-prong top with W->cb (top-b + c + b_W all in J,
        # gen flag ak8_match_tbqq_wcb) is NOT the resolved-b + merged-2-prong-W
        # signal topology.  The sig_wbc override above sends it to code 6
        # ("resolved" -> Rest); restore its own t^3 -> bbc sub-class (code 20).
        mat_cat[(match_tbqqwcb_lead == 1) & not_qcd] = 20

        # --- merged hadronic-Z -> qq, carved out of 'rest' ------------------
        # These ntuples carry no gen-Z direction (only genZ_pt) and no per-jet
        # Z match, so "merged" is a LOOSE pT proxy: the cb-candidate AK8 J
        # carries most of the Z momentum, pT(J) / pT(Z_gen) > 0.7.  Flavour
        # from z_decay (5 = bb, 4 = cc, 1/2/3 = light).  Only rewrites events
        # still in 'rest' (mat_cat == 3: J not W/top matched) and not V+jets
        # (is_qcd -> DY / W+jets stay in 'rest', same rule as the W-classes).
        # Hadronic-Z events that fail the pT proxy are "Z resolved" and stay
        # in 'rest' (mat_cat == 3), exactly like the resolved-W treatment.
        #   13 = merged Z->qq' (light)   14 = merged Z->cc   15 = merged Z->bb
        z_decay = np.asarray(
            raw["z_decay"] if "z_decay" in raw.fields else np.zeros(n),
            dtype=np.int32,
        )
        genZ_pt = np.asarray(
            raw["genZ_pt"] if "genZ_pt" in raw.fields else np.full(n, -1.0),
            dtype=np.float32,
        )
        _z_had = np.isin(z_decay, (1, 2, 3, 4, 5)) & not_qcd
        _z_ptratio = ak8_pt_0 / np.where(genZ_pt > 0.0, genZ_pt, 1.0)
        _z_merged = _z_had & (genZ_pt > 0.0) & (ak8_pt_0 > 0.0) & (_z_ptratio > 0.7)
        _in_rest = mat_cat == 3
        mat_cat[_in_rest & _z_merged & np.isin(z_decay, (1, 2, 3))] = 13
        mat_cat[_in_rest & _z_merged & (z_decay == 4)] = 14
        mat_cat[_in_rest & _z_merged & (z_decay == 5)] = 15
        # Everything else hadronic-Z stays mat_cat == 3 ("Z resolved" in 'rest',
        # like W-resolved).  gluon/light/c/b jets from every process also here.

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

        # -- cb-candidate-AK8 substructure (J) ---------------------------
        ak8_rawFactor_0 = np.asarray(_cand_ak8("ak8_rawFactor", -999.0), dtype=np.float32)
        ak8_tau21_0     = np.asarray(_cand_ak8("ak8_tau21", -999.0), dtype=np.float32)
        ak8_tau32_0     = np.asarray(_cand_ak8("ak8_tau32", -999.0), dtype=np.float32)
        # tau31 = tau3/tau1. No such branch is stored, so reconstruct it from
        # tau31 = (tau3/tau2) * (tau2/tau1) = tau32 * tau21 when both are valid.
        if "ak8_tau31" in raw.fields:
            ak8_tau31_0 = np.asarray(_cand_ak8("ak8_tau31", -999.0), dtype=np.float32)
        else:
            _tau_ok = (ak8_tau21_0 > 0) & (ak8_tau32_0 > 0)
            ak8_tau31_0 = np.where(
                _tau_ok, ak8_tau21_0 * ak8_tau32_0, -999.0
            ).astype(np.float32)
        ak8_nConst_0    = np.asarray(_cand_ak8("ak8_nConstituents", -999.0), dtype=np.float32)
        ak4_mass_0      = lead("ak4_mass")   # AK4: stays pT-leading

        # -- k-th AK8 jet under an arbitrary ranking ----------------------
        #    value_branch of the jet ranked k-th (0 = leading, 1 = sub-leading)
        #    when jets are sorted by order_branch descending. Used for the
        #    leading (J) and sub-leading (j_b) mSD under pT / mSD / bc-score
        #    orderings:
        #      j_1 / j_2  -> pT-leading / -sub-leading
        #      j^1 / j^2  -> highest / 2nd-highest mSD
        #      J / j_b  -> highest / 2nd-highest GloParT bc score
        #                   (J == the cb candidate == the *_0 fields)
        def nth_by(order_branch, value_branch, k, default=-999.0):
            if order_branch not in raw.fields or value_branch not in raw.fields:
                return np.full(n, default, dtype=np.float32)
            idx = ak.argsort(raw[order_branch], axis=1, ascending=False)
            ranked = raw[value_branch][idx]
            picked = ak.pad_none(ranked, k + 1, axis=1)[:, k]
            return np.asarray(ak.fill_none(picked, default), dtype=np.float32)

        ak8_sdmass_maxmass_0  = nth_by("ak8_sdmass", "ak8_sdmass", 0)   # j^1
        # pT-ordered j_1 / j_2 soft-drop masses -- specs commented out
        # 2026-09-04 (user), derivation disabled too.
        # ak8_sdmass_lead_pt_0  = nth_by("ak8_pt",     "ak8_sdmass", 0)   # j_1 (pT-leading; J == ak8_sdmass_0)
        # ak8_sdmass_sub_pt_0   = nth_by("ak8_pt",     "ak8_sdmass", 1)   # j_2
        ak8_sdmass_sub_mass_0 = nth_by("ak8_sdmass", "ak8_sdmass", 1)   # j^2
        ak8_sdmass_sub_bc_0   = nth_by("ak8_gpt_bc", "ak8_sdmass", 1)   # j_b
        ak8_tau21_sub_bc_0    = nth_by("ak8_gpt_bc", "ak8_tau21",  1)   # tau21 of j_b

        # -- Dbc BDT score for the OTHER ranked AK8 jets --------------------
        # Same 8-GloParT-node BDT as score_Dbc (J), re-evaluated on:
        #   j_b  = 2nd-highest bc-score AK8   (nth_by "ak8_gpt_bc", k=1)
        #   j^1  = highest-mSD AK8            (nth_by "ak8_sdmass",  k=0)
        #   j^2  = 2nd-highest-mSD AK8        (nth_by "ak8_sdmass",  k=1)
        # Missing-jet slots get 0 fed to the BDT, then the score is set to
        # SENTINEL where that jet does not exist (n_ak8 < 2 for j_b / j^2;
        # j^1 shares J's domain, i.e. n_ak8 >= 1).
        _DBC_NODES = ("bc", "bb", "cc", "qcd", "bs", "qq", "cs", "topbw")

        def _dbc_for(order_branch, k):
            _feat = [nth_by(order_branch, f"ak8_gpt_{_nd}", k, default=0.0)
                     for _nd in _DBC_NODES]
            return np.asarray(self.dbc_eval.get_Dbc(*_feat), dtype=np.float32)

        _n_ak8_i = np.asarray(n_ak8, dtype=np.float32)
        score_Dbc_jb    = np.where(_n_ak8_i >= 2,
                                   _dbc_for("ak8_gpt_bc", 1), SENTINEL).astype(np.float32)
        score_Dbc_jsup1 = np.where(_n_ak8_i >= 1,
                                   _dbc_for("ak8_sdmass", 0), SENTINEL).astype(np.float32)
        score_Dbc_jsup2 = np.where(_n_ak8_i >= 2,
                                   _dbc_for("ak8_sdmass", 1), SENTINEL).astype(np.float32)

        # -- leading-AK8 GloParT raw scores. g_bc/g_bb/... (the Dbc inputs)
        #    are already leading values from leading_or_zero() above.
        g_bqq  = leading_or_zero("ak8_gpt_bqq")
        g_topw = leading_or_zero("ak8_gpt_topw")
        # the finer GloParT sub-nodes that the 2final ntuples added
        # (topbw / topw / qcd sub-decays + ss + tauhtauh).  Missing branches
        # -> leading_or_zero returns 0.  ak8_gpt_<node>_0 for each.
        _gpt_extra = {("ak8_gpt_%s_0" % _n): leading_or_zero("ak8_gpt_" + _n)
                      for _n in _GPT_EXTRA_NODES}

        gpt_sum = (g_bc + g_bb + g_cc + g_qcd + g_bs + g_qq + g_cs
                   + g_topbw + g_bqq + g_topw + 1e-10)

        # discriminant-style GloParT ratios (W(cb) vs Top(bc) separation)
        gpt_bc_frac  = (g_bc / gpt_sum).astype(np.float32)
        # summed probability that J is ANY V->2-quark (2-prong) resonance:
        # bc + bb + cc + bs + cs + qq  (the six non-top, non-QCD GloParT nodes;
        # GloParT has no separate "ss" node).
        gpt_2prong_sum = (g_bc + g_bb + g_cc + g_bs + g_cs + g_qq).astype(np.float32)
        # bc vs all six 2-prong / resonance-like GloParT outputs (excludes the
        # top-like and QCD nodes): bc / (bc + cc + bb + qq + bs + cs)
        gpt_bc_vs_2prong = safe_ratio(
            g_bc, g_bc + g_cc + g_bb + g_qq + g_bs + g_cs)
        gpt_bc_vs_cc = safe_ratio(g_bc, g_bc + g_cc)
        gpt_bc_vs_bb = safe_ratio(g_bc, g_bc + g_bb)
        gpt_bc_vs_qq = safe_ratio(g_bc, g_bc + g_qq)
        gpt_cs_vs_qq = safe_ratio(g_cs, g_cs + g_qq)
        gpt_hf_w     = ((g_bc + g_cs + g_bs) / gpt_sum).astype(np.float32)
        gpt_top_frac = ((g_topbw + g_topw + g_bqq) / gpt_sum).astype(np.float32)
        gpt_w_vs_top = safe_ratio(g_bc + g_cs + g_qq,
                                  g_bc + g_cs + g_qq + g_topbw + g_topw + g_bqq)
        # same W-like/(W-like+top-like) GloParT ratio but evaluated on j_b (the
        # 2nd-highest bc-score AK8).  -999 (dropped) when n_ak8 < 2.
        _gb = lambda node: nth_by("ak8_gpt_bc", node, 1, default=0.0)
        _wlike_jb = _gb("ak8_gpt_bc") + _gb("ak8_gpt_cs") + _gb("ak8_gpt_qq")
        gpt_w_vs_top_jb = safe_ratio(
            _wlike_jb,
            _wlike_jb + _gb("ak8_gpt_topbw") + _gb("ak8_gpt_topw")
            + _gb("ak8_gpt_bqq"))

        # -- extra GloParT combination ratios for the MAT scan -------------
        #    All are X / (X + Y) forms, so bounded to [0, 1] and histogram
        #    cleanly on a linear x-axis; the discriminating structure sits
        #    near the edges, hence they are drawn with a log y-axis.
        #      bc vs top(bW)          : is the jet a genuine W(cb) or a
        #                              b+W merged top?
        #      top(bW) vs all top-like: which top merge topology
        #                              (b+W in jet / W only / b+q+q')?
        #      (bc+cs+cc) vs qq       : charm-tagged W decay vs light W->qq'
        #      (bc+bs+bb) vs qq       : b-tagged   W decay vs light W->qq'
        gpt_bc_vs_topbw  = safe_ratio(g_bc, g_bc + g_topbw)
        # W(cb) vs Top(->bc) discriminant: the prompt W->cb jet (no extra b in
        # cone) scores on the `bc` node, whereas a t->bc jet (top-b + W-c
        # merged) leaks onto ALL three top-like nodes (b+W in jet / W only /
        # b+q+q').  `gpt_bc_vs_topbw` only fights the topbw node; this pools the
        # full top hypothesis into the denominator -> sharper Wcb / topWcb split.
        gpt_bc_vs_top    = safe_ratio(g_bc, g_bc + g_topbw + g_topw + g_bqq)
        gpt_topbw_vs_top = safe_ratio(g_topbw, g_topbw + g_topw + g_bqq)
        gpt_cW_vs_qq     = safe_ratio(g_bc + g_cs + g_cc,
                                      g_bc + g_cs + g_cc + g_qq)
        gpt_bW_vs_qq     = safe_ratio(g_bc + g_bs + g_bb,
                                      g_bc + g_bs + g_bb + g_qq)
        # W(cb) vs all non-top hypotheses: gpt_bc_vs_2prong with the QCD score
        # folded into the denominator (all 2-prong W flavours + QCD, i.e.
        # everything except top(bW) / top(W) / bqq').
        gpt_bc_vs_nontop = safe_ratio(
            g_bc, g_bc + g_cc + g_bb + g_qq + g_bs + g_cs + g_qcd)
        # W(cb) vs QCD only: bc / (bc + QCD)
        gpt_bc_vs_qcd = safe_ratio(g_bc, g_bc + g_qcd)

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
        # Delta-phi / Delta-R between the leptonic W (v = lepton + MET; the
        # ntuple's v_{eta,phi}) and the cb-candidate AK8 J.
        _v_phi = ev("v_phi")
        _v_eta = ev("v_eta")
        _v_ok = np.abs(_v_phi) <= np.pi
        _dphi_w = dphi_pipi(_v_phi, ak8_phi_0)
        dphi_w_ak8 = np.where(ak8_ok & _v_ok, _dphi_w, -999).astype(np.float32)
        dR_w_ak8 = np.where(ak8_ok & _v_ok,
                            np.sqrt((_v_eta - ak8_eta_0) ** 2 + _dphi_w ** 2),
                            -999).astype(np.float32)
        dphi_ak4_ak8 = np.where(ak8_ok & ak4_ok, dphi_pipi(ak4_phi_0, ak8_phi_0), -999).astype(np.float32)
        deta_ak4_ak8 = np.where(ak8_ok & ak4_ok, np.abs(ak4_eta_0 - ak8_eta_0), -999).astype(np.float32)
        dR_ak4_ak8 = np.where(ak8_ok & ak4_ok,
                              np.sqrt(np.maximum(deta_ak4_ak8, 0)**2 + dphi_ak4_ak8**2), -999).astype(np.float32)
        ptrel_lep_ak8 = np.where(ak8_ok & lep_ok, lep1_pt / np.where(ak8_ok, ak8_pt_0, 1.0), -999).astype(np.float32)

        # -- reco b/c-tagged AK4 multiplicity within dR<1 of the cb-cand AK8 --
        # Per-jet ParticleNetAK4 `ak4_tag` code: c-region 40-44, b-region 50-54
        # (see the event-level n_ctag/n_btag scheme).  loose = tag>=40 (c) /
        # tag>=50 (b); medium = tag>=41 (c) / tag>=51 (b).  Count only jets with
        # sqrt(deta^2+dphi^2) < 1.0 from the candidate (J).  Sentinel -1 when
        # there is no valid candidate.
        _z16 = np.zeros(n, dtype=np.int16)
        if "ak4_tag" in raw.fields and "ak4_eta" in raw.fields:
            _at = raw["ak4_tag"]
            _ade = raw["ak4_eta"] - ak8_eta_0
            _adp = np.abs(raw["ak4_phi"] - ak8_phi_0)
            _adp = ak.where(_adp > np.pi, 2.0 * np.pi - _adp, _adp)
            _near = (_ade * _ade + _adp * _adp) < 1.0        # dR < 1
            _cnt = lambda msk: ak.to_numpy(
                ak.sum(ak.fill_none(msk, False), axis=1)).astype(np.int16)
            n_c_nearcb_L = _cnt(_near & (_at >= 40) & (_at < 50))
            n_c_nearcb_M = _cnt(_near & (_at >= 41) & (_at < 50))
            n_b_nearcb_L = _cnt(_near & (_at >= 50))
            n_b_nearcb_M = _cnt(_near & (_at >= 51))
        else:
            n_c_nearcb_L = n_c_nearcb_M = n_b_nearcb_L = n_b_nearcb_M = _z16
        n_c_nearcb_L = np.where(ak8_ok, n_c_nearcb_L, -1).astype(np.int16)
        n_c_nearcb_M = np.where(ak8_ok, n_c_nearcb_M, -1).astype(np.int16)
        n_b_nearcb_L = np.where(ak8_ok, n_b_nearcb_L, -1).astype(np.int16)
        n_b_nearcb_M = np.where(ak8_ok, n_b_nearcb_M, -1).astype(np.int16)

        # -- W->cb resolved-decay proxy: m(c-jet + nearest b-jet) --------------
        # Take the loose c-tagged AK4 jet with the highest ParticleNetAK4 c
        # score, then the loose b-tagged AK4 jet closest to it in dR; return
        # their invariant mass (massive 4-vectors).  For a resolved W->cb this
        # peaks near m_W.  -1 if the event has no loose c-tag or no loose b-tag.
        _need = {"ak4_tag", "ak4_pn_c", "ak4_pt", "ak4_eta", "ak4_phi", "ak4_mass"}
        if _need <= set(raw.fields):
            _tg   = raw["ak4_tag"]
            _cmsk = (_tg >= 40) & (_tg < 50)          # loose c
            _bmsk = _tg >= 50                         # loose b
            _has_c = ak.to_numpy(ak.sum(_cmsk, axis=1)) > 0
            _has_b = ak.to_numpy(ak.sum(_bmsk, axis=1)) > 0
            _NEG = -1.0e9
            _ic = ak.argmax(ak.where(_cmsk, raw["ak4_pn_c"], _NEG),
                            axis=1, keepdims=True)
            def _cpick(br):
                return ak.to_numpy(ak.fill_none(
                    ak.firsts(raw[br][_ic], axis=1), 0.0)).astype(np.float64)
            _c_pt, _c_eta, _c_phi, _c_m = (_cpick("ak4_pt"), _cpick("ak4_eta"),
                                           _cpick("ak4_phi"), _cpick("ak4_mass"))
            _dphi_cb = np.abs(raw["ak4_phi"] - _c_phi)
            _dphi_cb = ak.where(_dphi_cb > np.pi, 2.0 * np.pi - _dphi_cb, _dphi_cb)
            _dR2_cb = (raw["ak4_eta"] - _c_eta) ** 2 + _dphi_cb ** 2
            _ib = ak.argmin(ak.where(_bmsk, _dR2_cb, 1.0e18),
                            axis=1, keepdims=True)
            def _bpick(br):
                return ak.to_numpy(ak.fill_none(
                    ak.firsts(raw[br][_ib], axis=1), 0.0)).astype(np.float64)
            _b_pt, _b_eta, _b_phi, _b_m = (_bpick("ak4_pt"), _bpick("ak4_eta"),
                                           _bpick("ak4_phi"), _bpick("ak4_mass"))

            def _p4(pt, eta, phi, m):
                px = pt * np.cos(phi); py = pt * np.sin(phi)
                pz = pt * np.sinh(np.clip(eta, -10, 10))
                E = np.sqrt(px * px + py * py + pz * pz + np.maximum(m, 0) ** 2)
                return px, py, pz, E
            _cx, _cy, _cz, _cE = _p4(_c_pt, _c_eta, _c_phi, _c_m)
            _bx, _by, _bz, _bE = _p4(_b_pt, _b_eta, _b_phi, _b_m)
            _m2 = ((_cE + _bE) ** 2 - (_cx + _bx) ** 2
                   - (_cy + _by) ** 2 - (_cz + _bz) ** 2)
            mass_cL_nearb = np.where(_has_c & _has_b,
                                     np.sqrt(np.maximum(_m2, 0.0)), -1.0
                                     ).astype(np.float32)
        else:
            mass_cL_nearb = np.full(n, -1.0, dtype=np.float32)

        # -- m(b^L c^L) of the pair whose mass is closest to m_W (80 GeV) ------
        # Same b^L x c^L AK4 object set as mass_minDR_bc, but the pair is
        # chosen by |m(bc) - 80 GeV| minimal instead of by min dR.  For a
        # resolved W->cb this sharpens the m_W peak relative to the
        # dR-closest pick.  -1 if the event has no loose b-tag or no loose
        # c-tag AK4 jet.
        _needW = {"ak4_tag", "ak4_pt", "ak4_eta", "ak4_phi", "ak4_mass"}
        if _needW <= set(raw.fields):
            _tgW   = raw["ak4_tag"]
            _bselW = _tgW >= 50                       # loose b
            _cselW = (_tgW >= 40) & (_tgW < 50)       # loose c
            _bW = ak.zip({"pt":  raw["ak4_pt"][_bselW],
                          "eta": raw["ak4_eta"][_bselW],
                          "phi": raw["ak4_phi"][_bselW],
                          "m":   raw["ak4_mass"][_bselW]})
            _cW = ak.zip({"pt":  raw["ak4_pt"][_cselW],
                          "eta": raw["ak4_eta"][_cselW],
                          "phi": raw["ak4_phi"][_cselW],
                          "m":   raw["ak4_mass"][_cselW]})
            _prW = ak.cartesian({"b": _bW, "c": _cW}, axis=1)

            def _p4W(o):
                px = o["pt"] * np.cos(o["phi"])
                py = o["pt"] * np.sin(o["phi"])
                pz = o["pt"] * np.sinh(o["eta"])
                E = np.sqrt(px * px + py * py + pz * pz
                            + np.maximum(o["m"], 0.0) ** 2)
                return px, py, pz, E

            _bpx, _bpy, _bpz, _bE = _p4W(_prW["b"])
            _cpx, _cpy, _cpz, _cE = _p4W(_prW["c"])
            _mpairW = np.sqrt(np.maximum(
                (_bE + _cE) ** 2 - (_bpx + _cpx) ** 2
                - (_bpy + _cpy) ** 2 - (_bpz + _cpz) ** 2, 0.0))
            _hasW = ak.to_numpy(ak.num(_mpairW, axis=1)) > 0
            _iW = ak.argmin(np.abs(_mpairW - 80.0), axis=1, keepdims=True)
            _mW_pick = ak.to_numpy(ak.fill_none(
                ak.firsts(_mpairW[_iW], axis=1), -1.0))
            mass_bc_bestmW = np.where(_hasW, _mW_pick, -1.0).astype(np.float32)
        else:
            mass_bc_bestmW = np.full(n, -1.0, dtype=np.float32)

        # -- m(b^L b^L c^L) triplet top-mass reco -----------------------------
        # t -> b W, W -> c b  =>  the three tagged AK4 quarks are (b, b, c).
        # Form every (b^L, b^L, c^L) AK4 triplet and pick:
        #   mass_bbc_minDR  = m of the triplet with the smallest summed
        #                     pairwise dR  (most collimated triplet)
        #   mass_bbc_bestmt = m of the triplet whose mass is closest to
        #                     m_t (172.5 GeV)
        # -1 if the event has < 2 loose-b OR < 1 loose-c AK4 jet.
        _needT = {"ak4_tag", "ak4_pt", "ak4_eta", "ak4_phi", "ak4_mass"}
        if _needT <= set(raw.fields):
            _tgT   = raw["ak4_tag"]
            _bselT = _tgT >= 50                       # loose b
            _cselT = (_tgT >= 40) & (_tgT < 50)       # loose c
            _bT = ak.zip({"pt":  raw["ak4_pt"][_bselT],
                          "eta": raw["ak4_eta"][_bselT],
                          "phi": raw["ak4_phi"][_bselT],
                          "m":   raw["ak4_mass"][_bselT]})
            _cT = ak.zip({"pt":  raw["ak4_pt"][_cselT],
                          "eta": raw["ak4_eta"][_cselT],
                          "phi": raw["ak4_phi"][_cselT],
                          "m":   raw["ak4_mass"][_cselT]})
            _bpair = ak.combinations(_bT, 2, axis=1, fields=["b1", "b2"])
            _triT = ak.cartesian({"bb": _bpair, "c": _cT}, axis=1)
            _t_b1, _t_b2, _t_c = (_triT["bb"]["b1"], _triT["bb"]["b2"],
                                  _triT["c"])

            def _p4T(o):
                px = o["pt"] * np.cos(o["phi"])
                py = o["pt"] * np.sin(o["phi"])
                pz = o["pt"] * np.sinh(o["eta"])
                E = np.sqrt(px * px + py * py + pz * pz
                            + np.maximum(o["m"], 0.0) ** 2)
                return px, py, pz, E

            _x1, _y1, _z1, _e1 = _p4T(_t_b1)
            _x2, _y2, _z2, _e2 = _p4T(_t_b2)
            _x3, _y3, _z3, _e3 = _p4T(_t_c)
            _mtri = np.sqrt(np.maximum(
                (_e1 + _e2 + _e3) ** 2 - (_x1 + _x2 + _x3) ** 2
                - (_y1 + _y2 + _y3) ** 2 - (_z1 + _z2 + _z3) ** 2, 0.0))

            def _dRT(oa, ob):
                _dp = np.abs(oa["phi"] - ob["phi"])
                _dp = ak.where(_dp > np.pi, 2.0 * np.pi - _dp, _dp)
                return np.sqrt((oa["eta"] - ob["eta"]) ** 2 + _dp ** 2)

            _dRsum = (_dRT(_t_b1, _t_b2) + _dRT(_t_b1, _t_c)
                      + _dRT(_t_b2, _t_c))
            _hasT = ak.to_numpy(ak.num(_mtri, axis=1)) > 0

            _idr = ak.argmin(_dRsum, axis=1, keepdims=True)
            _mdr = ak.to_numpy(ak.fill_none(
                ak.firsts(_mtri[_idr], axis=1), -1.0))
            mass_bbc_minDR = np.where(_hasT, _mdr, -1.0).astype(np.float32)

            _imt = ak.argmin(np.abs(_mtri - 172.5), axis=1, keepdims=True)
            _mmt = ak.to_numpy(ak.fill_none(
                ak.firsts(_mtri[_imt], axis=1), -1.0))
            mass_bbc_bestmt = np.where(_hasT, _mmt, -1.0).astype(np.float32)
        else:
            mass_bbc_minDR = np.full(n, -1.0, dtype=np.float32)
            mass_bbc_bestmt = np.full(n, -1.0, dtype=np.float32)

        # -- min Delta-phi between a loose b-tag and a loose c-tag AK4 jet ------
        # Same object set as minDR_bc (all b^L x c^L AK4 pairs) but ranked by
        # |dphi| instead of dR.  -1 if the event has no loose b-tag or no loose
        # c-tag AK4 jet.
        _need2 = {"ak4_tag", "ak4_phi"}
        if _need2 <= set(raw.fields):
            _tg2  = raw["ak4_tag"]
            _bphi = raw["ak4_phi"][_tg2 >= 50]
            _cphi = raw["ak4_phi"][(_tg2 >= 40) & (_tg2 < 50)]
            _cp2  = ak.cartesian({"b": _bphi, "c": _cphi}, axis=1)
            _dphi_bc = np.abs(_cp2["b"] - _cp2["c"])
            _dphi_bc = ak.where(_dphi_bc > np.pi, 2.0 * np.pi - _dphi_bc, _dphi_bc)
            _has_bc  = ak.to_numpy(ak.num(_dphi_bc, axis=1)) > 0
            _mindphi = ak.to_numpy(ak.fill_none(
                ak.min(_dphi_bc, axis=1), -1.0))
            minDphi_bc = np.where(_has_bc, _mindphi, -1.0).astype(np.float32)
        else:
            minDphi_bc = np.full(n, -1.0, dtype=np.float32)

        # -- min Delta-R between a b-tag and a c-tag AK4 jet at tighter WP -----
        # combinations of the ParticleNetAK4 `ak4_tag` ladder.  The ev() branch
        # `minDR_bc` is the loose-b^L x loose-c^L version; these repeat it for:
        #   minDR_bc_LM : both AK4 pass loose AND at least one passes medium
        #                 (the "loose-medium || medium-loose" b/c combination)
        #   minDR_bc_MM : both AK4 pass medium (b: tag>=51 ; c: tag in [41,50))
        # dR = sqrt(deta^2 + dphi^2) over the full b x c cartesian product,
        # then the pair mask picks the WP combination.  -1 when no pair passes.
        _need3 = {"ak4_tag", "ak4_eta", "ak4_phi"}
        if _need3 <= set(raw.fields):
            _tg3 = raw["ak4_tag"]
            _b_jets = ak.zip({"eta": raw["ak4_eta"][_tg3 >= 50],
                              "phi": raw["ak4_phi"][_tg3 >= 50],
                              "tag": _tg3[_tg3 >= 50]})
            _c_sel = (_tg3 >= 40) & (_tg3 < 50)
            _c_jets = ak.zip({"eta": raw["ak4_eta"][_c_sel],
                              "phi": raw["ak4_phi"][_c_sel],
                              "tag": _tg3[_c_sel]})
            _pr = ak.cartesian({"b": _b_jets, "c": _c_jets}, axis=1)
            _dphi_pr = np.abs(_pr["b"]["phi"] - _pr["c"]["phi"])
            _dphi_pr = ak.where(_dphi_pr > np.pi, 2.0 * np.pi - _dphi_pr, _dphi_pr)
            _dR_pr = np.sqrt((_pr["b"]["eta"] - _pr["c"]["eta"]) ** 2 + _dphi_pr ** 2)
            _b_is_M = _pr["b"]["tag"] >= 51
            _c_is_M = _pr["c"]["tag"] >= 41

            def _min_over(mask):
                _d = _dR_pr[mask]
                _has = ak.to_numpy(ak.num(_d, axis=1)) > 0
                _mn = ak.to_numpy(ak.fill_none(ak.min(_d, axis=1), -1.0))
                return np.where(_has, _mn, -1.0).astype(np.float32)

            minDR_bc_LM = _min_over(_b_is_M | _c_is_M)
            minDR_bc_MM = _min_over(_b_is_M & _c_is_M)
        else:
            minDR_bc_LM = np.full(n, -1.0, dtype=np.float32)
            minDR_bc_MM = np.full(n, -1.0, dtype=np.float32)

        # -- mSD(J) - m(b^L c^L)_closest ------------------------------------
        # Soft-drop mass of the cb-candidate AK8 minus the invariant mass of the
        # dR-closest loose b-tag + loose c-tag AK4 pair (the mass_minDR_bc
        # branch).  ~0 when J is the merged image of that same resolved b+c.
        # -999 if either input is a sentinel (no J, or <2 such AK4 jets).
        _mbc = ev("mass_minDR_bc")
        _both_ok = (ak8_sdmass_0 > -900.0) & (_mbc > 0.0)
        mSD_minus_mbc = np.where(_both_ok, ak8_sdmass_0 - _mbc, -999.0).astype(np.float32)

        # ------------------------------------------------------------------
        # tt-reconstruction SEEDS -- the (up to) two b-candidate AK4 jets
        # OUTSIDE the W->cb candidate J (dR > 0.8), ranked by the
        # ParticleNetAK4 b-score (ak4_pn_b; falls back to the ak4_tag WP
        # ladder, then to pT).  Only the two seed 4-vectors + their b-info
        # are cached here.  The actual top pairing / assembly (dR rule,
        # pT-asymmetry tie-break, mass windows) is done at PLOT TIME in
        # DataManager._attach_runtime_fields from these seeds + the J and
        # W(lv) 4-vectors, so that algorithm can be retuned with NO
        # re-derive.  ("btt" = b for tt-reco.)
        # ------------------------------------------------------------------
        _bt_need = {"ak4_pt", "ak4_eta", "ak4_phi", "ak4_mass"}
        if _bt_need <= set(raw.fields):
            _a4eta, _a4phi = raw["ak4_eta"], raw["ak4_phi"]
            _dphi_ja = np.abs(_a4phi - ak8_phi_0)
            _dphi_ja = ak.where(_dphi_ja > np.pi, 2.0 * np.pi - _dphi_ja, _dphi_ja)
            _dR_ja = np.sqrt((_a4eta - ak8_eta_0) ** 2 + _dphi_ja ** 2)
            _out_ja = _dR_ja > 0.8                       # outside the AK8 cone
            if "ak4_pn_b" in raw.fields:
                _brank = raw["ak4_pn_b"]
            elif "ak4_tag" in raw.fields:
                _brank = ak.values_astype(raw["ak4_tag"], np.float32)
            else:
                _brank = raw["ak4_pt"]
            _order = ak.argsort(ak.where(_out_ja, _brank, -1.0e9),
                                axis=1, ascending=False)

            def _seed(branch, k, dflt=-999.0):
                if branch not in raw.fields:
                    return np.full(n, dflt, dtype=np.float32)
                col = ak.pad_none(raw[branch][_order], k + 1, axis=1, clip=True)
                return ak.to_numpy(ak.fill_none(col[:, k], dflt)).astype(np.float32)

            n_b_out_ja = ak.to_numpy(ak.sum(_out_ja, axis=1)).astype(np.int16)
            btt1_pt,  btt2_pt  = _seed("ak4_pt", 0),   _seed("ak4_pt", 1)
            btt1_eta, btt2_eta = _seed("ak4_eta", 0),  _seed("ak4_eta", 1)
            btt1_phi, btt2_phi = _seed("ak4_phi", 0),  _seed("ak4_phi", 1)
            btt1_m,   btt2_m   = _seed("ak4_mass", 0),  _seed("ak4_mass", 1)
            btt1_pnb, btt2_pnb = _seed("ak4_pn_b", 0),  _seed("ak4_pn_b", 1)
            btt1_tag, btt2_tag = _seed("ak4_tag", 0, -1.0), _seed("ak4_tag", 1, -1.0)

            # --- dR-only ranking: AK4 outside J ordered by dR(., J) -----
            # (for the simple t_a = J + nearest-AK4 reconstruction; no
            #  b-tag requirement -- pure geometry.  bdr1 = closest, bdr2 = 2nd.)
            _order_dr = ak.argsort(ak.where(_out_ja, _dR_ja, 1.0e18),
                                   axis=1, ascending=True)

            def _seed_dr(branch, k, dflt=-999.0):
                if branch not in raw.fields:
                    return np.full(n, dflt, dtype=np.float32)
                col = ak.pad_none(raw[branch][_order_dr], k + 1, axis=1,
                                  clip=True)
                return ak.to_numpy(ak.fill_none(col[:, k], dflt)
                                   ).astype(np.float32)

            bdr1_pt,  bdr2_pt  = _seed_dr("ak4_pt", 0),  _seed_dr("ak4_pt", 1)
            bdr1_eta, bdr2_eta = _seed_dr("ak4_eta", 0), _seed_dr("ak4_eta", 1)
            bdr1_phi, bdr2_phi = _seed_dr("ak4_phi", 0), _seed_dr("ak4_phi", 1)
            bdr1_m,   bdr2_m   = _seed_dr("ak4_mass", 0), _seed_dr("ak4_mass", 1)
            bdr1_pnb, bdr2_pnb = _seed_dr("ak4_pn_b", 0), _seed_dr("ak4_pn_b", 1)
            bdr1_pnuds = _seed_dr("ak4_pn_uds", 0)
            bdr1_pnc = _seed_dr("ak4_pn_c", 0)
            bdr1_tag, bdr2_tag = (_seed_dr("ak4_tag", 0, -1.0),
                                  _seed_dr("ak4_tag", 1, -1.0))

            # --- same, but restricted to loose-b AK4 (ak4_tag >= 50): the
            #     three dR-closest loose-b jets outside J (bL1 = nearest).
            #     Used by the b_L-based t_a attempts (nearest-b_L / best-mass).
            if "ak4_tag" in raw.fields:
                _bL_msk = _out_ja & (raw["ak4_tag"] >= 50)
            else:
                _bL_msk = _out_ja & (_out_ja == False)      # all-False, jagged
            n_bL_out_ja = ak.to_numpy(ak.sum(_bL_msk, axis=1)).astype(np.int16)
            _order_bL = ak.argsort(ak.where(_bL_msk, _dR_ja, 1.0e18),
                                   axis=1, ascending=True)

            def _seed_bL(branch, k, dflt=-999.0):
                if branch not in raw.fields:
                    return np.full(n, dflt, dtype=np.float32)
                col = ak.pad_none(raw[branch][_order_bL], k + 1, axis=1,
                                  clip=True)
                v = ak.to_numpy(ak.fill_none(col[:, k], dflt)).astype(np.float32)
                return np.where(n_bL_out_ja > k, v, dflt).astype(np.float32)

            bL1_pt = _seed_bL("ak4_pt", 0);  bL1_eta = _seed_bL("ak4_eta", 0)
            bL1_phi = _seed_bL("ak4_phi", 0); bL1_m = _seed_bL("ak4_mass", 0)
            bL2_pt = _seed_bL("ak4_pt", 1);  bL2_eta = _seed_bL("ak4_eta", 1)
            bL2_phi = _seed_bL("ak4_phi", 1); bL2_m = _seed_bL("ak4_mass", 1)
            bL3_pt = _seed_bL("ak4_pt", 2);  bL3_eta = _seed_bL("ak4_eta", 2)
            bL3_phi = _seed_bL("ak4_phi", 2); bL3_m = _seed_bL("ak4_mass", 2)

            # --- "light nearest" AK4: the dR-closest jet outside J that is
            #     NEITHER b_L (ak4_tag>=50) NOR c_L (40<=ak4_tag<50) tagged,
            #     i.e. ak4_tag < 40.  If the event has no such untagged jet
            #     outside J, fall back to the out-of-cone AK4 with the LOWEST
            #     ParticleNetAK4 (pn_b + pn_c).  SENTINEL if no AK4 outside J.
            if {"ak4_tag", "ak4_pn_b", "ak4_pn_c"} <= set(raw.fields):
                _lite_msk = _out_ja & (raw["ak4_tag"] < 40)
                _n_lite = ak.to_numpy(ak.sum(_lite_msk, axis=1)).astype(np.int16)
                _n_out = ak.to_numpy(ak.sum(_out_ja, axis=1)).astype(np.int16)
                _pnbc = raw["ak4_pn_b"] + raw["ak4_pn_c"]
                _ord_lite = ak.argsort(ak.where(_lite_msk, _dR_ja, 1.0e18),
                                       axis=1, ascending=True)
                _ord_lfb = ak.argsort(ak.where(_out_ja, _pnbc, 1.0e18),
                                      axis=1, ascending=True)

                def _seed_lite(branch, dflt=-999.0):
                    if branch not in raw.fields:
                        return np.full(n, dflt, dtype=np.float32)
                    _cp = ak.to_numpy(ak.fill_none(ak.pad_none(
                        raw[branch][_ord_lite], 1, axis=1, clip=True)[:, 0], dflt))
                    _cf = ak.to_numpy(ak.fill_none(ak.pad_none(
                        raw[branch][_ord_lfb], 1, axis=1, clip=True)[:, 0], dflt))
                    v = np.where(_n_lite >= 1, _cp, _cf)
                    return np.where(_n_out >= 1, v, dflt).astype(np.float32)

                blite1_pt = _seed_lite("ak4_pt");   blite1_eta = _seed_lite("ak4_eta")
                blite1_phi = _seed_lite("ak4_phi"); blite1_m = _seed_lite("ak4_mass")
            else:
                blite1_pt = blite1_eta = blite1_phi = blite1_m = \
                    np.full(n, -999.0, dtype=np.float32)

            # --- "not-b_L nearest" AK4: the dR-closest jet outside J that is
            #     NOT b_L tagged (ak4_tag < 50 -- c_L or untagged both OK).
            #     Fallback when the event has no such jet outside J: the
            #     out-of-cone AK4 with the LOWEST ParticleNetAK4 b-score
            #     (ak4_pn_b).  SENTINEL if no AK4 outside J.
            if {"ak4_tag", "ak4_pn_b"} <= set(raw.fields):
                _nb_msk = _out_ja & (raw["ak4_tag"] < 50)
                _n_nb = ak.to_numpy(ak.sum(_nb_msk, axis=1)).astype(np.int16)
                _ord_nb = ak.argsort(ak.where(_nb_msk, _dR_ja, 1.0e18),
                                     axis=1, ascending=True)
                _ord_nbfb = ak.argsort(ak.where(_out_ja, raw["ak4_pn_b"], 1.0e18),
                                       axis=1, ascending=True)

                def _seed_nb(branch, dflt=-999.0):
                    if branch not in raw.fields:
                        return np.full(n, dflt, dtype=np.float32)
                    _cp = ak.to_numpy(ak.fill_none(ak.pad_none(
                        raw[branch][_ord_nb], 1, axis=1, clip=True)[:, 0], dflt))
                    _cf = ak.to_numpy(ak.fill_none(ak.pad_none(
                        raw[branch][_ord_nbfb], 1, axis=1, clip=True)[:, 0], dflt))
                    v = np.where(_n_nb >= 1, _cp, _cf)
                    return np.where(n_b_out_ja >= 1, v, dflt).astype(np.float32)

                bnb1_pt = _seed_nb("ak4_pt");   bnb1_eta = _seed_nb("ak4_eta")
                bnb1_phi = _seed_nb("ak4_phi"); bnb1_m = _seed_nb("ak4_mass")
            else:
                bnb1_pt = bnb1_eta = bnb1_phi = bnb1_m = \
                    np.full(n, -999.0, dtype=np.float32)

            # --- "best-mass" AK4: the jet outside J (ANY flavour) that gives
            #     m(J + jet) closest to 172 GeV.  Full seed cached; its
            #     ak4_tag feeds the 3-bin flavour categorical bbestm1_flav.
            if {"ak4_pt", "ak4_eta", "ak4_phi", "ak4_mass"} <= set(raw.fields):
                _MT = 172.0
                _jpx = ak8_pt_0 * np.cos(ak8_phi_0)
                _jpy = ak8_pt_0 * np.sin(ak8_phi_0)
                _jpz = ak8_pt_0 * np.sinh(np.clip(ak8_eta_0, -10.0, 10.0))
                _jE = np.sqrt(_jpx ** 2 + _jpy ** 2 + _jpz ** 2
                              + np.maximum(ak8_sdmass_0, 0.0) ** 2)
                _a4pt, _a4m = raw["ak4_pt"], raw["ak4_mass"]
                _apx = _a4pt * np.cos(_a4phi)
                _apy = _a4pt * np.sin(_a4phi)
                # np.clip is not an awkward ufunc -> use sinh directly (AK4
                # |eta| < 5 always).
                _apz = _a4pt * np.sinh(_a4eta)
                _aE = np.sqrt(_apx ** 2 + _apy ** 2 + _apz ** 2
                              + np.maximum(_a4m, 0.0) ** 2)
                _cm2 = ((_jE + _aE) ** 2 - (_jpx + _apx) ** 2
                        - (_jpy + _apy) ** 2 - (_jpz + _apz) ** 2)
                _cm = np.sqrt(ak.where(_cm2 > 0.0, _cm2, 0.0))
                _dm_bm = ak.where(_out_ja, np.abs(_cm - _MT), 1.0e18)
                _ord_bm = ak.argsort(_dm_bm, axis=1, ascending=True)

                def _seed_bm(branch, dflt=-999.0):
                    if branch not in raw.fields:
                        return np.full(n, dflt, dtype=np.float32)
                    _c = ak.to_numpy(ak.fill_none(ak.pad_none(
                        raw[branch][_ord_bm], 1, axis=1, clip=True)[:, 0], dflt))
                    return np.where(n_b_out_ja >= 1, _c, dflt).astype(np.float32)

                bbestm1_pt = _seed_bm("ak4_pt");   bbestm1_eta = _seed_bm("ak4_eta")
                bbestm1_phi = _seed_bm("ak4_phi"); bbestm1_m = _seed_bm("ak4_mass")
                bbestm1_tag = _seed_bm("ak4_tag", -1.0)
                bbestm1_pnb = _seed_bm("ak4_pn_b", -1.0)
                bbestm1_pnuds = _seed_bm("ak4_pn_uds", -1.0)
                bbestm1_pnc = _seed_bm("ak4_pn_c", -1.0)

                # --- top-5 best-mass AK4 (sorted by |m(J+AK4) - 172|) with
                #     their "top-consistency" x = dR(AK4,J) * pT(J+AK4) /
                #     (2*172.5)  (~1 for a genuine 2-body decay).  The windowed
                #     picks mt8/9/10 (first best-mass jet with x < X_MAX) are
                #     built at PLOT TIME from these -> the X_MAX cuts are
                #     runtime-tunable with NO re-derive.
                _cpt = np.sqrt((_jpx + _apx) ** 2 + (_jpy + _apy) ** 2)
                _xcons = ak.where(_out_ja, _dR_ja * _cpt / (2.0 * 172.5), 1.0e9)

                def _seed_bmk(col_jag, k, dflt=-999.0):
                    _c = ak.to_numpy(ak.fill_none(ak.pad_none(
                        col_jag[_ord_bm], k + 1, axis=1, clip=True)[:, k], dflt))
                    return np.where(n_b_out_ja > k, _c, dflt).astype(np.float32)

                #     PNet b / uds / c of each are cached too, so the
                #     b/(b+uds) and (b+c)/(b+c+uds) discriminants can be
                #     evaluated at plot time for the EXACT AK4 that the final
                #     mt reco (first bmk with x<1.5) ends up using.
                _bmk = {}
                for _k in range(5):
                    _bmk[f"bmk{_k + 1}_pt"]  = _seed_bmk(raw["ak4_pt"], _k)
                    _bmk[f"bmk{_k + 1}_eta"] = _seed_bmk(raw["ak4_eta"], _k)
                    _bmk[f"bmk{_k + 1}_phi"] = _seed_bmk(raw["ak4_phi"], _k)
                    _bmk[f"bmk{_k + 1}_m"]   = _seed_bmk(raw["ak4_mass"], _k)
                    _bmk[f"bmk{_k + 1}_x"]   = _seed_bmk(_xcons, _k, 1.0e9)
                    # ak4_tag WP of each -> flavour categorical of j* (the
                    # windowed pick) at plot time -> jstar_flav.
                    _bmk[f"bmk{_k + 1}_tag"]   = _seed_bmk(raw["ak4_tag"], _k, -1.0) \
                        if "ak4_tag" in raw.fields else np.full(n, -1.0, np.float32)
                    _bmk[f"bmk{_k + 1}_pnb"]   = _seed_bmk(raw["ak4_pn_b"], _k, -1.0) \
                        if "ak4_pn_b" in raw.fields else np.full(n, -1.0, np.float32)
                    _bmk[f"bmk{_k + 1}_pnuds"] = _seed_bmk(raw["ak4_pn_uds"], _k, -1.0) \
                        if "ak4_pn_uds" in raw.fields else np.full(n, -1.0, np.float32)
                    _bmk[f"bmk{_k + 1}_pnc"]   = _seed_bmk(raw["ak4_pn_c"], _k, -1.0) \
                        if "ak4_pn_c" in raw.fields else np.full(n, -1.0, np.float32)
            else:
                bbestm1_pt = bbestm1_eta = bbestm1_phi = bbestm1_m = \
                    np.full(n, -999.0, dtype=np.float32)
                bbestm1_tag = bbestm1_pnb = bbestm1_pnuds = bbestm1_pnc = \
                    np.full(n, -1.0, dtype=np.float32)
                _bmk = {}
                for _k in range(5):
                    for _s in ("pt", "eta", "phi", "m"):
                        _bmk[f"bmk{_k + 1}_{_s}"] = np.full(n, -999.0, np.float32)
                    _bmk[f"bmk{_k + 1}_x"] = np.full(n, 1.0e9, np.float32)
                    for _s in ("tag", "pnb", "pnuds", "pnc"):
                        _bmk[f"bmk{_k + 1}_{_s}"] = np.full(n, -1.0, np.float32)
        else:
            _zf = np.full(n, -999.0, dtype=np.float32)
            n_b_out_ja = np.full(n, -1, dtype=np.int16)
            n_bL_out_ja = np.full(n, -1, dtype=np.int16)
            btt1_pt = btt2_pt = btt1_eta = btt2_eta = _zf
            btt1_phi = btt2_phi = btt1_m = btt2_m = _zf
            btt1_pnb = btt2_pnb = btt1_tag = btt2_tag = _zf
            bdr1_pt = bdr2_pt = bdr1_eta = bdr2_eta = _zf
            bdr1_phi = bdr2_phi = bdr1_m = bdr2_m = _zf
            bdr1_pnb = bdr2_pnb = bdr1_tag = bdr2_tag = bdr1_pnuds = bdr1_pnc = _zf
            bL1_pt = bL1_eta = bL1_phi = bL1_m = _zf
            bL2_pt = bL2_eta = bL2_phi = bL2_m = _zf
            bL3_pt = bL3_eta = bL3_phi = bL3_m = _zf
            blite1_pt = blite1_eta = blite1_phi = blite1_m = _zf
            bnb1_pt = bnb1_eta = bnb1_phi = bnb1_m = _zf
            bbestm1_pt = bbestm1_eta = bbestm1_phi = bbestm1_m = _zf
            bbestm1_tag = bbestm1_pnb = bbestm1_pnuds = bbestm1_pnc = \
                np.full(n, -1.0, dtype=np.float32)
            _bmk = {}
            for _k in range(5):
                for _s in ("pt", "eta", "phi", "m"):
                    _bmk[f"bmk{_k + 1}_{_s}"] = _zf
                _bmk[f"bmk{_k + 1}_x"] = np.full(n, 1.0e9, np.float32)
                for _s in ("tag", "pnb", "pnuds", "pnc"):
                    _bmk[f"bmk{_k + 1}_{_s}"] = np.full(n, -1.0, np.float32)

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

            # Jagged per-jet arrays for the "flatten" var_mode. Currently no
            # active plot uses var_mode="flatten" (build_plot_settings keeps
            # only the leading sd-mass + Dbc score, MAT is all event-level),
            # and they dominate the cache size / RSS (tt-semi: ~4 GB -> a few
            # hundred MB without them). Re-enable together with any flatten
            # plot -- the leading-jet aliases used by cuts come from the
            # *_0 fields below, not from these.
            # "ak8_pt": ak8_pt,
            # "ak8_eta": ak8_eta,
            # "ak8_phi": ak8_phi,
            # "ak8_sdmass": ak8_sdmass,
            # "ak4_pt": ak4_pt,
            # "ak4_eta": ak4_eta,
            # "ak4_phi": ak4_phi,

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
            "score_Dbc_jb": score_Dbc_jb,
            "score_Dbc_jsup1": score_Dbc_jsup1,
            "score_Dbc_jsup2": score_Dbc_jsup2,
            "score_SC": score_SC,

            "score_cata_w_qq_norm": score_cata_w_qq_norm,
            "score_cata_qcd_norm": score_cata_qcd_norm,
            "score_cata_top_bqq_norm": score_cata_top_bqq_norm,
            "score_cata_top_bc_norm": score_cata_top_bc_norm,
            "score_cata_top_bq_norm": score_cata_top_bq_norm,
            "score_cata_top_sum_norm": score_cata_top_sum_norm,
            "score_cata_non_norm": score_cata_non_norm,

            # ---- MAT jet-kinematics study ---------------------------------
            # leading-AK8 substructure
            "ak8_rawFactor_0": ak8_rawFactor_0,
            "ak8_tau21_0": ak8_tau21_0,
            "ak8_tau32_0": ak8_tau32_0,
            "ak8_tau31_0": ak8_tau31_0,
            "ak8_nConst_0": ak8_nConst_0,
            "ak8_sdmass_maxmass_0": ak8_sdmass_maxmass_0,
            # "ak8_sdmass_lead_pt_0" / "ak8_sdmass_sub_pt_0": commented out 2026-09-04 (user)
            "ak8_sdmass_sub_mass_0": ak8_sdmass_sub_mass_0,
            "ak8_sdmass_sub_bc_0": ak8_sdmass_sub_bc_0,
            "ak8_tau21_sub_bc_0": ak8_tau21_sub_bc_0,
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
            **_gpt_extra,

            # GloParT discriminant ratios
            "gpt_bc_frac": gpt_bc_frac,
            "gpt_2prong_sum": gpt_2prong_sum,
            "gpt_bc_vs_2prong": gpt_bc_vs_2prong,
            "gpt_bc_vs_nontop": gpt_bc_vs_nontop,
            "gpt_bc_vs_qcd": gpt_bc_vs_qcd,
            "gpt_bc_vs_cc": gpt_bc_vs_cc,
            "gpt_bc_vs_bb": gpt_bc_vs_bb,
            "gpt_bc_vs_qq": gpt_bc_vs_qq,
            "gpt_cs_vs_qq": gpt_cs_vs_qq,
            "gpt_hf_w": gpt_hf_w,
            "gpt_top_frac": gpt_top_frac,
            "gpt_w_vs_top": gpt_w_vs_top,
            "gpt_w_vs_top_jb": gpt_w_vs_top_jb,
            "gpt_bc_vs_topbw": gpt_bc_vs_topbw,
            "gpt_bc_vs_top": gpt_bc_vs_top,
            "gpt_topbw_vs_top": gpt_topbw_vs_top,
            "gpt_cW_vs_qq": gpt_cW_vs_qq,
            "gpt_bW_vs_qq": gpt_bW_vs_qq,

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
            "minDR_bc_LM": minDR_bc_LM,
            "minDR_bc_MM": minDR_bc_MM,
            "mass_minDR_bc": ev("mass_minDR_bc"),
            "maxMass_bc": ev("maxMass_bc"),

            # derived angular variables (leading AK8 vs lepton / MET / AK4)
            "dphi_lep_ak8": dphi_lep_ak8,
            "deta_lep_ak8": deta_lep_ak8,
            "dR_lep_ak8": dR_lep_ak8,
            "dphi_met_ak8": dphi_met_ak8,
            "dphi_lep_met": dphi_lep_met,
            "dphi_w_ak8": dphi_w_ak8,
            "dR_w_ak8": dR_w_ak8,
            "dphi_ak4_ak8": dphi_ak4_ak8,
            "deta_ak4_ak8": deta_ak4_ak8,
            "dR_ak4_ak8": dR_ak4_ak8,
            "ptrel_lep_ak8": ptrel_lep_ak8,
            "n_c_nearcb_L": n_c_nearcb_L,
            "n_c_nearcb_M": n_c_nearcb_M,
            "n_b_nearcb_L": n_b_nearcb_L,
            "n_b_nearcb_M": n_b_nearcb_M,
            "mass_cL_nearb": mass_cL_nearb,
            "mass_bc_bestmW": mass_bc_bestmW,
            "mass_bbc_minDR": mass_bbc_minDR,
            "mass_bbc_bestmt": mass_bbc_bestmt,
            "minDphi_bc": minDphi_bc,
            "mSD_minus_mbc": mSD_minus_mbc,

            # tt-reco seeds (2 highest-b-score AK4 outside J; the top
            # pairing itself is done in _attach_runtime_fields)
            "n_b_out_ja": n_b_out_ja,
            "btt1_pt": btt1_pt, "btt1_eta": btt1_eta, "btt1_phi": btt1_phi,
            "btt1_m": btt1_m, "btt1_pnb": btt1_pnb, "btt1_tag": btt1_tag,
            "btt2_pt": btt2_pt, "btt2_eta": btt2_eta, "btt2_phi": btt2_phi,
            "btt2_m": btt2_m, "btt2_pnb": btt2_pnb, "btt2_tag": btt2_tag,
            # dR-ranked AK4 outside J (closest / 2nd-closest in dR)
            "bdr1_pt": bdr1_pt, "bdr1_eta": bdr1_eta, "bdr1_phi": bdr1_phi,
            "bdr1_m": bdr1_m, "bdr1_pnb": bdr1_pnb, "bdr1_pnc": bdr1_pnc,
            "bdr1_pnuds": bdr1_pnuds, "bdr1_tag": bdr1_tag,
            "bdr2_pt": bdr2_pt, "bdr2_eta": bdr2_eta, "bdr2_phi": bdr2_phi,
            "bdr2_m": bdr2_m, "bdr2_pnb": bdr2_pnb, "bdr2_tag": bdr2_tag,
            # dR-ranked loose-b (ak4_tag>=50) AK4 outside J
            "n_bL_out_ja": n_bL_out_ja,
            "bL1_pt": bL1_pt, "bL1_eta": bL1_eta, "bL1_phi": bL1_phi, "bL1_m": bL1_m,
            "blite1_pt": blite1_pt, "blite1_eta": blite1_eta,
            "blite1_phi": blite1_phi, "blite1_m": blite1_m,
            "bnb1_pt": bnb1_pt, "bnb1_eta": bnb1_eta,
            "bnb1_phi": bnb1_phi, "bnb1_m": bnb1_m,
            "bbestm1_pt": bbestm1_pt, "bbestm1_eta": bbestm1_eta,
            "bbestm1_phi": bbestm1_phi, "bbestm1_m": bbestm1_m,
            "bbestm1_tag": bbestm1_tag, "bbestm1_pnb": bbestm1_pnb,
            "bbestm1_pnuds": bbestm1_pnuds, "bbestm1_pnc": bbestm1_pnc,
            # top-5 best-|m(J+AK4)-M_top| AK4 outside J + their
            # top-consistency x = dR*pT(J+AK4)/(2*172.5); the x-window
            # cut for mt8/9/10 is applied at plot time.
            **_bmk,
            "bL2_pt": bL2_pt, "bL2_eta": bL2_eta, "bL2_phi": bL2_phi, "bL2_m": bL2_m,
            "bL3_pt": bL3_pt, "bL3_eta": bL3_eta, "bL3_phi": bL3_phi, "bL3_m": bL3_m,

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
        # (id(arr), cut) -> dict(mask, weights, true_cat, mat_cat), all already
        # sliced by the event mask. Every MAT plot shares the same cut and
        # sample set, so this turns 73x re-evaluation into one pass.
        self._ctx_cache = {}

    def event_context(self, arr, cut, is_data):
        """Cut mask + masked weights / true_cat / mat_cat for one array, cached."""
        key = (id(arr), cut)
        hit = self._ctx_cache.get(key)
        if hit is not None:
            return hit

        mask = eval_cut(cut, arr)
        weights = ak.to_numpy(arr["weights"])[mask].astype(np.float32)
        if is_data:
            weights = np.ones_like(weights, dtype=np.float32)
        true_cat = (ak.to_numpy(arr["true_cat"])[mask]
                    if "true_cat" in arr.fields else None)
        mat_cat = (ak.to_numpy(arr["mat_cat"])[mask]
                   if "mat_cat" in arr.fields else None)

        # --- refine mat_cat code 6 (PLOT-TIME) -----------------------------
        # Code 6 is set ONLY for signal events (sig_wbc = is_wbc & is_signal)
        # whose J is NOT the merged 2-prong W->cb jet (that is codes 0/5).
        # These are STILL genuine t->bW, W->cb signal -- just a different
        # merge topology -- so split code 6 by the candidate's own truth type:
        #   true_cat Top_bc  (ak8_type 2, n_c=1) -> J = (b' c) merge ->
        #        signal code 21 "Wcb_tmrg_bc"  (t^2, drawn red dashed).
        #   true_cat Top_bq  (ak8_type 2, n_c=0) -> J = (b' b) merge ->
        #        signal code 22 "Wcb_tmrg_bb"  (t^2, drawn red dash-dot).
        #        (was: both -> one code 21; routed into the t->bc/bq PROXY
        #         lines 2/7 before that -- ~25% of the SR signal was hidden.)
        #   true_cat Top_bqq (ak8_type 4) -> J = (b' b c) full merge ->
        #        code 20 "Cat_Top_bbc" -- signal, t^3, drawn red dash-dot-dot.
        #   true_cat Other   -> W->cb fully RESOLVED (c and b are separate AK4
        #        jets, J is not the cb image) -> stays code 6 -> its own
        #        signal line "Wcb_res", drawn red dotted (2026-09-03, user:
        #        restored -- it is genuine t->bW, W->cb signal, just a
        #        non-merged topology; counts in S).
        # => the W->cb signal is 5 red lines: Wcb (solid), Wcb_tmrg_bc
        #    (dashed), Wcb_tmrg_bb (dash-dot), Cat_Top_bbc (dash-dot-dot),
        #    Wcb_res (dotted).
        if mat_cat is not None and true_cat is not None:
            mat_cat = np.asarray(mat_cat).copy()
            _cc = self.cfg.true_cat_codes
            _c6 = mat_cat == 6
            if _c6.any():
                mat_cat[_c6 & (true_cat == _cc["Cat_Top_bc"])] = 21
                mat_cat[_c6 & (true_cat == _cc["Cat_Top_bq"])] = 22
                mat_cat[_c6 & (true_cat == _cc["Cat_Top_bqq"])] = 20
            # split the fully-merged hadronic-top proxy t^3 (code 8) by
            # whether the merged W gave a c-quark prong:
            #   n_c(J) >= 1 -> code 23 "t^3(b',c,q)"   (W->cs / W->cd)
            #   n_c(J) == 0 -> stays code 8 "t^3(b',q,q)"  (W->ud)
            # ak8_n_c_in_jet counts W-daughter c quarks in J (top-b excluded).
            if "ak8_n_c_in_jet" in arr.fields:
                _nc = ak.to_numpy(arr["ak8_n_c_in_jet"])[mask]
                mat_cat[(mat_cat == 8) & (_nc >= 1)] = 23

        ctx = {"mask": mask, "weights": weights,
               "true_cat": true_cat, "mat_cat": mat_cat}
        self._ctx_cache[key] = ctx
        return ctx

    # MAT overlay category codes -> short label (matches draw_matching).
    # NB codes shown are POST event_context refinement: code 6 here is only
    # "W->cb resolved"; the two t^2 top-b+prong partial merges went to codes
    # 21 (b' c) / 22 (b' b), the full 3-prong merge to code 20.
    _MAT_CODES = [(0, "W->cb (J=jet)"), (5, "W->cb (J!=jet)"), (6, "W->cb resolved (signal)"),
                  (21, "t^2(b'c)"), (22, "t^2(b'b)"),
                  (1, "W->cq (=ja)"), (9, "W->cq (!=ja)"), (10, "W->cq resolved"),
                  (4, "W->qq' (=ja)"), (11, "W->qq' (!=ja)"), (12, "W->qq' resolved"),
                  (2, "t^2(b'c) proxy"), (7, "t^2(b'q)"),
                  (23, "t^3(b'cq)"), (8, "t^3(b'qq)"),
                  (20, "t->bbc (merged)"),
                  (13, "Z->qq' merged"), (14, "Z->cc merged"), (15, "Z->bb merged"),
                  (3, "rest")]

    def mat_report(self, samples, cut):
        """One-pass summary printed before the MAT draw loop: per-sample event
        yields after the cut, and the global leading-AK8 truth-category
        breakdown. Side effect: warms self._ctx_cache for every sample."""
        cat_n = {c: 0 for c, _ in self._MAT_CODES}
        cat_w = {c: 0.0 for c, _ in self._MAT_CODES}
        rows = []
        for s in samples:
            if s["is_data"]:
                continue
            arr = s["array"]
            ctx = self.event_context(arr, cut, False)
            m, w, mc = ctx["mask"], ctx["weights"], ctx["mat_cat"]
            rows.append((s["name"], s["group"], len(arr),
                         int(m.sum()), float(w.sum())))
            if mc is not None:
                for c, _ in self._MAT_CODES:
                    sel = mc == c
                    cat_n[c] += int(sel.sum())
                    cat_w[c] += float(w[sel].sum())

        print("=" * 100)
        print(f"[MAT] Sample event yields   (cut: {cut})")
        print(f"  {'sample':<30}{'group':<11}{'N raw':>13}{'N cut':>13}{'sum(w)':>15}")
        for name, grp, nraw, ncut, sumw in rows:
            print(f"  {name:<30}{grp:<11}{nraw:>13,}{ncut:>13,}{sumw:>15.3f}")
        tot_w = sum(cat_w.values()) or 1.0
        tot_n = sum(cat_n.values())
        print("-" * 100)
        print("[MAT] cb-candidate-AK8 (J) truth-category breakdown   (lumi-weighted)")
        print(f"  {'category':<18}{'N':>14}{'sum(w)':>15}{'yield':>10}")
        for c, lbl in self._MAT_CODES:
            print(f"  {lbl:<18}{cat_n[c]:>14,}{cat_w[c]:>15.3f}"
                  f"{100.0 * cat_w[c] / tot_w:>9.2f}%")
        print(f"  {'TOTAL':<18}{tot_n:>14,}{tot_w:>15.3f}{100.0:>9.2f}%")
        print("=" * 100)

    def get_values_weights_for_plot(self, arr, plot_cfg, is_data):
        var = plot_cfg["var"]
        var_mode = plot_cfg.get("var_mode", "event")
        cut = plot_cfg.get("cut", "1")

        if var not in arr.fields:
            return None, None, None, None, None
        # (5th return value is the per-event "no j*" flag, see below)

        if var_mode in ["event", "leading"]:
            ctx = self.event_context(arr, cut, is_data)
            event_mask = ctx["mask"]

            values = ak.to_numpy(arr[var])[event_mask]
            finite_mask = np.isfinite(values) & (values > SENTINEL_CUT)

            values = values[finite_mask]
            weights = ctx["weights"][finite_mask]
            true_cat = (ctx["true_cat"][finite_mask]
                        if ctx["true_cat"] is not None else None)
            mat_cat = (ctx["mat_cat"][finite_mask]
                       if ctx["mat_cat"] is not None else None)
            # "no j* candidate" flag aligned to the same event selection --
            # drives the dotted "J w/o-j*" overlay (Wcb_nomt /
            # Cat_Top_bc_nomt).  Prefer the explicit mt_nojs flag; fall back
            # to mt < 0 for caches built before it existed.
            if "mt_nojs" in arr.fields:
                aux_nojs = (ak.to_numpy(arr["mt_nojs"])[event_mask][finite_mask]
                            > 0.5)
            elif "mt" in arr.fields:
                aux_nojs = (ak.to_numpy(arr["mt"])[event_mask][finite_mask]
                            < 0.0)
            else:
                aux_nojs = None

            return values, weights, true_cat, mat_cat, aux_nojs

        # --- flatten path: uncached (not used by MAT) ---
        event_mask = eval_cut(cut, arr)
        event_weights = ak.to_numpy(arr["weights"])[event_mask]
        if is_data:
            event_weights = np.ones_like(event_weights, dtype=np.float32)

        def _field_event(field):
            if field not in arr.fields:
                return None
            return ak.to_numpy(arr[field])[event_mask]

        if var_mode == "flatten":
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

            return values, weights, true_cat, mat_cat, None

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

        def add_cv(key, c, v):
            hist_data[key] = hist_data.get(key, 0) + c
            hist_var[key] = hist_var.get(key, 0) + v

        for sample in samples:
            arr = sample["array"]
            group = sample["group"]
            is_data = sample["is_data"]

            values, weights, true_cat, mat_cat, aux_nojs = self.get_values_weights_for_plot(
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
                if true_cat is None:
                    continue
                codes = list(self.cfg.true_cat_codes.values())
                per = scatter_hist_by_category(values, weights, true_cat, bins, codes)
                for key in self.cfg.true_cat_order:
                    c, v = per[self.cfg.true_cat_codes[key]]
                    if not c.any() and not v.any():
                        continue
                    add_cv(key, c, v)

            elif mode == "mat":
                # Truth-matched cb-candidate-AK8 overlay, driven by `mat_cat`.
                # ONE red W->cb signal line = every event with a genuine merged
                # 2-prong W->cb AK8 jet, whether or not the selected J is that
                # jet (codes 0 + 5; the "J picked another jet" fraction is
                # <0.5%, so it is folded back in rather than shown separately).
                # Code 6 (no merged W->cb jet in the event) is refined in
                # event_context: J = t^2/t^3 top-substructure merge -> 21/22/20;
                # else (W->cb fully resolved -- c and b are separate AK4 jets) ->
                # stays 6 -> its own red dotted SIGNAL line "Wcb_res", counts
                # in S (2026-09-03, user: restored -- genuine W->cb signal,
                # non-merged topology).
                # Other flavours: =J codes 1/4 keep their key; the !=J codes
                # 9/11 pool into grey "W_sublead"; resolved codes 10/12 and code
                # 3 fold into "Rest".
                #   2 t->bc  7 t->bq  8 t->bqq'  20 t->bbc (W->cb merged top)
                if mat_cat is None:
                    continue

                _code_key = ((0, "Wcb"), (5, "Wcb"), (6, "Wcb_res"),
                             (21, "Wcb_tmrg_bc"), (22, "Wcb_tmrg_bb"),
                             # W background: two green merged lines (Wcq/Wqq,
                             # the 2 darkest of the V green ramp); the !=J
                             # (9/11) and resolved (10/12) W all fold into
                             # grey "Rest" (V-resolved), 2026-09-03.
                             (1, "Wcq"), (9, "Rest"), (10, "Rest"),
                             (4, "Wqq_light"), (11, "Rest"), (12, "Rest"),
                             # hadronic-top proxy: 4 blue lines
                             (2, "Cat_Top_bc"), (7, "Cat_Top_bq"),
                             (23, "Cat_Top_bqqc"), (8, "Cat_Top_bqq"),
                             (20, "Cat_Top_bbc"),
                             # merged Z: cc+bb -> "Zhf", light qq' -> "Zlight"
                             # (both green); resolved Z -> Rest (V-resolved).
                             (13, "Zlight"), (14, "Zhf"), (15, "Zhf"),
                             (3, "Rest"))
                per = scatter_hist_by_category(
                    values, weights, mat_cat, bins, [c for c, _ in _code_key])
                for code, key in _code_key:
                    c, v = per[code]
                    if not c.any() and not v.any():
                        continue
                    add_cv(key, c, v)

                # --- dotted "J w/o-j*" subsets of signal + proxy ---------
                # mat_cat 0/5 = merged W->cb signal ; 2 = t->bc proxy.  The
                # events for which the m_t reco found NO compatible j* (bmk1..5
                # all fail x < 2) are re-histogrammed into "Wcb_nomt" /
                # "Cat_Top_bc_nomt" and drawn later as dotted red / blue lines
                # -- shows WHERE those events land (their m_t is now mSD(J),
                # not -1).  SUBSETS of Wcb / Cat_Top_bc -> excluded from the
                # grand total and the S/sqrt(S+B) background sum.
                if aux_nojs is not None:
                    _nomt = np.asarray(aux_nojs, dtype=bool)
                    for _codes, _skey in (((0, 5, 21, 22), "Wcb_nomt"),
                                          ((2,),          "Cat_Top_bc_nomt")):
                        _sub = np.isin(mat_cat, _codes) & _nomt
                        if _sub.any():
                            add(_skey, values[_sub], weights[_sub])

            elif mode == "mixed":
                if group in self.cfg.signal_groups:
                    add(group, values, weights)
                elif true_cat is not None:
                    codes = list(self.cfg.true_cat_codes.values())
                    per = scatter_hist_by_category(values, weights, true_cat, bins, codes)
                    for key in self.cfg.true_cat_order:
                        c, v = per[self.cfg.true_cat_codes[key]]
                        if not c.any() and not v.any():
                            continue
                        add_cv(key, c, v)

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
        self._save_and_show(fig, outbase, tag=cfg.get("_progress", "PLOT"))

    def draw_matching(self, hist_data, plot_cfg, hist_var=None):
        """
        Truth-matched cb-candidate-AK8 overlay (--MAT mode).  The candidate is
        J = the AK8 jet with the highest GloParT bc score (see `_cand_ak8`).

        Step-line histograms driven by `mat_cat`.  y-axis set by the CLI `norm`
        arg: ABS (default) = absolute lumi-weighted yields with the red W->cb
        signal line auto-scaled to the t->bc proxy integral (the factor is
        quoted in the legend; SD-mass spectra forced to log);
        NORM = every line to unit area, per-spec lin/log.
        ONE red W->cb signal line = every event with a genuine merged 2-prong
        gen W->cb AK8 jet (codes 0 + 5; the "J picked another AK8" fraction
        is <0.5% and folded in).
        The 4 hadronic-V bkg lines W->(cq) / W->(qq) / Z->(cc/bb) / Z->(qq)
        share one GREEN shade ramp (darkest -> lightest in that order); the
        other-flavour !=J and resolved V events fold into grey "Rest".
        t colours: a 4-line blue ramp, pure blue (proxy, solid) getting
        lighter through (b',q) / t->(bcq) / t->(bqq) (lightest).
        t^2 = 2-prong (partial) top t->bc / t->bq ; t^3 = fully-merged 3-quark
        top t->bqq'.
        Rest -> grey (everything else: gluon/light/c/b jets, hadronic Z -- these
        ntuples have no accurate per-jet -> Z match -- and resolved hadronic W
        events where no AK8 merged the decay, codes 10/12).
        Legends: left = merged/sub-leading W components, right = top
        components + Rest.
        Lower panel: NORM -> shape ratio (W->cb =J)/(t->bc); ABS -> per-bin
        significance S_i/sqrt(S_i+B_i) with S = every W->cb topology
        (Wcb + Wcb_tmrg_bc + Wcb_tmrg_bb + Cat_Top_bbc + Wcb_res) and B =
        every other category.  Annotated with the
        quadrature sum sqrt(sum_i S_i^2/(S_i+B_i)) (top-left) and, from a
        brute-force 1-bin optimisation, the single contiguous window
        [x_lo, x_hi] that maximises the integrated S/sqrt(S+B) -- two magenta
        lines + text.  Specs flagged `truth_only` (gen jet-content: N_b, N_c,
        # partons -- the branch is unfilled for background) draw the raw
        S/sqrt(S+B) curve but skip the (biased) optimisation and annotation.

        Each legend entry carries the category's share (%) of the total
        luminosity-weighted yield, computed from `hist_data` before the shape
        normalization; the categories partition the sample so the
        percentages sum to ~100%.  The signal (red W->cb) entry is shown to
        3 dp, all others to 1 dp.

        `hist_var` (per-bin sum of w^2, under/overflow already folded in) gives
        the stat uncertainty err(shape_i) = sqrt(var_i) / N_tot, used only for
        the shaded band in the ratio panel (the overlay lines carry no band).
        """
        cfg = dict(self.cfg.default_plot)
        cfg.update(plot_cfg)

        bins = cfg["bins"]
        name = cfg["name"]
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        key_order = ["Wcb", "Wcb_tmrg_bc", "Wcb_tmrg_bb", "Cat_Top_bbc",
                     "Wcb_res",
                     "Wcb_nomt",
                     "Wcq", "Wqq_light",
                     "Cat_Top_bc", "Cat_Top_bc_nomt",
                     "Cat_Top_bq", "Cat_Top_bqqc", "Cat_Top_bqq",
                     "Rest", "Zhf", "Zlight"]
        # Subsets of Wcb / Cat_Top_bc (events with mt == -1) -- drawn as dotted
        # lines, but NOT counted in the grand total or the S/sqrt(S+B) bkg sum.
        _SUBSET_KEYS = ("Wcb_nomt", "Cat_Top_bc_nomt")

        # Red line "Wcb" = every event with a genuine merged 2-prong gen
        # W->cb AK8 jet (mat_cat 0 + 5).  Drives the auto-scale and is (with
        # the other 4 red lines) the S/sqrt(S+B) numerator.  Events with NO
        # merged W->cb jet: code 6 -> partial/full top merges reassigned to
        # t^2/t^3 (codes 21/22/20) in event_context; the rest (W->cb fully
        # resolved) stays code 6 -> its own red dotted "Wcb_res" signal line.
        # Other-flavour !=J W (codes 9/11) and resolved W (10/12) fold
        # into grey "Rest".
        # NOTATION (2026-09-03, user):
        #   parenthesis = quarks MERGED inside the AK8 jet J, e.g.
        #     W->(cb)  = the c and b of the W both inside J (2-prong jet)
        #     t->(bbc) = b' + (W->cb) all inside J  (fully-merged top)
        #     t->(b)qq = b' merged in J, the two light W quarks are
        #                separate AK4 jets
        #   a quark WITHOUT parenthesis = a separate resolved AK4 jet.
        #   b' = the b from the top decay ;  b_l = the b overlapping the
        #   lepton (leptonic-top side) ;  a bare b/c/q = anything else.
        #   t^2 = 2-prong merged top (2 quarks in J) ;
        #   t^3 = 3-quark fully-merged top (3 quarks in J).
        mat_labels = {
            # the 5 red W->cb SIGNAL lines
            "Wcb":         r"$W(cb)$",
            "Wcb_tmrg_bc": r"$t^2(b'c)$",
            "Wcb_tmrg_bb": r"$t^2(b'b)$",
            "Cat_Top_bbc": r"$t^3(b'bc)$",
            "Wcb_res":     r"$c,b,b'$ resolved",
            "Wcb_nomt": r"$J$ w/o-$j^{*}$",
            "Cat_Top_bc_nomt": r"$t^2(b'c)$ w/o-$j^{*}$",
            # W background -- 2 darkest of the green V ramp
            "Wcq": r"$W(cq)$",
            "Wqq_light": r"$W(qq)$",
            # hadronic-top proxy -- 4 blue lines
            "Cat_Top_bc":   r"$t^2(b'c)$ proxy",
            "Cat_Top_bq":   r"$t^2(b'q)$",
            "Cat_Top_bqqc": r"$t^3(b'cq)$",
            "Cat_Top_bqq":  r"$t^3(b'qq)$",
            "Rest": r"Rest: $g,q,c,b$, $V^{\mathrm{rsd}}$",
            # merged hadronic Z -- 2 lightest of the green V ramp
            "Zhf":    r"$Z(c\bar c/b\bar b)$",
            "Zlight": r"$Z(qq)$",
        }

        # ``NORM`` CLI arg -> shape comparison (every line to unit area).
        # Default -> absolute lumi-weighted yields, with the red W->cb signal
        # line (=J, code 0) scaled up so it is visible against the ~1000x
        # larger backgrounds.  The scale factor is computed automatically as
        # yield(t->bc proxy) / yield(W->cb =J) within the plotted range, i.e.
        # the signal line is normalised to the same integral as the t^2:t->bc
        # proxy line, and the factor is quoted in the legend.  The !=J and
        # resolved pools are NOT scaled (treated as background).
        # S/sqrt(S+B) numerator = every W->cb topology, 5 red lines:
        #   Wcb          W(cb)        merged 2-prong W->cb jet   (code 0/5)
        #   Wcb_tmrg_bc  t^2(b',c)   top-b + W's c merged       (code 21)
        #   Wcb_tmrg_bb  t^2(b',b)   top-b + W's b merged       (code 22)
        #   Cat_Top_bbc  t^3(b',b,c) fully-merged top t->(bbc)  (code 20)
        #   Wcb_res      W->cb (res.) c and b are separate AK4   (code 6)
        # all genuine t->bW, W->cb signal (user 2026-09-02 / 09-03).
        _SIG5 = ("Wcb", "Wcb_tmrg_bc", "Wcb_tmrg_bb", "Cat_Top_bbc", "Wcb_res")
        SIGNAL_KEYS = _SIG5
        SCALED_KEYS = _SIG5 + ("Wcb_nomt",)   # *SIGNAL_SCALE
        PROXY_KEY = "Cat_Top_bc"
        normalize = bool(cfg.get("_normalize", False))
        signal_only = bool(cfg.get("_signal_only", False))
        # MAT-SIGNAL: keep only the 5 red W->cb lines (+ their _nomt subset).
        if signal_only:
            key_order = [k for k in key_order
                         if k in SIGNAL_KEYS or k == "Wcb_nomt"]

        hist_var = hist_var or {}
        raw_tot = {}
        abs_hists = {}                 # raw lumi-weighted counts per category
        shape_hists = {}               # counts / area   (drives the ratio panel)
        shape_errs = {}
        for key in key_order:
            counts = hist_data.get(key)
            if counts is None:
                continue

            total = float(np.sum(counts))
            raw_tot[key] = total
            abs_hists[key] = counts
            var = hist_var.get(key)
            if total > 0:
                shape_hists[key] = counts / total
                shape_errs[key] = (np.sqrt(var) / total
                                   if var is not None else np.zeros_like(counts))
            else:
                shape_hists[key] = counts
                shape_errs[key] = np.zeros_like(counts)

        if "Wcb" not in shape_hists or (not signal_only
                                        and "Cat_Top_bc" not in shape_hists):
            print(f"[WARN] Missing Wcb or Cat_Top_bc for matching plot {name}, skip.")
            return

        # Auto signal scale: normalise the W->cb =J line to the t->bc proxy
        # integral in the plotted range.  Fall back to 1.0 if either yield is
        # empty (e.g. a truth_only var with no proxy entries).
        _sig_yield = sum(raw_tot.get(k, 0.0) for k in SIGNAL_KEYS)
        _proxy_yield = raw_tot.get(PROXY_KEY, 0.0)
        SIGNAL_SCALE = (_proxy_yield / _sig_yield
                        if (_sig_yield > 0 and _proxy_yield > 0) else 1.0)

        # Lines actually drawn + what sets the y-axis.
        if normalize:
            draw_hists = dict(shape_hists)
        else:
            draw_hists = {k: (v * SIGNAL_SCALE if k in SCALED_KEYS else v)
                          for k, v in abs_hists.items()}

        # Legend: per-line suffix = % share of the total (1 dp, both modes,
        # 2026-09-04 user); the absolute expected event yields are quoted
        # once each on the "Signal" / "BKG" group headers.  raw_tot is the
        # true (unscaled) yield; the drawn W->cb lines are *SIGNAL_SCALE
        # (the "xN" part of the label).
        _grand = sum(v for k, v in raw_tot.items()
                     if v > 0 and k not in _SUBSET_KEYS)
        # totals for the group headers
        _sig_tot = sum(raw_tot.get(k, 0.0) for k in _SIG5)
        _bkg_tot = sum(v for k, v in raw_tot.items()
                       if v > 0 and k not in _SIG5 and k not in _SUBSET_KEYS)

        def _pct_suffix(key):
            v = raw_tot.get(key, 0.0)
            if _grand <= 0 or v <= 0:
                return ""
            frac = 100.0 * v / _grand
            # left (signal) legend: 2 dp; middle/right: 1 dp (2026-09-04 user)
            _nd = 2 if key in SCALED_KEYS else 1
            _lim = 10.0 ** (-_nd)
            return (f"  <{_lim:g}%" if frac < 0.5 * _lim
                    else f"  {frac:.{_nd}f}%")

        def _fmt_scale(x):
            # always an integer, no decimal digits (round to nearest)
            return f"{x:,.0f}"

        # the "signal drawn xN for visibility" factor -- quoted ONCE on the
        # Signal header (2026-09-04, user: was appended to every signal line,
        # which overran the legend column).
        _sig_scale_txt = (rf"  $\times{_fmt_scale(SIGNAL_SCALE)}$"
                          if (not normalize)
                          and abs(SIGNAL_SCALE - 1.0) > 1e-6 else "")

        def _leg_label(key):
            base = mat_labels.get(key, self.label(key))
            return base + _pct_suffix(key)

        def _stepify(y):
            return np.repeat(y, 2)

        step_x = np.repeat(bins, 2)[1:-1]

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            # lower pad = 1/2 the height of the upper pad
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
            sharex=True,
        )
        ax = axs[0]
        ax_ratio = axs[1]

        # MAT-overlay styling.  =J lines: flavour colour, solid.  The pooled
        # !=J W line is grey, solid (continuous), thinner than the =J lines.
        #   W->cb (signal)  red  |  W->(cq)/W->(qq)/Z->(cc/bb)/Z->(qq)  green ramp
        #   t : proxy = blue (solid); bq/bqq = blue shades (dashed)
        #   Rest : grey (dashed)
        _ls2 = (0, (5, 2))        # grey "Rest" line: ROOT linestyle 2 (dashed)
        # The 4 background-top lines -> ONE blue ramp, starting at pure blue
        # (ROOT colour 4) for the proxy and getting lighter: proxy -> (b',q)
        # -> t->(bcq) -> t->(bqq) lightest.  The 4 hadronic V bkg lines
        # (W->(cq), W->(qq), Z->(cc/bb), Z->(qq)) -> ONE green ramp,
        # darkest -> lightest in that order (2026-09-03, user).
        # 2026-09-04 (user): the 5 W->cb signal topologies now get DISTINCT
        # colours, all SOLID (was one red + 5 ROOT dash styles).  Exact ROOT
        # colour indices 2 / 95 / 92 / 5 / 41 (red -> orange -> amber ->
        # yellow -> pale khaki), top->bottom of the W legend:
        # Wcb / t^2(b'c) / t^2(b'b) / t^3(b'bc) / c,b,b' resolved.
        _c_sig = {
            "Wcb":         "#ff0000",   # ROOT 2   red        -- W(cb) on J
            "Wcb_tmrg_bc": "#ff7700",   # ROOT 95  orange     -- t^2(b',c)
            "Wcb_tmrg_bb": "#ffbe00",   # ROOT 92  amber      -- t^2(b',b)
            "Cat_Top_bbc": "#ffff00",   # ROOT 5   yellow     -- t^3(b',b,c)
            "Wcb_res":     "#d4cf87",   # ROOT 41  pale khaki -- c,b,b' resolved
        }
        # hadronic-top proxy -- ROOT colour indices 4 / 63 / 65 / 67
        # (blue -> cyan ramp), upper->lower of the middle legend (2026-09-04,
        # user): proxy t^2(b'c) / t^2(b'q) / t^3(b'cq) / t^3(b'qq).  All SOLID.
        _c_t1, _c_t2 = "#0000ff", "#008bff"           # ROOT 4  / ROOT 63
        _c_t3, _c_t4 = "#00bbff", "#00ebff"           # ROOT 65 / ROOT 67
        # hadronic-V background -- ROOT colour indices 209 / 210 / 211 / kSpring
        # (dark -> light green), upper->lower of the right legend (2026-09-04,
        # user): W(cq) / W(qq) / Z(cc/bb) / Z(qq).  All SOLID.
        _c_wq_a, _c_wq_b = "#0f8a0f", "#14b814"       # ROOT 209 / ROOT 210
        _c_z1,   _c_z2   = "#47eb47", "#33ff00"       # ROOT 211 / kSpring (820)
        style_overrides = {
            # the W->cb SIGNAL -- 5 lines, one per topology, distinct colours,
            # all SOLID, uniform linewidth 2 (2026-09-04, user).  Top->bottom
            # of the W legend:
            #   Wcb          W(cb)        red
            #   Wcb_tmrg_bc  t^2(b',c)   orange
            #   Wcb_tmrg_bb  t^2(b',b)   amber
            #   Cat_Top_bbc  t^3(b',b,c) yellow
            #   Wcb_res      c,b,b' res.  pale khaki
            "Wcb":          {"color": _c_sig["Wcb"],         "linestyle": "-", "linewidth": 2.0, "zorder": 9},
            "Wcb_tmrg_bc":  {"color": _c_sig["Wcb_tmrg_bc"], "linestyle": "-", "linewidth": 2.0, "zorder": 9},
            "Wcb_tmrg_bb":  {"color": _c_sig["Wcb_tmrg_bb"], "linestyle": "-", "linewidth": 2.0, "zorder": 9},
            # W->cb with c and b RESOLVED (separate AK4 jets) -- SIGNAL
            # (5th W->cb signal line; 2026-09-03 user: restored from Rest).
            "Wcb_res":      {"color": _c_sig["Wcb_res"],     "linestyle": "-", "linewidth": 2.0, "zorder": 9},
            # "no m_t candidate" subsets -- dotted, same colour as the parent
            # line (red = signal Wcb, blue = t->bc proxy).
            # "no m_t candidate" (mt == -1) subsets -- dotted.  Wcb_nomt is the
            # signal subset, drawn GREY (2026-09-04, user) so it reads as a
            # "failed-reco" sideband rather than a signal line; Cat_Top_bc_nomt
            # (proxy) stays blue-dotted.
            "Wcb_nomt":     {"color": "#6E6E6E", "linestyle": (0, (1, 1)), "linewidth": 1.5, "zorder": 10},
            "Cat_Top_bc_nomt": {"color": "blue", "linestyle": (0, (1, 1)), "linewidth": 1.5, "zorder": 10},
            # W background -- 2 darkest of the 4-line green V ramp, SOLID lw 2.
            "Wcq":          {"color": _c_wq_a, "linestyle": "-", "linewidth": 2.0, "zorder": 6},
            "Wqq_light":    {"color": _c_wq_b, "linestyle": "-", "linewidth": 2.0, "zorder": 6},
            # background top -- 4-line blue->cyan ramp, all SOLID lw 2
            # (2026-09-04, user): proxy / (b'q) / (b'cq) / (b'qq).
            "Cat_Top_bc":   {"color": _c_t1, "linestyle": "-", "linewidth": 2.0, "zorder": 8},
            "Cat_Top_bq":   {"color": _c_t2, "linestyle": "-", "linewidth": 2.0, "zorder": 4},
            "Cat_Top_bqqc": {"color": _c_t3, "linestyle": "-", "linewidth": 2.0, "zorder": 4},
            "Cat_Top_bqq":  {"color": _c_t4, "linestyle": "-", "linewidth": 2.0, "zorder": 4},
            # t^3(b',b,c) fully-merged W->cb top -- SIGNAL, yellow solid
            # (2026-09-04 user: distinct colour, all 5 signal lines solid lw 2).
            "Cat_Top_bbc": {"color": _c_sig["Cat_Top_bbc"], "linestyle": "-", "linewidth": 2.0, "zorder": 9},
            "Rest":        {"color": "#9C9CA1", "linestyle": _ls2, "linewidth": 1.5, "zorder": 2},
            # merged hadronic Z -- the 2 LIGHTEST of the green V ramp, SOLID
            # lw 2.  Resolved Z stays in grey Rest.
            "Zhf":         {"color": _c_z1, "linestyle": "-", "linewidth": 2.0, "zorder": 3},
            "Zlight":      {"color": _c_z2, "linestyle": "-", "linewidth": 2.0, "zorder": 3},
        }

        for key in key_order:
            if key not in draw_hists:
                continue

            style = {"color": self.color(key, "true"), "linestyle": "-", "linewidth": 2}
            style.update(style_overrides.get(key, {}))

            hep.histplot(
                draw_hists[key],
                bins=bins,
                histtype="step",
                label=_leg_label(key),
                ax=ax,
                **style,
            )

        # -------- lower pad ----------------------------------------------
        if signal_only:
            # MAT-SIGNAL: no background -> no significance.  Show the per-bin
            # fraction of the signal that is the clean 2-prong (cb) image of J
            # (i.e. Wcb / sum of all 5 red lines).
            _z = np.zeros(len(bins) - 1)
            _tot = sum((abs_hists[k] for k in SIGNAL_KEYS if k in abs_hists),
                       start=_z.copy())
            _cb = abs_hists.get("Wcb", _z)
            valid = _tot > 0
            frac = np.full(len(_tot), np.nan)
            frac[valid] = _cb[valid] / _tot[valid]
            ax_ratio.plot(np.repeat(bins, 2)[1:-1],
                          _stepify(np.where(valid, frac, np.nan)),
                          color="black", linewidth=1.3)
            ax_ratio.axhline(1.0, color="0.6", linestyle=":", linewidth=1)
            ax_ratio.set_ylim(0.0, 1.05)
            ax_ratio.set_ylabel(r"$(cb)$ / all signal", fontsize=10, labelpad=2)
        elif normalize:
            # NORM: shape ratio  (W->cb =J) / (t->bc), both unit-area
            wcb, top = shape_hists["Wcb"], shape_hists["Cat_Top_bc"]
            e_wcb = shape_errs.get("Wcb", np.zeros_like(wcb))
            e_top = shape_errs.get("Cat_Top_bc", np.zeros_like(top))

            valid = top > 0
            ratio = np.full_like(wcb, np.nan, dtype=np.float64)
            ratio[valid] = wcb[valid] / top[valid]

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
            ax_ratio.plot(bin_centers[valid], ratio[valid],
                          drawstyle="steps-mid", color="black", linewidth=1.3)
            ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1)
            ax_ratio.set_ylim(cfg.get("ratio_ylim", (0.0, 4.0)))
            ax_ratio.set_ylabel(r"$W\to cb$ (merged) / $t\to(cb)$", fontsize=10, labelpad=2)
        else:
            # ABS: per-bin significance  S_i / sqrt(S_i + B_i).
            #   S = every W->cb topology (SIGNAL_KEYS: Wcb merged 2-prong +
            #       Wcb_tmrg_bc/bb + Cat_Top_bbc + Wcb_res -- the 5 W->cb
            #       lines), UNSCALED
            #       (the auto-scale on the overlay lines does NOT enter here).
            #   B = every other category.
            _z = np.zeros(len(bins) - 1)
            _sig_keys = list(SIGNAL_KEYS)          # the 5 red W->cb lines
            # the *_nomt subsets are parts of Wcb / Cat_Top_bc already ->
            # excluded from B to avoid double count.
            _bkg_keys = [k for k in key_order
                         if k not in _sig_keys and k not in _SUBSET_KEYS]
            S = sum((abs_hists[k] for k in _sig_keys if k in abs_hists), start=_z.copy())
            B = sum((abs_hists[k] for k in _bkg_keys if k in abs_hists), start=_z.copy())
            varS = sum((np.asarray(hist_var.get(k, _z)) for k in _sig_keys), start=_z.copy())
            varB = sum((np.asarray(hist_var.get(k, _z)) for k in _bkg_keys), start=_z.copy())

            _N = S + B
            valid = _N > 0
            metric = np.full(len(B), np.nan)
            metric[valid] = S[valid] / np.sqrt(_N[valid])
            # Z = S / sqrt(S+B):  dZ/dS = (S/2 + B) / (S+B)^1.5,
            #                     dZ/dB = -(S/2) / (S+B)^1.5
            # var(Z) = [ (S/2 + B)^2 varS + (S/2)^2 varB ] / (S+B)^3
            merr = np.full(len(B), np.nan)
            merr[valid] = np.sqrt(
                ((0.5 * S[valid] + B[valid]) ** 2 * varS[valid]
                 + (0.5 * S[valid]) ** 2 * varB[valid]) / _N[valid] ** 3
            )
            m_band = valid & np.isfinite(merr) & (merr > 0)
            if np.any(m_band):
                ax_ratio.fill_between(
                    step_x,
                    _stepify(np.where(m_band, metric - merr, np.nan)),
                    _stepify(np.where(m_band, metric + merr, np.nan)),
                    color="0.6", alpha=0.35, linewidth=0,
                )
            # Draw as a full-width step over *every* bin (0 where B==0), so the
            # curve stays continuous -- plotting only bin_centers[valid] makes
            # matplotlib bridge dropped bins with slanted segments / gaps.
            metric_draw = np.where(valid, metric, 0.0)
            hep.histplot(metric_draw, bins=bins, histtype="step",
                         color="black", linewidth=1.3, ax=ax_ratio)

            # Fixed y-axis for the S/sqrt(S+B) pad -- constant across every MAT
            # plot so the significance is comparable by eye between observables.
            ax_ratio.set_ylim(0.0, 0.5)
            ax_ratio.set_ylabel(r"$S/\sqrt{S+B}$", fontsize=12, labelpad=2)

            if cfg.get("truth_only"):
                # gen-truth variable -- either the branch is unfilled (-1) for
                # non-W/top jets so background piles into the sentinel bin
                # (N_b, N_c, ...), OR the x-axis is a gen-level composition
                # label (ja_truth_cat) that cannot be cut on in real data.
                # Either way the per-bin S/sqrt(S+B) curve is shown for
                # information but the 1-bin window optimisation is meaningless
                # -> skipped, and the quad-sum annotation is dropped.
                ax_ratio.text(0.06, 0.86,
                              "gen-truth axis: per-bin $S/\\sqrt{S+B}$ shown\n"
                              r"but not a cuttable observable $\to$ no optimisation",
                              transform=ax_ratio.transAxes, fontsize=9.0,
                              va="top", ha="left", color="0.35")
                print(f"[MAT-opt] {name:28s} gen-truth variable -- "
                      f"significance optimisation skipped")
            else:
              # quadrature combination of the per-bin values drawn above
              _z_quad = float(np.sqrt(np.nansum(metric[valid] ** 2)))
              ax_ratio.text(0.055, 0.90,
                            rf"$\sqrt{{\sum_i S_i^2/(S_i+B_i)}} = {_z_quad:.3f}$"
                            "\n" r"($S$: all $W\!\to\!cb$ (merged, $b'$-mrg, $t\!\to\!(bbc)$);   $B$: rest)",
                            transform=ax_ratio.transAxes, fontsize=9.5,
                            va="top", ha="left", linespacing=1.6)

              # 1-bin optimisation (Root_plot.py optimize_n_bins(1)): the
              # single contiguous window [x_lo, x_hi] that maximises the
              # *integrated* S/sqrt(S+B).  Brute force over every window, require
              # integral(B) >= 1.  Reported: window edges, S/sqrt(S+B),
              # improvement vs the no-cut full-range S/sqrt(S+B), signal
              # efficiency eps_S = S_win/S_tot and background rejection
              # 1 - eps_B = 1 - B_win/B_tot.  Two magenta lines + text + stdout.
              _S_tot, _B_tot = float(S.sum()), float(B.sum())
              _sig_full = (_S_tot / np.sqrt(_S_tot + _B_tot)
                           if (_S_tot + _B_tot) > 0 else 0.0)
              _cS = np.concatenate([[0.0], np.cumsum(S)])
              _cB = np.concatenate([[0.0], np.cumsum(B)])
              _i, _j = np.triu_indices(len(S) + 1, k=1)     # 0 <= i < j <= nbins
              _bw = _cB[_j] - _cB[_i]
              _sw = _cS[_j] - _cS[_i]
              _ok = (_bw >= 1.0) & (_sw > 0.0)
              if np.any(_ok):
                  _sig_w = np.where(
                      _ok, _sw / np.sqrt(np.where(_ok, _sw + _bw, 1.0)),
                      -np.inf)
                  _k = int(np.argmax(_sig_w))
                  _x_lo, _x_hi = float(bins[_i[_k]]), float(bins[_j[_k]])
                  _sig_opt = float(_sig_w[_k])
                  _S_w, _B_w = float(_sw[_k]), float(_bw[_k])
                  _eff_s = _S_w / _S_tot if _S_tot > 0 else float("nan")
                  _rej_b = 1.0 - (_B_w / _B_tot) if _B_tot > 0 else float("nan")
                  _gain = 100.0 * (_sig_opt / _sig_full - 1.0) if _sig_full > 0 else float("nan")
                  for _x in (_x_lo, _x_hi):
                      ax_ratio.axvline(_x, ymin=0.0, ymax=0.42, color="magenta",
                                       linewidth=1.6, zorder=5)
                  ax_ratio.text(0.94, 0.88,
                                rf"$S/\sqrt{{S+B}}={_sig_opt:.3f}$ (+{_gain:.0f}%)",
                                transform=ax_ratio.transAxes, fontsize=9.0,
                                va="top", ha="right", color="magenta")
                  ax_ratio.text(0.94, 0.72,
                                rf"$\varepsilon_S={_eff_s*100:.0f}\%$" "\n"
                                rf"$1-\varepsilon_B={_rej_b*100:.1f}\%$",
                                transform=ax_ratio.transAxes, fontsize=9.0,
                                va="top", ha="right", color="magenta",
                                linespacing=1.5)
                  print(f"[MAT-opt] {name:28s} window [{_x_lo:.4g}, {_x_hi:.4g}]  "
                        f"S/sqrt(S+B)={_sig_opt:.4f}  (full-range {_sig_full:.4f}, "
                        f"{_gain:+.0f}%)  effS={_eff_s*100:.1f}%  "
                        f"rej(1-effB)={_rej_b*100:.2f}%")

        ax_ratio.set_xlabel(cfg.get("xlabel", cfg.get("var", "")), labelpad=1)
        # force round 1/2/5-style ticks (0.01, 0.02, 0.05, 0.10 ...) so the
        # labels stay 2-decimal and short -- no 0.015 / 0.045.
        ax_ratio.yaxis.set_major_locator(
            plt.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], prune="both"))
        ax_ratio.tick_params(labelsize=10)

        ax.set_ylabel("Normalized to unity" if normalize else
                      "Events / bin  (lumi-weighted)")
        ax.tick_params(labelsize=11)

        # y-axis. NORM mode: the fixed ranges tuned for unit-area shapes (log
        # 5e-4..4, linear auto). Absolute mode: auto-scale from the drawn
        # histograms (incl. the scaled signal), with generous headroom for the
        # in-axes legends.  The SD-mass spectra are forced to log in absolute
        # mode (they span many decades once the QCD / "resolved" bulk sits at
        # its true scale); NORM keeps them linear per spec.
        _logy = cfg.get("logy", False) or (
            (not normalize) and "sdmass" in str(cfg.get("var", "")))
        _all_h = list(draw_hists.values())
        if _logy:
            ax.set_yscale("log")
            if normalize:
                ax.set_ylim(cfg.get("ylim_bottom", 5e-4), cfg.get("ylim_top", 4.0))
            else:
                _pos = np.concatenate([h[h > 0] for h in _all_h
                                       if h is not None and len(h)]
                                      or [np.array([1.0])])
                # bottom: fixed floor -- 0.1 for MAT-SIGNAL (signal-only, small
                # t^2 / resolved / sub-percent tails must stay on-axis), 1.0 for
                # MAT (signal + background) so the sub-event bkg noise is hidden
                # (2026-09-04, user).
                # top: 100x the tallest drawn bin (incl. the scaled signal) --
                # 2 clear decades of headroom for the in-axes legends.
                ax.set_ylim(0.1 if signal_only else 1.0, 100.0 * _pos.max())
        else:
            _ymax = max((float(np.max(h)) for h in _all_h
                         if h is not None and len(h)), default=1.0)
            # extra headroom for the enlarged in-axes legends -- more for the
            # categorical composition plot (ja_truth) whose tallest bin is a
            # narrow spike that would otherwise poke into the 3 legends.
            _hr = 2.6 if cfg.get("xticklabels") else 1.7
            ax.set_ylim(0.0, _ymax * _hr)
            # powers-of-10 tick labels (1e5, ...) rather than "200000"
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.yaxis.get_offset_text().set_size(9)

        # Three separate legends across the top (2026-09-04, user):
        #   left   = the 5 W->cb SIGNAL lines + "J w/o-j*"
        #   middle = the hadronic-top proxy lines (blue)
        #   right  = the hadronic-V background (green) + Rest
        # Kept as distinct artists so each packs tightly to its width.
        _sig_keys = ["Wcb", "Wcb_tmrg_bc", "Wcb_tmrg_bb", "Cat_Top_bbc",
                     "Wcb_res", "Wcb_nomt"]
        _t_keys   = ["Cat_Top_bc", "Cat_Top_bc_nomt",
                     "Cat_Top_bq", "Cat_Top_bqqc", "Cat_Top_bqq"]
        _v_keys   = ["Wcq", "Wqq_light", "Zhf", "Zlight", "Rest"]
        _hmap = {l: h for h, l in zip(*ax.get_legend_handles_labels())}

        # Legend style -- IDENTICAL for MAT and MAT-SIGNAL (2026-09-04, user):
        # non-bold, ~10% larger than before, no frame.  The "Signal" / "BKG"
        # group headers are drawn ONCE each as ax.text above the legends
        # (not per-legend titles) so a single "BKG" spans the middle+right
        # pair.
        _leg_fs = 9.1                       # was 8.3  (+10%)
        _hdr_fs = 11.0                      # +5% (2026-09-04, user)
        _leg_kw = dict(frameon=False, labelspacing=0.18,
                       handlelength=1.5, borderaxespad=0.4,
                       prop={"size": _leg_fs, "weight": "normal"})
        _leg_y = 0.95                       # legend-box top (axes frac) -- +3%
        _hdr_y = 0.915                      # header baseline (2026-09-04 user)
        _hdr_x0 = 0.06                      # "Signal" x
        _hdr_x1 = 0.77                      # "BKG" x

        def _make_legend(keys, loc, anchor):
            pairs = [(_hmap[_leg_label(k)], _leg_label(k))
                     for k in keys if _leg_label(k) in _hmap]
            if not pairs:
                return None
            h, l = zip(*pairs)
            return ax.legend(h, l, loc=loc, bbox_to_anchor=anchor,
                             bbox_transform=ax.transAxes, **_leg_kw)

        _c_sig_hdr, _c_bkg_hdr = "#E42536", "#0000FF"
        _legs = [_make_legend(k, loc, anch) for k, loc, anch in (
            (_sig_keys, "upper left",   (0.01, _leg_y)),
            (_t_keys,   "upper center", (0.50, _leg_y)),
            (_v_keys,   "upper right",  (1.00, _leg_y)))]
        _created = [lg for lg in _legs if lg is not None]
        for _lg in _created[:-1]:            # last stays as ax._legend
            ax.add_artist(_lg)

        # group headers: "Signal" once (over the left legend), "BKG" once
        # (centred over the middle+right pair; only if any bkg line is drawn).
        # Each carries the total expected (unscaled) event yield of its group.
        ax.text(_hdr_x0, _hdr_y, f"Signal  {_sig_tot:.1f}{_sig_scale_txt}",
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=_hdr_fs, fontweight="normal",
                color=_c_sig_hdr, clip_on=False)
        if _legs[1] is not None or _legs[2] is not None:
            ax.text(_hdr_x1, _hdr_y, f"BKG  {_bkg_tot:.1f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=_hdr_fs,
                    fontweight="normal", color=_c_bkg_hdr, clip_on=False)

        # Selection tag: shown after "Simulation" in the CMS label and appended
        # to the PNG filename so PRE / SR plot sets don't clobber each other.
        _sel = str(cfg.get("_sel", "PRE")).upper()
        _sel_label = {"PRE": "Preselection", "SR": "SR",
                      "JB": "jb-Region"}.get(_sel, _sel)
        if signal_only:
            _sel_label += " (signal only)"
        # "CMS Simulation" + lumi via mplhep (no supplementary text) ...
        hep.cms.label(
            "",
            data=False,
            lumi=round(self.cfg.lumi, 1),   # 2017: 41.5 fb^-1, shown top-right
            ax=ax,
            fontsize=CMS_LABEL_FONTSIZE,
        )
        # ... then the selection tag as its OWN artist: ~5% higher than the
        # "CMS Simulation" baseline and UPRIGHT (non-italic) (2026-09-04, user).
        ax.text(0.36, 1.065, _sel_label, transform=ax.transAxes,
                ha="left", va="baseline", clip_on=False,
                fontsize=CMS_LABEL_FONTSIZE * 0.85,
                fontstyle="normal", fontweight="normal")

        ax.set_xlim(cfg.get("xlim", (bins[0], bins[-1])))
        if cfg.get("logx", False):
            ax.set_xscale("log")
            ax_ratio.set_xscale("log")

        # optional categorical x-tick labels (e.g. the sdwin_tag_nj2 4-bin
        # in/out plot).  sharex -> set positions once; label the bottom pad.
        _xticks = cfg.get("xticks")
        if _xticks is not None:
            ax.set_xticks(_xticks)
            _xtl = cfg.get("xticklabels")
            if _xtl is not None:
                # many narrow categories -> labels FULLY VERTICAL, centred
                # directly under their bin (2026-09-04, user).
                if len(_xtl) > 8:
                    ax_ratio.set_xticklabels(
                        _xtl, fontsize=10.5, rotation=90,
                        ha="center", va="top")
                else:
                    ax_ratio.set_xticklabels(
                        _xtl, fontsize=9.5, ha="center", va="top")

        # Optional explanatory caption (spec "caption" key): a small grey text
        # block under the lower pad.  Used by the composition / categorical
        # specs to spell out, on the figure itself, what each bin means and
        # the merged/resolved (parenthesis) notation.
        _cap = cfg.get("caption")
        _cap_nl = _cap.count("\n") + 1 if _cap else 0
        # categorical x-labels need a deeper bottom margin: a lot for the
        # >8-bin vertical case, a little for the horizontal 3-8-bin case.
        _rot_pad = 0.0
        if _xticks is not None and cfg.get("xticklabels"):
            _rot_pad = 0.17 if len(cfg["xticklabels"]) > 8 else 0.055
        fig.subplots_adjust(left=0.125, right=0.965, top=0.91,
                            bottom=(0.085 + _rot_pad + 0.033 * _cap_nl))
        if _cap:
            fig.text(0.015, 0.012, _cap, fontsize=6.6, va="bottom", ha="left",
                     color="0.30", linespacing=1.45)

        # Filename suffixes: per-selection ("_PRE" preselection / "_SR" signal
        # region) so the two plot sets never clobber each other; "_norm" for
        # NORM-mode so it doesn't clobber the ABS-mode PNG of the same observable.
        _sel_suffix = {"PRE": "_PRE", "SR": "_SR", "JB": "_JB"}.get(_sel, "")
        _suffix = _sel_suffix + ("_sig" if signal_only else "") + ("_norm" if normalize else "")
        outbase = os.path.join(self.cfg.figure_path, name + _suffix)
        self._save_and_show(fig, outbase, tag=cfg.get("_progress", "MAT"))

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
    # cb-candidate jet plots  (J = highest GloParT bc score)
    # ------------------------------------------------------------------
    leading_vars = [
        {
            "var": "ak8_pt_0",
            "bins": np.linspace(200, 800, 31),
            "xlabel": "cb-cand AK8 jet $p_{T}$ [GeV]  ($J$)",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_eta_0",
            "bins": np.linspace(-2.5, 2.5, 31),
            "xlabel": "cb-cand AK8 jet $\\eta$  ($J$)",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_phi_0",
            "bins": np.linspace(-3.2, 3.2, 33),
            "xlabel": "cb-cand AK8 jet $\\phi$  ($J$)",
            "cut": "ak8_pt[0] > 200",
        },
        {
            "var": "ak8_sdmass_0",
            "bins": np.linspace(0, 250, 26),
            "xlabel": "cb-cand AK8 soft-drop mass [GeV]  ($J$)",
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
            "xlabel": r"EC no-match score (not $W$/top/QCD)",
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
            "xlabel": "$J$ soft-drop mass [GeV], $D_{bc} > 0.985$",
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


def build_mat_plot_settings(sel="PRE", signal_only=False):
    """
    Truth-matched cb-candidate-AK8 overlay (--MAT mode). Step histograms,
    driven by `mat_cat` (built in build_derived_array).  The candidate jet is
    J = highest GloParT bc score (was: j_1, pT-leading).

    NOTATION: a parenthesis means the quarks are MERGED inside J
    (W->(cb) = both W quarks in one AK8 jet); a quark with no parenthesis is
    a separate resolved AK4 jet (t->(b)qq = b merged, the 2 light quarks are
    AK4).  b' = b from the top decay, b_l = b overlapping the lepton, a bare
    b/c/q = any other.  t^2 = 2 quarks merged in J, t^3 = 3 quarks merged.

    SIGNAL: 5 RED lines (all W->cb, all in S, all auto-scaled to the
    t->(bc) proxy), one per topology, split by line style:
      Wcb          W(cb)         solid   -- J IS the 2-prong W->(cb) jet
                   (mat_cat 0 + 5; <0.5% "J picked another AK8" folded in).
      Wcb_tmrg_bc  t^2(b',c)    dashed  -- J = (top-b + W's c) merge (21).
      Wcb_tmrg_bb  t^2(b',b)    dash-dot-- J = (top-b + W's b) merge (22).
      Cat_Top_bbc  t^3(b',b,c)  dash-dot-dot -- J = fully-merged top
                   t->(bbc), W->cb (code 20).
      Wcb_res      W->cb (res.)  dotted  -- W->cb with c and b RESOLVED
                   (separate AK4 jets), J not the cb image (code 6;
                   2026-09-03 user: restored from Rest, counts in S).
      *_nomt    : dotted red / blue -- the subset of Wcb / t->(bc) proxy
                  whose m_t reco fails (mt == -1, no AK4 with x<2).  Subsets,
                  not categories -> excluded from the % total and the bkg sum.

    BACKGROUND:
      Wcq/Wqq'  : green (the 2 darkest of the 4-line V green ramp) -- the
                  merged 2-prong W jet IS the cb candidate.  Their "!=J"
                  events (codes 9/11) pool into grey "W_sublead"; "resolved"
                  (codes 10/12) fold into Rest.
      t         : proxy = blue solid t->(bc), blue shades for t->(bq) /
                  t->(bqq').  t^2 = 2-prong top, t^3 = 3-quark merged top.
      Rest      : grey -- everything else: gluon/light/c/b jets, hadronic Z,
                  resolved hadronic W (codes 9/10/11/12).  (W->cb fully-
                  resolved, code 6, is now its own red "Wcb_res" signal line.)
    W and top match flags carry no process guard, so W/top jets from tt,
    single-top, tW, ttZ, twZ and diboson all enter the same lines.
    Lower panel: NORM -> (W->cb merged) / (t->bc) shape ratio;
    ABS -> per-bin significance S/sqrt(S+B), S = every W->cb topology
    (the 5 red W->cb lines, unscaled), B = every other category;
    + the 1-bin-optimal S/sqrt(S+B)
    window (2 magenta lines).  `truth_only` specs (N_b, N_c, # partons) skip
    the optimisation -- the gen jet-content branch is unfilled for background.

    All entries below describe the *cb-candidate AK8 jet* (J) or the event it
    sits in.  The list is a broad kinematic scan (~65 observables) meant to
    expose where a genuine W->cb jet differs from a W->cq or a Top->bc jet:
    GloParT tagger outputs, jet substructure, angular/dR variables and event
    activity. Baseline phase space: leading AK8 pt > 200 GeV (which every AK8 in
    these ntuples already satisfies).  No tagger cut -- bare preselection.

    Each spec row: (short_name, var, lo, hi, nbins, xlabel, logy).
    SENTINEL (-999) fills are removed before histogramming; genuine
    out-of-range entries are folded into the first/last bin by
    hist_with_flow (ROOT UnderOverFlow1D style).

    `sel` picks the event selection (3rd CLI token, `Make_plots.py MAT SR`):
      PRE (default)  : boosted AK8 only, no tagger cut.
      SR             : + score_Dbc > 0.9  (the D_bc BDT; NOT the raw
                         ak8_gpt_bc_0 node -- its softmax score rarely exceeds
                         ~0.93 for signal).  No S_EVT cut.  (0.95 -> 0.9
                         2026-08-31.)
      JB (jb-Region) : score_Dbc(J) > 0.9  AND  S_EVT > 0.05
                         (S_EVT = score_SC = EC W->qq' norm score).
                         (= SR loosened + the S_EVT cut.  The former
                         mSD(j_b) in [65,105] and tau21(j_b) < 0.5 clauses
                         were both removed 2026-08-31, so JB no longer
                         forces n_ak8 >= 2.)

      ALL THREE also carry the common preselection block:
        0 < tau21(J) < 0.65       (2-prong-ness of the cb candidate;
                                     relaxed 0.6 -> 0.65 2026-09-03)
        dR(lepton, J) > 1.2       (lepton clear of the fat jet;
                                     1.5 -> 1.0 2026-09-03, -> 1.5 same day,
                                     -> 1.2 2026-09-04 user)

      PRE and SR (NOT JB) additionally carry (2026-09-04, user):
        m(J)   > 40 GeV           (the cb-candidate AK8 -- ak8_sdmass_0)
        m(j^2) < 100 GeV          (2nd-heaviest AK8 -- ak8_sdmass_sub_mass_0;
                                     = -999 when n_ak8 < 2, so the veto only
                                     fires on a genuine 2nd heavy AK8)
    """
    # Common preselection applied to ALL three selections.
    # tau21(J) is guarded with > 0 so SENTINEL (-999) jets are not let
    # through by the "< 0.6" test.
    _COMMON = ("(ak8_tau21_0 > 0) and (ak8_tau21_0 < 0.65) and "
               "(dR_lep_ak8 > 1.2)")
    # PRE and SR only (2026-09-04, user):
    #   m(J) > 40  (cb-candidate AK8)   AND   m(j^2) < 100  (2nd-heaviest AK8).
    _MJ12 = "(ak8_sdmass_0 > 40) and (ak8_sdmass_sub_mass_0 < 100)"
    _SELECTIONS = {
        "PRE": "(ak8_pt[0] > 200) and " + _MJ12 + " and " + _COMMON,
        "SR":     ("(ak8_pt[0] > 200) and (score_Dbc > 0.9) and "
                   + _MJ12 + " and " + _COMMON),
        "JB":     ("(ak8_pt[0] > 200) and (score_Dbc > 0.9) and "
                   "(score_SC > 0.05) and " + _COMMON),
    }
    CUT = _SELECTIONS.get(str(sel).upper(), _SELECTIONS["PRE"])
    PI = float(np.pi)

    def m(name, var, lo, hi, nb, xlabel, logy, truth_only=False, logx=False,
          xticklabels=None, caption=None):
        # logx: log-spaced bin edges + log x-axis (lo must be > 0).
        # xticklabels: list of nb strings -> categorical x-axis (ticks placed
        # at the bin centres, bottom pad labelled with these).
        _bins = (np.logspace(np.log10(lo), np.log10(hi), nb + 1)
                 if logx else np.linspace(lo, hi, nb + 1))
        _ctr = 0.5 * (_bins[1:] + _bins[:-1])
        return {
            "name": f"MAT_{name}",
            "var": var,
            "var_mode": "event",
            "mode": "mat",
            "bins": _bins,
            "logx": logx,
            "xlabel": xlabel,
            "xlim": (lo, hi),
            "xticks": (list(_ctr) if xticklabels is not None else None),
            "xticklabels": xticklabels,
            "caption": caption,
            "cut": CUT,
            "logy": logy,
            # gen-level jet-content: the branch is only filled for W/top-matched
            # jets, so background lands in the sentinel bin -> S/sqrt(S+B) is
            # biased.  Suppress the significance optimisation for these.
            "truth_only": truth_only,
            "ylim_bottom": 5e-4,
            "ylim_top": 4.0,
            "ratio_ylim": (0.0, 4.0),
        }

    # =====================================================================
    # m_t RECONSTRUCTION  (2026-09-04, user)
    # =====================================================================
    # J   = the cb-candidate AK8 jet (mass = its soft-drop mass, m(J)).
    # j*  = one AK4 jet from OUTSIDE the J cone (dR > 0.8) -- meant to be the
    #       top's b.  Picked from the 5 AK4 with m(J+AK4) closest to 172.5,
    #       the FIRST that also passes an angular gate (below).
    #
    #   m_t = m(J + j*)   ONLY when   40 < m(J) < 130 GeV   AND a j* is found
    #   m_t = m(J)        otherwise
    #
    # The m(J) window (env WCB_TTRECO_MRG_MSD_LO / _MRG_MSD, default 40/130):
    #   m(J) < 40   -> J is a light / "Rest" jet, no W or t^2 candidate --
    #                  do NOT reconstruct a top (would pile fakes on the
    #                  peaks); m_t = m(J) parks them below 40.
    #   m(J) > 130  -> J already IS the merged top; m_t = m(J).
    #   40..130     -> J is a W / partly-merged jet: combine with j*.
    #
    # THE j* GATE -- two versions, kept in parallel (differ ONLY here):
    #
    #  (A) PRIMARY -- ttreco_mt -- fixed reference mass
    #        keep j*  if   dR(j*,J) * pT(J+j*) / 2  <  2 * 172.5
    #        (env WCB_TTRECO_XMT, the "2").  A TOP-MASS hypothesis test:
    #        "is this pair arranged like the daughters of a 172.5-GeV top of
    #        this pT?"  Kinematics: dR*pT ~ 2M for a 2-body decay, so the LHS
    #        ~ M for a real top and "< 2*172.5" = "within 2x".  Permissive.
    #        BEST FOR a tt sample: every event has a hadronic top, so
    #        assuming ~172 is fine and the loose gate recovers the t^2
    #        partial-merge topologies (m(J + b') ~ 172 even when J is only
    #        the 2-prong W).
    #
    #  (B) ALT -- ttreco_mt_xself -- the pair's OWN mass as reference
    #        keep j*  if   dR(J,j*) * pT(J+j*) / 2  <  2 * m(J+j*)
    #        (env WCB_TTRECO_XSELF).  No 172 assumption -- the LHS ~ M and the
    #        RHS ~ M for a genuine 2-body decay at ANY mass, so this tags a
    #        real pairing regardless of mass; a pair too wide to be a
    #        resonance (random / ISR / the other top's b) fails.
    #        BEST FOR shape-based categorisation / template fits where the
    #        m_t distribution itself carries the information.
    #
    # DIAGNOSTICS (all follow the PRIMARY gate's j*):
    #   dR_ja_mt   = dR(J, j*)                 reported for any j* CANDIDATE
    #   xself_jstar = x_self of that j*        (-1 = no candidate at all,
    #   dRpt_jstar  = dR(J,j*)*pT(J+j*) [GeV]   lines up with the "w/o-j*"
    #   pt_jstar    = pT(J+j*)  (reco top pT)   overlay)
    #   xself_bmk1  = x_self of bmk1 (raw best-mass AK4, for tuning gate B)
    #   bmk1_x      = the old x of bmk1 (fixed 172.5, for tuning gate A)
    #   {b,c,bc}vuds_bestm / jstar_flav = ParticleNetAK4 flavour of the j*
    #                                    that built the PRIMARY m_t
    specs = [
        # ---- top-mass reconstruction (SINGLE kept method) -------------------
        # m_t = m(J + j*), J taken with its measured soft-drop mass
        # (ak8_sdmass_0) -- NOT an m_W-constrained mass: J is not always a
        # 2-prong hadronic W (partial/full merged tops, light jets), so
        # forcing m_W would mismodel every non-W topology (2026-09-03, user).
        # m_t = m(J+j*)  only when  m(J) in [40,130]  AND a j* is found;
        # otherwise  m_t = m(J).  TWO versions, differing ONLY in the j* gate:
        #   ttreco_mt        -- PRIMARY: keep j* if  dR*pT(J+j*)/2 < 2*172.5
        #   ttreco_mt_xself  -- alt:     keep j* if  dR*pT(J+j*)/2 < 2*m(J+j*)
        ("ttreco_mt",           "mt",                 -20.0, 300.0, 32, r"$m_{t}$ [GeV]   ($\Delta R\,p_T(J{+}j^{*})/2<1.35\cdot173$, else $m(J)$)", True),
        ("ttreco_mt_xself",     "mt_xself",           -20.0, 300.0, 32, r"$m_{t}$ [GeV]   ($\Delta R\,p_T(J{+}j^{*})/2<1.10\,m(J{+}j^{*})$, else $m(J)$)", True),
        ("ttreco_dRpt_jstar",   "dRpt_jstar",         -10.0, 600.0, 61, r"$\Delta R(J,j^{*})\,p_T(J{+}j^{*})/2$ [GeV]   ($\approx 172$ for $t^{2}$ \& $W$;  $-1$ = no $j^{*}$)", True),
        ("ttreco_pt_jstar",     "pt_jstar",             0.0, 800.0, 40, r"$p_T(J{+}j^{*})$ [GeV]   (reconstructed top $p_T$)", True),
        # tune the two gates here -- keep bmk1 if the plotted ratio is below
        # 1.35 (fixed-173 ref) / 1.10 (self ref).
        ("ttreco_bmk1_dRpt_173",  "bmk1_x",           0.0, 4.0, 40, r"$\Delta R\,p_T(J{+}\mathrm{best}\text{-}m_t\ \mathrm{AK4})\,/\,(2\cdot173)$   (keep $<1.35$)", True),
        # ("ttreco_bmk1_dRpt_mJj",  "xself_bmk1",       -0.1, 5.0, 51, r"$\Delta R\,p_T(J{+}\mathrm{bmk1})\,/\,(2\,m(J{+}\mathrm{bmk1}))$   (keep $<1.10$)", True),  # removed on request 2026-09-04
        # ("ttreco_jstar_dRpt_mJj", "xself_jstar",      -0.1, 5.0, 51, r"$\Delta R\,p_T(J{+}j^{*})\,/\,(2\,m(J{+}j^{*}))$ of the used $j^{*}$   ($-1$ = none)", True),  # removed on request 2026-09-04
        # dR(J, j*), j* = the AK4 that reconstructs m_t (first of bmk1..5
        # passing x_self < 1.5).  -1 when no compatible j* (folds into the
        # underflow bin).  Angular separation of the cb candidate from the top's b.
        ("ttreco_dR_ja_mt",     "dR_ja_mt",            0.0, 5.0, 50, r"$\Delta R(J,\,j^{*})$   ($-1$ = no $j^{*}$)", False),
        # Standalone composition plot: what J / the event IS at gen level
        # (13 bins).  ( ) = quarks merged inside the AK8 jet J; a bare
        # quark = a separate AK4 jet.  b' = b from the top.
        #   SIGNAL (all W->cb) -- 5 categories: 0 (cb) 2-prong merge |
        #     1 (b' c) | 2 (b' b) partial merges | 3 (b' b c) full top
        #     t->(bbc) | 4 (cb) resolved.
        #   BACKGROUND: 5 proxy hadronic top (no c prong) |
        #     6 t->(b' c)q (hadronic top, c prong merged in J) |
        #     7 W->(cq) merged | 8 W->(qq) merged | 9 Z->(cc/bb) merged |
        #     10 Z->(qq) merged | 11 resolved W bkg (W->cq / W->qq' unmerged) |
        #     12 rest (incl. resolved Z -- no z_decay in the cache).
        # truth_only=True: gen-truth axis, not a cuttable observable ->
        # the S/sqrt(S+B) 1-bin window optimisation is suppressed.
        ("ttreco_ja_truth",     "ja_truth_cat",       -0.5, 13.5, 14, r"$J$ / event gen content", False, True, False,
         [r"$W(cb)$", r"$t^2(b'c)$", r"$t^2(b'b)$", r"$t^3(b'bc)$", r"$c,b,b'$ resolved",
          r"$t^2(b'c)$ proxy", r"$t^2(b'q)$", r"$t^3(b'cq)$", r"$t^3(b'qq)$",
          r"$W(cq)$", r"$W(qq)$",
          r"$Z(c\bar c/b\bar b)$", r"$Z(qq)$",
          r"rest"]),
        # 4-bin categorical: ParticleNetAK4 flavour ladder of a chosen AK4.
        #   jstar_flav   = j* (the AK4 that reconstructs m_t) -- the "modified"
        #                  version of bdr1_flav (2026-09-03, user).
        #   bbestm1_flav = bmk1 (raw best-|m-172.5| AK4, BEFORE the x window).
        #   bdr1_flav (dR-closest AK4 to J) is commented out -- superseded
        #   by jstar_flav.
        # ("ttreco_bdr1_flav",       "bdr1_flav",              -0.5, 3.5, 4, r"flavour of nearest AK4 to $J$", False, False, False,
        #  [r"light/untag.", r"$c_L$", r"$b_L$ (not $b_M$)", r"$b_M$"]),  # superseded by jstar_flav
        ("ttreco_jstar_flav",      "jstar_flav",             -1.5, 2.5, 4, r"ParticleNetAK4 tag of $j^{*}$ (AK4 used by $m_t$)", False, False, False,
         [r"w/o $j^{*}$", r"non-$b/c_L$-tagged", r"$c_L$", r"$b_L$"]),
        # ("ttreco_bbestm1_flav",    "bbestm1_flav",           -0.5, 3.5, 4, r"ParticleNetAK4 tag of best-$m$ AK4 (bmk1)", False, False, False,
        #  [r"light/untag.", r"$c_L$", r"$b_L$ (not $b_M$)", r"$b_M$"]),  # commented out on request 2026-09-04
        # ("ttreco_bvuds_near",      "bvuds_near",              0.0, 1.0, 50, r"$b/(b{+}uds)$, nearest AK4", False),  # commented out on request
        # x-range starts at -0.05 (55 bins) so the -1 "no j*" pile sits in a
        # visibly detached first bin, clear of the [0,1] score range.
        ("ttreco_bvuds_bestm",     "bvuds_bestm",           -0.05, 1.0, 55, r"$b/(b{+}uds)$ of $j^{*}$ ($-1$ = no $j^{*}$)", True),
        # Same jet j* (the one m_t uses), other PNetAK4 score ratios:
        #   cvuds_bestm  = c / (c + uds)          -- is it a c jet?
        #   bcvuds_bestm = (b + c) / (b + c + uds) -- heavy-flavour vs light
        #   cvsb_bestm   = c / (c + b)            -- c-vs-b discriminant
        # For a real top-b j* should be b-like (bvuds high, cvuds low,
        # cvsb low); the (b+c) combo tags it as "not a light / gluon jet".
        ("ttreco_cvuds_bestm",     "cvuds_bestm",           -0.05, 1.0, 55, r"$c/(c{+}uds)$ of $j^{*}$ ($-1$ = no $j^{*}$)", True),
        ("ttreco_bcvuds_bestm",    "bcvuds_bestm",          -0.05, 1.0, 55, r"$(b{+}c)/(b{+}c{+}uds)$ of $j^{*}$ ($-1$ = no $j^{*}$)", True),
        ("ttreco_cvsb_bestm",      "cvsb_bestm",            -0.05, 1.0, 55, r"$c/(c{+}b)$ of $j^{*}$ (PNetAK4 c-vs-b;  $-1$ = no $j^{*}$)", True),
        # m_t re-drawn under a flavour cut on j* (2026-09-04, user): same
        # m(J+j*), x<2 reco as ttreco_mt, but kept only when j*'s PNetAK4
        # score ratio passes the stated cut; -1 (same sentinel as ttreco_mt)
        # both when there is no j* and when j* fails the cut.
        ("ttreco_mt_bjstar",   "mt_bjstar",   -20.0, 300.0, 32, r"$m_{t}$ [GeV],  $j^{*}$ $b^{L}$-tagged   ($-1$ = no $j^{*}$ / fails cut)", True),
        ("ttreco_mt_cjstar",   "mt_cjstar",   -20.0, 300.0, 32, r"$m_{t}$ [GeV],  $j^{*}$ $c^{L}$-tagged   ($-1$ = no $j^{*}$ / fails cut)", True),
        ("ttreco_mt_udsjstar", "mt_udsjstar", -20.0, 300.0, 32, r"$m_{t}$ [GeV],  $j^{*}$ non-$b/c_L$-tagged   ($-1$ = no $j^{*}$ / fails cut)", True),
        ("ttreco_mt_bcjstar",  "mt_bcjstar",  -20.0, 300.0, 32, r"$m_{t}$ [GeV],  $j^{*}$ $b^{L}$ or $c^{L}$ tagged   ($-1$ = no $j^{*}$ / fails cut)", True),
        # hfuds_near / hfuds_bestm  (b+c)/(b+c+uds) specs removed on request.
        # 2026-09-04 (user): no-j* investigation.  dR_ja_bdr1 = dR(J, nearest
        # AK4 outside the cone) -- the cleanest single tag of the "isolated
        # boosted W-jet" no-j* population (~2.8 vs ~1.5 for a normal top).
        # dR_ja_bL12 = dR between the 2 loose-b AK4 nearest J (bL1,bL2 seeds)
        # -- "dR of the nearest 2 b^L jets".  -1 = no such jet / < 2 loose-b.
        ("ttreco_dR_ja_bdr1",      "dR_ja_bdr1",              -0.2, 5.0,   52, r"$\Delta R(J,\,$nearest AK4 outside cone$)$   ($-1$ = none)", False),
        # ("ttreco_dR_ja_bL12",      "dR_ja_bL12",              -0.2, 5.0,   52, r"$\Delta R$ of the nearest 2 $b^{L}$ AK4 (outside cone)   ($-1$ = $<2$)", False),  # removed on request 2026-09-04
        # ("ttreco_dR_ja_bL1",       "dR_ja_bL1",               0.0, 5.0,   50, r"$\Delta R(J,\,$nearest loose-$b$ AK4 outside cone$)$", True),  # Delta-R var, commented out on request
        # ("ttreco_n_bL_out_ja",     "n_bL_out_ja",            -0.5, 5.5,   6, r"# loose-$b$ AK4 jets outside the $J$ cone", False),  # commented out on request

        # ---- classifier / discriminant outputs ------------------------------
        ("score_Dbc",              "score_Dbc",              0.0, 1.0, 50, r"$D_{bc}(J)$",                 True),
        # Same D_bc BDT re-run on the other ranked AK8 jets (j_b = 2nd-bc-score;
        # j^1 / j^2 = 1st / 2nd highest-mSD).  SENTINEL -> dropped when the jet
        # doesn't exist (n_ak8 < 2 for j_b / j^2).
        # ("score_Dbc_jb",           "score_Dbc_jb",           0.0, 1.0, 50, r"$D_{bc}(j_b)$   (2nd-highest-$bc$-score AK8)", True),  # commented out on request 2026-09-04
        # ("score_Dbc_jsup1",        "score_Dbc_jsup1",        0.0, 1.0, 50, r"$D_{bc}(j^{1})$   (highest-$m_{\mathrm{SD}}$ AK8)", True),  # commented out on request
        # ("score_Dbc_jsup2",        "score_Dbc_jsup2",        0.0, 1.0, 50, r"$D_{bc}(j^{2})$   (2nd-highest-$m_{\mathrm{SD}}$ AK8)", True),  # commented out on request
        ("score_SC",               "score_SC",             1e-4, 1.0, 50, r"$S_{\mathrm{EVT}}$",            True, False, True),
        # EventClassifier scores + their sums/ratios -- all commented out on request:
        # ("evtcls_w_qq",            "score_cata_w_qq_norm",  1e-4, 1.0, 50, r"EventClassifier $W\to qq'$ score",   True, False, True),
        # ("evtcls_top_bc",          "score_cata_top_bc_norm",  0.0, 1.0, 50, r"EventClassifier Top($bc$) score",    True),
        # ("evtcls_top_bqq",         "score_cata_top_bqq_norm", 1e-5, 0.1, 50, r"EventClassifier Top($bqq'$) score",  False, False, True),
        # ("evtcls_top_bq",          "score_cata_top_bq_norm",  0.0, 1.0, 50, r"EventClassifier Top($bq$) score",    True),
        # ("evtcls_top_like",        "score_cata_top_sum_norm", 0.0, 1.0, 50, r"top-like score  (EC Top($bc$)$+$Top($bq$)$+$Top($bqq'$))", True),
        # ("evtcls_top_like_2",      "score_cata_top_sum2_norm", 0.0, 1.0, 50, r"top-like-2 score  (EC Top($bq$)$+$Top($bqq'$),  no Top($bc$))", True),
        # ("evtcls_qcd",             "score_cata_qcd_norm",   1e-5, 1.0, 50, r"EventClassifier QCD score",          True, False, True),
        # "none-of-the-above" = the EventClassifier's 6th output (`score_cata_non`,
        # gen flag `ak8_match_non`): the leading AK8 jet is matched to NONE of the
        # five physics categories the classifier targets -- not W->qq', not QCD,
        # not top(bqq'), not top(bc), not top(bq). In practice a jet with no clean
        # generator-level W/top content (out-of-acceptance decay products,
        # accidental jet, ...). Renamed here to make "above" explicit.
        # ("evtcls_nomatch",         "score_cata_non_norm",     0.0, 1.0, 50, r"EventClassifier no-match score  ($j_1$ unmatched)", True),

        # ---- leading-AK8 GloParT raw scores --------------------------------
        # 9 live GloParT node scores of J (gpt_bqq is a dead node -> stays
        # off).  Re-enabled 2026-09-03 on request.
        ("gpt_bc",                 "ak8_gpt_bc_0",            0.0, 1.0, 50, r"GloParT $bc$",                  True),
        ("gpt_bb",                 "ak8_gpt_bb_0",            0.0, 1.0, 50, r"GloParT $bb$",                  True),
        ("gpt_cc",                 "ak8_gpt_cc_0",          1e-4, 1.0, 50, r"GloParT $cc$",                  True, False, True),
        ("gpt_cs",                 "ak8_gpt_cs_0",          1e-4, 1.0, 50, r"GloParT $cs$",                  True, False, True),
        ("gpt_bs",                 "ak8_gpt_bs_0",          1e-4, 1.0, 50, r"GloParT $bs$",                  True, False, True),
        ("gpt_qq",                 "ak8_gpt_qq_0",          1e-6, 1.0, 50, r"GloParT $qq$",                  True, False, True),
        ("gpt_qcd",                "ak8_gpt_qcd_0",         1e-3, 1.0, 50, r"GloParT QCD",                   True, False, True),
        ("gpt_topbw",              "ak8_gpt_topbw_0",       1e-4, 1.0, 50, r"GloParT top score ($b{+}W$ in jet)", True, False, True),
        ("gpt_topw",               "ak8_gpt_topw_0",        1e-4, 1.0, 50, r"GloParT top-$W$ score ($b$ outside jet)", True, False, True),
        # ("gpt_bqq",              "ak8_gpt_bqq_0",         1e-4, 1.0, 50, r"GloParT $bqq'$ score (3-prong $t\to bqq'$)", True, False, True),  # DROPPED 2026-09-01: ak8_gpt_bqq is identically 0.0 in every jet of every sample (dead GloParT node -- 3-prong hadronic top lives in gpt_topbw).  Branch still loaded + kept in the ratio denominators (contributes 0) in case a future ntuple fills it.
        # finer GloParT sub-nodes from the 2final ntuples (2026-09-04, user) --
        # ak8_gpt_<node>_0 for every node in _GPT_EXTRA_NODES.  log-x 1e-4..1.
        # ONLY in MAT-SIGNAL (1 sample) or with WCB_MAT_GPT_EXTRA=1 -- loading
        # all 24 extra columns for the 17-sample MAT run OOMs the cgroup.
        *([("gpt_" + _n, "ak8_gpt_%s_0" % _n, 1e-4, 1.0, 50,
            r"GloParT $%s$ score" % _n.replace("topbw", "t\\to bW,").replace(
                "topw", "t\\to W,"), True, False, True)
           for _n in _GPT_EXTRA_NODES]
          if (signal_only or os.environ.get("WCB_MAT_GPT_EXTRA"))
          else []),

        # ---- GloParT discriminant ratios ----------------------------------
        # ENTIRE BLOCK commented out on request 2026-09-02 (derived vars +
        # cache entries + _attach_runtime_fields all stay; just no MAT plots).
        # ("gpt_bc_frac",            "gpt_bc_frac",             0.0, 1.0, 50, r"$bc/\sum$ GloParT",             True),
        # ("gpt_2prong_sum",         "gpt_2prong_sum",          0.0, 1.0, 50, r"$\sum$ GloParT V$\to$2q  ($bc{+}bb{+}cc{+}bs{+}cs{+}qq$)", True),
        # ("gpt_bc_vs_2prong",       "gpt_bc_vs_2prong",        0.0, 1.0, 50, r"$bc/(bc{+}cc{+}bb{+}qq{+}bs{+}cs)$", True),
        # ("gpt_bc_vs_nontop",       "gpt_bc_vs_nontop",        0.0, 1.0, 50, r"$bc/(bc{+}cc{+}bb{+}qq{+}bs{+}cs{+}\mathrm{QCD})$", True),
        # ("gpt_bc_vs_qcd",          "gpt_bc_vs_qcd",           0.0, 1.0, 50, r"$bc/(bc+\mathrm{QCD})$", True),
        # ("gpt_bc_vs_cc",           "gpt_bc_vs_cc",            0.0, 1.0, 50, r"$bc/(bc+cc)$",                  True),
        # ("gpt_bc_vs_ccbb",         "gpt_bc_vs_ccbb",          0.0, 1.0, 50, r"$bc/(bc+cc+bb)$",               True),
        # ("gpt_hfmix_vs_2prong",    "gpt_hfmix_vs_2prong",     0.0, 1.0, 50, r"$(bc{+}bs{+}cs)/(bc{+}bs{+}cs{+}cc{+}bb{+}qq)$", True),
        # ("gpt_bc_vs_hf2prong",     "gpt_bc_vs_hf2prong",      0.0, 1.0, 50, r"$bc/(bc{+}bs{+}cs{+}cc{+}bb)$", True),
        # ("gpt_bc_vs_bb",           "gpt_bc_vs_bb",            0.0, 1.0, 50, r"$bc/(bc+bb)$",                  False),  # commented out on request
        # ("gpt_bc_vs_qq",           "gpt_bc_vs_qq",            0.0, 1.0, 50, r"$bc/(bc+qq)$",                  True),
        # ("gpt_cs_vs_qq",         "gpt_cs_vs_qq",            0.0, 1.0, 50, r"$cs/(cs+qq)$",                  False),  # dropped 2026-08-31
        # ("gpt_hf_w",             "gpt_hf_w",                0.0, 1.0, 50, r"$(bc+cs+bs)/\sum$ GloParT",     True),  # dropped 2026-08-31
        # ("gpt_top_frac",           "gpt_top_frac",          1e-4, 1.0, 50, r"top-like fraction GloParT",     True, False, True),
        # ("gpt_w_vs_top",           "gpt_w_vs_top",            0.0, 1.0, 50, r"W-like $/$ (W-like $+$ top-like)   ($J$)", True),
        # ("gpt_w_vs_top_jb",        "gpt_w_vs_top_jb",         0.0, 1.0, 50, r"W-like $/$ (W-like $+$ top-like)   ($j_b$)", True),
        # ("gpt_bc_vs_topbw",      "gpt_bc_vs_topbw",         0.0, 1.0, 50, r"$bc/(bc+t(bW))$",              True),  # dropped 2026-08-31 (superseded by gpt_bc_vs_top)
        # ("gpt_bc_vs_top",        "gpt_bc_vs_top",           0.0, 1.0, 50, r"$bc/(bc+t(bW)+t(W)+bqq')$   (W$cb$ vs t$\to bc$)", True),  # dropped 2026-08-31
        # ("gpt_topbw_vs_top",       "gpt_topbw_vs_top",        0.0, 1.0, 50, r"$t(bW)/(t(bW)+t(W)+bqq)$",     True),
        # ("gpt_cW_vs_qq",         "gpt_cW_vs_qq",            0.0, 1.0, 50, r"$(bc{+}cs{+}cc)/((bc{+}cs{+}cc)+qq)$", True),  # dropped 2026-08-31
        # ("gpt_bW_vs_qq",         "gpt_bW_vs_qq",            0.0, 1.0, 50, r"$(bc{+}bs{+}bb)/((bc{+}bs{+}bb)+qq)$", True),  # dropped 2026-08-31

        # ---- cb-candidate-AK8 kinematics & substructure  (J) ------------
        # ("ak8_pt_0",               "ak8_pt_0",              200.0, 1000.0, 40, r"$J$ $p_T$ [GeV]",   True),  # commented out on request
        # eta spectra dropped -> no separation power (detector-frame, ~flat).
        # ("ak8_eta_0",              "ak8_eta_0",             -2.5,  2.5,   50, r"$J$ $\eta$",          False),
        # phi is flat / detector-frame only -> no separation power, dropped.
        # ("ak8_phi_0",              "ak8_phi_0",             -PI,   PI,    50, r"$J$ $\phi$",          False),
        # SD masses: 25 bins over 0-250 GeV -> 10 GeV bins (rebinned x2 from 50).
        ("ak8_sdmass_0",           "ak8_sdmass_0",            0.0, 250.0, 25, r"$m_{\mathrm{SD}}(J)$ [GeV]  (highest-$bc$-score AK8)", False),
        ("ak8_sdmass_maxmass_0",   "ak8_sdmass_maxmass_0",    0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j^{1})$ [GeV]  (highest-$m_{\mathrm{SD}}$ AK8)", False),
        # ("ak8_sdmass_lead_pt_0",   "ak8_sdmass_lead_pt_0",    0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j_{1})$ [GeV]  (highest-$p_T$ AK8)", False),  # commented out on request 2026-09-04
        # ("ak8_sdmass_sub_pt_0",    "ak8_sdmass_sub_pt_0",     0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j_{2})$ [GeV]  (sub-leading $p_T$ AK8)", False),  # commented out on request 2026-09-04
        ("ak8_sdmass_sub_mass_0",  "ak8_sdmass_sub_mass_0",   0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j^{2})$ [GeV]  (2nd-highest-$m_{\mathrm{SD}}$ AK8)", False),
        ("ak8_sdmass_sub_bc_0",    "ak8_sdmass_sub_bc_0",     0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j_{b})$ [GeV]  (2nd-highest-$bc$-score AK8)", False),
        # same four SD masses but ONLY for events with >= 2 AK8 jets (value is
        # SENTINEL -> dropped when n_ak8 < 2).  Fields attached at load time by
        # DataManager._attach_runtime_fields, NOT in the parquet cache.
        # ("ak8_sdmass_ja_nj2",      "ak8_sdmass_ja_nj2",       0.0, 250.0, 25, r"$m_{\mathrm{SD}}(J)$ [GeV]  ($N_{\mathrm{AK8}}\!\geq\!2$; highest-$bc$-score)", False),  # commented out on request 2026-09-04
        # ("ak8_sdmass_jb_nj2",      "ak8_sdmass_jb_nj2",       0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j_{b})$ [GeV]  ($N_{\mathrm{AK8}}\!\geq\!2$; 2nd-highest-$bc$-score)", False),  # commented out on request 2026-09-04
        # ("ak8_sdmass_jsup1_nj2",   "ak8_sdmass_jsup1_nj2",    0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j^{1})$ [GeV]  ($N_{\mathrm{AK8}}\!\geq\!2$; highest-$m_{\mathrm{SD}}$)", False),  # commented out on request 2026-09-04
        ("ak8_sdmass_jsup2_nj2",   "ak8_sdmass_jsup2_nj2",    0.0, 250.0, 25, r"$m_{\mathrm{SD}}(j^{2})$ [GeV]  ($N_{\mathrm{AK8}}\!\geq\!2$; 2nd-highest-$m_{\mathrm{SD}}$)", False),
        # 4-bin categorical: which (mSD(J), mSD(j_b)) quadrant of the W
        # window [65,105] GeV the event sits in.  Filled for every n_ak8>=2
        # event (all truth classes incl. signal), so the overlay shows where
        # each class lands.  value = [ mSD(j_b) in [65,105] ]  (0 = out, 1 = in),
        # filled only when mSD(J) is ALSO in the window.
        # ("sdwin_tag_nj2",          "sdwin_tag_nj2",          -0.5, 1.5, 2, r"$m_{\mathrm{SD}}(j_b)$ in $W$-window [65,105] GeV  ($N_{\mathrm{AK8}}\!\geq\!2$, $m_{\mathrm{SD}}(J)$ in window)", True, False, False,
        #  [r"$j_b$ out", r"$j_b$ in"]),  # commented out on request 2026-09-04
        # ("ak8_rawFactor_0",        "ak8_rawFactor_0",         0.0, 1.0,   50, r"$J$ raw factor",      False),
        ("ak8_tau21_0",            "ak8_tau21_0",             0.0, 1.0,   50, r"$\tau_{21}(J)$",     False),
        ("ak8_tau32_0",            "ak8_tau32_0",             0.0, 1.0,   50, r"$\tau_{32}(J)$",     False),
        ("ak8_tau31_0",            "ak8_tau31_0",             0.0, 1.0,   50, r"$\tau_{31}(J) = \tau_{32}\,\tau_{21}$", False),
        # ("ak8_tau21_sub_bc_0",     "ak8_tau21_sub_bc_0",      0.0, 1.0,   50, r"$\tau_{21}(j_b)$   (2nd-highest-$bc$-score AK8)", False),  # commented out on request 2026-09-04
        # ("ak8_nConst_0",         "ak8_nConst_0",            0.0, 120.0, 60, r"$J$ $N_{\mathrm{const}}$", False),  # dropped 2026-08-31
        # ("ak8_n_b_in_jet",         "ak8_n_b_in_jet",         -0.5, 5.5,    6, r"$N_b$ in $J$ (gen)",    False, True),  # commented out on request
        # ("ak8_n_c_in_jet",         "ak8_n_c_in_jet",         -0.5, 5.5,    6, r"$N_c$ in $J$ (gen)",    False, True),  # commented out on request
        # ("ak8_n_in_jet",           "ak8_n_in_jet",           -0.5, 10.5,  11, r"# partons in $J$ (gen)", False, True),  # commented out on request

        # ---- leading-AK4 -------------------------------------------------
        # ("ak4_pt_0",             "ak4_pt_0",                0.0, 600.0, 50, r"Leading AK4 $p_T$ [GeV]",     True),  # dropped 2026-08-31
        # ("ak4_eta_0",              "ak4_eta_0",              -2.5, 2.5,   50, r"Leading AK4 $\eta$",          False),
        # ("ak4_mass_0",           "ak4_mass_0",              0.0, 60.0,  40, r"Leading AK4 mass [GeV]",      False),  # dropped 2026-08-31

        # ---- angular / dR variables ------------------------------------
        #   J == the leading AK8 jet, i.e. the Wcb / top-bc candidate jet
        #   whose truth match defines the MAT overlay categories.
        #   All bare Delta-phi variables are commented out: azimuthal
        #   differences carry little separation on top of the Delta-R /
        #   Delta-eta pairs kept below, and phi itself is detector-frame flat.
        # Delta-eta / Delta-phi / Delta-R angular-separation vars -- all commented out on request:
        ("dR_lep_ak8",             "dR_lep_ak8",              0.0, 5.0,   50, r"$\Delta R(\ell,\,J)$",        False),  # re-enabled: m_t=-1 study
        # ("dphi_lep_ak8",           "dphi_lep_ak8",            0.0, PI,    40, r"$\Delta\phi(\ell,\,J)$",                  False),
        # ("dphi_w_ak8",             "dphi_w_ak8",              0.0, PI,    40, r"$\Delta\phi(W(\ell\nu),\,J)$   [$W$ = $\ell+p_T^{\mathrm{miss}}$]", True),
        ("dR_w_ak8",               "dR_w_ak8",                0.0, 5.0,   50, r"$\Delta R(W(\ell\nu),\,J)$   [$W$ = $\ell+p_T^{\mathrm{miss}}$]", False),  # re-enabled: m_t=-1 study
        # ("deta_lep_ak8",           "deta_lep_ak8",            0.0, 5.0,   50, r"$\Delta\eta(\ell,\,J)$",      False),
        # ("dphi_met_ak8",           "dphi_met_ak8",            0.0, PI,    40, r"$\Delta\phi(p_T^{\mathrm{miss}},\,J)$",   False),
        # ("dphi_lep_met",           "dphi_lep_met",            0.0, PI,    40, r"$\Delta\phi(\ell,\,p_T^{\mathrm{miss}})$",  False),
        # ("dR_ak4_ak8",             "dR_ak4_ak8",              0.0, 5.0,   50, r"$\Delta R(j_{\mathrm{AK4}},\,J)$", True),
        # ("dphi_ak4_ak8",           "dphi_ak4_ak8",            0.0, PI,    40, r"$\Delta\phi(j_{\mathrm{AK4}},\,J)$", False),
        # ("deta_ak4_ak8",           "deta_ak4_ak8",            0.0, 1.0,   50, r"$\Delta\eta(j_{\mathrm{AK4}},\,J)$", True),
        # ("ptrel_lep_ak8",          "ptrel_lep_ak8",           0.0, 2.0,   40, r"$p_T^{\ell}/p_T^{J}$",        False),

        # ---- reco b/c-tagged AK4 multiplicity within dR<1 of the cb-cand AK8 --
        # ("n_c_nearcb_L",           "n_c_nearcb_L",           -0.5, 5.5,    6, r"$N_{c^{L}}$ within $\Delta R\!<\!1$ of $J$   ($c^{L}$ = ParticleNetAK4 loose $c$-tag)", False),  # commented out on request
        # ("n_c_nearcb_M",           "n_c_nearcb_M",           -0.5, 5.5,    6, r"$N_{c^{M}}$ within $\Delta R\!<\!1$ of $J$   ($c^{M}$ = ParticleNetAK4 medium $c$-tag)", False),  # commented out on request
        # ("n_b_nearcb_L",           "n_b_nearcb_L",           -0.5, 5.5,    6, r"$N_{b^{L}}$ within $\Delta R\!<\!1$ of $J$   ($b^{L}$ = ParticleNetAK4 loose $b$-tag)", False),  # commented out on request
        # ("n_b_nearcb_M",           "n_b_nearcb_M",           -0.5, 5.5,    6, r"$N_{b^{M}}$ within $\Delta R\!<\!1$ of $J$   ($b^{M}$ = ParticleNetAK4 medium $b$-tag)", False),  # commented out on request

        # ---- dR / invariant-mass observables from tagged AK4 jets --------
        #   b^{L/M/T} , c^{L/M/T} = an AK4 jet ParticleNetAK4-tagged as b / c at
        #   the loose / medium / tight WP (superscript = WP).  These minDR/mass
        #   observables all use the LOOSE WP.  NB the AK8 bc-score ranking uses
        #   SUBSCRIPTS J / j_b / j_c -- unrelated.
        #   minDR_X = smallest dR among all pairs of the >=2 X-tagged AK4 jets;
        #   mass_minDR_X = invariant mass of *that* dR-closest pair
        #     (-1 when the event has <2 such jets);
        #   mass_cL_nearb (derived here, not an ev() branch) = m( the
        #     highest-pn_c c^{L} AK4  +  its nearest-dR b^{L} AK4 ).
        # min Delta-R / Delta-phi over tagged-AK4 pairs -- Delta vars, commented out on request:
        ("minDR_b",                "minDR_b",                 0.0, 5.0,   50, r"$\Delta R$ of the closest $b^{L}b^{L}$ AK4 jet pair", True),  # re-enabled: m_t=-1 study
        # ("minDR_c",                "minDR_c",                 0.0, 5.0,   50, r"$\Delta R$ of the closest $c^{L}c^{L}$ AK4 jet pair", True),
        # ("minDR_bc",               "minDR_bc",                0.0, 5.0,   50, r"$\Delta R$ of the closest $b^{L}c^{L}$ AK4 jet pair",   True),
        # ("minDR_bc_LM",            "minDR_bc_LM",             0.0, 5.0,   50, r"$\Delta R$ of the closest $b\,c$ AK4 pair, $b^{L}c^{M}$ or $b^{M}c^{L}$", True),
        # ("minDR_bc_MM",            "minDR_bc_MM",             0.0, 5.0,   50, r"$\Delta R$ of the closest $b^{M}c^{M}$ AK4 jet pair",   True),
        # ("minDphi_bc",             "minDphi_bc",              0.0, PI,    50, r"min $\Delta\phi$ over $b^{L}c^{L}$ AK4 jet pairs",     True),
        # AK4-only W->cb ("m(qq)") and t->bqq ("m(qqq)") reconstruction
        # attempts -- ALL commented out on request to focus on the m_t
        # reconstructed WITH J (mt1..mt10 above).  Derived fields stay cached.
        # ("mass_minDR_b",           "mass_minDR_b",            0.0, 300.0, 50, r"$m(b^{L}b^{L})$ of the $\Delta R$-closest such AK4 jet pair [GeV]",  True),
        # ("mass_minDR_c",           "mass_minDR_c",            0.0, 300.0, 50, r"$m(c^{L}c^{L})$ of the $\Delta R$-closest such AK4 jet pair [GeV]",  True),
        # ("mass_minDR_bc",          "mass_minDR_bc",           0.0, 300.0, 50, r"$m(b^{L}c^{L})$ of the $\Delta R$-closest such AK4 jet pair [GeV]", True),
        # ("mSD_minus_mbc",          "mSD_minus_mbc",        -150.0, 150.0, 50, r"$m_{\mathrm{SD}}(J) - m(b^{L}c^{L})_{\Delta R\text{-closest}}$ [GeV]", True),
        # ("mass_cL_nearb",          "mass_cL_nearb",           0.0, 300.0, 50, r"$m(c^{L}b^{L})$: highest-score $c^{L}$ AK4 $+$ its nearest-$\Delta R$ $b^{L}$ AK4 [GeV]", True),
        # ("mass_bc_bestmW",         "mass_bc_bestmW",          0.0, 300.0, 50, r"$m(b^{L}c^{L})$ of the $b^{L}c^{L}$ AK4 pair with mass closest to 80 GeV [GeV]", True),
        # ("mass_bbc_minDR",         "mass_bbc_minDR",          0.0, 500.0, 50, r"$m(b^{L}b^{L}c^{L})$ of the most collimated (min $\sum\Delta R$) AK4 triplet [GeV]", False),
        # ("mass_bbc_bestmt",        "mass_bbc_bestmt",         0.0, 500.0, 50, r"$m(b^{L}b^{L}c^{L})$ of the AK4 triplet with mass closest to 172.5 GeV [GeV]", True),

        # ---- top pT asymmetry  |pt_top - pt_top^lep| / sum  (needs the
        #      leptonic-side pairing internally; the leptonic top mass itself
        #      is not plotted).  Differ only in the hadronic-side b choice:
        #   ptasym_tt    : the 2 highest-PNet-b AK4 outside J, split by
        #                  dR + a pT-balance tie-break (WCB_TTRECO_AMB).
        #   ptasym_dr    : b = loose-b AK4 nearest J in dR.
        #   ptasym_bestm : b = loose-b AK4 giving m_t closest to 172.5.
        # ("ttreco_ptasym",          "ptasym_tt",               0.0, 1.0,   40, r"top $p_T$ asymmetry   ($b$-score seeds $+$ $\Delta R$/$p_T$-balance)", False),  # commented out on request
        # ("ttreco_ptasym_dr",       "ptasym_dr",               0.0, 1.0,   40, r"top $p_T$ asymmetry   ($b$ = loose-$b$ AK4 nearest $J$ in $\Delta R$)", False),  # commented out on request
        # ("ttreco_ptasym_bestm",    "ptasym_bestm",            0.0, 1.0,   40, r"top $p_T$ asymmetry   ($b$ = loose-$b$ AK4 giving $m_{t}$ closest to 172.5 GeV)", False),  # removed on request

        # ---- event activity -------------------------------------------
        # ("ht",                     "ht",                      0.0, 2000.0, 50, r"$H_T$ [GeV]",               True),  # commented out on request
        ("n_ak4",                  "n_ak4",                  -0.5, 14.5,   15, r"# AK4 jets",                False),  # re-enabled: m_t=-1 study
        # ("ht_bc",                  "ht_bc",                   0.0, 1500.0, 50, r"$\sum p_T$ of $b$- and $c$-tagged AK4 jets [GeV]", True),
        # n_ak4 re-enabled for the m_t=-1 study -- moved up next to ttreco_mt.
        ("n_ak8",                  "n_ak8",                  -0.5, 6.5,     7, r"$N_{\mathrm{AK8}}$",         False),
        # ParticleNetAK4 exclusive b/c ladder (per-jet `ak4_tag`), with
        # p_bc = P(b)+P(c) and r_b = P(b)/p_bc.  b-tag WPs: L = p_bc>0.5 &
        # r_b>0.40 ; M = & r_b>0.70 ; T = & r_b>0.88.  c-tag WPs:
        # L = p_bc>0.1 (not b) ; M = p_bc>0.2 ; T = p_bc>0.5 & r_b<=0.40.
        # ("n_btag",                 "n_btag",                 -0.5, 8.5,     9, r"$N_{b^{L}}$  (# ParticleNetAK4 loose-$b$-tagged AK4)",   False),  # commented out on request
        # ("n_btagM",                "n_btagM",                -0.5, 8.5,     9, r"$N_{b^{M}}$  (# ParticleNetAK4 medium-$b$-tagged AK4)",  False),  # commented out on request
        # ("n_btagT",                "n_btagT",                -0.5, 8.5,     9, r"$N_{b^{T}}$  (# ParticleNetAK4 tight-$b$-tagged AK4)",   False),  # commented out on request
        # ("n_ctag",                 "n_ctag",                 -0.5, 8.5,     9, r"$N_{c^{L}}$  (# ParticleNetAK4 loose-$c$-tagged AK4)",   False),  # commented out on request
        # ("n_ctagM",                "n_ctagM",                -0.5, 8.5,     9, r"$N_{c^{M}}$  (# ParticleNetAK4 medium-$c$-tagged AK4)",  False),  # commented out on request
        # ("n_ctagT",                "n_ctagT",                -0.5, 8.5,     9, r"$N_{c^{T}}$  (# ParticleNetAK4 tight-$c$-tagged AK4)",   False),  # commented out on request
        # ("met",                    "met",                     0.0, 400.0,  40, r"$p_T^{\mathrm{miss}}$ [GeV]", False),
        # ("mTW",                    "mTW",                     0.0, 250.0,  50, r"$m_T^W$ [GeV]",             False),  # commented out on request
        # ("w_mass",                 "w_mass",                  0.0, 400.0,  50, r"$m(\ell,\,p_T^{\mathrm{miss}})$, $p_z^{\nu}\!=\!0$ [GeV]", False),
        # ("v_pt",                   "v_pt",                    0.0, 600.0,  50, r"$p_T(\ell + p_T^{\mathrm{miss}})$ [GeV]",  False),
        # v_mass dropped: the ntuple's v_{pt,eta,phi,mass} vector is literally
        # (lepton + (met_x, met_y, 0)), so v_mass is bit-identical to the
        # inline w_mass above (verified: corr 0.9999999, same percentiles).
        # It is NOT the transverse mass -- that is mTW.
        # ("lep1_pt",                "lep1_pt",                 0.0, 400.0,  40, r"Lepton $p_T$ [GeV]",        True),
        # ("lep1_eta",               "lep1_eta",               -2.5, 2.5,   50, r"Lepton $\eta$",             False),
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


def run_mat_montage(figure_dir, suffix=""):
    """Assemble the MAT summary panels (mSD, raw GloParT scores, pT/energy
    scale) by invoking ``montage_mat.sh`` next to this script once every MAT
    PNG has been written. ``suffix`` ("_PRE" / "_SR") selects which per-
    selection plot set to montage. No-op if the helper script or ImageMagick's
    ``montage`` is missing -- never fails the run."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "montage_mat.sh")
    if not os.path.isfile(script):
        print(f"[MONTAGE] {script} not found -- skipping panels.")
        return
    print("=" * 100)
    print(f"[MONTAGE] {script} {figure_dir} {suffix}")
    try:
        subprocess.run(["bash", script, os.path.abspath(figure_dir), suffix],
                       check=False)
    except OSError as e:
        print(f"[MONTAGE] could not run {script}: {e}")


# Every raw GloParT ak8_gpt_* node that has a (non-dead) MAT spec, panel order.
# NB: ak8_gpt_bqq is identically 0 in these ntuples -> excluded.
GPT_SCORE_NODES = ["bc", "bb", "cc", "cs", "bs", "qq", "qcd",
                   "topbw", "topw"]

# Finer GloParT sub-nodes added by the 2final (2026-09-02) ntuples -- read,
# stored as ak8_gpt_<node>_0, and each given a MAT spec (all verified live).
_GPT_EXTRA_NODES = [
    # t -> b W, split by the W decay
    "topbwcs", "topbwqq", "topbwc", "topbwq", "topbws",
    "topbwtauhv", "topbwev", "topbwmv", "topbwtauev", "topbwtaumv",
    # t -> W  (b lost / merged elsewhere), split by W decay
    "topwqq", "topwcs", "topwtauhv", "topwev", "topwmv",
    "topwtauev", "topwtaumv",
    # QCD sub-flavour + the missing 2-prong "ss" + di-tau
    "qcdbb", "qcdb", "qcdcc", "qcdc", "qcdothers", "ss", "tauhtauh",
]


def run_gpt_score_montage(figure_dir, suffix=""):
    """End-of-MAT-run summary panel of ALL raw GloParT score plots
    (``MAT_gpt_<node><suffix>.png`` for every node in ``GPT_SCORE_NODES``)
    -> ``montage_MAT_gpt_scores<suffix>.png``.  No-op if ImageMagick's
    ``montage`` is missing or no such PNGs exist; never fails the run."""
    d = os.path.abspath(figure_dir)
    files = [os.path.join(d, f"MAT_gpt_{n}{suffix}.png")
             for n in GPT_SCORE_NODES]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print(f"[MONTAGE] no MAT_gpt_*{suffix}.png found -- gpt panel skipped.")
        return
    if shutil.which("montage") is None:
        print("[MONTAGE] ImageMagick 'montage' not on PATH -- gpt panel skipped.")
        return
    out = os.path.join(d, f"montage_MAT_gpt_scores{suffix}.png")
    cmd = (["montage"] + files
           + ["-tile", "3x3", "-geometry", "+4+4", "-background", "white", out])
    print("=" * 100)
    try:
        subprocess.run(cmd, check=False)
        print(f"[MONTAGE] wrote {out}  ({len(files)}/{len(GPT_SCORE_NODES)} "
              f"raw-gpt score plots)")
    except OSError as e:
        print(f"[MONTAGE] could not run montage: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Data/MC plotting framework")
    parser.add_argument(
        "mode",
        nargs="?",
        default="ALL",
        type=str.upper,
        choices=["ALL", "MAT", "MAT-SIGNAL"],
        help="ALL (default): run the normal Data/MC batch plots. "
             "MAT: matching-truth overlay only (Wcb vs Cat_Top_bc vs Rest, "
             "with a ratio panel).  "
             "MAT-SIGNAL: same overlay but the SIGNAL sample only -- draws "
             "just the 5 red W->cb lines (loads only ttbar-powheg, fast).",
    )
    # 2nd/3rd positional tokens for MAT: one picks the y-axis (ABS|NORM), the
    # other the selection (PRE|SR|JB).  Order-independent, both optional, e.g.
    #   Make_plots.py MAT SR          Make_plots.py MAT NORM JB
    #   Make_plots.py MAT SR NORM     Make_plots.py MAT ABS
    parser.add_argument("opt_a", nargs="?", default=None, type=str.upper,
                        help="MAT: y-axis ABS (default) | NORM, and/or "
                             "selection PRE (default) | SR | JB (jb-Region).")
    parser.add_argument("opt_b", nargs="?", default=None, type=str.upper,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    _NORMS = {"ABS", "NORM"}
    _SELS = {"PRE", "SR", "JB"}
    _toks = [t for t in (args.opt_a, args.opt_b) if t]
    _bad = [t for t in _toks if t not in _NORMS | _SELS]
    if _bad:
        parser.error(f"unrecognized MAT option(s) {_bad}; expected one of "
                     f"ABS/NORM and/or PRE/SR/JB")
    args.norm = next((t for t in _toks if t in _NORMS), "ABS")
    args.sel = next((t for t in _toks if t in _SELS), "PRE")
    return args


def main():
    t_total0 = time.time()

    args = parse_args()
    cfg = Config()
    # let the loader prune the derived-cache columns to just what this
    # mode/selection needs (see columns_for_cache) -- keeps the load-phase
    # RSS under the lxplus memory cgroup.
    cfg.plot_mode = args.mode
    cfg.mat_sel = getattr(args, "sel", "PRE")
    cfg.signal_only = (args.mode == "MAT-SIGNAL")
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
    if args.mode in ("MAT", "MAT-SIGNAL"):
        mat_settings = build_mat_plot_settings(args.sel, cfg.signal_only)
        mat_cut = mat_settings[0]["cut"] if mat_settings else "1"
        _normalize = (args.norm == "NORM")
        print(f"[INFO] MAT selection: {args.sel}   (cut: {mat_cut})")
        print(f"[INFO] MAT y-axis   : {'shape-normalized (NORM)' if _normalize else 'absolute yields, W->cb auto-scaled to t->bc proxy (ABS)'}")

        # Fail fast with a clear message if the selection's cut needs a field
        # that isn't in the loaded cache (e.g. running MAT JB against a cache
        # tag older than v15m -- the tau21(j_b) branch is missing).  `ak8_pt`
        # is exempt: `ak8_pt[0]` is turned into a leading alias downstream.
        if samples:
            _need = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", mat_cut))
            _have = set(samples[0]["array"].fields)
            _miss = sorted(f for f in _need if f not in _have
                           and f not in ("and", "or", "not", "abs", "ak8_pt"))
            if _miss:
                # Graceful degrade: drop every cut clause that references a
                # missing field, keep the rest.  Lets e.g. MAT JB run with just
                # the mSD(j_b) window while the tau21(j_b) branch is absent
                # (pre-v15m cache / MC ROOT area down).
                _kept, _dropped = [], []
                for _cl in re.split(r"\s+and\s+", mat_cut):
                    _ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _cl))
                    (_dropped if (_ids & set(_miss)) else _kept).append(_cl)
                if not _kept:
                    raise SystemExit(
                        f"[ERROR] selection '{args.sel}': every cut clause "
                        f"needs a field missing from cache '{cfg.cache_tag}' "
                        f"({_miss}).")
                mat_cut = " and ".join(_kept)
                print(f"[WARN] selection '{args.sel}': cache '{cfg.cache_tag}' "
                      f"lacks {_miss} -> DROPPED clause(s) {_dropped}.")
                print(f"[WARN] running a PARTIAL '{args.sel}' with cut: {mat_cut}")
                for _p in mat_settings:
                    _p["cut"] = mat_cut

        # One-pass summary (also warms the per-sample event-context cache that
        # every MAT plot below reuses).
        hist_maker.mat_report(samples, mat_cut)

        t_draw0 = time.time()
        plot_times = []

        print("=" * 100)
        print(f"[INFO] Start drawing MAT (matching-truth) plots: {len(mat_settings)} plots")
        print("=" * 100)

        for i, p in enumerate(mat_settings):
            p["_progress"] = f"MAT {i + 1}/{len(mat_settings)}"
            p["_normalize"] = _normalize
            p["_sel"] = args.sel
            p["_signal_only"] = cfg.signal_only
            t_p0 = time.time()
            hist_data, hist_var = hist_maker.make_histograms(samples, p)

            if len(hist_data) == 0:
                print(f"[WARN] {i + 1}/{len(mat_settings)}  {p['name']}: empty, skip.")
                continue

            plotter.draw_matching(hist_data, p, hist_var)
            plot_times.append(time.time() - t_p0)

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

        # end-of-run panel of every raw GloParT ak8_gpt_* score plot.
        _msfx = ({"PRE": "_PRE", "SR": "_SR", "JB": "_JB"}.get(args.sel, "")
                 + ("_norm" if args.norm == "NORM" else ""))
        run_gpt_score_montage(cfg.figure_path, _msfx)
        # full mSD / pT panels: run `montage_mat.sh <dir> <suffix>` by hand.
        # run_mat_montage(cfg.figure_path, _msfx)
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
        p["_progress"] = f"PLOT {i + 1}/{len(plot_settings)}"
        t_p0 = time.time()
        hist_data, hist_var = hist_maker.make_histograms(samples, p)

        if len(hist_data) == 0:
            print(f"[WARN] {i + 1}/{len(plot_settings)}  {p['name']}: empty, skip.")
            continue

        plotter.draw_datamc(hist_data, p)
        plot_times.append(time.time() - t_p0)

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