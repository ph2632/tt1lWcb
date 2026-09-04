import sys, os, re, glob, time, numexpr, argparse, hashlib, gc
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import awkward as ak

def _import_uproot():
    # Try uproot4 (explicit v4 alias) first
    try:
        import uproot4 as _up
        if hasattr(_up, 'open'):
            return _up
    except ImportError:
        pass
    # Try uproot (v4+) which has uproot.open directly
    try:
        import uproot as _up
        if hasattr(_up, 'open') and _up.__file__ is not None:
            return _up
    except ImportError:
        pass
    # Older uproot where open lives in the reading submodule
    try:
        import uproot as _up
        from uproot import reading as _reading
        _up.open = _reading.open
        return _up
    except (ImportError, AttributeError):
        pass
    print("ERROR: Cannot import a working uproot (v4+). Activate the correct conda/venv environment.")
    raise ImportError("No working uproot found")

uproot = _import_uproot()

# Disable .rootlogon.C BEFORE importing ROOT to avoid FWLiteEnabler errors
os.environ['ROOTLOGON'] = '/dev/null'
os.environ['ROOTLOGONOFF'] = '1'
os.environ['ROOT_HIST'] = '0'
os.environ["MPLCONFIGDIR"] = "/tmp"

import ROOT
import math
from Summary_plots import plot_combined_9SR, plot_combined_nSR_mlpmass
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def robust_glob(pattern, attempts=5, delay=3.0):
    """glob.glob with retries. A glob over AFS/EOS returns [] both when nothing matches AND when
    the directory listing transiently fails — the latter is common when many jobs run in parallel.
    Retrying turns that transient empty result back into the real file list."""
    for _i in range(attempts):
        _hits = glob.glob(pattern)
        if _hits:
            return _hits
        if _i < attempts - 1:
            print(f"  [glob] empty result for '{pattern}' (attempt {_i+1}/{attempts}) — retrying in {delay:.0f}s ...")
            time.sleep(delay)
    return []


ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetEndErrorSize(0)  # No horizontal end-caps on error bars (plot markers + legend glyphs)
ROOT.gErrorIgnoreLevel = ROOT.kError  # Suppress ROOT warnings

# ============================================================================
# QUICK MODE: Set to False to skip loading MC background files (faster testing)
# ============================================================================
LOAD_BKG_MC = False  # Set to False to skip WJets, Top, VV MC loading; overridden by --loadmc in main()
_SIG_NORM_DIV = 2.0  # overridden to 1.0 in main() when not PRED3/PRED12

# X-score (MLPf_X) SR quantile boundaries — drives both the PRED3/PRED12 SR/VR bins and the
# -Region SR1..SR4 selections (module-level so both can share one definition).
_PRED_SR_edges = [0.85, 0.97, 0.99, 0.997, 1.0]  # 5 values = 4 SR bins

# 9 SR BINNING SCHEME:
# True  = 1D binning on (BDT_ParT+BDT_KIN)/2 with 9 uniform bins: 0.55-1.00
# False = 2D binning on BDT_ParTxBDT_KIN (3x3 grid with channel-specific boundaries)
USE_1D_AVG_BINNING = True

# ROC CURVES: compare BDT/MLP/DNN discriminators (signal vs data background)
PLOT_ROC = False        # Overridden to True automatically when ROC mode is passed on CLI

# MASS SCULPTING: m_SD distributions under different discriminator cuts
PLOT_SCULPTING = False  # Set to True to enable sculpting plots

# =============================================================================
# VARIABLE DOCUMENTATION (searchable): 0L=217, 1L=223, 2L=228 branches
# =============================================================================
# ALL_VARIABLES_0L = [run, luminosityBlock, event, year, channel, genWeight, nPSWeight, PSWeight, LHE_Vpt, puWeight, puWeightUp, puWeightDown, PV_npvs, l1PreFiringWeight, l1PreFiringWeightUp, l1PreFiringWeightDown, lumiwgt, topptWeight, topptWeightNNLO, vptWeightEWK, wptWeightEWK, zptWeightEWK, vhWeightEWK, vhWeightEWK_UP, vhWeightEWK_DOWN, vvWeightNNLO, vvWeightNNLO_UP, vvWeightNNLO_DOWN, elEffWeight, elEffWeight_UP, elEffWeight_DOWN, muEffWeight, muEffWeight_UP, muEffWeight_DOWN, pileupJetIdWeight, pileupJetIdWeight_UP, pileupJetIdWeight_DOWN, passTrigMET, passTrigEl, passTrigMu, passmetfilters, HLT_Ele27_WPTight_Gsf, HLT_Ele32_WPTight_Gsf, HLT_Ele35_WPTight_Gsf, HLT_Ele32_WPTight_Gsf_L1DoubleEG, HLT_IsoMu24, HLT_IsoMu27, HLT_PFMET110_PFMHT110_IDTight, HLT_PFMET120_PFMHT120_IDTight, HLT_PFMET130_PFMHT130_IDTight, HLT_PFMET140_PFMHT140_IDTight, HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET120_PFMHT120_IDTight_PFHT60, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60, HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60, HLT_PFMETTypeOne110_PFMHT110_IDTight, HLT_PFMETTypeOne120_PFMHT120_IDTight, HLT_PFMETTypeOne130_PFMHT130_IDTight, HLT_PFMETTypeOne140_PFMHT140_IDTight, HLT_PFMETNoMu110_PFMHTNoMu110_IDTight, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, HLT_PFMETNoMu130_PFMHTNoMu130_IDTight, HLT_PFMETNoMu140_PFMHTNoMu140_IDTight, HLT_PFMET100_PFMHT100_IDTight_PFHT60, HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60, HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60, met, met_phi, v_pt, v_eta, v_phi, v_mass, n_ak15, ak15_pt, ak15_eta, ak15_abseta, ak15_phi, ak15_mass, ak15_sdmass, ak15_regressed_mass, ak15_rawFactor, ak15_tau21, ak15_tau32, ak15_rndm, ak15_deltaR_sj12, ak15_sj1_pt, ak15_sj2_pt, ak15_nBHadrons, ak15_nCHadrons, ak15_sj1_nBHadrons, ak15_sj1_nCHadrons, ak15_sj2_nBHadrons, ak15_sj2_nCHadrons, ak15_ParticleNetMD_HccVsQCD, ak15_ParticleNetMD_HbbVsQCD, ak15_ParticleNetMD_Xbb, ak15_ParticleNetMD_Xcc, ak15_ParticleNetMD_Xqq, ak15_ParticleNetMD_QCD, ak15_ParticleNetMD_HccVsQCD_cat, ak15_ParTMDV2_Hgg, ak15_ParTMDV2_Hbb, ak15_ParTMDV2_Hcc, ak15_ParTMDV2_Hss, ak15_ParTMDV2_Hqq, ak15_ParTMDV2_Hbc, ak15_ParTMDV2_Hbs, ak15_ParTMDV2_Hcs, ak15_ParTMDV2_Hee, ak15_ParTMDV2_Hmm, ak15_ParTMDV2_Htauhtaue, ak15_ParTMDV2_Htauhtauh, ak15_ParTMDV2_Htauhtaum, ak15_ParTMDV2_QCDbb, ak15_ParTMDV2_QCDb, ak15_ParTMDV2_QCDcc, ak15_ParTMDV2_QCDc, ak15_ParTMDV2_QCDothers, ak15_ParTMDV2_TopbWcs, ak15_ParTMDV2_TopbWqq, ak15_ParTMDV2_TopbWc, ak15_ParTMDV2_TopbWs, ak15_ParTMDV2_TopbWq, ak15_ParTMDV2_TopbWev, ak15_ParTMDV2_TopbWmv, ak15_ParTMDV2_TopbWtauev, ak15_ParTMDV2_TopbWtaumv, ak15_ParTMDV2_TopbWtauhv, ak15_ParTMDV2_TopWcs, ak15_ParTMDV2_TopWqq, ak15_ParTMDV2_TopWev, ak15_ParTMDV2_TopWmv, ak15_ParTMDV2_TopWtauev, ak15_ParTMDV2_TopWtaumv, ak15_ParTMDV2_TopWtauhv, ak15_ParTMDV2_resonanceMassCorr, ak15_ParTMDV2_visiableMassCorr, n_ak4, ak4_1_pt, ak4_1_eta, ak4_1_phi, ak4_1_mass, n_jet, jet_1_pt, jet_1_eta, jet_1_phi, jet_1_mass, deta_V_ak15, dphi_V_ak15, min_dr_ak15_ak4, min_deta_ak15_ak4, dphi_met_tkmet, min_dphi_V_ak4, min_dphi_met_jet, nboostedTau, boostedTau_pt, boostedTau_eta, boostedTau_phi, boostedTau_mass, boostedTau_charge, boostedTau_decayMode, boostedTau_chargedIso, boostedTau_neutralIso, boostedTau_rawIso, boostedTau_rawIsodR03, boostedTau_puCorr, boostedTau_rawAntiEle2018, boostedTau_rawAntiEleCat2018, boostedTau_idAntiEle2018, boostedTau_idAntiMu, boostedTau_rawMVAnewDM2017v2, boostedTau_rawMVAoldDM2017v2, boostedTau_rawMVAoldDMdR032017v2, boostedTau_idMVAnewDM2017v2, boostedTau_idMVAoldDM2017v2, boostedTau_idMVAoldDMdR032017v2, boostedTau_leadTkDeltaEta, boostedTau_leadTkDeltaPhi, boostedTau_leadTkPtOverTauPt, boostedTau_photonsOutsideSignalCone, boostedTau_jetIdx, boostedTau_genPartIdx, boostedTau_genPartFlav, genTtbarId, genH_pt, genZ_pt, genW_pt, genV_pt, v1_pt, v2_pt, dr_ak15_hdaus, dr_ak15_zdaus, dr_ak15_wdaus, h2bb, h2cc, z2bb, z2cc, z2qq, w2cq, w2qq, dr_ak15_hadtop_b, dr_ak15_hadtop_wqmax, dr_ak15_hadtop_wqmin, hadtop_wqmax_pdgId, hadtop_wqmin_pdgId, dr_ak15_leptop1_b, dr_ak15_leptop1_wlep, leptop1_wlep_pdgId, dr_ak15_leptop2_b, dr_ak15_leptop2_wlep, leptop2_wlep_pdgId, kinBDT, flavBDT]
# ALL_VARIABLES_1L = [run, luminosityBlock, event, year, channel, genWeight, nPSWeight, PSWeight, LHE_Vpt, puWeight, puWeightUp, puWeightDown, PV_npvs, l1PreFiringWeight, l1PreFiringWeightUp, l1PreFiringWeightDown, lumiwgt, topptWeight, topptWeightNNLO, vptWeightEWK, wptWeightEWK, zptWeightEWK, vhWeightEWK, vhWeightEWK_UP, vhWeightEWK_DOWN, vvWeightNNLO, vvWeightNNLO_UP, vvWeightNNLO_DOWN, elEffWeight, elEffWeight_UP, elEffWeight_DOWN, muEffWeight, muEffWeight_UP, muEffWeight_DOWN, pileupJetIdWeight, pileupJetIdWeight_UP, pileupJetIdWeight_DOWN, passTrigMET, passTrigEl, passTrigMu, passmetfilters, HLT_Ele27_WPTight_Gsf, HLT_Ele32_WPTight_Gsf, HLT_Ele35_WPTight_Gsf, HLT_Ele32_WPTight_Gsf_L1DoubleEG, HLT_IsoMu24, HLT_IsoMu27, HLT_PFMET110_PFMHT110_IDTight, HLT_PFMET120_PFMHT120_IDTight, HLT_PFMET130_PFMHT130_IDTight, HLT_PFMET140_PFMHT140_IDTight, HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET120_PFMHT120_IDTight_PFHT60, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60, HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60, HLT_PFMETTypeOne110_PFMHT110_IDTight, HLT_PFMETTypeOne120_PFMHT120_IDTight, HLT_PFMETTypeOne130_PFMHT130_IDTight, HLT_PFMETTypeOne140_PFMHT140_IDTight, HLT_PFMETNoMu110_PFMHTNoMu110_IDTight, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, HLT_PFMETNoMu130_PFMHTNoMu130_IDTight, HLT_PFMETNoMu140_PFMHTNoMu140_IDTight, HLT_PFMET100_PFMHT100_IDTight_PFHT60, HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60, HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60, met, v_pt, v_eta, v_phi, v_mass, lep1_pt, lep1_eta, lep1_phi, lep1_mass, lep1_pdgId, n_ak15, ak15_pt, ak15_eta, ak15_abseta, ak15_phi, ak15_mass, ak15_sdmass, ak15_regressed_mass, ak15_rawFactor, ak15_tau21, ak15_tau32, ak15_rndm, ak15_deltaR_sj12, ak15_sj1_pt, ak15_sj2_pt, ak15_nBHadrons, ak15_nCHadrons, ak15_sj1_nBHadrons, ak15_sj1_nCHadrons, ak15_sj2_nBHadrons, ak15_sj2_nCHadrons, ak15_ParticleNetMD_HccVsQCD, ak15_ParticleNetMD_HbbVsQCD, ak15_ParticleNetMD_Xbb, ak15_ParticleNetMD_Xcc, ak15_ParticleNetMD_Xqq, ak15_ParticleNetMD_QCD, ak15_ParticleNetMD_HccVsQCD_cat, ak15_ParTMDV2_Hgg, ak15_ParTMDV2_Hbb, ak15_ParTMDV2_Hcc, ak15_ParTMDV2_Hss, ak15_ParTMDV2_Hqq, ak15_ParTMDV2_Hbc, ak15_ParTMDV2_Hbs, ak15_ParTMDV2_Hcs, ak15_ParTMDV2_Hee, ak15_ParTMDV2_Hmm, ak15_ParTMDV2_Htauhtaue, ak15_ParTMDV2_Htauhtauh, ak15_ParTMDV2_Htauhtaum, ak15_ParTMDV2_QCDbb, ak15_ParTMDV2_QCDb, ak15_ParTMDV2_QCDcc, ak15_ParTMDV2_QCDc, ak15_ParTMDV2_QCDothers, ak15_ParTMDV2_TopbWcs, ak15_ParTMDV2_TopbWqq, ak15_ParTMDV2_TopbWc, ak15_ParTMDV2_TopbWs, ak15_ParTMDV2_TopbWq, ak15_ParTMDV2_TopbWev, ak15_ParTMDV2_TopbWmv, ak15_ParTMDV2_TopbWtauev, ak15_ParTMDV2_TopbWtaumv, ak15_ParTMDV2_TopbWtauhv, ak15_ParTMDV2_TopWcs, ak15_ParTMDV2_TopWqq, ak15_ParTMDV2_TopWev, ak15_ParTMDV2_TopWmv, ak15_ParTMDV2_TopWtauev, ak15_ParTMDV2_TopWtaumv, ak15_ParTMDV2_TopWtauhv, ak15_ParTMDV2_resonanceMassCorr, ak15_ParTMDV2_visiableMassCorr, n_ak4, ak4_1_pt, ak4_1_eta, ak4_1_phi, ak4_1_mass, n_jet, jet_1_pt, jet_1_eta, jet_1_phi, jet_1_mass, deta_V_ak15, dphi_V_ak15, min_dr_ak15_ak4, min_deta_ak15_ak4, deta_lep_ak15, dphi_lep_met, mt_lep_met, min_dr_lep_ak4, min_deta_lep_ak4, nboostedTau, boostedTau_pt, boostedTau_eta, boostedTau_phi, boostedTau_mass, boostedTau_charge, boostedTau_decayMode, boostedTau_chargedIso, boostedTau_neutralIso, boostedTau_rawIso, boostedTau_rawIsodR03, boostedTau_puCorr, boostedTau_rawAntiEle2018, boostedTau_rawAntiEleCat2018, boostedTau_idAntiEle2018, boostedTau_idAntiMu, boostedTau_rawMVAnewDM2017v2, boostedTau_rawMVAoldDM2017v2, boostedTau_rawMVAoldDMdR032017v2, boostedTau_idMVAnewDM2017v2, boostedTau_idMVAoldDM2017v2, boostedTau_idMVAoldDMdR032017v2, boostedTau_leadTkDeltaEta, boostedTau_leadTkDeltaPhi, boostedTau_leadTkPtOverTauPt, boostedTau_photonsOutsideSignalCone, boostedTau_jetIdx, boostedTau_genPartIdx, boostedTau_genPartFlav, genTtbarId, genH_pt, genZ_pt, genW_pt, genV_pt, v1_pt, v2_pt, dr_ak15_hdaus, dr_ak15_zdaus, dr_ak15_wdaus, h2bb, h2cc, z2bb, z2cc, z2qq, w2cq, w2qq, dr_ak15_hadtop_b, dr_ak15_hadtop_wqmax, dr_ak15_hadtop_wqmin, hadtop_wqmax_pdgId, hadtop_wqmin_pdgId, dr_ak15_leptop1_b, dr_ak15_leptop1_wlep, leptop1_wlep_pdgId, dr_ak15_leptop2_b, dr_ak15_leptop2_wlep, leptop2_wlep_pdgId, kinBDT, flavBDT]
# ALL_VARIABLES_2L = [run, luminosityBlock, event, year, channel, genWeight, nPSWeight, PSWeight, LHE_Vpt, puWeight, puWeightUp, puWeightDown, PV_npvs, l1PreFiringWeight, l1PreFiringWeightUp, l1PreFiringWeightDown, lumiwgt, topptWeight, topptWeightNNLO, vptWeightEWK, wptWeightEWK, zptWeightEWK, vhWeightEWK, vhWeightEWK_UP, vhWeightEWK_DOWN, vvWeightNNLO, vvWeightNNLO_UP, vvWeightNNLO_DOWN, elEffWeight, elEffWeight_UP, elEffWeight_DOWN, muEffWeight, muEffWeight_UP, muEffWeight_DOWN, pileupJetIdWeight, pileupJetIdWeight_UP, pileupJetIdWeight_DOWN, passTrigMET, passTrigEl, passTrigMu, passmetfilters, HLT_Ele27_WPTight_Gsf, HLT_Ele32_WPTight_Gsf, HLT_Ele35_WPTight_Gsf, HLT_Ele32_WPTight_Gsf_L1DoubleEG, HLT_IsoMu24, HLT_IsoMu27, HLT_PFMET110_PFMHT110_IDTight, HLT_PFMET120_PFMHT120_IDTight, HLT_PFMET130_PFMHT130_IDTight, HLT_PFMET140_PFMHT140_IDTight, HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1, HLT_PFMET120_PFMHT120_IDTight_PFHT60, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60, HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60, HLT_PFMETTypeOne110_PFMHT110_IDTight, HLT_PFMETTypeOne120_PFMHT120_IDTight, HLT_PFMETTypeOne130_PFMHT130_IDTight, HLT_PFMETTypeOne140_PFMHT140_IDTight, HLT_PFMETNoMu110_PFMHTNoMu110_IDTight, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, HLT_PFMETNoMu130_PFMHTNoMu130_IDTight, HLT_PFMETNoMu140_PFMHTNoMu140_IDTight, HLT_PFMET100_PFMHT100_IDTight_PFHT60, HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60, HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60, met, v_pt, v_eta, v_phi, v_mass, lep1_pt, lep1_eta, lep1_phi, lep1_mass, lep1_pdgId, lep2_pt, lep2_eta, lep2_phi, lep2_mass, lep2_pdgId, n_ak15, ak15_pt, ak15_eta, ak15_abseta, ak15_phi, ak15_mass, ak15_sdmass, ak15_regressed_mass, ak15_rawFactor, ak15_tau21, ak15_tau32, ak15_rndm, ak15_deltaR_sj12, ak15_sj1_pt, ak15_sj2_pt, ak15_nBHadrons, ak15_nCHadrons, ak15_sj1_nBHadrons, ak15_sj1_nCHadrons, ak15_sj2_nBHadrons, ak15_sj2_nCHadrons, ak15_ParticleNetMD_HccVsQCD, ak15_ParticleNetMD_HbbVsQCD, ak15_ParticleNetMD_Xbb, ak15_ParticleNetMD_Xcc, ak15_ParticleNetMD_Xqq, ak15_ParticleNetMD_QCD, ak15_ParticleNetMD_HccVsQCD_cat, ak15_ParTMDV2_Hgg, ak15_ParTMDV2_Hbb, ak15_ParTMDV2_Hcc, ak15_ParTMDV2_Hss, ak15_ParTMDV2_Hqq, ak15_ParTMDV2_Hbc, ak15_ParTMDV2_Hbs, ak15_ParTMDV2_Hcs, ak15_ParTMDV2_Hee, ak15_ParTMDV2_Hmm, ak15_ParTMDV2_Htauhtaue, ak15_ParTMDV2_Htauhtauh, ak15_ParTMDV2_Htauhtaum, ak15_ParTMDV2_QCDbb, ak15_ParTMDV2_QCDb, ak15_ParTMDV2_QCDcc, ak15_ParTMDV2_QCDc, ak15_ParTMDV2_QCDothers, ak15_ParTMDV2_TopbWcs, ak15_ParTMDV2_TopbWqq, ak15_ParTMDV2_TopbWc, ak15_ParTMDV2_TopbWs, ak15_ParTMDV2_TopbWq, ak15_ParTMDV2_TopbWev, ak15_ParTMDV2_TopbWmv, ak15_ParTMDV2_TopbWtauev, ak15_ParTMDV2_TopbWtaumv, ak15_ParTMDV2_TopbWtauhv, ak15_ParTMDV2_TopWcs, ak15_ParTMDV2_TopWqq, ak15_ParTMDV2_TopWev, ak15_ParTMDV2_TopWmv, ak15_ParTMDV2_TopWtauev, ak15_ParTMDV2_TopWtaumv, ak15_ParTMDV2_TopWtauhv, ak15_ParTMDV2_resonanceMassCorr, ak15_ParTMDV2_visiableMassCorr, n_ak4, ak4_1_pt, ak4_1_eta, ak4_1_phi, ak4_1_mass, n_jet, jet_1_pt, jet_1_eta, jet_1_phi, jet_1_mass, deta_V_ak15, dphi_V_ak15, min_dr_ak15_ak4, min_deta_ak15_ak4, deltaR_ll, deta_z_ak15, min_deta_ak15_lep, min_deta_z_ak4, min_dr_z_ak4, nboostedTau, boostedTau_pt, boostedTau_eta, boostedTau_phi, boostedTau_mass, boostedTau_charge, boostedTau_decayMode, boostedTau_chargedIso, boostedTau_neutralIso, boostedTau_rawIso, boostedTau_rawIsodR03, boostedTau_puCorr, boostedTau_rawAntiEle2018, boostedTau_rawAntiEleCat2018, boostedTau_idAntiEle2018, boostedTau_idAntiMu, boostedTau_rawMVAnewDM2017v2, boostedTau_rawMVAoldDM2017v2, boostedTau_rawMVAoldDMdR032017v2, boostedTau_idMVAnewDM2017v2, boostedTau_idMVAoldDM2017v2, boostedTau_idMVAoldDMdR032017v2, boostedTau_leadTkDeltaEta, boostedTau_leadTkDeltaPhi, boostedTau_leadTkPtOverTauPt, boostedTau_photonsOutsideSignalCone, boostedTau_jetIdx, boostedTau_genPartIdx, boostedTau_genPartFlav, genTtbarId, genH_pt, genZ_pt, genW_pt, genV_pt, v1_pt, v2_pt, dr_ak15_hdaus, dr_ak15_zdaus, dr_ak15_wdaus, h2bb, h2cc, z2bb, z2cc, z2qq, w2cq, w2qq, dr_ak15_hadtop_b, dr_ak15_hadtop_wqmax, dr_ak15_hadtop_wqmin, hadtop_wqmax_pdgId, hadtop_wqmin_pdgId, dr_ak15_leptop1_b, dr_ak15_leptop1_wlep, leptop1_wlep_pdgId, dr_ak15_leptop2_b, dr_ak15_leptop2_wlep, leptop2_wlep_pdgId, kinBDT, flavBDT]
# CHANNEL-SPECIFIC DIFFERENCES:
# 0L ONLY: met_phi, dphi_met_tkmet, min_dphi_V_ak4, min_dphi_met_jet (4 branches)
# 1L ONLY: lep1_pt, lep1_eta, lep1_phi, lep1_mass, lep1_pdgId, deta_lep_ak15, dphi_lep_met, mt_lep_met, min_dr_lep_ak4, min_deta_lep_ak4 (10 branches)
# 2L ONLY: lep1_*, lep2_* (10 lep branches), deltaR_ll, deta_z_ak15, min_deta_ak15_lep, min_deta_z_ak4, min_dr_z_ak4 (15 branches)

def compute_sdmass_correlations(events, data_name, mc_names, var_list, sel_func, output_prefix="corr_sdmass", channel=None, max_vars=15):
    """
    Compute and plot correlations of ak15_sdmass with other variables.
    Shows Pearson (bars) and Spearman (markers) for Data and MC.
    """
    from scipy.stats import spearmanr

    def get_correlations(sample_name):
        """Return (pearson_dict, spearman_dict) for one sample."""
        if sample_name not in events: return {}, {}
        sample = events[sample_name]
        if "ak15_sdmass" not in sample.fields: return {}, {}
        mask = sel_func(sample)
        sdmass = ak.to_numpy(sample["ak15_sdmass"][mask])
        pearson, spearman = {}, {}
        for var in var_list:
            var_name = var.name if hasattr(var, 'name') else var
            if var_name not in sample.fields or var_name == "ak15_sdmass": continue
            try:
                var_data = ak.to_numpy(sample[var_name][mask])
                valid = np.isfinite(sdmass) & np.isfinite(var_data)
                if np.sum(valid) > 10:
                    corr_p = np.corrcoef(sdmass[valid], var_data[valid])[0, 1]
                    corr_s, _ = spearmanr(sdmass[valid], var_data[valid])
                    if np.isfinite(corr_p): pearson[var_name] = corr_p
                    if np.isfinite(corr_s): spearman[var_name] = corr_s
            except: continue
        return pearson, spearman

    # Get Data correlations
    data_pearson, data_spearman = get_correlations(data_name)
    if not data_pearson: print("  WARNING: No Data correlations"); return None

    # Get MC correlations (combine all MC samples)
    mc_pearson_lists, mc_spearman_lists = {}, {}
    for mc_name in mc_names:
        mc_p, mc_s = get_correlations(mc_name)
        for var, corr in mc_p.items():
            mc_pearson_lists.setdefault(var, []).append(corr)
        for var, corr in mc_s.items():
            mc_spearman_lists.setdefault(var, []).append(corr)
    mc_pearson = {var: np.mean(corrs) for var, corrs in mc_pearson_lists.items() if corrs}
    mc_spearman = {var: np.mean(corrs) for var, corrs in mc_spearman_lists.items() if corrs}
    has_mc = len(mc_pearson) > 0

    # Sort by absolute Data Pearson correlation, take top max_vars
    sorted_vars = sorted(data_pearson.keys(), key=lambda v: abs(data_pearson[v]), reverse=True)[:max_vars]

    # Print to terminal
    print(f"\n  --- Correlations with ak15_sdmass (Pearson rp / Spearman rs) ---")
    if has_mc:
        print(f"  {'Variable':35s} | {'Data rp':>8s} {'Data rs':>8s} | {'MC rp':>8s} {'MC rs':>8s}")
        for var in sorted_vars:
            dp = data_pearson.get(var, 0); ds = data_spearman.get(var, 0)
            mp = mc_pearson.get(var, 0); ms = mc_spearman.get(var, 0)
            print(f"  {var:35s} | {dp:+.3f}   {ds:+.3f}   | {mp:+.3f}   {ms:+.3f}")
    else:
        print(f"  {'Variable':35s} | {'Data rp':>8s} {'Data rs':>8s}")
        for var in sorted_vars:
            dp = data_pearson.get(var, 0); ds = data_spearman.get(var, 0)
            print(f"  {var:35s} | {dp:+.3f}   {ds:+.3f}")

    # ROOT plot: bars for Pearson, markers for Spearman
    n_vars = len(sorted_vars)
    # Scale canvas height with number of variables (min 500, ~30px per variable)
    canvas_height = max(500, int(30 * n_vars))
    c = ROOT.TCanvas("c_corr", "c_corr", 500, canvas_height)
    c.SetLeftMargin(0.056); c.SetRightMargin(0.03); c.SetTopMargin(0.08 * 500.0 / canvas_height); c.SetBottomMargin(0.14 * 500.0 / canvas_height)
    c.SetTickx(1)

    # Scale text sizes inversely with canvas height
    scale = 500.0 / canvas_height
    # Create frame with x-range -0.6 to 0.6
    frame = ROOT.TH2F("frame_corr", "", 100, -0.6, 0.6, n_vars, 0, n_vars)
    frame.GetXaxis().SetTitle("Correlation with m_{SD}  (bars = Pearson, markers = Spearman)")
    frame.GetXaxis().SetTitleSize(0.035 * scale); frame.GetXaxis().SetLabelSize(0.035 * scale); frame.GetXaxis().SetTitleOffset(1.2)
    frame.GetYaxis().SetLabelSize(0.056 * scale)
    import re
    var_labels_inside = []
    for i, var in enumerate(reversed(sorted_vars)):
        clean_var = var.replace("ak15_", "").replace("ParTMDV2_", "")
        clean_var = re.sub(r'\*\*(\d+\.?\d*)', r'^{\1}', clean_var)
        clean_var = clean_var[:25]
        rank = n_vars - i
        frame.GetYaxis().SetBinLabel(i + 1, f"({rank})")
        var_labels_inside.append((i, clean_var))
    frame.Draw()

    # Draw vertical dashed grid lines
    grid_lines = []
    for x_val in [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]:
        gl = ROOT.TLine(x_val, 0, x_val, n_vars)
        gl.SetLineColor(ROOT.kGray+1); gl.SetLineStyle(2); gl.SetLineWidth(1)
        gl.Draw(); grid_lines.append(gl)

    # Draw Pearson bars
    boxes_data, boxes_mc = [], []
    bar_height = 0.35 if has_mc else 0.4
    for i, var in enumerate(reversed(sorted_vars)):
        y_center = i + 0.5
        d_corr = data_pearson.get(var, 0)
        if has_mc:
            box_d = ROOT.TBox(0 if d_corr > 0 else d_corr, y_center, d_corr if d_corr > 0 else 0, y_center + bar_height)
        else:
            box_d = ROOT.TBox(0 if d_corr > 0 else d_corr, y_center - bar_height/2, d_corr if d_corr > 0 else 0, y_center + bar_height/2)
        box_d.SetFillColor(30); box_d.SetLineColor(30); box_d.SetLineWidth(1)
        box_d.Draw("same"); boxes_data.append(box_d)
        if has_mc:
            m_corr = mc_pearson.get(var, 0)
            box_m = ROOT.TBox(0 if m_corr > 0 else m_corr, y_center - bar_height, m_corr if m_corr > 0 else 0, y_center)
            box_m.SetFillColor(93); box_m.SetLineColor(93); box_m.SetLineWidth(1)
            box_m.Draw("same"); boxes_mc.append(box_m)

    # Draw Spearman markers (triangles for Data, circles for MC)
    gr_spearman_data = ROOT.TGraph(n_vars)
    gr_spearman_mc = ROOT.TGraph(n_vars) if has_mc else None
    for i, var in enumerate(reversed(sorted_vars)):
        y_center = i + 0.5
        ds = data_spearman.get(var, 0)
        gr_spearman_data.SetPoint(i, ds, y_center + (0.17 if has_mc else 0))
        if has_mc:
            ms = mc_spearman.get(var, 0)
            gr_spearman_mc.SetPoint(i, ms, y_center - 0.17)
    gr_spearman_data.SetMarkerStyle(22)  # upward triangle
    gr_spearman_data.SetMarkerSize(1.0); gr_spearman_data.SetMarkerColor(ROOT.kGreen+3)
    gr_spearman_data.Draw("P same")
    if has_mc:
        gr_spearman_mc.SetMarkerStyle(23)  # downward triangle
        gr_spearman_mc.SetMarkerSize(1.0); gr_spearman_mc.SetMarkerColor(ROOT.kOrange+7)
        gr_spearman_mc.Draw("P same")

    # Vertical line at 0
    line = ROOT.TLine(0, 0, 0, n_vars); line.SetLineColor(ROOT.kBlack); line.SetLineWidth(1); line.Draw()

    # Draw variable names inside plot frame
    var_name_texts = []
    for i, clean_var in var_labels_inside:
        y_pos = i + 0.5
        txt = ROOT.TLatex(-0.58, y_pos, clean_var)
        txt.SetTextFont(42); txt.SetTextSize(0.047 * scale); txt.SetTextAlign(12)
        txt.Draw(); var_name_texts.append(txt)

    # Legend (position in NDC, scale text)
    n_entries = 2 + (2 if has_mc else 0)
    leg_top = 1.0 - c.GetTopMargin() - 0.01
    leg_bot = leg_top - 0.055 * scale * n_entries
    leg = ROOT.TLegend(0.62, leg_bot, 0.95, leg_top)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035 * scale)
    leg.AddEntry(boxes_data[0], "Data (Pearson)", "f")
    leg.AddEntry(gr_spearman_data, "Data (Spearman)", "p")
    if has_mc:
        leg.AddEntry(boxes_mc[0], "MC (Pearson)", "f")
        leg.AddEntry(gr_spearman_mc, "MC (Spearman)", "p")
    leg.Draw()

    # CMS label (left), channel (middle), lumi (right)
    top_y = 1.0 - c.GetTopMargin() + 0.01
    cms = ROOT.TLatex(); cms.SetNDC(); cms.SetTextFont(62); cms.SetTextSize(0.055 * scale)
    cms.DrawLatex(0.056, top_y, "CMS")
    prel = ROOT.TLatex(); prel.SetNDC(); prel.SetTextFont(42); prel.SetTextSize(0.042 * scale)
    prel.DrawLatex(0.17, top_y, "#it{Preliminary}")
    chan = ROOT.TLatex(); chan.SetNDC(); chan.SetTextFont(42); chan.SetTextSize(0.045 * scale)
    chan.DrawLatex(0.50, top_y, f"{channel}")
    lumi = ROOT.TLatex(); lumi.SetNDC(); lumi.SetTextFont(42); lumi.SetTextSize(0.040 * scale)
    lumi.DrawLatex(0.72, top_y, "59.7 fb^{-1} (13 TeV)")

    output_file = f"{output_prefix}_{channel}.png"
    c.SaveAs(output_file); print(f"  --> Correlation plot: {output_file}"); c.Close()
    return output_file

def UnderOverFlow1D(h, reject_underflow=False, reject_overflow=False):
    """Move underflow/overflow bin contents to first/last visible bins, with proper error propagation.
    Also zeroes out the underflow/overflow bins to maintain total event count.
    reject_underflow/reject_overflow: discard instead of merging — the first/last visible bin
    then holds only its own genuine in-range fill (used when the display range's edge bin is
    also the variable's natural domain boundary, so anything beyond it is not meaningful data)."""
    Bins = h.GetNbinsX()
    # Underflow (bin 0) -> first bin (bin 1)
    if not reject_underflow:
        h.SetBinContent(1, h.GetBinContent(1) + h.GetBinContent(0))
        h.SetBinError(1, math.sqrt(h.GetBinError(1)**2 + h.GetBinError(0)**2))
    h.SetBinContent(0, 0)
    h.SetBinError(0, 0)
    # Overflow (bin Bins+1) -> last bin (bin Bins)
    if not reject_overflow:
        h.SetBinContent(Bins, h.GetBinContent(Bins) + h.GetBinContent(Bins + 1))
        h.SetBinError(Bins, math.sqrt(h.GetBinError(Bins)**2 + h.GetBinError(Bins + 1)**2))
    h.SetBinContent(Bins + 1, 0)
    h.SetBinError(Bins + 1, 0)
    return h

# Signal cross-sections x BR (fb) - Taken from table 27-28, 176, https://www.arxiv.org/pdf/1610.07922
# W^+H-->lv(gg)  WplusH_HToGG_WToLNu   (3*94.26*0.08187=) 23.151 fb
# W^-H-->vl(gg)  WminusH_HToGG_WToLNu  (3*59.83*0.08187=) 14.695 fb
# ZH-->vv(gg)    ZH_HToGG_ZToNuNu      (0.2*0.08187*883.4=) 14.468 fb (BR(Z->vv)=0.2, sigma(qq->ZH)=883.4 fb at 13 TeV)
# ggZH-->vv(gg)  ggZH_HToGG_ZToNuNu    (0.2*0.08187*135.3=) 2.216 fb (gg->ZH cross-section = 135.3 fb)
# ZH-->ll(gg)    ZH_HToGG_ZToLL        (0.0673*0.08187*883.4=) 4.867 fb (BR(Z->ll)=0.0673, l=e,mu,tau)
# ggZH-->ll(gg)  ggZH_HToGG_ZToLL      (0.0673*0.08187*135.3=) 0.745 fb (gg->ZH cross-section = 135.3 fb)

#xsec_x_BR = {
#    "ZH_ZToNuNu": 14.468, "ggZH_ZToNuNu": 2.216,  # 0l channel: ZH->vv(gg)
#    "WplusH"    : 23.151, "WminusH"     : 14.695,          # 1l channel: WH->lv(gg)
#    "ZH_ZToLL"  : 4.867 , "ggZH_ZToLL"  : 0.745,       # 2l channel: ZH->ll(gg)
#}

# xsec_x_BR0: signal xsec values from xsecWeight branch in ntuples
xsec_x_BR = {
    "ZH_ZToNuNu": 14.468, "ggZH_ZToNuNu":  2.216,  # 0l channel: ZH->vv(gg)
    "WplusH"    : 23.151, "WminusH"     : 14.695,    # 1l channel: from xsecWeight branch
    "ZH_ZToLL"  :  6.310, "ggZH_ZToLL"  :  1.017,    # 2l channel: from xsecWeight branch
}

# WJets cross-sections (fb) - matched to xsecWeight branch in ntuples (0l)
# WARNING: 1l cross-check shows different xsecWeight for some samples (same ROOT file!):
#   Pt-250To400: 1l branch gives 32406 fb (ratio 1.2061) vs 0l branch 26868 fb — xsecWeight not constant?
#   Pt-600ToInf: 1l branch gives 543 fb (ratio 0.9883) vs 0l branch 549 fb
xsec_WJets = {
    "WJetsToLNu_Pt-100To250": 776.653 * 1000,  # was 770.8 pb — 0l/1l consistent
    "WJetsToLNu_Pt-250To400": 26.868 * 1000,   # was 28.06 pb — 0l matched; 1l gives 32.4 pb (20% higher!)
    "WJetsToLNu_Pt-400To600": 3.592 * 1000,    # was 3.591 pb — 0l/1l consistent
    "WJetsToLNu_Pt-600ToInf": 0.5494 * 1000,   # was 0.5491 pb — 0l matched; 1l gives 0.543 pb (1.2% lower)
}
# Z(->vv)+jets cross-sections (fb) for 0l channel - matched to xsecWeight branch in ntuples
xsec_ZJets = {
    "Z1JetsToNuNu_PtZ-50To150": 596.269 * 1000,   # was 596.3 pb (unchanged)
    "Z1JetsToNuNu_PtZ-150To250": 17.430 * 1000,   # was 17.98 pb (-3%)
    "Z1JetsToNuNu_PtZ-250To400": 2.088 * 1000,    # was 2.057 pb (+2%)
    "Z1JetsToNuNu_PtZ-400ToInf": 0.2242 * 1000,   # was 0.476 pb (was 2x too high!)
    "Z2JetsToNuNu_PtZ-50To150": 325.604 * 1000,   # was 139.5 pb (was 2.3x too low!)
    "Z2JetsToNuNu_PtZ-150To250": 26.662 * 1000,   # was 17.8 pb (was 1.5x too low!)
    "Z2JetsToNuNu_PtZ-250To400": 5.168 * 1000,    # was 2.64 pb (was 2x too low!)
    "Z2JetsToNuNu_PtZ-400ToInf": 0.8471 * 1000,   # was 0.351 pb (was 2.4x too low!)
}
# DY (Z->ll)+jets cross-sections (fb) for 2l channel - matched to xsecWeight branch in ntuples
xsec_DY = {
    "DYJetsToLL_PtZ-0To50": 1485.0 * 1000,       # ~1485 pb — matched branch (1.0000)
    "DYJetsToLL_PtZ-50To100": 397.4 * 1000,      # ~397 pb  — matched branch (1.0000)
    "DYJetsToLL_PtZ-100To250": 97.519 * 1000,    # was 97.2 pb (+0.3%) — matched branch
    "DYJetsToLL_PtZ-250To400": 3.746 * 1000,     # was 3.701 pb (+1.2%) — matched branch
    "DYJetsToLL_PtZ-400To650": 0.5131 * 1000,    # was 0.5086 pb (+0.9%) — matched branch
    "DYJetsToLL_PtZ-650ToInf": 0.04723 * 1000,   # was 0.04728 pb (-0.1%) — matched branch
#    "DY1JetsToLL": 877.8 * 1000,                  # ~878 pb (inclusive 1-jet)
#    "DY2JetsToLL": 304.4 * 1000,                  # ~304 pb (inclusive 2-jet)
}
# Top cross-sections (fb) - matched to xsecWeight branch in ntuples (0l)
# WARNING: 1l cross-check shows different xsecWeight for TTbar (same ROOT file!):
#   TTToSemiLeptonic: 1l branch gives 336271 fb (ratio 0.8813) vs 0l branch 381546 fb
#   TTTo2L2Nu: 1l branch gives 88286 fb (ratio 0.9949) vs 0l branch 88734 fb
xsec_Top = {
    # TTbar
    "TTTo2L2Nu": 88.734 * 1000,             # was 88.29 pb — 0l matched; 1l gives 88.29 pb (0.5% lower)
    "TTToSemiLeptonic": 381.546 * 1000,     # was 365.34 pb — 0l matched; 1l gives 336.3 pb (11.9% lower!)
    # Single top t-channel
    "ST_t-channel_top": 135.918 * 1000,     # was 136.02 pb (unchanged)
    "ST_t-channel_antitop": 80.997 * 1000,  # was 80.95 pb (unchanged)
    # Single top tW-channel
    "ST_tW_top": 35.852 * 1000,             # was 35.85 pb (unchanged)
    "ST_tW_antitop": 35.838 * 1000,         # was 35.85 pb (unchanged)
    # Single top s-channel
    "ST_s-channel": 3.365 * 1000,           # was 3.36 pb (unchanged)
}
# Diboson (VV) cross-sections (fb) - matched to xsecWeight branch in ntuples
xsec_VV = {
    # WW
    "WWTo1L1Nu2Q": 53.404 * 1000,           # was 49.997 pb (+6.8%)
    "WWTo2L2Nu": 12.875 * 1000,             # was 12.178 pb (+5.7%)
    # WZ
    "WZTo1L1Nu2Q": 11.396 * 1000,           # was 10.71 pb (+6.4%)
    "WZTo1L3Nu": 3.221 * 1000,              # was 3.05 pb (+5.6%)
    "WZTo3LNu": 4.940 * 1000,               # was 4.941 pb (unchanged)
    "WZTo2Q2L": 6.193 * 1000,               # was 6.331*0.978=6.192 pb (unchanged)
    "WZTo2Q2Nu": 6.713 * 1000,              # was 6.858*0.978=6.707 pb (+0.1%)
    # ZZ
    "ZZTo2Q2L": 4.182 * 1000,               # was 3.688*1.134=4.182 pb (unchanged)
    "ZZTo2L2Nu": 0.6809 * 1000,             # was 0.6008*1.134=0.681 pb (unchanged)
    "ZZTo2Q2Nu": 5.183 * 1000,              # was 4.561*1.134=5.172 pb (+0.2%)
    "ZZTo4L"   : 1.5025 * 1000,             # was 1.325*1.134=1.503 pb (unchanged)
}


PI = np.pi  # Use exact pi value: 3.141592653589793

#R = 0.75
#HEM_veto = lambda e: ~(((e["run"] >= 319077) | ((e["run"] == 1) & (np.random.uniform() < 0.632))) & (e["year"] == 2018) &
#                        (((-3.2 < e["lep1_eta"]) & (e["lep1_eta"] < -1.3) & (-1.63 < e["lep1_phi"]) & (e["lep1_phi"] < -0.57)) | (((-3.2 - R) < e["ak15_eta"]) &
#                        (e["ak15_eta"] < (-1.3 + R)) & ((-1.63 - R) < e["ak15_phi"]) & (e["ak15_phi"] < (-0.57 + R)))));

# HEM veto for 0L channel (deterministic, no random fluctuations)
# For Data (run >= 319077): veto if V (vector boson) in HEM region (phi in [-1.63, -0.57])
# For MC (run == 1): no hard veto, use HEM_weight_0l for lumi-weighted scaling

# HEM region phi bounds: [-1.63, -0.57] (no eta cut for 0L since MET is 2D)
HEM_veto_0l   = lambda e: ~( (e["year"] == 2018) & (e["run"] >= 319077) & (e["v_phi"] > -1.66) & (e["v_phi"] < -0.54) )
HEM_weight_0l = lambda e: np.where( (e["year"] == 2018) & (e["run"] == 1) & (e["v_phi"] > -1.66) & (e["v_phi"] < -0.54), 0.3529, 1.0 )

# ── MET trigger SFs for 0l channel (Data/BKG MC, 2018) ──────────────────────
# Derived from Trigger.py: python3 Trigger.py 0l
# Toggle with APPLY_MET_TRIG_SF = True/False
APPLY_MET_TRIG_SF = True

# Keys: (MET_lo, MET_hi), Values: (SF, err)
_MET_TRIG_SF = {
    ( 50,  60): (0.9408, 0.0021),
    ( 60,  70): (0.9363, 0.0021),
    ( 70,  80): (0.9314, 0.0021),
    ( 80,  90): (0.9284, 0.0021),
    ( 90, 100): (0.9233, 0.0021),
    (100, 110): (0.9207, 0.0021),
    (110, 120): (0.9203, 0.0021),
    (120, 130): (0.9244, 0.0020),
    (130, 140): (0.9285, 0.0020),
    (140, 150): (0.9337, 0.0019),
    (150, 160): (0.9408, 0.0017),
    (160, 170): (0.9483, 0.0016),
    (170, 180): (0.9543, 0.0015),
    (180, 190): (0.9690, 0.0014),
    (190, 200): (0.9732, 0.0013),
    (200, 210): (0.9796, 0.0012),
    (210, 220): (0.9832, 0.0011),
    (220, 230): (0.9874, 0.0010),
    (230, 240): (0.9914, 0.0010),
    (240, 260): (0.9929, 0.0007),
    (260, 280): (0.9945, 0.0007),
    (280, 300): (0.9973, 0.0007),
    (300, 320): (0.9978, 0.0007),
    (320, 340): (0.9963, 0.0010),
    (340, 360): (0.9973, 0.0010),
    (360, 380): (0.9960, 0.0013),
    (380, 400): (0.9979, 0.0012),
    (400, 420): (0.9985, 0.0013),
    (420, 440): (0.9991, 0.0015),
    (440, 460): (1.0001, 0.0014),
    (460, 480): (0.9992, 0.0020),
    (480, 500): (0.9999, 0.0019),
}
# Build sorted numpy arrays for fast vectorised lookup
_sf_edges = np.array(sorted(set(lo for lo,hi in _MET_TRIG_SF) | set(hi for lo,hi in _MET_TRIG_SF)), dtype=np.float64)
_sf_vals  = np.array([_MET_TRIG_SF[(lo, hi)][0]
                      for lo, hi in zip(_sf_edges[:-1], _sf_edges[1:])], dtype=np.float64)

def met_trig_sf(met_array):
    """Return per-event MET trigger SF array for 0l 2018.
    Events outside [50,500] GeV get SF=1 (plateau or below turn-on, conservative).
    """
    idx = np.searchsorted(_sf_edges, met_array, side='right') - 1
    idx = np.clip(idx, 0, len(_sf_vals) - 1)
    sf  = _sf_vals[idx]
    # below first bin → 1.0 (no SF defined, below analysis MET threshold)
    sf[met_array < _sf_edges[0]]   = 1.0
    # above last bin → last bin value (plateau, keep last SF rather than defaulting to 1)
    sf[met_array >= _sf_edges[-1]] = _sf_vals[-1]
    return sf

def get_channel_mc_weight(e, sel, channel):
    """Channel-specific MC weight: HEM correction (all 0l) + MET trigger SF (0l, toggleable)."""
    if channel == "0l":
        w = HEM_weight_0l(e)[sel]
        if APPLY_MET_TRIG_SF:
            w = w * met_trig_sf(np.asarray(e["met"][sel], dtype=np.float64))
        return w
    return 1.0

# ============================================================================
# PRESELECTION CUTS
# ============================================================================
# ---------- BASE CUTS (common to all channels) ----------
# z = pT2/(pT1+pT2): select z<0.1 (sqrt(r)<0.333) OR z>0.5 (sqrt(r)>1, impossible since pT2<=pT1)
BASE_PRECUT_STR = "ak15_pt>200 & v_pt>150 & dphi_V_ak15>2.5 & ak15_sj1_pt>80 & ak15_sj2_pt>20 & (ak15_sj2_pt/(ak15_sj2_pt+ak15_sj1_pt)>0.1) & (ak15_sj2_pt/(ak15_sj2_pt+ak15_sj1_pt)<0.5)"
Base_Preselection = lambda e: (e["ak15_pt"] > 200) & (e["v_pt"] > 150) & (e["dphi_V_ak15"] > 2.5) & (e["ak15_sj1_pt"] > 80) & (e["ak15_sj2_pt"] > 20) & ((e["ak15_sj2_pt"]/(e["ak15_sj2_pt"]+e["ak15_sj1_pt"]) > 0.1) & (e["ak15_sj2_pt"]/(e["ak15_sj2_pt"]+e["ak15_sj1_pt"]) < 0.5)) #& TR_CUT["TR1"](e)


# ---------- TEST REGIONS (SDmass ranges for investigation) ----------
TR_CUT_STR = {
    "TR1": "ak15_sdmass> 0 & ak15_sdmass< 30",
    "TR2": "ak15_sdmass>30 & ak15_sdmass< 90",
    "TR3": "ak15_sdmass>90 & ak15_sdmass<170",
    "TR4": "ak15_sdmass>170",
    "TR5": "(ak15_sdmass<100 | ak15_sdmass>170)",  # sidebands
}
TR_CUT = {
    "TR1": lambda e: (e["ak15_sdmass"] > 0) & (e["ak15_sdmass"] < 30),
    "TR2": lambda e: (e["ak15_sdmass"] > 30) & (e["ak15_sdmass"] < 90),
    "TR3": lambda e: (e["ak15_sdmass"] > 90) & (e["ak15_sdmass"] < 170),
    "TR4": lambda e: (e["ak15_sdmass"] > 170),
    "TR5": lambda e: (e["ak15_sdmass"] < 100) | (e["ak15_sdmass"] > 170),  # sidebands
}

# HEM veto string for 0L ROOT TTreeFormula
# Using De Morgan's law to avoid negation operator which can cause TTreeFormula parsing issues in TMVA
# De Morgan: (year!=2018 || run<319077 || v_phi<=-1.66 || v_phi>=-0.54)
HEM_VETO_STR_0l = "(year!=2018 | run<319077 | v_phi<=-1.66 | v_phi>=-0.54)"

# ---------- PRECUT STRINGS (for ROOT TTreeFormula) ----------
PRECUT_STR_0l = BASE_PRECUT_STR + " & met>150 & passTrigMET & passmetfilters & " + HEM_VETO_STR_0l
PRECUT_STR_1l = BASE_PRECUT_STR + " & lep1_pt>25 & met>60 & (passTrigMu|passTrigMET|passTrigEl) & passmetfilters"
PRECUT_STR_2l = BASE_PRECUT_STR + " & v_mass>70 & v_mass<110 & (passTrigEl|passTrigMu) & passmetfilters"

# ---------- PRESELECTION LAMBDAS (for awkward arrays) ----------
Preselection_0l = lambda e: Base_Preselection(e) & (e["met"] > 150) & (e["passTrigMET"] > 0) & (e["passmetfilters"] > 0) & HEM_veto_0l(e)
Preselection_1l = lambda e: Base_Preselection(e) & (e["lep1_pt"] > 25) & (e["met"] > 60) & ((e["passTrigMu"] > 0) | (e["passTrigMET"] > 0) | (e["passTrigEl"] > 0)) & (e["passmetfilters"] > 0)
Preselection_2l = lambda e: Base_Preselection(e) & (e["v_mass"] > 70) & (e["v_mass"] < 110) & ((e["passTrigEl"] > 0) | (e["passTrigMu"] > 0)) & (e["passmetfilters"] > 0)

# 2l combo score quantile window applied on top of Preselection_2l.
# Comment out the line below to disable (no combo cut applied).
COMBO_Q_RANGE = [96,100]   # [lo%, hi%] wrt data — computed on the fly at runtime

# ---------- PRESELECTION LOOKUP DICTIONARIES ----------
PRECUT_STR = {"0l": PRECUT_STR_0l, "1l": PRECUT_STR_1l, "2l": PRECUT_STR_2l}

# Kinematic-only precuts safe for uproot cut= parameter.
# Trigger/flag branches (passTrigMET, passmetfilters, etc.) are stored as Float_t in ROOT
# and cause bitwise_and failures in awkward; full selection applied later via Preselection_*.
_UPROOT_PRECUT = {
    "0l": BASE_PRECUT_STR + " & met>150",
    "1l": BASE_PRECUT_STR + " & lep1_pt>25 & met>60",
    "2l": BASE_PRECUT_STR + " & v_mass>70 & v_mass<110",
}
PRESELECTION = {"0l": Preselection_0l, "1l": Preselection_1l, "2l": Preselection_2l}

# ---------- TEMPORARY EXTRA CUTS (comment out PRESELECTION_EXTRA lines to disable) ----------
_EXTRA_CUT_STR = "BDT_KIN>0.5 && BDT_ParT>0.5"
_EXTRA_CUT     = lambda e: (e["BDT_KIN"] > 0.5) & (e["BDT_ParT"] > 0.5)
PRESELECTION_EXTRA = {ch: (lambda e, base=f: base(e) & _EXTRA_CUT(e)) for ch, f in PRESELECTION.items()}

# Signal region (1l only for now - extend for other channels as needed)
SR = lambda e: Preselection_1l(e) & (e["ak15_ParTMDV2_Hgg"] > 0.1) & (e["flavBDT"]>0) & (e["kinBDT"]>0.52) & (e["ak15_deltaR_sj12"]<1.2) & (e["ak15_sdmass"] > 100) & (e["ak15_sdmass"] < 170)




# Var(name, bins, xlabel, channels, logy, cdf_flat) - channels/logy/cdf_flat have defaults
# Pass logy="F" (or cdf_flat=True) to apply a CDF flat-background transform at plot time.
_VarBase = namedtuple('_VarBase', ['name', 'bins', 'xlabel', 'channels', 'logy', 'cdf_flat'])
def Var(name, bins, xlabel, channels='', logy=0, cdf_flat=False):
    """Var factory. logy='F' is shorthand for cdf_flat='Data' (log scale stays off).
    cdf_flat accepts: False (off), True/'Data' (flat wrt data), 'MC' (flat wrt total MC), 'Signal' (flat wrt weighted signal)."""
    if logy in ('F', 'f'):
        logy, cdf_flat = 0, "Data"
    if cdf_flat is True:
        cdf_flat = "Data"
    return _VarBase(name, bins, xlabel, channels, logy, cdf_flat)


variables_to_plot = [
# ---- M-score input variables (15, channel-independent) ----
#Var("ak15_sdmass",                                                        (50, 0, 250),   "m_{SD}(H) [GeV]",                            "0l,1l,2l"),
#Var("ak15_tau21",                                                         (50, 0, 1),     "#tau_{21}(H)",                                "0l,1l,2l"),
#Var("ak15_tau32",                                                         (50, 0, 1),     "#tau_{32}(H)",                                "0l,1l,2l"),
#Var("ak15_tau32*ak15_tau21",                                              (50, 0, 1),     "#tau_{31}(H) = #tau_{32}#times#tau_{21}",     "0l,1l,2l"),
#Var("ak15_deltaR_sj12",                                                   (32, 0, 1.6),   "#DeltaR(sj_{1},sj_{2})",                     "0l,1l,2l"),
#Var("ak15_sj1_pt",                                                        (40, 0, 400),   "p_{T}(sj_{1}) [GeV]",                        "0l,1l,2l"),
#Var("ak15_sj2_pt",                                                        (40, 0, 300),   "p_{T}(sj_{2}) [GeV]",                        "0l,1l,2l"),
#Var("ak15_sj1_pt/(ak15_sj2_pt+1e-6)",                                    (50, 0, 10),    "p_{T}(sj_{1})/p_{T}(sj_{2})",                "0l,1l,2l"),
#Var("(ak15_sj1_pt-ak15_sj2_pt)/(ak15_sj1_pt+ak15_sj2_pt+1e-6)",         (50, 0, 1),     "p_{T} asymmetry sj_{1}/sj_{2}",              "0l,1l,2l"),
#Var("ak15_mass",                                                          (50, 0, 350),   "m_{AK15} [GeV]",                             "0l,1l,2l"),
#Var("jet_1_mass",                                                         (36, 0, 180),   "m(J_{1}^{AK4}) [GeV]",                       "0l,1l,2l"),
#Var("ak4_1_mass",                                                         (20, -1, 39),   "m(j_{2}^{AK4}) [GeV]",                       "0l,1l,2l"),
#Var("jet_1_pt",                                                           (30, 50, 350),  "p_{T}(J_{1}^{AK4}) [GeV]",                   "0l,1l,2l"),
#Var("ak15_pt",                                                            (30, 200, 500), "p_{T}(H) [GeV]",                             "0l,1l,2l"),
#Var("v_pt",                                                               (40, 150, 450), "p_{T}(V) [GeV]",                             "0l,1l,2l"),

#Var("BDT_ParT",        (40, 0, 1),      "ParT score BDT finetuned",              "0l,1l,2l"),
#Var("BDT_KIN",          (40, 0, 1),      "KIN score BDT",                         "0l,1l,2l"),
##Var("BDT_MASS",        (40, 50, 250),   "Regressed mass BDT finetuned",          "0l,1l,2l"),
#Var("BDT_Mscore",      (40, 0, 1),      "M-score BDT finetuned",              "0l,1l,2l"),

#Var("BDTf_ParT",        (40, 0, 1),      "ParT score BDT finetuned (flat)",  "0l,1l,2l"),
#Var("BDTf_KIN",          (40, 0, 1),      "KIN score BDT (flat)",              "0l,1l,2l"),
#Var("BDT_MASS",        (50, 0, 250),    "Regressed mass BDT [GeV]",            "0l,1l,2l"),
#Var("BDTf_Mscore",      (40, 0, 1),      "M-score BDT finetuned (flat)",      "0l,1l,2l"),

#Var("MLP_ParT",        (50, 0, 1),      "ParT score MLP finetuned",   "0l,1l,2l"),
#Var("MLP_KIN",          (40, 0, 1),      "KIN score MLP",                         "0l,1l,2l"),
#Var("MLP_MASS",        (40, 50, 250),   "Regreesed mass MLP finetuned","0l,1l,2l"),
#Var("MLP_Mscore",      (40, 0, 1),      "M-score MLP finetuned",    "0l,1l,2l"),

#Var("MLPf_ParT",       (50, 0, 1),      "ParT score MLP finetuned (flat)",   "0l,1l,2l"),
#Var("MLPf_KIN",         (40, 0, 1),      "KIN score MLP (flat)",              "0l,1l,2l"),
#Var("MLP_MASS",        (50, 0, 250),    "Regressed mass MLP [GeV]",            "0l,1l,2l"),
#Var("MLPf_Mscore",     (40, 0, 1),      "M-score MLP finetuned (flat)",    "0l,1l,2l"),
#Var("MLPf_Mscore",     (20, 0, 1),      "M-score MLP finetuned (flat)",    "0l,1l,2l", cdf_flat="MC"),
#Var("MLPf_Mscore",     (500, 0, 1),      "M-score MLP finetuned (flat)",    "0l,1l,2l"),

#Var("DNN_ParT",        (40, 0, 1),      "ParT score DNN finetuned",              "0l,1l,2l"),
#Var("DNN_KIN",          (40, 0, 1),      "KIN score DNN",                         "0l,1l,2l"),
#Var("DNN_MASS",        (40, 50, 250),   "Regressed mass DNN finetuned",          "0l,1l,2l"),
#Var("DNN_Mscore",      (40, 0, 1),      "M-score DNN finetuned",              "0l,1l,2l"),

#Var("DNNf_ParT",        (40, 0, 1),      "ParT score DNN finetuned (flat)",  "0l,1l,2l"),
#Var("DNNf_KIN",          (40, 0, 1),      "KIN score DNN (flat)",              "0l,1l,2l"),
#Var("DNN_MASS",         (50, 0, 250),    "Regressed mass DNN [GeV] ",            "0l,1l,2l"),
#Var("DNNf_Mscore",      (40, 0, 1),      "M-score DNN finetuned (flat)",  "0l,1l,2l"),

#Var("(BDT_KIN*BDT_ParT)**0.5", (40, 0, 1.),      "(BDT_{KIN} #times BDT_{ParT})^0.5", "0l,1l,2l"),
#Var("(BDTf_KIN*BDTf_ParT)**0.5", (40, 0, 1.),      "(BDTf_{KIN} #times BDTf_{ParT})^0.5", "0l,1l,2l"),
#Var("(2*BDTf_ParT+BDTf_KIN)/3", (50, 0, 1),      "(BDTf_{KIN}+2BDTf_{ParT})/3", "0l,1l,2l"),

#Var("(2*BDTf_Mscore+BDTf_KIN)/3", (50, 0, 1),      "(BDTf_{KIN}+2BDTf_{Mscore})/3", "0l,1l,2l"),
#Var("(BDTf_ParT+BDTf_Mscore)/2", (50, 0, 1),      "(BDTf_{Mscore}+BDTf_{ParT})/2", "0l,1l,2l"),
#Var("(2*BDTf_ParT+2*BDTf_Mscore+BDTf_KIN)/5", (50, 0, 1), "(2BDTf_{ParT}+2BDTf_{Mscore}+BDTf_{KIN})/5", "0l,1l,2l", cdf_flat=True),

#Var("(2.8*MLPf_ParT+MLPf_KIN)/3.8", (100, 0., 1),      "(MLPf_{KIN}+2.8MLPf_{ParT})/3.8", "0l,1l,2l"),
#Var("(2.8*MLPf_ParT+MLPf_KIN)/3.8", (100, 0., 1),      "(MLPf_{KIN}+2.8MLPf_{ParT})/3.8", "0l,1l,2l",cdf_flat="Data"),

#Var("MLP_X",  (50, 0., 1), "X MLP (Par T &KIN combined)",      "0l,1l,2l"),
#Var("MLPf_X", (50, 0., 1), "X MLP (ParT & KIN combined, flat)", "0l,1l,2l", cdf_flat="Data"),
#Var("MLPf_X", (50, 0., 1), "X MLP (ParT & KIN combined, flat)", "0l,1l,2l",),
#Var("MLPf_X",                     (50, 0.,   1.),   "X-score MLP (flat)",           "0l,1l,2l"),
# Duplicated variables carry a "*1" / "*1*1" factor purely so the sanitised output filename
# differs (MLPf_Mscore, MLPf_Mscore1, ...) — the plotted values are unchanged.
Var("MLPf_Mscore",               (50, 0.,   1.),   "M-score MLP (flat)",           "0l,1l,2l"),
Var("MLPf_Mscore*1",             (50, 0.75, 1.),   "M-score MLP (flat)",           "0l,1l,2l"),
Var("ak15_sdmass",                (50, 0,  250),    "m_{SD}(H) [GeV]",              "0l,1l,2l"),
Var("ak15_sdmass*1",              (20, 100, 200),   "m_{SD}(H) [GeV]",              "0l,1l,2l"),
Var("1-abs(ak15_sdmass-133)/133",(50, 0.,   1.),   "1-|m_{SD}-133|/133",           "0l,1l,2l"),
Var("1-abs(ak15_sdmass-133)/133",(30, 0.4,   1.),   "1-|m_{SD}-133|/133",           "0l,1l,2l"),


#Var("MLP_X",  (100, 0., 1), "X MLP (signal-flat)", "0l,1l,2l", cdf_flat="Signal"),
#Var("MLPf_X", (100, 0., 1), "X MLP (flat, signal-flat)", "0l,1l,2l", cdf_flat="Signal"),

#Var("(2*MLPf_Mscore+MLPf_KIN)/3", (40, 0, 1),      "(MLPf_{KIN}+2MLPf_{Mscore})/3", "0l,1l,2l"),
#Var("(MLPf_ParT+MLPf_Mscore)/2", (40, 0, 1),      "(MLPf_{Mscore}+MLPf_{ParT})/2", "0l,1l,2l"),
#Var("(2.8*MLPf_ParT+2.2*MLPf_Mscore+MLPf_KIN)/6", (20, 0, 1), "(2.8MLPf_{ParT}+2.2MLPf_{Mscore}+MLPf_{KIN})/6", "0l,1l,2l"),

#Var("(2*MLPf_ParT+MLPf_KIN)/3.0", (40, 0, 1),      "(MLPf_{KIN}+2MLPf_{ParT})/3", "0l,1l,2l",cdf_flat=True),
#Var("(2*MLPf_Mscore+MLPf_KIN)/3.0", (40, 0, 1),      "(MLPf_{KIN}+2MLPf_{Mscore})/3", "0l,1l,2l",cdf_flat=True),
#Var("(MLPf_ParT+MLPf_Mscore)/2.0", (40, 0, 1),      "(MLPf_{Mscore}+MLPf_{ParT})/2", "0l,1l,2l",cdf_flat=True),
#Var("(2*MLPf_ParT+2*MLPf_Mscore+MLPf_KIN)/5.0", (40, 0, 1), "(2MLPf_{ParT}+2MLPf_{Mscore}+MLPf_{KIN})/5", "0l,1l,2l", cdf_flat=True),

#Var("(2*DNNf_ParT+DNNf_KIN)/3", (40, 0, 1),      "(DNNf_{KIN}+2DNNf_{ParT})/3", "0l,1l,2l"),
#Var("(2*DNNf_Mscore+DNNf_KIN)/3", (40, 0, 1),      "(DNNf_{KIN}+2DNNf_{Mscore})/3", "0l,1l,2l"),
#Var("(DNNf_ParT+DNNf_Mscore)/2", (40, 0, 1),      "(DNNf_{Mscore}+DNNf_{ParT})/2", "0l,1l,2l"),
#Var("(2*DNNf_ParT+2*DNNf_Mscore+DNNf_KIN)/5", (40, 0, 1), "(2DNNf_{ParT}+2DNNf_{Mscore}+DNNf_{KIN})/5", "0l,1l,2l", cdf_flat=True),

#Var("(2*BDTf_ParT+BDTf_KIN)/3", (40, 0, 1),      "(BDTf_{KIN}+2BDTf_{ParT})/3", "0l,1l,2l", cdf_flat=True),
#Var("(BDTf_ParT+BDTf_KIN)/2", (40, 0, 1),      "(BDTf_{KIN}+BDTf_{ParT})/2", "0l,1l,2l"),
#Var("(BDTf_ParT+BDTf_KIN)*0.5", (40, 0, 1),      "0.5(BDTf_{KIN}+BDTf_{ParT})", "0l,1l,2l"),
#Var("((BDT_KIN**2+BDT_ParT**2)/2)**0.5"  , (40, 0, 1.),      "[(BDT_{KIN}^2+BDT_{ParT}^2)/2]^0.5", "0l,1l,2l"),
#Var("2*BDT_KIN*BDT_ParT/(BDT_KIN+BDT_ParT+1e-6)", (40, 0, 1), "Harmonic mean 2*BDT_{KIN}*BDT_{ParT}/(BDT_{KIN}+BDT_{ParT})", "0l,1l,2l"),

#Var("(MLP_KIN*MLP_ParT)**0.5",           (40, 0, 1.),  "(MLP_{KIN} #times MLP_{ParT})^{0.5}", "0l,1l,2l"),
#Var("(MLP_ParT+MLP_KIN)*0.5",            (40, 0, 1),   "0.5(MLP_{KIN}+MLP_{ParT})",           "0l,1l,2l"),
#Var("((MLP_KIN**2+MLP_ParT**2)/2)**0.5", (40, 0, 1.),  "[(MLP_{KIN}^2+MLP_{ParT}^2)/2]^0.5", "0l,1l,2l"),

#Var("(DNN_KIN*DNN_ParT)**0.5",           (40, 0, 1.),  "(DNN_{KIN} #times DNN_{ParT})^{0.5}", "0l,1l,2l"),
#Var("(DNN_ParT+DNN_KIN)*0.5",            (40, 0, 1),   "0.5(DNN_{KIN}+DNN_{ParT})",           "0l,1l,2l"),
#Var("((DNN_KIN**2+DNN_ParT**2)/2)**0.5", (40, 0, 1.),  "[(DNN_{KIN}^2+DNN_{ParT}^2)/2]^0.5", "0l,1l,2l"),

# Radial proximity to signal peak (BDT_ParT=0.9, BDT_KIN=0.8); score=1 at peak, ~0 at origin
#Var("1 - ((BDT_ParT-0.9)**2 + (BDT_KIN-0.8)**2)**0.5 / 1.2042", (40, 0, 1.), "1 - d_{sig}(BDT_{ParT}, BDT_{KIN})", "0l,1l,2l"),
#Var("1 - ((MLP_ParT-0.9)**2 + (MLP_KIN-0.8)**2)**0.5 / 1.2042", (40, 0, 1.), "1 - d_{sig}(MLP_{ParT}, MLP_{KIN})", "0l,1l,2l"),
#Var("1 - ((DNN_ParT-0.9)**2 + (DNN_KIN-0.8)**2)**0.5 / 1.2042", (40, 0, 1.), "1 - d_{sig}(DNN_{ParT}, DNN_{KIN})", "0l,1l,2l"),

#Var("ak15_sdmass",      (50, 0, 250),    "m_{SD}(H) [GeV]",  "0l,1l,2l"),  # for display
#Var("ak15_mass",        (50, 0, 250),    "m_{ungroomed} [GeV]",  "0l,1l,2l"),
#Var("ak15_regressed_mass",        (50, 0, 250),    "m_{regressed} [GeV]",  "0l,1l,2l"),

#Var("1-abs(ak15_sdmass - 135)/135",    (40, 0, 1),  " 1-|ak15_sdmass - 135|/135",  "0l,1l,2l"),  # for display
#Var("1-abs(ak15_sdmass - 133)/133",    (50, 0, 1),  " 1-|ak15_sdmass - 133|/133",  "0l,1l,2l", cdf_flat=True),  # for display
#Var("1-abs(BDT_MASS - 125)/125",       (40, 0.3, 1),  " 1-|BDT_MASS - 125|/125",  "0l,1l,2l"),  # for display

# ── SDmass/DR symmetrized variables (signal near 1, background near 0) ──────
#Var("1-abs(ak15_sdmass-133)/133",                                                         (50, 0, 1),   "1-|m_{SD}-133|/133",                               "0l,1l,2l"),
#Var("1-abs(ak15_deltaR_sj12-0.9)/1.8",                                                    (50, 0, 1),   "1-|#DeltaR(sj_{1},sj_{2})-0.9|/1.8",               "0l,1l,2l"),
#Var("0.5*(1-abs(ak15_sdmass-133)/133) + 0.5*(1-abs(ak15_deltaR_sj12-0.9)/1.8)",                (50, 0, 1),   "0.5|m_{SD}-133|/133 + 0.5|#DeltaR-0.9|/1.8",    "0l,1l,2l"),
#Var("((1-abs(ak15_sdmass-133)/133)**2 + (1-abs(ak15_deltaR_sj12-0.9)/1.8)**2)**0.5",                (50, 0, 1),   "0.5|m_{SD}-133|/133 + 0.5|#DeltaR-0.9|/1.8",    "0l,1l,2l"),

#Var("abs(ak15_sdmass-133)/133 + 0.5*abs(ak15_deltaR_sj12-0.9)/1.8",                (50, 0, 1),   "0.5|m_{SD}-133|/133 + 0.5|#DeltaR-0.9|/1.8",    "0l,1l,2l"),


#Var("1-abs(BDT_MASS - 126)/126",    (40, 0, 1),  " 1-|BDT_MASS - 126|/126",  "0l,1l,2l", cdf_flat=True),  # for display


# ---------- HEM15/16 check: phi and eta of relevant objects per channel ----------
# 0l: ak15 jet + MET (v_phi = met_phi in 0l);  1l/2l: ak15 jet + leading lepton
#Var("ak15_eta",   (40, -2.6,  2.6),  "#eta(H)",         "0l,1l,2l"),
#Var("ak15_phi",   (40, -3.2,  3.2),  "#phi(H)",         "0l,1l,2l"),
#Var("v_phi",      (32, -3.15, 3.15), "#phi(MET)",       "0l"),        # MET phi (v = MET in 0l)
#Var("lep1_eta",   (40, -2.6,  2.6),  "#eta(l1)",        "1l,2l"),
#Var("lep1_phi",   (32, -3.15, 3.15), "#phi(l1)",        "1l,2l"),
# ---------- end HEM check ----------

#Var("ak15_mass*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 250),    "m #times ParT [GeV]",  "0l,1l,2l"),
#Var("ak15_sdmass*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 250),    "m_{SD} #times ParT [GeV]",  "0l,1l,2l"),
#Var("ak15_regressed_mass*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 250),    "m_{reg} #times ParT [GeV]",  "0l,1l,2l"),  # if double correction needed

#Var("ak15_mass*(1-ak15_rawFactor)",  (50, 0, 250),    "m_{raw} [GeV]",  "0l,1l,2l"),  # raw ungroomed (undo JEC)
#Var("ak15_mass*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 250),    "m_{corr} #times ParT [GeV]",  "0l,1l,2l"),  # corrected + ParT regression
#Var("ak15_sdmass*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 250),    "m_{SD} #times ParT [GeV]",  "0l,1l,2l"),  # soft-drop + ParT regression
#Var("ak15_regressed_mass*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 250),    "m_{reg} #times ParT [GeV]",  "0l,1l,2l"),  # if double correction needed

#Var("ak15_ParTMDV2_visiableMassCorr/ak15_ParTMDV2_resonanceMassCorr",  (50, 0.8, 1.1),    "/",  "0l,1l,2l"),
#Var("ak15_ParTMDV2_visiableMassCorr*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 2),    "*",  "0l,1l,2l"),

#Var("ak15_ParTMDV2_visiableMassCorr",  (50, 0, 2),    "ak15_ParTMDV2_visiableMassCorr",  "0l,1l,2l"),
#Var("1/ak15_ParTMDV2_visiableMassCorr",  (50, 0, 2),    "1/ak15_ParTMDV2_visiableMassCorr",  "0l,1l,2l"),
#Var("(1-ak15_rawFactor)",  (50, 0.8, 1),    "(1-rawF)",  "0l,1l,2l"),
#Var("1/ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 2),    "1/resMassCorr",  "0l,1l,2l"),
#Var("1*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 2),    "resMassCorr",  "0l,1l,2l"),
#Var("(1-ak15_rawFactor)/ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 2),    "(1-rawFac)/resMassCorr",  "0l,1l,2l"),
#Var("(1-ak15_rawFactor)*ak15_ParTMDV2_resonanceMassCorr",  (50, 0, 2),    "(1-rawFac)*resMassCorr",  "0l,1l,2l"),

#Var("ak15_sdmass",      (50, 0, 250),    "m_{SD}(H) [GeV]",  "0l,1l,2l"),  # for display
#Var("ak15_mass",      (50, 0, 250),    "ak15_mass [GeV]",  "0l,1l,2l"),  # for display
#Var("ak15_regressed_mass",      (50, 0, 250),    "ak15_regressed_mass [GeV]",  "0l,1l,2l"),  # for display
#Var("ak15_mass/ak15_sdmass",      (50, 0, 2),    "ak15_mass/ak15_sdmass",  "0l,1l,2l"),  # for display
#Var("ak15_regressed_mass/ak15_sdmass",      (50, 0, 2),    "ak15_regressed_mass/ak15_sdmass",  "0l,1l,2l"),  # for display
#Var("ak15_regressed_mass/ak15_regressed_mass",      (50, 0, 2),    "ak15_regressed_mass/ak15_regressed_mass",  "0l,1l,2l"),  # for display

#  Var("ak15_pt",           (30, 200, 500),  "p_{T}(H) [GeV]",       "0l,1l,2l"),
#Var("ak15_sdmass",      (50, 0, 250),    "m_{SD}(H) [GeV]",  "0l,1l,2l"),  # for display
#  Var("ak15_abseta",       (27, -0.1, 2.6), "|#eta(H)|",            "0l,1l,2l"),
#  Var("ak15_phi",         (32, -3.15, 3.15), "#phi(H)","0l,1l,2l"),
#  Vaar("jet_1_pt",          (30, 50, 350),   "p_{T}(J1) [GeV]",      "0l,1l,2l"),
#  Var("jet_1_mass",        (36, 0, 180),    "m(J1) [GeV]",          "0l,1l,2l", 1),
#  #Var("abs(jet_1_eta)",   (27,-0.1, 2.6),  "|#eta(J1)|",           "1l,2l"),  # not in training
#  #Var("jet_1_phi",        (32, -3.15, 3.15), "#phi(J1)"),
#  Var("ak4_1_pt",         (30, -10, 140),  "p_{T}(j1) [GeV]",      "1l,2l", 1),
#  Var("ak4_1_mass",       (20, -1, 39),    "m(j1) [GeV]",          "0l,1l,2l", 1),
#  #Var("abs(ak4_1_eta)",   (47, -0.1, 4.6), "|#eta(j1)|",           "1l,2l", 1),  # not in training
#  #Var("ak4_1_phi",        (32, -3.15, 3.15), "#phi(j1)"),
#  Var("lep1_pt",          (50, 0, 300),    "p_{T}(l1) [GeV]",      "1l,2l"),
#  #Var("lep1_mass",        (30, 0, 0.15),   "m(l1) [GeV]",          "1l,2l"),  # not in training
#  #Var("abs(lep1_eta)",    (27, -1, 2.5),   "|#eta(l1)|",           "1l,2l"),  # use lep1_eta directly
#  Var("lep1_phi",         (32, -3.15, 3.15), "#phi(l1)"),  # not in training
#  Var("lep2_pt",          (50, 0, 200),    "p_{T}(l2) [GeV]",      "2l"),
#  #Var("lep2_mass",        (30, 0, 0.15),   "m(l2) [GeV]",          "2l"),  # not in training
#  #Var("abs(lep2_eta)",    (27, -0.1, 2.5), "|#eta(l2)|",           "2l"),  # use lep2_eta directly
#  #Var("lep2_phi",         (32, -3.15, 3.15), "#phi(l2)"),
#  Var("met",               (50, 0, 500),    "MET [GeV]",            "0l,1l,2l"),
#  Var("(met+ak15_sdmass)/(v_pt+ak15_pt)",              (40, 0, 0.4),  "(mH+MET)/(p_{T}(V)+p_{T}(H))",       "0l,1l,2l"),
#  Var("met_phi",          (32, -3.15, 3.15), "#phi(MET)",          "0l,1l,2l"),  # not in training
#  Var("v_pt",              (40, 150, 450),  "p_{T}(V) [GeV]",       "0l,1l,2l"),
#  Var("v_mass",            (30, 75, 105),  "(v_mass - m_{Z}) [GeV]","2l"),
#  #Var("abs(v_eta)",       (27, -0.1, 2.6), "|#eta(V)|",            "1l,2l"),  # not in training
#  Var("v_phi",            (32, -3.15, 3.15), "#phi(V)","0l,1l,2l"),
#  Var("jet_1_pt/ak15_pt",  (50, 0, 1.5),    "p_{T}(J1)/p_{T}(H)",   "0l,1l,2l"),
#  Var("ak4_1_pt/ak15_pt", (50, 0, 1),      "p_{T}(j1)/p_{T}(H)",   "1l,2l"),
#  Var("lep1_pt/ak15_pt",  (50, 0, 1),      "p_{T}(l1)/p_{T}(H)",   "1l,2l"),
#  Var("lep2_pt/ak15_pt",  (50, 0, 0.5),    "p_{T}(l2)/p_{T}(H)",   "2l"),
#  Var("met/ak15_pt",      (50, 0, 1),    "MET/p_{T}(H)",         "0l,1l,2l"),
#  Var("v_pt/ak15_pt",      (20, 0.5, 1.5),  "p_{T}(V)/p_{T}(H)",    "1l,2l"),
#  #Var("ak4_1_pt/jet_1_pt",(50, 0, 1),      "p_{T}(j1)/p_{T}(J1)",  "1l,2l"),  # not in training
#  #Var("lep1_pt/jet_1_pt", (50, 0, 1),      "p_{T}(l1)/p_{T}(J1)",  "1l,2l"),  # not in training
#  Var("lep2_pt/jet_1_pt", (50, 0, 0.5),    "p_{T}(l2)/p_{T}(J1)",  "2l"),
#  Var("met/jet_1_pt",      (50, 0.5, 2.5),      "MET/p_{T}(J1)",        "0l,1l,2l"),
#  Var("met/jet_1_mass",      (50, 2, 7),      "MET/m(J1)",        "0l,1l,2l"),
#  Var("(met+ak15_sdmass)/jet_1_pt",      (50, 0, 5),      "MET+m(H)/jet_1_pt",        "0l,1l,2l"),
#  Var("v_pt/jet_1_pt",    (50, 0, 2),      "p_{T}(V)/p_{T}(J1)",   "1l"),
#  Var("ak4_1_pt/v_pt",    (50, 0, 1),      "p_{T}(j1)/p_{T}(V)",   "1l"),
#  Var("lep1_pt/v_pt",     (50, 0, 1),      "p_{T}(l1)/p_{T}(V)",   "1l,2l"),
#  Var("lep2_pt/v_pt",     (50, 0, 0.5),    "p_{T}(l2)/p_{T}(V)",   "2l"),
#  Var("met/v_pt",         (50, 0, 1.5),    "MET/p_{T}(V)",         "1l,2l"),
#  #Var("ak4_1_pt/met",     (50, 0, 1.5),    "p_{T}(j1)/MET",        "1l,2l"),  # not in training
#  #Var("lep1_pt/met",      (50, 0, 1.5),    "p_{T}(l1)/MET",        "1l,2l"),  # not in training
#  #Var("lep2_pt/met",      (50, 0, 1),      "p_{T}(l2)/MET",        "2l"),  # not in training
#  Var("lep1_pt/ak4_1_pt", (25, 0, 5),      "p_{T}(l1)/p_{T}(j1)",  "1l"),
#  #Var("lep2_pt/ak4_1_pt", (50, 0, 1.5),    "p_{T}(l2)/p_{T}(j1)",  "2l"),  # not in training
#  Var("lep2_pt/lep1_pt",  (50, 0, 1),      "p_{T}(l2)/p_{T}(l1)",  "2l"),
#  #
#  #Var("(lep1_pt + lep2_pt)/met",     (50, 0, 3),   "(p_{T}(l1) + p_{T}(l2))/MET",        "1l,2l"),  # not in training
#  Var("(lep1_pt + lep2_pt)/v_pt",    (50, 0, 1.5), "(p_{T}(l1) + p_{T}(l2))/p_{T}(V)",   "2l"),
#  Var("(lep1_pt + lep2_pt)/ak15_pt", (50, 0, 1),   "(p_{T}(l1) + p_{T}(l2))/p_{T}(H)",   "2l"),
#  #Var("ak15_pt + lep1_pt + lep2_pt + met", (50, 300, 1200), "p_{T}(H) + p_{T}(l1) + p_{T}(l2) + MET [GeV]", "1l,2l"),  # not in training
#  Var("ak15_pt + v_pt",    (25, 350, 850),  "p_{T}(H) + p_{T}(V) [GeV]", "0l,1l,2l"),
#  Var("ak15_tau21",        (50, 0, 1),      "#tau_{21}(H)",         "0l,1l,2l"),
#  Var("ak15_tau32",        (50, 0, 1),      "#tau_{32}(H)",         "0l,1l,2l"),
#  Var("ak15_tau32*ak15_tau21", (50, 0, 1), "#tau_{31}(H)",           "0l,1l,2l"),
#  Var("ak15_deltaR_sj12",  (32, 0, 1.6),    "#DeltaR(sj1,sj2)",     "0l,1l,2l"),
#  #Var("2*ak15_sdmass/ak15_pt", (25, 0, 2),  "2m_{SD}(H)/p_{T}(H)",  "0l,1l,2l"),  # ak15_sdmass not in training
#  Var("2*jet_1_mass/jet_1_pt", (25, 0, 0.6), "2m(J1)/p_{T}(J1)",    "0l,1l,2l"),
#  #Var("2*ak4_1_mass/ak4_1_pt", (25, 0, 0.6), "2m(j1)/p_{T}(j1)",    "1l,2l"),  # not in training
#  
#  ## Deltaeta variables - keep only those used in training
#  Var("ak15_eta", (40, -2.6, 2.6), "#eta(H)", "0l,1l,2l"),
#  Var("ak15_phi", (40, -3.2, 3.2), "#phi(H)", "0l,1l,2l"),  
#  Var("abs(ak15_eta - jet_1_eta)", (20, 0, 0.5), "|#Delta#eta(H, J1)|", "0l,1l,2l"),
#  #Var("abs(ak15_eta - ak4_1_eta)", (50, 0, 5),   "|#Delta#eta(H, j1)|", "1l,2l"),  # not in training
#  #Var("abs(ak15_eta - lep1_eta)",  (50, 0, 5),   "|#Delta#eta(H, l1)|", "1l,2l"),  # not in training
#  #Var("abs(ak15_eta - lep2_eta)",  (50, 0, 5),   "|#Delta#eta(H, l2)|", "1l,2l"),  # not in training
#  #Var("abs(ak15_eta - v_eta)",     (20, 2.4, 3.15), "|#Delta#eta(H, V)|"),  # v_eta-ak15_eta in training
#  #Var("abs(jet_1_eta - ak4_1_eta)",(50, 0, 5),   "|#Delta#eta(J1, j1)|", "1l,2l"),  # not in training
#  Var("abs(jet_1_eta - lep1_eta)", (50, 0, 5),   "|#Delta#eta(J1, l1)|", "1l"),
#  Var("abs(jet_1_eta - lep2_eta)", (50, 0, 5),   "|#Delta#eta(J1, l2)|", "2l"),
#  Var("abs(jet_1_eta - v_eta)",    (30, 0, 3),   "|#Delta#eta(J1, V)|",  "0l,1l"),
#  #Var("abs(ak4_1_eta - lep1_eta)", (50, 0, 5),   "|#Delta#eta(j1, l1)|", "1l,2l"),  # not in training
#  #Var("abs(ak4_1_eta - lep2_eta)", (50, 0, 5),   "|#Delta#eta(j1, l2)|", "1l,2l"),  # not in training
#  #Var("abs(ak4_1_eta - v_eta)",    (50, 0, 5),   "|#Delta#eta(j1, V)|",  "1l,2l"),  # not in training
#  Var("abs(lep1_eta - lep2_eta)",  (50, 0, 4),   "|#Delta#eta(l1, l2)|", "2l"),
#  Var("abs(lep1_eta - v_eta)",     (50, 0, 4),   "|#Delta#eta(l1, V)|",  "1l"),
#  #Var("abs(lep2_eta - v_eta)",     (50, 0, 4),   "|#Delta#eta(l2, V)|",  "2l"),  # not in training
#  ## Deltaphi variables - keep only those used in training
  # NOTE: use arccos(cos(dphi)) instead of (dphi+PI)%(2*PI)-PI — numexpr does not support % operator
#  Var("arccos(cos(ak15_phi - jet_1_phi))",  (20, 0, .5),    "|#Delta#phi(H, J1)|",    "0l,1l,2l"),
  #Var("arccos(cos(ak15_phi - ak4_1_phi))", (50, 0, 3.15),  "|#Delta#phi(H, j1)|",    "1l,2l"),  # not in training
  #Var("arccos(cos(ak15_phi - lep1_phi))",  (50, 0, 3.15),  "|#Delta#phi(H, l1)|",    "1l,2l"),  # not in training
  #Var("arccos(cos(ak15_phi - lep2_phi))",  (50, 0, 3.15),  "|#Delta#phi(H, l2)|",    "1l,2l"),  # not in training
#  Var("arccos(cos(ak15_phi - v_phi))",      (20, 2.5, 3.15),"|#Delta#phi(H, V)|",      "0l,1l,2l"),
  #Var("arccos(cos(jet_1_phi - ak4_1_phi))",(50, 0, 3.15),  "|#Delta#phi(J1, j1)|"),              # not in training
#  Var("arccos(cos(jet_1_phi - lep1_phi))",  (50, 0, 3.15),  "|#Delta#phi(J1, l1)|",   "1l"),
  #Var("arccos(cos(jet_1_phi - lep2_phi))", (50, 0, 3.15),  "|#Delta#phi(J1, l2)|",   ""),       # not in training
#  Var("arccos(cos(jet_1_phi - v_phi))",     (20, 2.4, 3.15),"|#Delta#phi(J1, V)|",    "0l,1l,2l"),
  #Var("arccos(cos(ak4_1_phi - lep1_phi))", (50, 0, 3.15),  "|#Delta#phi(j1, l1)|",   "1l,2l"),  # not in training
  #Var("arccos(cos(ak4_1_phi - lep2_phi))", (50, 0, 3.15),  "|#Delta#phi(j1, l2)|",   "1l,2l"),  # not in training
#  Var("arccos(cos(ak4_1_phi - v_phi))",     (50, 0, 3.15),  "|#Delta#phi(j1, V)|",    "0l,1l,2l"),
#  Var("arccos(cos(lep1_phi - lep2_phi))",   (50, 0, 3.15),  "|#Delta#phi(l1, l2)|",   "2l"),
#  Var("arccos(cos(lep1_phi - v_phi))",      (50, 0, 3.15),  "|#Delta#phi(l1, V)|",    "1l"),
  #Var(f"abs((lep2_phi - v_phi + {PI}) % (2*{PI}) - {PI})",     (50, 0, 3.15), "|#Delta#phi(l2, V)|",  ""),  # not in training
  #Var(f"abs((ak15_phi - met_phi + {PI}) % (2*{PI}) - {PI})",   (20, 0, 1), "|#Delta#phi(H, MET)|",   "1l,2l"),  # not in training
  #Var(f"abs((jet_1_phi - met_phi + {PI}) % (2*{PI}) - {PI})",  (20, 0, 1), "|#Delta#phi(J1, MET)|",  "1l,2l"),  # not in training
  #Var(f"abs((ak4_1_phi - met_phi + {PI}) % (2*{PI}) - {PI})",  (50, 0, 3.15), "|#Delta#phi(j1, MET)|", "1l,2l"),  # not in training
  #Var(f"abs((lep1_phi - met_phi + {PI}) % (2*{PI}) - {PI})",   (50, 0, 3.15), "|#Delta#phi(l1, MET)|", "1l,2l"),  # not in training
  #Var(f"abs((lep2_phi - met_phi + {PI}) % (2*{PI}) - {PI})",   (50, 0, 3.15), "|#Delta#phi(l2, MET)|", "1l,2l"),  # not in training
  #Var(f"abs((v_phi - met_phi + {PI}) % (2*{PI}) - {PI})",      (20, 0, 1), "|#Delta#phi(V, MET)|",   "1l,2l"),  # not in training
  #
#  ## DeltaR variables - keep only those used in training (use dR^2 in training, but show sqrt for plots)
#  #Var(f"sqrt((ak15_eta - jet_1_eta)**2 + ((ak15_phi - jet_1_phi + {PI}) % (2*{PI}) - {PI})**2)", (20, 0, 1), "#DeltaR(H, J1)", "0l,1l,2l"),  # use deta/dphi instead
#  #Var(f"sqrt((ak15_eta - ak4_1_eta)**2 + ((ak15_phi - ak4_1_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(H, j1)", "1l,2l"),  # not in training
#  Var(f"sqrt((ak15_eta - lep1_eta)**2 + ((ak15_phi - lep1_phi + {PI}) % (2*{PI}) - {PI})**2)", (20, 2, 4), "#DeltaR(H, l1)", "2l"),
#  Var(f"sqrt((ak15_eta - lep2_eta)**2 + ((ak15_phi - lep2_phi + {PI}) % (2*{PI}) - {PI})**2)", (20, 2, 4), "#DeltaR(H, l2)", "2l"),
#  Var(f"sqrt((ak15_eta - v_eta)**2 + ((ak15_phi - v_phi + {PI}) % (2*{PI}) - {PI})**2)", (30, 2.5, 4), "#DeltaR(H, V)", "0l,1l,2l"),
#  #Var(f"sqrt((jet_1_eta - ak4_1_eta)**2 + ((jet_1_phi - ak4_1_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(J1, j1)", "1l,2l"),  # not in training
#  #Var(f"sqrt((jet_1_eta - lep1_eta)**2 + ((jet_1_phi - lep1_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(J1, l1)", ""),  # not in training
#  #Var(f"sqrt((jet_1_eta - lep2_eta)**2 + ((jet_1_phi - lep2_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(J1, l2)", ""),  # not in training
#  Var(f"sqrt((jet_1_eta - v_eta)**2 + ((jet_1_phi - v_phi + {PI}) % (2*{PI}) - {PI})**2)", (20, 2, 4), "#DeltaR(J1, V)", "0l,1l"),
#  #Var(f"sqrt((ak4_1_eta - lep1_eta)**2 + ((ak4_1_phi - lep1_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(j1, l1)", "1l,2l"),  # not in training
#  #Var(f"sqrt((ak4_1_eta - lep2_eta)**2 + ((ak4_1_phi - lep2_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(j1, l2)", "2l"),  # not in training
#  #Var(f"sqrt((ak4_1_eta - v_eta)**2 + ((ak4_1_phi - v_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(j1, V)", "1l,2l"),  # not in training
#  #Var(f"sqrt((lep1_eta - lep2_eta)**2 + ((lep1_phi - lep2_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(l1, l2)", ""),  # use deltaR_ll instead
#  #Var(f"sqrt((lep1_eta - v_eta)**2 + ((lep1_phi - v_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(l1, V)", ""),  # not in training
#  #Var(f"sqrt((lep2_eta - v_eta)**2 + ((lep2_phi - v_phi + {PI}) % (2*{PI}) - {PI})**2)", (50, 0, 5), "#DeltaR(l2, V)", ""),  # not in training
#  ## Mass ratios - comment out those using ak15_sdmass or lep masses (not in training)
#  #Var("jet_1_mass/ak15_sdmass", (34, 0, 1.7), "m_{SD}(H)/m(J1)", "0l,1l,2l"),  # ak15_sdmass not in training
#  #Var("ak4_1_mass/ak15_sdmass",  (50, 0, 0.5),  "m(j1)/m_{SD}(H)", "1l,2l"),  # ak15_sdmass not in training
#  #Var("ak4_1_mass/jet_1_mass",   (50, 0, 0.5),  "m(j1)/m(J1)",     "1l,2l"),  # not in training
#  #Var("ak4_1_mass/v_mass",       (50, 0, 0.3),  "m(j1)/m(V)",      "1l,2l"),  # not in training
#  #Var("ak15_sdmass/v_mass",      (50, 0, 2),    "m_{SD}(H)/m(V)",  "1l,2l"),  # ak15_sdmass not in training
#  #Var("jet_1_mass/v_mass",       (50, 0, 2),    "m(J1)/m(V)",      "1l,2l"),  # not in training
#  #Var("lep1_mass/ak15_sdmass",   (50, 0, 0.01), "m(l1)/m_{SD}(H)", "1l,2l"),  # not in training
#  #Var("lep1_mass/jet_1_mass",    (50, 0, 0.01), "m(l1)/m(J1)",     "1l,2l"),  # not in training
#  #Var("lep1_mass/ak4_1_mass",    (50, 0, 0.05), "m(l1)/m(j1)",     "1l,2l"),  # not in training
#  #Var("lep1_mass/v_mass",        (50, 0, 0.01), "m(l1)/m(V)",      "1l,2l"),  # not in training
#  #Var("lep2_mass/ak15_sdmass",   (50, 0, 0.01), "m(l2)/m_{SD}(H)", "2l"),  # not in training
#  #Var("lep2_mass/jet_1_mass",    (50, 0, 0.01), "m(l2)/m(J1)",     "2l"),  # not in training
#  #Var("lep2_mass/ak4_1_mass",    (50, 0, 0.05), "m(l2)/m(j1)",     "2l"),  # not in training
#  #Var("lep2_mass/v_mass",        (50, 0, 0.01), "m(l2)/m(V)",      "2l"),  # not in training
#  #Var("lep2_mass/lep1_mass",     (50, 0, 2),    "m(l2)/m(l1)",     "2l"),  # not in training
#  #
#  #Var("np.sqrt(2 * lep1_pt * met * (1 - np.cos(lep1_phi - met_phi)))", (50, 0, 200), "m_{T}(l1, MET) [GeV]", "1l,2l"),  # use mt_lep_met branch
#  #Var("np.sqrt(2 * lep2_pt * met * (1 - np.cos(lep2_phi - met_phi)))", (50, 0, 150), "m_{T}(l2, MET) [GeV]", "2l"),  # not in training
#  #Var("np.sqrt(2 * (lep1_pt + lep2_pt) * met * (1 - np.cos(v_phi - met_phi)))", (50, 0, 300), "m_{T}(l1+l2, MET) [GeV]", "2l"),  # not in training
#  #
#  Var("lep1_pdgId", (30, -15, 15), "PDG ID (l1)", "1l"),
#  #Var("lep2_pdgId", (30, -15, 15), "PDG ID (l2)", "2l"),  # not in training
#  #
#  Var("n_ak4",   (7, -.5, 6.5), "N(ak4)",  "0l,1l,2l"),
#  Var("n_jet",   (10, -.5, 9.5), "N(jets)", "0l,1l,2l"),
#  #
#  #Var("dphi_met_tkmet",   (26, 0, 0.52), "#Delta#phi(MET, tkMET)"),
#  Var("min_dphi_V_ak4", (50, 0.3, 6.3), "min#Delta#phi(V, ak4)", "0l"),
#  Var("min_dphi_met_jet", (30, 0.4, 3.4), "min#Delta#phi(MET, jet)", "0l"),
#  Var("deta_lep_ak15",    (50, 0, 5),    "|#Delta#eta(l, H)|",      "1l"),
#  Var("dphi_lep_met",     (50, 0, 3.15), "#Delta#phi(l, MET)",      "1l"),
#  Var("mt_lep_met",       (50, 0, 200),  "m_{T}(l, MET) [GeV]",     "1l"),
#  Var("min_dr_lep_ak4",   (50, 0, 5),    "min#DeltaR(l, ak4)",      "1l"),
#  #Var("min_deta_lep_ak4", (50, 0, 5),    "min|#Delta#eta(l, ak4)|", "1l"),  # not in training
#  ##2L specific:
#  Var("deltaR_ll",        (50, 0, 5),    "#DeltaR(l1, l2)",         "2l"),
#  Var("deta_z_ak15",      (50, 0, 5),    "|#Delta#eta(Z, H)|",      "2l"),
#  Var("min_deta_ak15_lep",(30, 0, 3),    "min|#Delta#eta(H, l)|",   "2l"),
#  #Var("min_deta_z_ak4",   (50, 0, 5),    "min|#Delta#eta(Z, ak4)|", "2l"),  # not in training
#  Var("min_dr_z_ak4",     (25, 0, 10),    "min#DeltaR(Z, ak4)",      "1l"),
#  ##Subjet pT:
#  Var("ak15_sj1_pt",              (30, 80, 380),  "p_{T}(sj1) [GeV]", "0l,1l,2l"),
#  Var("ak15_sj2_pt",              (25, 20, 220),  "p_{T}(sj2) [GeV]", "0l,1l,2l"),
#  Var("(ak15_sj1_pt + ak15_sj2_pt)/ak15_pt", (25, 0.6, 1.6), "(p_{T}(sj1)+p_{T}(sj2))/p_{T}(H)", "0l,1l,2l"),
#  #Var("(ak15_sj1_pt + ak15_sj2_pt)/jet_1_pt", (25, 0.8, 2.3), "(p_{T}(sj1)+p_{T}(sj2))/p_{T}(J1)", "0l,1l,2l"),  # not in training
#  Var("(ak15_sj1_pt - ak15_sj2_pt)/ak15_pt", (20, 0, 1), "(p_{T}(sj1)-p_{T}(sj2))/p_{T}(H)", "0l,1l,2l"),
#  #Var("(ak15_sj1_pt - ak15_sj2_pt)/jet_1_pt", (20, 0, 1), "(p_{T}(sj1)-p_{T}(sj2))/p_{T}(J1)", "0l,1l,2l"),  # not in training
#  
#  Var("ak15_sj2_pt/ak15_sj1_pt",  (45, 0.1, 1),    "p_{T}(sj2)/p_{T}(sj1)", "0l,1l,2l"),
#  Var("ak15_sj2_pt/(ak15_sj1_pt + ak15_sj2_pt)", (45, 0.1, 0.5), "z = p_{T}(sj2)/(p_{T}(sj1)+p_{T}(sj2))", "0l,1l,2l"),
#  Var("ak15_sj1_pt/ak15_pt",  (100, 0., 1),    "p_{T}(sj1)/p_{T}(H)", "0l,1l,2l"),
#  Var("ak15_sj2_pt/ak15_pt",  (25, 0, 0.8),    "p_{T}(sj2)/p_{T}(H)", "0l,1l,2l"),
#  
#  ## Invariant mass: H + V - comment out (use ak15_sdmass which is not in training)
#  #Var(f"sqrt(ak15_sdmass**2 + v_mass**2 + 2*ak15_pt*v_pt*(cosh(ak15_eta - v_eta) - cos((ak15_phi - v_phi + {PI}) % (2*{PI}) - {PI})))", (40, 300, 1300), "m(H+V) [GeV]", "0l,1l,2l"),  # ak15_sdmass not in training
#  #Var(f"sqrt(ak15_sdmass**2 + lep1_mass**2 + 2*ak15_pt*lep1_pt*(cosh(ak15_eta - lep1_eta) - cos((ak15_phi - lep1_phi + {PI}) % (2*{PI}) - {PI})) + 2*ak15_pt*met*(cosh(ak15_eta) - cos((ak15_phi - met_phi + {PI}) % (2*{PI}) - {PI})) + 2*lep1_pt*met*(cosh(lep1_eta) - cos((lep1_phi - met_phi + {PI}) % (2*{PI}) - {PI})))", (50, 300, 1500), "m(H+l1+MET) [GeV]", "1l,2l"),  # ak15_sdmass not in training
#  #Var(f"sqrt(ak15_sdmass**2 + v_mass**2 + 2*ak15_pt*v_pt*(cosh(ak15_eta - 0) - cos((ak15_phi - v_phi + {PI}) % (2*{PI}) - {PI})) + 2*ak15_pt*met*(cosh(ak15_eta) - cos((ak15_phi - met_phi + {PI}) % (2*{PI}) - {PI})) + 2*v_pt*met*(cosh(v_eta) - cos((v_phi - met_phi + {PI}) % (2*{PI}) - {PI})))", (40, 300, 1300), "m(H+V+MET) [GeV]", "1l,2l"),  # ak15_sdmass not in training
#  
# === 12 MLP_ParT training inputs (var_specs_ParT in Train.py, same for 0l/1l/2l) ===
#Var("ak15_ParTMDV2_Hgg",        (40, 0, 1), "ParT Hgg", "0l,1l,2l", logy=1),
#Var("ak15_ParTMDV2_Hgg**0.25",  (50, 0, 1), "ParT Hgg^{0.25}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_Hss**0.15",  (50, 0, 1), "ParT Hss^{0.15}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_Hqq**0.15",  (50, 0, 1), "ParT Hqq^{0.15}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_Hbs**0.1",   (50, 0, 1), "ParT Hbs^{0.1}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_Hbc**0.1",   (50, 0, 1), "ParT Hbc^{0.1}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_Hcs**0.13",  (50, 0, 1), "ParT Hcs^{0.13}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_QCDcc**0.13", (50, 0, 1), "ParT QCDcc^{0.13}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_QCDc**0.13", (50, 0, 1), "ParT QCDc^{0.13}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_QCDothers**0.3", (50, 0, 1), "ParT QCDothers^{0.3}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_TopbWqq**0.15", (50, 0, 1), "ParT TopbWqq^{0.15}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_TopbWcs**0.15", (50, 0, 1), "ParT TopbWcs^{0.15}", "0l,1l,2l"),
#Var("ak15_ParTMDV2_TopbWev**0.1",  (50, 0, 1), "ParT TopbWev^{0.1}", "0l,1l,2l"),
# === Remaining ParT variables (not used as BDT inputs) ===
#  Var("ak15_ParTMDV2_Hgg/(ak15_ParTMDV2_Hgg+ak15_ParTMDV2_QCDothers+ak15_ParTMDV2_QCDbb+ak15_ParTMDV2_QCDb+ak15_ParTMDV2_QCDcc+ak15_ParTMDV2_QCDc)",        (40, 0, 1), "ParT Hgg/(Hgg+QCD)", "0l,1l,2l",logy=1),
#  Var("1*ak15_ParTMDV2_Hgg/(ak15_ParTMDV2_Hgg+ak15_ParTMDV2_QCDothers+ak15_ParTMDV2_QCDbb+ak15_ParTMDV2_QCDb+ak15_ParTMDV2_QCDcc+ak15_ParTMDV2_QCDc+ak15_ParTMDV2_TopbWcs+ak15_ParTMDV2_TopbWqq+ak15_ParTMDV2_TopbWc+ak15_ParTMDV2_TopbWs+ak15_ParTMDV2_TopbWq+ak15_ParTMDV2_TopbWev+ak15_ParTMDV2_TopbWmv+ak15_ParTMDV2_TopbWtauev+ak15_ParTMDV2_TopbWtaumv+ak15_ParTMDV2_TopbWtauhv+ak15_ParTMDV2_TopWcs+ak15_ParTMDV2_TopWqq+ak15_ParTMDV2_TopWev+ak15_ParTMDV2_TopWmv+ak15_ParTMDV2_TopWtauev+ak15_ParTMDV2_TopWtaumv+ak15_ParTMDV2_TopWtauhv)",        (40, 0, 1), "ParT Hgg/(Hgg+QCD+top)", "0l,1l,2l",logy=1),
#  Var("ak15_ParTMDV2_Hbb**0.08",        (50, 0, 1), "ParT Hbb^{0.08}", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_Hcc**0.13",        (50, 0, 1), "ParT Hcc^{0.13}", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_Htauhtaue**0.07",  (50, 0, 1), "ParT Htauhtaue^0.07", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_Htauhtauh**0.07",  (50, 0, 1), "ParT Htauhtauh^0.07", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_Htauhtaum**0.07",  (50, 0, 1), "ParT Htauhtaum^0.07", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_Hee**0.07",        (50, 0, 1), "ParT Hee^0.07", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_Hmm**0.07",        (50, 0, 1), "ParT Hmm^0.07", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_QCDbb**0.15",      (50, 0, 1), "ParT QCDbb^{0.15}", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_QCDb**0.15",       (50, 0, 1), "ParT QCDb^{0.15}", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_TopbWcs**0.15",    (50, 0, 1), "ParT TopbWcs^{0.15}", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_TopbWc**0.1",      (50, 0, 1), "ParT TopbWc^{0.1}", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopbWs**0.1",      (50, 0, 1), "ParT TopbWs^0.1", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_TopbWq**0.1",      (50, 0, 1), "ParT TopbWq^{0.1}", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopbWev**0.1",     (50, 0, 1), "ParT TopbWev^0.1", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopbWmv**0.1",     (50, 0, 1), "ParT TopbWmv^0.1", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopbWtauev**0.1",  (50, 0, 1), "ParT TopbWtauev^0.1", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopbWtaumv**0.1",  (50, 0, 1), "ParT TopbWtaumv^0.1", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_TopbWtauhv**0.1",  (50, 0, 1), "ParT TopbWtauhv^{0.1}", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_TopWcs**0.1",      (50, 0, 1), "ParT TopWcs^{0.1}", "0l,1l,2l"),
#  Var("ak15_ParTMDV2_TopWqq**0.1",      (50, 0, 1), "ParT TopWqq^{0.1}", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopWev**0.08",     (50, 0, 1), "ParT TopWev^0.08", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopWmv**0.08",     (50, 0, 1), "ParT TopWmv^0.08", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopWtauev**0.08",  (50, 0, 1), "ParT TopWtauev^0.08", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopWtaumv**0.08",  (50, 0, 1), "ParT TopWtaumv^0.08", "0l,1l,2l"),
#  #Var("ak15_ParTMDV2_TopWtauhv**0.08",  (50, 0, 1), "ParT TopWtauhv^0.08", "0l,1l,2l"),
# === N-subjettiness (not used as BDT inputs) ===
#  Var("ak15_tau21",        (50, 0, 1),      "#tau_{21}(H)",         "0l,1l,2l"),
#  Var("ak15_tau32",        (50, 0, 1),      "#tau_{32}(H)",         "0l,1l,2l"),
#  Var("ak15_tau32*ak15_tau21", (50, 0, 1), "#tau_{31}(H)",           "0l,1l,2l"),
#  #===============================================================================
#  #Var("boostedTau_chargedIso",             (50, 0, 10),       "boostedTau charged isolation", "", 1),
#  ##Var("boostedTau_eta",                   (50, -2.5, 2.5),   "boostedTau #eta"),
#  ##Var("boostedTau_leadTkDeltaEta",        (51, -0.001, 0.001), "#Delta#eta(leadTk, boostedTau)", "", 1),
#  ##Var("boostedTau_leadTkDeltaPhi",        (51, -0.001, 0.001), "#Delta#phi(leadTk, boostedTau)", "", 1),
#  ##Var("boostedTau_leadTkPtOverTauPt",     (51, 0.995, 1.005), "leadTk pT / boostedTau pT"),
#  #Var("boostedTau_mass",                   (50, 0, 2.5),      "boostedTau mass (GeV)", "", 1),
#  ##Var("boostedTau_neutralIso",            (50, 0, 5),        "boostedTau neutral isolation", "", 1),
#  ##Var("boostedTau_phi",                   (64, -3.2, 3.2),   "boostedTau #phi"),
#  #Var("boostedTau_photonsOutsideSignalCone", (15, 0, 15),     "# photons outside signal cone", "", 1),
#  #Var("boostedTau_pt",                     (50, 20, 300),     "boostedTau pT (GeV)"),
#  ##Var("boostedTau_puCorr",                (100, 0, 100),     "boostedTau pileup correction"),
#  #Var("boostedTau_rawAntiEle2018",         (50, -1, 1),       "boostedTau rawAntiEle2018"),
#  #Var("boostedTau_rawIso",                 (50, 0, 5),        "boostedTau raw isolation"),
#  #Var("boostedTau_rawIsodR03",             (50, 0, 4),        "boostedTau raw isolation dR03", "", 1),
#  ###Var("boostedTau_rawMVAnewDM2017v2",    (50, -1, 1),       "boostedTau raw MVA newDM2017v2", "", 1),
#  ###Var("boostedTau_rawMVAoldDM2017v2",    (50, -1, 1),       "boostedTau raw MVA oldDM2017v2"),
#  ##Var("boostedTau_rawMVAoldDMdR032017v2", (50, -1, 1),       "boostedTau raw MVA oldDMdR03 2017v2"),
#  #Var("boostedTau_charge",                 (5, -2, 2),        "boostedTau charge"),
#  #Var("boostedTau_decayMode",              (12, -0.5, 11.5),  "boostedTau decay mode"),
#  ##Var("boostedTau_jetIdx",                (50, -1, 1),       "boostedTau jet index"),
#  #Var("boostedTau_rawAntiEleCat2018",      (20, -10, 10),     "boostedTau rawAntiEleCat2018"),
#  #Var("boostedTau_idAntiEle2018",          (20, -10, 10),     "boostedTau idAntiEle2018"),
#  #Var("boostedTau_idAntiMu",               (5, -1, 4),        "boostedTau idAntiMu"),
#  ##Var("boostedTau_idMVAnewDM2017v2",      (5, -1, 4),        "boostedTau idMVAnewDM2017v2"),
#  ##Var("boostedTau_idMVAoldDM2017v2",      (5, -1, 4),        "boostedTau idMVAoldDM2017v2"),
#  ##Var("boostedTau_idMVAoldDMdR032017v2",  (5, -1, 4),        "boostedTau idMVAoldDMdR032017v2"),
#  ##Var("boostedTau_genPartIdx",            (20, -1, 19),      "boostedTau genPartIdx"),
#  ##Var("boostedTau_genPartFlav",           (7, 0, 6),         "boostedTau genPartFlav"),

#Var("HLT_Ele27_WPTight_Gsf",                             (2, -0.5, 1.5), "HLT Ele27_WPTight_Gsf"),
#Var("HLT_Ele32_WPTight_Gsf",                             (2, -0.5, 1.5), "HLT Ele32_WPTight_Gsf"),
#Var("HLT_Ele35_WPTight_Gsf",                             (2, -0.5, 1.5), "HLT Ele35_WPTight_Gsf"),
#Var("HLT_Ele32_WPTight_Gsf_L1DoubleEG",                  (2, -0.5, 1.5), "HLT Ele32_WPTight_Gsf_L1DoubleEG"),
#Var("HLT_IsoMu24",                                       (2, -0.5, 1.5), "HLT IsoMu24"),
#Var("HLT_IsoMu27",                                       (2, -0.5, 1.5), "HLT IsoMu27"),
#Var("HLT_PFMET110_PFMHT110_IDTight",                     (2, -0.5, 1.5), "HLT PFMET110_PFMHT110_IDTight"),
#Var("HLT_PFMET120_PFMHT120_IDTight",                     (2, -0.5, 1.5), "HLT PFMET120_PFMHT120_IDTight"),
#Var("HLT_PFMET130_PFMHT130_IDTight",                     (2, -0.5, 1.5), "HLT PFMET130_PFMHT130_IDTight"),
#Var("HLT_PFMET140_PFMHT140_IDTight",                     (2, -0.5, 1.5), "HLT PFMET140_PFMHT140_IDTight"),
#Var("HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1", (2, -0.5, 1.5), "HLT PFMET100 CaloBTagDeepCSV_3p1"),
#Var("HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1", (2, -0.5, 1.5), "HLT PFMET110 CaloBTagDeepCSV_3p1"),
#Var("HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1", (2, -0.5, 1.5), "HLT PFMET120 CaloBTagDeepCSV_3p1"),
#Var("HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1", (2, -0.5, 1.5), "HLT PFMET130 CaloBTagDeepCSV_3p1"),
#Var("HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1", (2, -0.5, 1.5), "HLT PFMET140 CaloBTagDeepCSV_3p1"),
#Var("HLT_PFMET120_PFMHT120_IDTight_PFHT60",              (2, -0.5, 1.5), "HLT PFMET120 PFHT60"),
#Var("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",      (2, -0.5, 1.5), "HLT PFMETNoMu120 PFHT60"),
#Var("HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60",       (2, -0.5, 1.5), "HLT PFMETTypeOne120 PFHT60"),
#Var("HLT_PFMETTypeOne110_PFMHT110_IDTight",              (2, -0.5, 1.5), "HLT PFMETTypeOne110_IDTight"),
#Var("HLT_PFMETTypeOne120_PFMHT120_IDTight",              (2, -0.5, 1.5), "HLT PFMETTypeOne120_IDTight"),
#Var("HLT_PFMETTypeOne130_PFMHT130_IDTight",              (2, -0.5, 1.5), "HLT PFMETTypeOne130_IDTight"),
#Var("HLT_PFMETTypeOne140_PFMHT140_IDTight",              (2, -0.5, 1.5), "HLT PFMETTypeOne140_IDTight"),
#Var("HLT_PFMETNoMu110_PFMHTNoMu110_IDTight",             (2, -0.5, 1.5), "HLT PFMETNoMu110_IDTight"),
#Var("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",             (2, -0.5, 1.5), "HLT PFMETNoMu120_IDTight"),
#Var("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight",             (2, -0.5, 1.5), "HLT PFMETNoMu130_IDTight"),
#Var("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",             (2, -0.5, 1.5), "HLT PFMETNoMu140_IDTight"),
#Var("HLT_PFMET100_PFMHT100_IDTight_PFHT60",              (2, -0.5, 1.5), "HLT PFMET100 PFHT60"),
#Var("HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60",      (2, -0.5, 1.5), "HLT PFMETNoMu100 PFHT60"),
#Var("HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60",       (2, -0.5, 1.5), "HLT PFMETTypeOne100 PFHT60"),

#Var("year",           (5, 2016, 2021),   "Year"),
#Var("channel",        (120, -20, 100),   "Channel"),
#Var("passmetfilters", (2, -0.5, 1.5),    "Pass MET filters"),

#Var("h2bb",           (40, 0, 1), "h2bb"),
#Var("h2cc",           (40, 0, 1), "h2cc"),
#Var("z2bb",           (40, 0, 1), "z2bb"),
#Var("z2cc",           (40, 0, 1), "z2cc"),
#Var("z2qq",           (40, 0, 1), "z2qq"),
#Var("w2cq",           (40, 0, 1), "w2cq"),
#Var("w2qq",           (40, 0, 1), "w2qq"),
#Var("genZ_pt",        (50, 0, 500), "gen Z pT (GeV)"),
#Var("genW_pt",        (50, 0, 500), "gen W pT (GeV)"),
#Var("dr_ak15_hdaus",  (50, 0, 5), "DeltaR(ak15, H daughters)"),
#Var("dr_ak15_zdaus",  (50, 0, 5), "DeltaR(ak15, Z daughters)"),
#Var("dr_ak15_wdaus",  (50, 0, 5), "DeltaR(ak15, W daughters)"),
#Var("elEffWeight_UP",         (50, 0.8, 1.2), "elEffWeight_UP"),
#Var("elEffWeight_DOWN",       (50, 0.8, 1.2), "elEffWeight_DOWN"),
#Var("muEffWeight_UP",         (50, 0.8, 1.2), "muEffWeight_UP"),
#Var("muEffWeight_DOWN",       (50, 0.8, 1.2), "muEffWeight_DOWN"),
#Var("pileupJetIdWeight_UP",   (50, 0.8, 1.2), "pileupJetIdWeight_UP"),
#Var("pileupJetIdWeight_DOWN", (50, 0.8, 1.2), "pileupJetIdWeight_DOWN"),
#Var("v1_pt",          (50, 0, 500), "v1 pT (GeV)"),
#Var("v2_pt",          (50, 0, 500), "v2 pT (GeV)"),
#Var("vhWeightEWK_UP",     (50, 0.8, 1.2), "vhWeightEWK_UP"),
#Var("vhWeightEWK_DOWN",   (50, 0.8, 1.2), "vhWeightEWK_DOWN"),
#Var("vvWeightNNLO_UP",    (50, 0.8, 1.2), "vvWeightNNLO_UP"),
#Var("vvWeightNNLO_DOWN",  (50, 0.8, 1.2), "vvWeightNNLO_DOWN"),
#Var("puWeightUp",         (50, 0.8, 1.2), "puWeightUp"),
#Var("puWeightDown",       (50, 0.8, 1.2), "puWeightDown"),


# === KIN variables — matches Train.py var_specs_KIN_{0l,1l,2l} exactly === (Figs 17-19)
# --- shared all channels ---
#Var("ak15_eta",                                    (40, -2.6,  2.6),  "#eta(H)",                                        "0l,1l,2l"),
#Var("dphi_V_ak15",                                 (40,  2.2,  3.2),  "#Delta#phi(V,H)",                                "0l,1l,2l"),
#Var("arccos(cos(ak4_1_phi-v_phi))",                (50,  0,    3.15), "|#Delta#phi(j_{1},V)|",                          "0l,1l,2l"),
#Var("n_ak4",                                       (8,  -0.5,  7.5),  "N_{jets} (AK4)",                                 "0l,1l,2l"),
#Var("ak15_eta - jet_1_eta",                        (50, -1.75, 1.75), "#Delta#eta(H,jet_{1})",                          "0l,1l,2l"),
#Var("arccos(cos(ak15_phi - jet_1_phi))",            (50,  0,    3.15), "|#Delta#phi(H,jet_{1})|",                        "0l,1l,2l"),
# --- shared 0l,1l ---
#Var("(ak15_sj1_pt - ak15_sj2_pt)/ak15_pt",        (40,  0,    1.0),  "sj p_{T} asym. (sj_{1}-sj_{2})/p_{T}(H)",       "0l,1l"),
# --- shared 0l,2l ---
#Var("met/ak15_pt",                                 (40,  0,    2.0),  "MET/p_{T}(H)",                                   "0l,2l"),
#Var("arccos(cos(jet_1_phi - v_phi))",               (50,  0,    3.15), "|#Delta#phi(jet_{1},V)|",                        "0l,2l"),
# --- shared 1l,2l ---
#Var("v_pt/ak15_pt",                                (40,  0,    2.0),  "p_{T}(V)/p_{T}(H)",                              "1l,2l"),
#Var("lep1_eta",                                    (40, -2.6,  2.6),  "#eta(l_{1})",                                    "1l,2l"),
# --- 0l only ---
#Var("met",                                         (50,  0,  600),    "MET [GeV]",                                      "0l"),
#Var("dphi_met_tkmet",                              (30,  0,    0.6),  "#Delta#phi(MET,tkMET)",                          "0l"),
#Var("v_pt",                                        (50, 150,  650),   "p_{T}^{miss} proxy [GeV]",                       "0l"),
#Var("jet_1_eta",                                   (40, -3.0,  3.0),  "#eta(jet_{1})",                                  "0l"),
# --- 1l only ---
#Var("met",                                         (50,  0,  600),    "MET [GeV]",                                      "1l"),
#Var("met/v_pt",                                    (40,  0,    1.5),  "MET/p_{T}(W)",                                   "1l"),
#Var("v_pt",                                        (50, 150,  650),   "p_{T}(W) [GeV]",                                 "1l"),
#Var("ak15_pt + v_pt",                              (50, 350, 1200),   "p_{T}(H)+p_{T}(W) [GeV]",                       "1l"),
#Var("lep1_pt",                                     (50,   0,  400),   "p_{T}(l_{1}) [GeV]",                             "1l"),
#Var("lep1_pdgId",                                  (29, -14.5, 14.5), "l_{1} PDG ID",                                   "1l"),
#Var("lep1_pt/ak15_pt",                             (40,  0,    1.5),  "p_{T}(l_{1})/p_{T}(H)",                          "1l"),
#Var("dphi_lep_met",                                (50,  0,    3.15), "#Delta#phi(l,MET)",                              "1l"),
#Var("deta_lep_ak15",                               (50,  0,    4.0),  "#Delta#eta(l,H)",                                "1l"),
#Var("mt_lep_met",                                  (50,  0,  300),    "m_{T}(l,MET) [GeV]",                             "1l"),
# --- 2l only ---
#Var("(91-v_mass)/91",                              (40, -0.5,  0.5),  "(91-m_{ll})/91",                                 "2l"),
#Var("v_eta",                                       (40, -2.6,  2.6),  "#eta(Z)",                                        "2l"),
#Var("(lep1_pt - lep2_pt)/(lep1_pt + lep2_pt)",    (40,  0,    1.0),  "l p_{T} asym. (l_{1}-l_{2})/(l_{1}+l_{2})",    "2l"),
#Var("lep1_pt/v_pt",                                (40,  0,    1.5),  "p_{T}(l_{1})/p_{T}(Z)",                          "2l"),
#Var("lep2_pt/v_pt",                                (40,  0,    1.0),  "p_{T}(l_{2})/p_{T}(Z)",                          "2l"),
#Var("deltaR_ll",                                   (50,  0,    2.5),  "#Delta R(l_{1},l_{2})",                          "2l"),
#Var("deta_z_ak15",                                 (50,  0,    5.0),  "#Delta#eta(Z,H)",                                "2l"),
#Var("ak4_1_pt",                                    (40,  0,  400),    "p_{T}(j_{1}) [GeV]",                             "2l"),
#Var("lep2_eta - jet_1_eta",                        (40, -5.0,  5.0),  "#Delta#eta(l_{2},jet_{1})",                      "2l"),
#Var("min_dr_z_ak4",                                (40,  0,   10.0),  "min #Delta R(Z,ak4)",                            "2l"),

# === Preselection kinematic variables — for Figs 4-6 in the AN ===
# --- Row 1: H jet mass/pT/substructure (all channels; sdmass & regressed_mass active above) ---
#Var("ak15_pt",                              (50, 200,  800),  "p_{T}(H) [GeV]",                    "0l,1l,2l"),
#Var("ak15_tau21",                           (40,   0,    1),  "#tau_{21}",                          "0l,1l,2l"),
# --- Row 2: H jet geometry/subjets ---
#Var("abs(ak15_eta)",                        (30,   0,  2.6),  "|#eta(H)|",                          "0l,1l,2l"),
#Var("ak15_deltaR_sj12",                     (40,   0,  2.0),  "#Delta R(sj_{1},sj_{2})",            "0l,1l,2l"),
#Var("(ak15_sj1_pt-ak15_sj2_pt)/ak15_pt",   (40,   0,    1),  "sj p_{T} asym.",                     "0l,1l,2l"),
#Var("ak15_sj1_pt/(ak15_sj2_pt+1e-6)",       (40,   1,   10),  "p_{T}(sj_{1})/p_{T}(sj_{2})",       "0l,1l"),
# --- Row 3: V boson / MET system (channel-specific) ---
#Var("met",                                  (50,   0,  600),  "MET [GeV]",                          "0l"),
#Var("met/ak15_pt",                          (40,   0,    2),  "MET/p_{T}(H)",                       "0l"),
#Var("ak15_phi",                             (32, -3.15, 3.15),"#phi(H)",                            "0l,2l"),
#Var("min_dphi_met_jet",                     (40,   0,  3.2),  "min #Delta#phi(MET,jet)",            "0l"),
#Var("v_pt",                                 (50, 150,  650),  "p_{T}(W) [GeV]",                    "1l"),
#Var("met",                                  (50,   0,  600),  "MET [GeV]",                          "1l"),
#Var("v_mass",                               (50,   0,  300),  "m_{T}(W) [GeV]",                    "1l"),
#Var("mt_lep_met",                           (50,   0,  300),  "m_{T}(l,MET) [GeV]",                "1l"),
#Var("v_pt",                                 (50,  50,  600),  "p_{T}(Z) [GeV]",                    "2l"),
#Var("met",                                  (50,   0,  300),  "MET [GeV]",                          "2l"),
#Var("v_mass",                               (50,  65,  115),  "m(ll) [GeV]",                        "2l"),
#Var("deltaR_ll",                            (50,   0,  2.5),  "#Delta R(l_{1},l_{2})",              "2l"),
# --- Row 4: Angular/topology ---
#Var("dphi_V_ak15",                          (40, 2.2,  3.2),  "#Delta#phi(V,H)",                    "0l,1l,2l"),
#Var("ak15_tau32*ak15_tau21",                (40,   0,    1),  "#tau_{31}",                          "0l,1l,2l"),
#Var("lep1_pdgId",                           (29, -14.5, 14.5),"l_{1} PDG ID",                      "1l"),
#Var("(abs(lep1_pdgId)+abs(lep2_pdgId))/2", (3, 10.5, 13.5),  "BINLABELS:ee,e#mu,#mu#mu",          "2l"),
#Var("n_ak4",                                (8,  -0.5,  7.5), "N_{jets} (AK4)",                     "0l,1l,2l"),
#Var("ak15_eta - jet_1_eta",                 (50, -1.75, 1.75),"#Delta#eta(H, jet_{1})",             "0l,1l,2l"),
# --- Row 5: Jet and lepton kinematics ---
#Var("jet_1_pt",                             (50,  50,  550),  "p_{T}(jet_{1}) [GeV]",              "0l,1l,2l"),
#Var("n_ak15",                               (5,  -0.5,  5.5), "N_{AK15}",                           "0l"),
#Var("v_phi",                                (40, -3.2,  3.2), "#phi(MET) [rad]",                    "0l"),
#Var("jet_1_eta",                            (40, -3.0,  3.0), "#eta(jet_{1})",                      "0l"),
#Var("lep1_pt",                              (50,   0,  400),  "p_{T}(l_{1}) [GeV]",                "1l,2l"),
#Var("lep2_pt",                              (50,   0,  300),  "p_{T}(l_{2}) [GeV]",                "2l"),
#Var("deta_lep_ak15",                        (50,   0,  4.0),  "#Delta#eta(l,H)",                    "1l"),
]

# ============================================================================
# BDT binned plotting configuration
# ============================================================================
# Define BDT score bins for separate plots: (file_suffix, display_label, cut_function)
BDT_ParT_bins = [
#    ("ParT_0p57_0p86", "PS+ParT:0.57-0.86", lambda e: (e["BDT_ParT"] > 0.57) & (e["BDT_ParT"] < 0.86)),
#    ("ParT_0p86_0p95", "PS+ParT:0.86-0.95", lambda e: (e["BDT_ParT"] > 0.86) & (e["BDT_ParT"] < 0.95)),
#    ("ParT_gt0p95",   "PS+ParT:>0.95",    lambda e: e["BDT_ParT"] > 0.95),
]
BDT_KIN_bins = [
#    ("KIN_0p5_0p8", "PS+KIN:0.5-0.8", lambda e: (e["BDT_KIN"] > 0.5) & (e["BDT_KIN"] < 0.8)),
#    ("KIN_0p8_0p9", "PS+KIN:0.8-0.9", lambda e: (e["BDT_KIN"] > 0.8) & (e["BDT_KIN"] < 0.9)),
#    ("KIN_gt0p9",   "PS+KIN:>0.9",    lambda e: e["BDT_KIN"] > 0.9),
]

# Channel-specific BDT boundaries: [ParT_low, ParT_mid, ParT_high, KIN_low, KIN_mid, KIN_high]
# ParT bins: 1=low-mid, 2=mid-high, 3=>high
# KIN bins:  a=low-mid, b=mid-high, c=>high
BDT_BOUNDARIES = {
    #           -------ParT------, ------KIN------- 
    "0l": [0.52, 0.84, 0.92, 0.44, 0.76, 0.88],  # 0l  ZH->nunu
    "1l": [0.52, 0.84, 0.92, 0.56, 0.84, 0.92],  # 1l  WH->lnu
    "2l": [0.52, 0.84, 0.92, 0.5, 0.8, 0.9],  # 2l  ZH->ll
}

def make_BDT_ParT_KIN_bins(channel):
    """Generate 9 SR bins based on channel-specific BDT boundaries."""
    P1, P2, P3, K1, K2, K3 = BDT_BOUNDARIES[channel]
    return [
        ("region_1a", "SR1a", lambda e, p1=P1, p2=P2, k1=K1, k2=K2: (e["BDT_ParT"] > p1) & (e["BDT_ParT"] < p2) & (e["BDT_KIN"] > k1) & (e["BDT_KIN"] < k2)),
        ("region_1b", "SR1b", lambda e, p1=P1, p2=P2, k2=K2, k3=K3: (e["BDT_ParT"] > p1) & (e["BDT_ParT"] < p2) & (e["BDT_KIN"] > k2) & (e["BDT_KIN"] < k3)),
        ("region_1c", "SR1c", lambda e, p1=P1, p2=P2, k3=K3: (e["BDT_ParT"] > p1) & (e["BDT_ParT"] < p2) & (e["BDT_KIN"] > k3)),
        ("region_2a", "SR2a", lambda e, p2=P2, p3=P3, k1=K1, k2=K2: (e["BDT_ParT"] > p2) & (e["BDT_ParT"] < p3) & (e["BDT_KIN"] > k1) & (e["BDT_KIN"] < k2)),
        ("region_2b", "SR2b", lambda e, p2=P2, p3=P3, k2=K2, k3=K3: (e["BDT_ParT"] > p2) & (e["BDT_ParT"] < p3) & (e["BDT_KIN"] > k2) & (e["BDT_KIN"] < k3)),
        ("region_2c", "SR2c", lambda e, p2=P2, p3=P3, k3=K3: (e["BDT_ParT"] > p2) & (e["BDT_ParT"] < p3) & (e["BDT_KIN"] > k3)),
        ("region_3a", "SR3a", lambda e, p3=P3, k1=K1, k2=K2: (e["BDT_ParT"] > p3) & (e["BDT_KIN"] > k1) & (e["BDT_KIN"] < k2)),
        ("region_3b", "SR3b", lambda e, p3=P3, k2=K2, k3=K3: (e["BDT_ParT"] > p3) & (e["BDT_KIN"] > k2) & (e["BDT_KIN"] < k3)),
        ("region_3c", "SR3c", lambda e, p3=P3, k3=K3: (e["BDT_ParT"] > p3) & (e["BDT_KIN"] > k3)),
    ]

def make_BDT_AVG_bins():
    """Generate SR bins based on (BDT_ParT+BDT_KIN)/2 with configurable edges.
    Edit 'edges' below to change number and boundaries of bins.
    """
#    edges = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]  # 9 bins
#    edges = [0.52, 0.60, 0.68, 0.76, 0.84, 0.92, 1.00]  # 6 bins
    edges = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 1.00];   #7 bins
    n_bins = len(edges) - 1
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        region_id = f"region_avg{i+1}"
        label = f"SR{i+1}: {lo:.2f}-{hi:.2f}"
        if i < n_bins - 1:
            # All bins except the last: lo <= avg < hi
            bins.append((region_id, label, lambda e, lo=lo, hi=hi:
                ((e["BDT_ParT"] + e["BDT_KIN"]) / 2 >= lo) &
                ((e["BDT_ParT"] + e["BDT_KIN"]) / 2 < hi)))
        else:
            # Last bin: lo <= avg <= hi (inclusive upper bound)
            bins.append((region_id, label, lambda e, lo=lo, hi=hi:
                ((e["BDT_ParT"] + e["BDT_KIN"]) / 2 >= lo) &
                ((e["BDT_ParT"] + e["BDT_KIN"]) / 2 <= hi)))
    return bins

# Placeholder - will be set after channel is known
BDT_ParT_KIN_bins = []

# Variables to plot in BDT bins (separate plot for each bin)
variables_binned_plot = [
#Var("ak15_sdmass",                  (50, 0, 250),    "m_{SD,ak15} (GeV)"),
#Var("1-abs(MLP_MASS - 125)/125",    (20, 0, 1),      "1-|m_{MLP}-125|/125"),
#Var("1-abs(ak15_sdmass - 135)/135", (50, 0, 1),      "1-|ak15_sdmass - 135|/135"),
#Var("BDT_ParT",                     (100, 0, 1),     "BDT ParT"),
#Var("BDT_KIN",                      (100, 0, 1),     "BDT KIN"),
#Var("1-abs(ak15_sdmass - 135)/35",  (30, -2, 1),     "m_{SD,ak15} (GeV)"),
#Var("v_mass + ak15_sdmass",         (50, 0, 500),    "m(V) + m_{SD}(H) (GeV)"),
#Var("v_pt + ak15_pt",               (50, 300, 1100), "p_{T}(V) + p_{T}(ak15) (GeV)"),
#Var("lep1_pt / v_pt",               (50, 0, 1),      "p_{T}(l) / p_{T}(V)"),
#Var("met / v_pt",                   (50, 0, 1),      "MET / p_{T}(V)"),
]

# 2D plot configuration: (x_var, y_var, bins_x, bins_y, xlabel, ylabel)
# If this list is empty, no 2D plots will be made
variables_2D_plot = [

#{"x_name": "1-abs(ak15_sdmass-133)/133",
# "y_name": "1-abs(ak15_deltaR_sj12-0.9)/0.9",
# "bins_x": (50, 0, 1), "bins_y": (50, 0, 1),
# "xlabel": "1-|m_{SD}-133|/133", "ylabel": "1-|#DeltaR-0.9|/0.9",
# "y_tag": "DR_sym"},

# ── SDmass vs all Mscore input variables ────────────────────────────────────
#{"x_name": "ak15_sdmass", "y_name": "ak15_tau21",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 1.0),
# "xlabel": "m_{SD} (GeV)", "ylabel": "#tau_{21}"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_tau32",
# "bins_x": (40, 30, 230), "bins_y": (40, 0.3, 1.0),
# "xlabel": "m_{SD} (GeV)", "ylabel": "#tau_{32}"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_tau32*ak15_tau21",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 0.85),
# "xlabel": "m_{SD} (GeV)", "ylabel": "#tau_{32} #times #tau_{21}",
# "y_tag": "tau31"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_deltaR_sj12",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 2.0),
# "xlabel": "m_{SD} (GeV)", "ylabel": "#DeltaR(sj_{1},sj_{2})"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_sj1_pt",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 400),
# "xlabel": "m_{SD} (GeV)", "ylabel": "sj_{1} p_{T} (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_sj2_pt",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 200),
# "xlabel": "m_{SD} (GeV)", "ylabel": "sj_{2} p_{T} (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_sj1_pt/(ak15_sj2_pt+1e-6)",
# "bins_x": (40, 30, 230), "bins_y": (40, 1, 8),
# "xlabel": "m_{SD} (GeV)", "ylabel": "p_{T}(sj_{1})/p_{T}(sj_{2})",
# "y_tag": "sj1_over_sj2_pt"},

#{"x_name": "ak15_sdmass",
# "y_name": "(ak15_sj1_pt - ak15_sj2_pt) / (ak15_sj1_pt + ak15_sj2_pt + 1e-6)",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 0.85),
# "xlabel": "m_{SD} (GeV)", "ylabel": "p_{T} asymmetry (sj_{1}-sj_{2})/(sj_{1}+sj_{2})",
# "y_tag": "sj_pt_asymmetry"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_mass",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 300),
# "xlabel": "m_{SD} (GeV)", "ylabel": "m_{AK15} ungroomed (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "jet_1_mass",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 80),
# "xlabel": "m_{SD} (GeV)", "ylabel": "m_{AK4,1} (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "ak4_1_mass",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 30),
# "xlabel": "m_{SD} (GeV)", "ylabel": "m_{AK4,2} (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "jet_1_pt",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 500),
# "xlabel": "m_{SD} (GeV)", "ylabel": "p_{T}(AK4,1) (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "ak15_pt",
# "bins_x": (40, 30, 230), "bins_y": (40, 200, 700),
# "xlabel": "m_{SD} (GeV)", "ylabel": "p_{T}(AK15) (GeV)"},

#{"x_name": "ak15_sdmass", "y_name": "v_pt",
# "bins_x": (40, 30, 230), "bins_y": (40, 0, 500),
# "xlabel": "m_{SD} (GeV)", "ylabel": "p_{T}(V) (GeV)"},

#{"x_name": "MLPf_X", "y_name": "MLPf_Mscore", "bins_x": (50, 0., 1), "bins_y": (50, 0., 1), "xlabel": "MLPf_{X}", "ylabel": "MLPf_{Mscore}", "cdf_flat_x": False},

#{"x_name": "BDTf_ParT", "y_name": "BDTf_KIN", "bins_x": (20, 0, 1), "bins_y": (20, 0, 1), "xlabel": "BDTf_{ParT}", "ylabel": "BDTf_{KIN}"},
#{"x_name": "BDTf_ParT", "y_name": "BDTf_Mscore", "bins_x": (20, 0, 1), "bins_y": (20, 0, 1), "xlabel": "BDTf_{ParT}", "ylabel": "BDTf_{Mscore}"},
#{"x_name": "BDTf_KIN", "y_name": "BDTf_Mscore", "bins_x": (20, 0, 1), "bins_y": (20, 0, 1), "xlabel": "BDTf_{KIN}", "ylabel": "BDTf_{Mscore}"},

#{"x_name": "MLPf_ParT", "y_name": "MLPf_KIN", "bins_x": (25, 0., 1), "bins_y": (25, 0., 1), "xlabel": "MLPf_{ParT}", "ylabel": "MLPf_{KIN}"},
#{"x_name": "MLPf_ParT", "y_name": "MLPf_Mscore", "bins_x": (25, 0., 1), "bins_y": (25, 0., 1), "xlabel": "MLPf_{ParT}", "ylabel": "MLPf_{Mscore}"},
#{"x_name": "MLPf_KIN", "y_name": "MLPf_Mscore", "bins_x": (25, 0., 1), "bins_y": (25, 0., 1), "xlabel": "MLPf_{KIN}", "ylabel": "MLPf_{Mscore}"},

]
# ============================================================================

def compute_significance_metrics(B, S, norm, xmin, xmax, Nbins=3, sig_xrange=None):
    """Compute significance metrics: initial, 1-bin, ..., Nbins-bin optimal.
    Args:
        B, S: Background and Signal histograms
        norm: Signal normalization factor
        xmin, xmax: x-axis range (for reference)
        Nbins: Maximum number of bins for optimization (default 3)
        sig_xrange: if (lo, hi), restrict initial S/sqrt(D) integral to that range
    Returns a dictionary with all computed values, or None if no valid data.
    NOTE: For histograms with >25 bins, uses coarser rebinning to avoid O(n^Nbins+1) slowdown."""
    from itertools import combinations
    eps = 1e-9
    hist_nbins = B.GetNbinsX()
    if sig_xrange is not None:
        b_lo = B.FindBin(sig_xrange[0]); b_hi = B.FindBin(sig_xrange[1] - 1e-9)
    else:
        b_lo, b_hi = 1, hist_nbins
    # Bin-by-bin sqrt(Σ s_i²/d_i): per-bin shape metric (upper bound on significance)
    _initial_sum = 0.0
    for _ib in range(b_lo, b_hi + 1):
        _d = B.GetBinContent(_ib); _s = S.GetBinContent(_ib) / norm
        if _d > 0: _initial_sum += _s * _s / _d
    initial = _initial_sum ** 0.5
    S0 = S.Integral(b_lo, b_hi) / norm
    if initial == 0 or S0 == 0: return None
    # Bulk S/sqrt(D): naive full-range count — used as "0 bins" gain reference.
    # n-bin optima always >= this, so gain% vs gain_ref is always positive.
    D0 = B.Integral(b_lo, b_hi)
    gain_ref = S0 / (D0 + eps)**0.5 if D0 > 0 else 0.0

    # For histograms with many bins, rebin to avoid O(n^(Nbins+1)) complexity.
    # Threshold 60: avoids rebinning typical 51-bin histograms (PRED3 Mscore) so
    # the last bin (Mscore→1, most sensitive) is never lost to overflow.
    MAX_BINS_FOR_OPTIM = 60
    if hist_nbins > MAX_BINS_FOR_OPTIM:
        rebin_factor = hist_nbins // MAX_BINS_FOR_OPTIM
        B = B.Clone("B_rebinned"); B.Rebin(rebin_factor)
        S = S.Clone("S_rebinned"); S.Rebin(rebin_factor)
        hist_nbins = B.GetNbinsX()
        # b_lo/b_hi were computed on the original histogram — recompute for rebinned bins
        if sig_xrange is not None:
            b_lo = max(1, B.FindBin(sig_xrange[0]))
            b_hi = min(hist_nbins, B.FindBin(sig_xrange[1] - 1e-9))
        else:
            b_lo, b_hi = 1, hist_nbins

    def bin_sig(left, right):
        """Compute S/sqrt(D) for bin range [left, right] where D=Data."""
        b = B.Integral(left, right); s = S.Integral(left, right) / norm
        return (s / (b + eps)**0.5, b, s) if b >= 1 else (0, b, s)

    def make_result(sig, edges):
        """Create result dict with edges from bin numbers.
        edges: list of bin boundaries [left, mid1, mid2, ..., right]
        gain: % improvement of sig over gain_ref (bulk S/sqrt(D)) — always >= 0."""
        gain = 100.0 * (sig / gain_ref - 1.0) if gain_ref > 0 else 0.0
        d = {"sig": sig, "gain": gain, "left_bin": edges[0], "right_bin": edges[-1]}
        d["left_edge"] = B.GetBinLowEdge(edges[0])
        d["right_edge"] = B.GetBinLowEdge(edges[-1] + 1)
        for i, mid in enumerate(edges[1:-1], start=1):
            d[f"mid{i}_bin"] = mid
            d[f"mid{i}_edge"] = B.GetBinLowEdge(mid)
        return d

    def optimize_n_bins(n):
        """Find optimal n-bin partition maximizing sqrt(sum(sig_i^2)).
        n=1: best single contiguous window (left/right free).
        n>1: best N-way split of the FULL range [b_lo, b_hi]; only N-1 internal cuts vary."""
        best_result = None; best_sig = 0
        if n == 1:
            # Best single window: both left and right are free
            for right in range(hist_nbins, 0, -1):
                for left in range(1, right + 1):
                    sig, b, s = bin_sig(left, right)
                    if b >= 1 and sig >= best_sig:
                        best_sig = sig
                        best_result = make_result(sig, [left, right])
        else:
            # Fix full range [b_lo, b_hi], search only internal midpoints
            left, right = b_lo, b_hi
            for mids in combinations(range(left, right), n - 1):
                sigs = []; all_valid = True
                for i in range(n):
                    l = left if i == 0 else mids[i-1] + 1
                    r = mids[i] if i < n - 1 else right
                    sig_i, b_i, _ = bin_sig(l, r)
                    if b_i < 1: all_valid = False; break
                    sigs.append(sig_i)
                if not all_valid: continue
                sig_quad = sum(s**2 for s in sigs)**0.5
                if sig_quad > best_sig:
                    best_sig = sig_quad
                    edges = [left] + [m + 1 for m in mids] + [right]
                    best_result = make_result(sig_quad, edges)
        return best_result

    # Compute optimizations for 1 to Nbins
    results = {"initial": initial, "gain_ref": gain_ref, "nbins": hist_nbins}
    for n in range(1, Nbins + 1):
        opt = optimize_n_bins(n)
        results[f"opt_{n}bin"] = opt
        if n == 1 and opt is None:
            return None  # No valid bins at all

    return results


def build_expressions(variable_list):
    expressions = set()
    for var in variable_list:
        if var.name.isidentifier():
            expressions.add(var.name)
        else:
            tokens = re.findall(r"\b[a-zA-Z_]\w*\b", var.name)
            expressions.update(tokens)
    return list(expressions)


def safe_get(e, field, default):
    if field in e.fields:
        return e[field]
    else:
        print(f"   >> Warning: field \"{field}\" not found - using default")
        return default


def short_fname(fname, max_len=38):
    """Shorten filename: keep first 35 chars + '...' (no _tree.root suffix)"""
    # Remove _tree.root or _merged_tree.root suffix
    base = fname.replace("_merged_tree.root", "").replace("_tree.root", "")
    if len(base) > max_len - 3:
        short = base[:max_len-3] + "..."
    else:
        short = base
    return short.ljust(max_len)


def _make_pseudodata_hist(h_data, seed, fit_lo=0.60, fit_hi=0.90, sr_lo=0.90, sr_hi=None, n_fit_bins=10, template_hist=None):
    """Poisson-sample pseudodata in [sr_lo, sr_hi] (sr_hi=None means open-ended) from
    template_hist (Bernstein fit) when provided, otherwise fit Poly1 to the last n_fit_bins
    bins with center < fit_hi.
    Returns (h_pseudo, h_poly_pred, fit_x_lo, fit_x_hi).
    Seed is deterministic — same Xbin boundaries always give the same pseudodata."""
    n = h_data.GetNbinsX()
    h_pseudo    = h_data.Clone(f"h_pseudo_{h_data.GetName()}");    h_pseudo.Reset();    h_pseudo.SetDirectory(0)
    h_poly_pred = h_data.Clone(f"h_polypred_{h_data.GetName()}"); h_poly_pred.Reset(); h_poly_pred.SetDirectory(0)
    # Garwood/Neyman asymmetric Poisson errors (CMS Stat. Committee recommendation) for the
    # pseudodata counts, matching h_data's error convention.
    h_pseudo.SetBinErrorOption(ROOT.TH1.kPoisson)
    rng = np.random.default_rng(int(seed) % (2**31))

    def _in_blind(xc):
        return xc >= sr_lo and (sr_hi is None or xc <= sr_hi)

    if template_hist is not None:
        # Always use the Bernstein template when available — it's already the correct,
        # properly-extrapolated prediction. A low predicted yield in the blind bins is real
        # information (small/zero Poisson-sampled counts, pulls near 0), not a reason to drop
        # pseudodata in favor of an inferior Poly1 linear extrapolation.
        for ib in range(1, n + 1):
            xc = h_data.GetBinCenter(ib)
            if _in_blind(xc):
                mu = max(0.0, float(template_hist.GetBinContent(ib)))
                h_poly_pred.SetBinContent(ib, mu)
                n_ps = int(rng.poisson(mu))
                h_pseudo.SetBinContent(ib, n_ps)
        return h_pseudo, h_poly_pred, -1., -1.

    # Fallback: Poly1 fit over last n_fit_bins unblinded bins
    all_bins = []
    for ib in range(1, n + 1):
        xc = h_data.GetBinCenter(ib)
        yc = h_data.GetBinContent(ib)
        ye = h_data.GetBinError(ib)
        if xc < fit_hi and ye > 0:
            all_bins.append((xc, yc, ye))
    fit_bins = all_bins[-n_fit_bins:] if len(all_bins) >= n_fit_bins else all_bins
    if len(fit_bins) < 2:
        return None, None, None, None
    xs  = np.array([b[0] for b in fit_bins])
    ys  = np.array([b[1] for b in fit_bins])
    yes = np.array([b[2] for b in fit_bins])
    coeffs = np.polyfit(xs, ys, deg=1, w=1.0 / yes)
    poly   = np.poly1d(coeffs)
    fit_x_lo, fit_x_hi = float(xs[0]), float(xs[-1])
    for ib in range(1, n + 1):
        xc = h_data.GetBinCenter(ib)
        if _in_blind(xc):
            mu = max(0.0, float(poly(xc)))
            h_poly_pred.SetBinContent(ib, mu)
            n_ps = int(rng.poisson(mu))
            h_pseudo.SetBinContent(ib, n_ps)
    return h_pseudo, h_poly_pred, fit_x_lo, fit_x_hi


def fit_bernstein_to_hist(h_data, order, fit_lo=0.1, fit_hi=0.9, fit_exclude_lo=None, fit_exclude_hi=None):
    """Fit Bernstein polynomial of given order to h_data in [fit_lo, fit_hi].
    Bins in [fit_exclude_lo, fit_exclude_hi] are skipped during fitting (but polynomial is
    evaluated there — used for interior blind regions like the Higgs mSD window).
    x is normalised to [0,1] using the histogram axis range for numerical stability.
    Returns (h_fit, chi2, ndf) or (None, None, None) on failure.
    h_fit has fit values over the full histogram range."""
    from math import comb as _comb
    n = h_data.GetNbinsX()
    h_xlo = h_data.GetXaxis().GetXmin()
    h_xhi = h_data.GetXaxis().GetXmax()
    _span = h_xhi - h_xlo if h_xhi > h_xlo else 1.0
    def _norm(x): return (x - h_xlo) / _span

    xc = np.array([h_data.GetBinCenter(i)  for i in range(1, n + 1)])
    yc = np.array([h_data.GetBinContent(i) for i in range(1, n + 1)])
    ye = np.array([h_data.GetBinError(i)   for i in range(1, n + 1)])
    _nonempty = ye > 0  # True for bins with data; empty bins excluded from chi² and ndf
    ye = np.where(_nonempty, ye, 1.0)  # safe value for extrapolation drawing; not used in fit
    mask = (xc >= fit_lo) & (xc <= fit_hi) & _nonempty
    if fit_exclude_lo is not None and fit_exclude_hi is not None:
        mask &= ~((xc >= fit_exclude_lo) & (xc <= fit_exclude_hi))
    if mask.sum() <= order + 1:
        return None, None, None
    xf_norm = _norm(xc[mask]); yf = yc[mask]; ef = ye[mask]
    k = np.arange(order + 1)
    B = np.array([[_comb(order, ki) * x**ki * (1 - x)**(order - ki) for ki in k] for x in xf_norm])
    W = 1.0 / ef**2
    coeffs, _, _, _ = np.linalg.lstsq(np.sqrt(W)[:, None] * B, np.sqrt(W) * yf, rcond=None)
    chi2 = float(np.sum(W * (yf - B @ coeffs)**2))
    ndf  = int(mask.sum()) - (order + 1)
    h_fit = h_data.Clone(f"h_bern{order}_{h_data.GetName()}")
    h_fit.SetDirectory(0)
    h_fit.Reset()
    for i in range(1, n + 1):
        t = _norm(h_data.GetBinCenter(i))
        val = float(sum(coeffs[ki] * _comb(order, ki) * t**ki * (1 - t)**(order - ki) for ki in range(order + 1)))
        h_fit.SetBinContent(i, max(0.0, val))
        h_fit.SetBinError(i, 0)
    return h_fit, chi2, ndf


def load_select_branches(files, expressions, n=None, bdt_type=None, verbose=False, channel=None, warn=True, mlp_mode=False, dnn_mode=False, flat=False, precut=None):
    events_list = []
    file_info = []  # Store (filename, nevents) for verbose output
    flat_suffix = "_FLAT" if flat else ""
    # Support BDT, MLP, and DNN modes
    if dnn_mode:
        score_branch_name = f"DNN_{bdt_type}" if bdt_type else None
        score_dir = f"DNN_scores_{channel}_{bdt_type}{flat_suffix}" if (bdt_type and channel) else None
    elif mlp_mode:
        score_branch_name = f"MLP_{bdt_type}" if bdt_type else None
        score_dir = f"MLP_scores_{channel}_{bdt_type}{flat_suffix}" if (bdt_type and channel) else None
    else:
        score_branch_name = f"BDT_{bdt_type}" if bdt_type else None
        score_dir = f"BDT_scores_{channel}_{bdt_type}{flat_suffix}" if (bdt_type and channel) else ("BDT_scores" if bdt_type else None)
    # Keep backward compatibility with bdt_branch_name and bdt_dir
    bdt_branch_name = score_branch_name
    bdt_dir = score_dir
    missing_branches = set()

    # Accumulate genEventSumw from Runs tree for proper MC normalization
    total_genEventSumw = 0.0

    for f in files:
        # Open ROOT file with uproot
        file_handle = uproot.open(f)
        tree = file_handle["Events"]

        # Read genEventSumw from Runs tree for proper MC normalization
        if "Runs" in file_handle:
            runs_tree = file_handle["Runs"]
            # Try different possible branch names
            for branch_name in ["genEventSumw", "genEventSumw_"]:
                if branch_name in runs_tree.keys():
                    genEventSumw_arr = runs_tree[branch_name].array()
                    total_genEventSumw += float(np.sum(genEventSumw_arr))
                    break

        sel_branches = []
        for b in expressions:
            if b in tree.keys():
                sel_branches.append(b)
            else:
                missing_branches.add(b)

        # Get number of entries in the tree
        n_entries = tree.num_entries if n is None else min(tree.num_entries, n)

        # Load branches from main file (may be empty if only BDT branches requested)
        # Apply preselection cut at read time to avoid loading O(10M) raw events into RAM.
        # Try/except: some BKG files store ak15_* as std::vector<float> (jagged), which
        # causes bitwise_and failures in awkward when used in uproot cut= expressions.
        _cut = precut if precut and all(b in tree.keys() for b in ["ak15_pt", "v_pt", "dphi_V_ak15"]) else None
        if sel_branches:
            try:
                events = tree.arrays(sel_branches, entry_stop=n, cut=_cut)
            except TypeError:
                events = tree.arrays(sel_branches, entry_stop=n)
        else:
            events = None

        if verbose:
            file_info.append((os.path.basename(f), n_entries))

        # Try to load BDT scores if bdt_type is specified and BDT branch is requested
        if bdt_type and bdt_branch_name in expressions:
            # Construct path to BDT score file
            filename = os.path.basename(f)
            bdt_file_path = os.path.join(bdt_dir, filename)

            # Handle XRootD paths
            if f.startswith("root://"):
                # For XRootD paths, use local BDT file in current directory
                bdt_file_path = os.path.join(bdt_dir, filename)

            if os.path.exists(bdt_file_path):
                try:
                    bdt_file_handle = uproot.open(bdt_file_path)
                    bdt_tree = bdt_file_handle["Events"]
                    _score_cut = precut if precut and all(b in bdt_tree.keys() for b in ["ak15_pt", "v_pt", "dphi_V_ak15"]) else None
                    bdt_scores = bdt_tree.arrays([bdt_branch_name], entry_stop=n, cut=_score_cut)
                    # If no other branches were loaded, use BDT scores directly
                    if events is None:
                        events = bdt_scores
                    else:
                        events[bdt_branch_name] = bdt_scores[bdt_branch_name]
                    print(f"    Loaded {len(bdt_scores)} {bdt_branch_name} scores from {bdt_file_path}")
                except Exception as e:
                    print(f"  Warning: Could not load BDT scores from {short_fname(os.path.basename(bdt_file_path))}: {e}")
                    # Create dummy array with -999 values
                    if events is not None and sel_branches:
                        events[bdt_branch_name] = ak.full_like(events[sel_branches[0]], -0.5)
                    else:
                        events = ak.Array({bdt_branch_name: np.full(n_entries, -0.5)})
            else:
                print(f"  Warning: BDT score file not found: {bdt_file_path}")
                print(f"  Run: python3 Compute_BDT.py  (computes both KIN and ParT by default)")
                # Create dummy array with -999 values
                if events is not None and sel_branches:
                    events[bdt_branch_name] = ak.full_like(events[sel_branches[0]], -0.5)
                else:
                    events = ak.Array({bdt_branch_name: np.full(n_entries, -0.5)})

        # Ensure events is not None before appending
        if events is not None:
            events_list.append(events)

    # Collect missing branches (excluding known non-branch tokens) for consolidated warning
    non_branch_tokens = {'abs', 'np', 'pi', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'arctan2', 'PI'}
    real_missing = missing_branches - non_branch_tokens

    if not events_list:
        raise RuntimeError(
            f"load_select_branches: no events loaded from {len(files)} file(s). "
            f"Either the file list was empty (a glob over AFS/EOS can silently return [] if the "
            f"directory listing transiently fails — common when many jobs run concurrently), or "
            f"none of the requested branches exist in the trees. Files: {list(files)[:3]}"
        )
    result = ak.concatenate(events_list, axis=0)

    # Return file_info, genEventSumw, and missing branches if verbose for caller to print
    if verbose:
        return result, file_info, total_genEventSumw, real_missing
    return result, total_genEventSumw, real_missing


def _has_var(W, expr):
    """Return True if all fields referenced in expr are present in W."""
    if expr in W.fields:
        return True
    needed = [f for f in W.fields if re.search(r'\b' + re.escape(f) + r'\b', expr)]
    return len(needed) > 0

def _get_vals(W, expr, sel_mask):
    """Return numpy array of expr evaluated on W[sel_mask]. Handles raw fields and numpy expressions."""
    if expr in W.fields:
        return W[expr][sel_mask]
    local = {f: ak.to_numpy(W[f][sel_mask]) for f in W.fields if re.search(r'\b' + re.escape(f) + r'\b', expr)}
    local.update({'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp, 'arccos': np.arccos})
    return eval(expr, {"__builtins__": {}}, local)

def _apply_cdf_map(vals, cdf_map):
    """Apply a CDF map (cdf_x, cdf_y) to vals, remapping them to [0,1] with flat background.
    Identity operation when cdf_map is None."""
    if cdf_map is None:
        return vals
    arr = np.asarray(vals, dtype=np.float64)
    return np.interp(arr, cdf_map[0], cdf_map[1])

def plot_variable(events, var_name, selection, bins, xlabel, output_prefix, logy=False, region_label=None, blind_range=None, genEventSumw=None, signal_names=None, zjets_samples=None, wjets_samples=None, xsec_zjets=None, xsec_wjets=None, zjets_label="Z+jets", wjets_label="W+jets", signal_legend_label=None, channel=None, plot_index=0, cdf_flat=False, blinded_label=None, overlay_hist=None, cr_template_hist=None, tf_hist=None, pred_params_label=None, overlay_hist2=None, pred_extra_label=None, bernstein_fits=None, bernstein_fit_range=(0.1, 0.9), val_transform=None, bernstein_winner_idx=None, pseudodata_seed=None, ratio_yrange=None, upper_pad_only=False, sig_xrange=None, zero_edge_bins=None, pred3_yrange=False, sig_scale_factor=None):
    """
    Plot a variable for signal and data.

    Args:
        genEventSumw: dict with signal sample names as keys for proper MC normalization
        signal_names: list of signal sample names (e.g., ["WminusH", "WplusH"] for 1l)
        zjets_samples: list of Z+jets sample names
        wjets_samples: list of W+jets sample names
        xsec_zjets: dict of cross-sections for Z+jets samples
        xsec_wjets: dict of cross-sections for W+jets samples
        zjets_label: label for Z+jets in legend (e.g., "Z+jets", "DY")
        channel: channel for channel-specific MC weights (e.g., "0l", "1l", "2l")
        wjets_label: label for W+jets in legend
    """
    if signal_names is None:
        signal_names = ["WminusH", "WplusH"]  # Default for backward compatibility
    nbins, xmin, xmax = bins; NORM = 1;
    # BINLABELS:label1,label2,... in xlabel → replace numeric tick labels with text
    _bin_labels = None
    if xlabel.startswith("BINLABELS:"):
        _bin_labels = xlabel[len("BINLABELS:"):].split(",")
        xlabel = ""
    histos = {};
    # Colors for signal samples (first two) and combined total
    colors = ({signal_names[0]: 62, signal_names[1]: 8} if len(signal_names) >= 2 else {})
    colors["signal_total"] = 2
    skip_weights_for = ["genWeight","lumiwgt","puWeight","l1PreFiringWeight","elEffWeight","muEffWeight","pileupJetIdWeight","topptWeight","topptWeightNNLO","vptWeightEWK","wptWeightEWK","zptWeightEWK","vhWeightEWK","vvWeightNNLO"];

    # Helper function for vectorized histogram filling with underflow/overflow
    def fill_hist_vectorized(h, values, weights=None):
        """Fill ROOT histogram using vectorized numpy operations, including underflow/overflow."""
        # Flatten awkward arrays if needed
        if hasattr(values, 'layout'):
            values = ak.to_numpy(ak.flatten(values, axis=None))
        values = np.asarray(values, dtype=np.float64)

        if weights is not None:
            if hasattr(weights, 'layout'):
                weights = ak.to_numpy(ak.flatten(weights, axis=None))
            weights = np.asarray(weights, dtype=np.float64)
            # Broadcast weights if values were flattened from jagged array
            if len(weights) != len(values):
                weights = np.ones(len(values), dtype=np.float64)

        # Use numpy histogram for speed, then transfer to ROOT
        bin_edges = np.linspace(xmin, xmax, nbins + 1)

        # Compute underflow (values < xmin)
        underflow_mask = values < xmin
        # Compute overflow (values >= xmax)
        overflow_mask = values >= xmax
        # In-range values
        inrange_mask = ~underflow_mask & ~overflow_mask

        if weights is not None:
            # Underflow
            underflow_content = np.sum(weights[underflow_mask])
            underflow_error = math.sqrt(np.sum(weights[underflow_mask]**2))
            # Overflow
            overflow_content = np.sum(weights[overflow_mask])
            overflow_error = math.sqrt(np.sum(weights[overflow_mask]**2))
            # In-range histogram
            counts, _ = np.histogram(values[inrange_mask], bins=bin_edges, weights=weights[inrange_mask])
            weights_sq, _ = np.histogram(values[inrange_mask], bins=bin_edges, weights=weights[inrange_mask]**2)
            errors = np.sqrt(weights_sq)
        else:
            # Underflow
            underflow_content = np.sum(underflow_mask)
            underflow_error = math.sqrt(underflow_content)
            # Overflow
            overflow_content = np.sum(overflow_mask)
            overflow_error = math.sqrt(overflow_content)
            # In-range histogram
            counts, _ = np.histogram(values[inrange_mask], bins=bin_edges)
            errors = np.sqrt(counts)  # Poisson errors for unweighted

        # Fill ROOT histogram with both content and errors
        for i, (c, e) in enumerate(zip(counts, errors)):
            h.SetBinContent(i + 1, c)
            h.SetBinError(i + 1, e)

        # Fill underflow bin (bin 0) and overflow bin (bin nbins+1)
        h.SetBinContent(0, underflow_content)
        h.SetBinError(0, underflow_error)
        h.SetBinContent(nbins + 1, overflow_content)
        h.SetBinError(nbins + 1, overflow_error)

    # --- Build CDF map (once, before filling any histogram) ---
    cdf_map = None
    _score_prefixes = ("BDT_","BDTf_","MLP_","MLPf_","DNN_","DNNf_")
    def _get_valid_vals(sample, mask=None):
        """Return finite raw values for var_name from sample, optionally pre-masked."""
        if not _has_var(sample, var_name): return np.array([])
        m = selection(sample) if mask is None else mask
        if any(var_name.startswith(p) for p in _score_prefixes) and var_name in sample.fields:
            m = m & (sample[var_name] >= 0)
        v = np.asarray(_get_vals(sample, var_name, m), dtype=np.float64)
        v = v[np.isfinite(v)]
        if val_transform is not None:
            v = val_transform(v)
        return v

    if cdf_flat == "Data":
        arr = _get_valid_vals(events["data_2018"])
        if len(arr) > 1:
            arr_s = np.sort(arr)
            cdf_map = (arr_s, np.linspace(0.0, 1.0, len(arr_s)))

    elif cdf_flat == "MC":
        # Collect weighted (value, weight) pairs from all MC background samples
        _all_mc = list(zjets_samples or []) + list(wjets_samples or [])
        _all_mc += ["TTTo2L2Nu","TTToSemiLeptonic","ST_t-channel_top","ST_t-channel_antitop",
                    "ST_tW_top","ST_tW_antitop","ST_s-channel"]
        _all_mc += ["WWTo1L1Nu2Q","WWTo2L2Nu","WZTo1L1Nu2Q","WZTo1L3Nu","WZTo3LNu",
                    "WZTo2Q2L","WZTo2Q2Nu","ZZTo2Q2L","ZZTo2L2Nu","ZZTo2Q2Nu","ZZTo4L"]
        _mc_vals, _mc_wgts = [], []
        for _sname in _all_mc:
            if _sname not in events or not _has_var(events[_sname], var_name): continue
            W = events[_sname]; S = selection(W)
            if any(var_name.startswith(p) for p in _score_prefixes) and var_name in W.fields:
                S = S & (W[var_name] >= 0)
            v = np.asarray(_get_vals(W, var_name, S), dtype=np.float64)
            fin = np.isfinite(v)
            v = v[fin]
            if len(v) == 0: continue
            # Compute per-event weight (xsec × lumi × pu × ...) / sumGenW
            _xsec = {**xsec_zjets, **xsec_wjets, **xsec_Top, **xsec_VV}.get(_sname, None)
            if _xsec is None: continue
            _sumw = genEventSumw.get(_sname, 0) if genEventSumw else 0
            if _sumw <= 0: _sumw = float(ak.sum(W["genWeight"]))
            w = (ak.to_numpy(W["genWeight"][S]) * _xsec
                 * ak.to_numpy(W["lumiwgt"][S])
                 * ak.to_numpy(safe_get(W, "puWeight", 1.0)[S])) / _sumw
            w = w[fin]
            _mc_vals.append(v); _mc_wgts.append(w)
        if _mc_vals:
            all_v = np.concatenate(_mc_vals); all_w = np.concatenate(_mc_wgts)
            idx = np.argsort(all_v)
            all_v, all_w = all_v[idx], all_w[idx]
            cumw = np.cumsum(all_w); cumw /= cumw[-1]  # normalised weighted CDF
            cdf_map = (all_v, cumw)

    elif cdf_flat == "Signal":
        _sig_vals, _sig_wgts = [], []
        for _sname in signal_names:
            if _sname not in events or not _has_var(events[_sname], var_name): continue
            W = events[_sname]; S = selection(W)
            if any(var_name.startswith(p) for p in _score_prefixes) and var_name in W.fields:
                S = S & (W[var_name] >= 0)
            v = np.asarray(_get_vals(W, var_name, S), dtype=np.float64)
            fin = np.isfinite(v); v = v[fin]
            if len(v) == 0: continue
            _xsec = xsec_x_BR.get(_sname, None)
            if _xsec is None: continue
            _sumw = genEventSumw.get(_sname, 0) if genEventSumw else 0
            if _sumw <= 0: _sumw = float(ak.sum(W["genWeight"]))
            w = (ak.to_numpy(W["genWeight"][S]) * _xsec
                 * ak.to_numpy(W["lumiwgt"][S])
                 * ak.to_numpy(safe_get(W, "puWeight", 1.0)[S])) / _sumw
            w = w[fin]
            _sig_vals.append(v); _sig_wgts.append(w)
        if _sig_vals:
            all_v = np.concatenate(_sig_vals); all_w = np.concatenate(_sig_wgts)
            idx = np.argsort(all_v); all_v, all_w = all_v[idx], all_w[idx]
            cumw = np.cumsum(all_w); cumw /= cumw[-1]
            cdf_map = (all_v, cumw)

    def _get_val(sample, var, mask):
        """Get values for var from sample[mask], apply CDF remap then val_transform."""
        v = _apply_cdf_map(_get_vals(sample, var, mask), cdf_map)
        if val_transform is not None:
            v = val_transform(np.asarray(v, dtype=np.float64))
        return v

    if cdf_flat == "Data":
        xlabel = xlabel + "  #it{(data-flat)}"
    elif cdf_flat == "MC":
        xlabel = xlabel + "  #it{(MC-flat)}"
    elif cdf_flat == "Signal":
        xlabel = xlabel + "  #it{(signal-flat)}"

    h_data = None;
    if _has_var(events["data_2018"], var_name):
        h_data = ROOT.TH1F(f"data_2018_{plot_index}", "", nbins, xmin, xmax);
        h_data.SetDirectory(0);
        sel_mask = selection(events["data_2018"])
        if any(var_name.startswith(p) for p in _score_prefixes) and var_name in events["data_2018"].fields:
            score_valid = events["data_2018"][var_name] >= 0
            sel_mask = sel_mask & score_valid
        val = _get_val(events["data_2018"], var_name, sel_mask)
        fill_hist_vectorized(h_data, val)
        UnderOverFlow1D(h_data, reject_underflow=(zero_edge_bins in ("first", "both")),
                                 reject_overflow=(zero_edge_bins == "both"))  # Move (or reject) under/overflow
        h_data.SetMarkerStyle(20); h_data.SetMarkerSize(0.5); h_data.SetLineColor(ROOT.kBlack); h_data.SetMarkerColor(ROOT.kBlack); h_data.SetLineWidth(2);
        # Garwood/Neyman asymmetric Poisson errors (CMS Stat. Committee recommendation),
        # not the symmetric sqrt(N) default — avoids undercoverage at low counts.
        # fill_hist_vectorized/UnderOverFlow1D call SetBinError() explicitly, which silently
        # activates Sumw2 — and SetBinErrorOption(kPoisson) has no effect while Sumw2 is active,
        # so it must be cleared first (matches the CMS recipe's h1->Sumw2(kFALSE) step).
        h_data.Sumw2(ROOT.kFALSE)
        h_data.SetBinErrorOption(ROOT.TH1.kPoisson)

    for sample in signal_names:
        if sample not in events or not _has_var(events[sample], var_name): continue;
        W = events[sample];
        S           = selection(W);
        # Filter out invalid scores (-999 = invalid input vars) for all scorer variables
        if any(var_name.startswith(p) for p in ("BDT_","BDTf_","MLP_","MLPf_","DNN_","DNNf_")) and var_name in W.fields:
            score_valid = W[var_name] >= 0
            S = S & score_valid
        val         = _get_val(W, var_name, S);
        lumi        = W["lumiwgt"][S];
        pu          = W["puWeight"][S];
        l1PreFiring = safe_get(W, "l1PreFiringWeight", 1.0)[S];
        muEffWeight = safe_get(W, "muEffWeight", 1.0)[S];
        elEffWeight = safe_get(W, "elEffWeight", 1.0)[S];
        pileupJetIdWeight = safe_get(W, "pileupJetIdWeight", 1.0)[S];
        hem_weight = get_channel_mc_weight(W, S, channel)  # HEM weight for 0L MC
        # Use genEventSumw from Runs tree for proper normalization (if available)
        sum_gen_wgt = genEventSumw[sample] if (genEventSumw and sample in genEventSumw and genEventSumw[sample] > 0) else ak.sum(W["genWeight"])
        if var_name not in skip_weights_for:
            weights = W["genWeight"][S] * xsec_x_BR[sample] * lumi * ( l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight * hem_weight ) / sum_gen_wgt;
        else:
            weights = None

        h = ROOT.TH1F(f"{sample}_{plot_index}", "", nbins, xmin, xmax);
        h.SetDirectory(0);
        fill_hist_vectorized(h, val, weights)
        UnderOverFlow1D(h, reject_underflow=(zero_edge_bins in ("first", "both")),
                           reject_overflow=(zero_edge_bins == "both"))  # Move (or reject) under/overflow

        h.SetLineColor(colors[sample]);
        h.SetLineWidth(2);
        histos[sample] = h;

    # Create Z+jets background histogram
    h_zjets = None
    n_zjets_events = 0.0
    if zjets_samples and xsec_zjets:
        for zj_sample in zjets_samples:
            if zj_sample not in events: continue
            if not _has_var(events[zj_sample], var_name): continue
            W = events[zj_sample]
            S = selection(W)
            # Filter out invalid scores (-999 = invalid input vars) for all scorer variables
            if any(var_name.startswith(p) for p in ("BDT_","BDTf_","MLP_","MLPf_","DNN_","DNNf_")) and var_name in W.fields:
                score_valid = W[var_name] >= 0
                S = S & score_valid
            val = _get_val(W, var_name, S)
            lumi = W["lumiwgt"][S]
            pu = W["puWeight"][S]
            l1PreFiring = safe_get(W, "l1PreFiringWeight", 1.0)[S]
            muEffWeight = safe_get(W, "muEffWeight", 1.0)[S]
            elEffWeight = safe_get(W, "elEffWeight", 1.0)[S]
            pileupJetIdWeight = safe_get(W, "pileupJetIdWeight", 1.0)[S]
            hem_weight = get_channel_mc_weight(W, S, channel)  # HEM weight for 0L MC
            sum_gen_wgt = genEventSumw[zj_sample] if (genEventSumw and zj_sample in genEventSumw and genEventSumw[zj_sample] > 0) else ak.sum(W["genWeight"])
            if var_name not in skip_weights_for:
                weights = W["genWeight"][S] * xsec_zjets[zj_sample] * lumi * (l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight * hem_weight) / sum_gen_wgt
            else:
                weights = None

            h_zj = ROOT.TH1F(f"h_{zj_sample}", "", nbins, xmin, xmax)
            fill_hist_vectorized(h_zj, val, weights)
            UnderOverFlow1D(h_zj, reject_underflow=(zero_edge_bins in ("first", "both")),
                                   reject_overflow=(zero_edge_bins == "both"))

            if h_zjets is None:
                h_zjets = h_zj.Clone("ZJets")
            else:
                h_zjets.Add(h_zj)

    if h_zjets:
        n_zjets_events = h_zjets.Integral()
        zjets_color = ROOT.TColor.GetColor("#f89c20")  # Orange
        h_zjets.SetFillColor(zjets_color)
        h_zjets.SetFillStyle(1001)  # Solid fill
        h_zjets.SetLineColor(zjets_color)  # Same as fill - no outline
        h_zjets.SetLineWidth(0)
        histos["ZJets"] = h_zjets

    # Create W+jets background histogram
    h_wjets = None
    n_wjets_events = 0.0
    if wjets_samples and xsec_wjets:
        for wj_sample in wjets_samples:
            if wj_sample not in events: continue
            if not _has_var(events[wj_sample], var_name): continue
            W = events[wj_sample]
            S = selection(W)
            # Filter out invalid scores (-999 = invalid input vars) for all scorer variables
            if any(var_name.startswith(p) for p in ("BDT_","BDTf_","MLP_","MLPf_","DNN_","DNNf_")) and var_name in W.fields:
                score_valid = W[var_name] >= 0
                S = S & score_valid
            val = _get_val(W, var_name, S)
            lumi = W["lumiwgt"][S]
            pu = W["puWeight"][S]
            l1PreFiring = safe_get(W, "l1PreFiringWeight", 1.0)[S]
            muEffWeight = safe_get(W, "muEffWeight", 1.0)[S]
            elEffWeight = safe_get(W, "elEffWeight", 1.0)[S]
            pileupJetIdWeight = safe_get(W, "pileupJetIdWeight", 1.0)[S]
            hem_weight = get_channel_mc_weight(W, S, channel)  # HEM weight for 0L MC
            sum_gen_wgt = genEventSumw[wj_sample] if (genEventSumw and wj_sample in genEventSumw and genEventSumw[wj_sample] > 0) else ak.sum(W["genWeight"])
            if var_name not in skip_weights_for:
                weights = W["genWeight"][S] * xsec_wjets[wj_sample] * lumi * (l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight * hem_weight) / sum_gen_wgt
            else:
                weights = None

            h_wj = ROOT.TH1F(f"h_{wj_sample}", "", nbins, xmin, xmax)
            fill_hist_vectorized(h_wj, val, weights)
            UnderOverFlow1D(h_wj, reject_underflow=(zero_edge_bins in ("first", "both")),
                                   reject_overflow=(zero_edge_bins == "both"))

            if h_wjets is None:
                h_wjets = h_wj.Clone("WJets")
            else:
                h_wjets.Add(h_wj)

    if h_wjets:
        n_wjets_events = h_wjets.Integral()
        wjets_color = ROOT.TColor.GetColor("#7a21dd")  # Purple
        h_wjets.SetFillColor(wjets_color)
        h_wjets.SetFillStyle(1001)  # Solid fill
        h_wjets.SetLineColor(wjets_color)  # Same as fill - no outline
        h_wjets.SetLineWidth(0)
        histos["WJets"] = h_wjets

    # Create Top background histogram (ttbar + single top)
    h_top = None
    n_top_events = 0.0
    top_samples = ["TTTo2L2Nu", "TTToSemiLeptonic", "ST_t-channel_top", "ST_t-channel_antitop", "ST_tW_top", "ST_tW_antitop", "ST_s-channel"]
    for top_sample in top_samples:
        if top_sample not in events: continue
        if not _has_var(events[top_sample], var_name): continue
        W = events[top_sample]
        S = selection(W)
        # Filter out invalid scores (-999 = invalid input vars) for all scorer variables
        if any(var_name.startswith(p) for p in ("BDT_","BDTf_","MLP_","MLPf_","DNN_","DNNf_")) and var_name in W.fields:
            score_valid = W[var_name] >= 0
            S = S & score_valid
        val = _get_val(W, var_name, S)
        lumi = W["lumiwgt"][S]
        pu = W["puWeight"][S]
        l1PreFiring = safe_get(W, "l1PreFiringWeight", 1.0)[S]
        muEffWeight = safe_get(W, "muEffWeight", 1.0)[S]
        elEffWeight = safe_get(W, "elEffWeight", 1.0)[S]
        pileupJetIdWeight = safe_get(W, "pileupJetIdWeight", 1.0)[S]
        hem_weight = get_channel_mc_weight(W, S, channel)  # HEM weight for 0L MC
        sum_gen_wgt = genEventSumw[top_sample] if (genEventSumw and top_sample in genEventSumw and genEventSumw[top_sample] > 0) else ak.sum(W["genWeight"])
        if var_name not in skip_weights_for:
            weights = W["genWeight"][S] * xsec_Top[top_sample] * lumi * (l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight * hem_weight) / sum_gen_wgt
        else:
            weights = None

        h_t = ROOT.TH1F(f"h_{top_sample}", "", nbins, xmin, xmax)
        fill_hist_vectorized(h_t, val, weights)
        UnderOverFlow1D(h_t, reject_underflow=(zero_edge_bins in ("first", "both")),
                              reject_overflow=(zero_edge_bins == "both"))

        if h_top is None:
            h_top = h_t.Clone("Top")
        else:
            h_top.Add(h_t)

    if h_top:
        n_top_events = h_top.Integral()
        top_color = ROOT.TColor.GetColor("#5790fc")
        h_top.SetFillColor(top_color)
        h_top.SetFillStyle(1001)  # Solid fill
        h_top.SetLineColor(top_color)  # Same as fill - no outline
        h_top.SetLineWidth(0)
        histos["Top"] = h_top

    # Create VV (diboson) background histogram
    h_vv = None
    n_vv_events = 0.0
    vv_samples = ["WWTo1L1Nu2Q", "WWTo2L2Nu", "WZTo1L1Nu2Q", "WZTo1L3Nu", "WZTo3LNu", "WZTo2Q2L", "WZTo2Q2Nu", "ZZTo2Q2L", "ZZTo2L2Nu", "ZZTo2Q2Nu", "ZZTo4L"]
    for vv_sample in vv_samples:
        if vv_sample not in events: continue
        if not _has_var(events[vv_sample], var_name): continue
        W = events[vv_sample]
        S = selection(W)
        # Filter out invalid scores (-999 = invalid input vars) for all scorer variables
        if any(var_name.startswith(p) for p in ("BDT_","BDTf_","MLP_","MLPf_","DNN_","DNNf_")) and var_name in W.fields:
            score_valid = W[var_name] >= 0
            S = S & score_valid
        val = _get_val(W, var_name, S)
        lumi = W["lumiwgt"][S]
        pu = W["puWeight"][S]
        l1PreFiring = safe_get(W, "l1PreFiringWeight", 1.0)[S]
        muEffWeight = safe_get(W, "muEffWeight", 1.0)[S]
        elEffWeight = safe_get(W, "elEffWeight", 1.0)[S]
        pileupJetIdWeight = safe_get(W, "pileupJetIdWeight", 1.0)[S]
        hem_weight = get_channel_mc_weight(W, S, channel)  # HEM weight for 0L MC
        sum_gen_wgt = genEventSumw[vv_sample] if (genEventSumw and vv_sample in genEventSumw and genEventSumw[vv_sample] > 0) else ak.sum(W["genWeight"])
        if var_name not in skip_weights_for:
            weights = W["genWeight"][S] * xsec_VV[vv_sample] * lumi * (l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight * hem_weight) / sum_gen_wgt
        else:
            weights = None

        h_v = ROOT.TH1F(f"h_{vv_sample}", "", nbins, xmin, xmax)
        fill_hist_vectorized(h_v, val, weights)
        UnderOverFlow1D(h_v, reject_underflow=(zero_edge_bins in ("first", "both")),
                              reject_overflow=(zero_edge_bins == "both"))

        if h_vv is None:
            h_vv = h_v.Clone("VV")
        else:
            h_vv.Add(h_v)

    if h_vv:
        n_vv_events = h_vv.Integral()
        vv_color = ROOT.TColor.GetColor("#e42536")
        h_vv.SetFillColor(vv_color)
        h_vv.SetFillStyle(1001)  # Solid fill
        h_vv.SetLineColor(vv_color)  # Same as fill - no outline
        h_vv.SetLineWidth(0)
        histos["VV"] = h_vv

    # Create h_total ONCE (removed duplicate block)
    n_data_events = h_data.Integral() if h_data else 0
    n_signal_events = 0.0
    if len(signal_names) >= 2 and signal_names[0] in histos and signal_names[1] in histos:
        h_total = histos[signal_names[0]].Clone("signal_total"); h_total.Add(histos[signal_names[1]]);
        h_total.SetLineColor(colors["signal_total"]); h_total.SetLineWidth(2);
        h_total.SetLineStyle(1); h_total.SetMarkerStyle(1); h_total.SetMarkerSize(0);
        histos["signal_total"] = h_total;
        n_signal_events = h_total.Integral()  # Store unscaled signal integral

        if h_data and h_total.Integral() > 0 and h_data.Integral() > 0:
            NORM = h_data.Integral() / h_total.Integral() / _SIG_NORM_DIV;

    for key in signal_names + ["signal_total"]:
        if key in histos:
            if histos[key].Integral() > 0: histos[key].Scale(NORM);

    # Collect info for consolidated print at end
    n_total_bkg = n_zjets_events + n_wjets_events + n_top_events + n_vv_events
    plot_info = {"data": n_data_events, "signal": n_signal_events, "bkg": n_total_bkg}

    # NOTE: zero_edge_bins no longer needs a post-hoc "blank the whole edge bin" pass — the
    # underflow/overflow rejection now happens at fill time (UnderOverFlow1D(reject_underflow=...,
    # reject_overflow=...) above), so bin 1 / bin nbins already hold only their genuine in-range
    # fill, with out-of-domain events discarded rather than leaking in and then being hidden.

    # Create canvas — upper-pad-only mode uses a shorter canvas (1:0.7 aspect ratio)
    c=ROOT.TCanvas(f"c_{plot_index}", f"c_{plot_index}", 490, 343 if upper_pad_only else 490)
    c.cd()

    # Upper pad for main plot
    if upper_pad_only:
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0, 1, 1.0)
        pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.035)
        pad1.SetTopMargin(0.11)
        pad1.SetBottomMargin(0.14)  # Show x-axis labels
    else:
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0.36, 1, 1.0)
        pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.035)
        pad1.SetTopMargin(0.08)
        pad1.SetBottomMargin(0.02)  # Small gap, no x-axis labels
    pad1.Draw()
    # ROOT only clips drawn primitives to the pad's physical rectangle, not to the frame's
    # axis range. When y_min > 0 (e.g. pred3_yrange), histogram content below y_min would
    # otherwise paint straight through pad1's bottom margin toward pad2. Clip to the frame box.
    pad1.SetBit(ROOT.TPad.kClipFrame)

    # Lower pad for S/sqrt(D) and Data/MC ratio (skipped in upper_pad_only mode)
    if not upper_pad_only:
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.36)
        pad2.SetLeftMargin(0.12)
        pad2.SetRightMargin(0.035)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.25)  # Optimised bottom margin for x-axis labels
        pad2.Draw()

    # Draw main plot on upper pad
    pad1.cd()
    if logy: pad1.SetLogy()
    pad1.SetTickx(1)
    pad1.SetTicky(1)

    all_maxima = [h.GetMaximum() for h in histos.values()];
    if h_data: all_maxima.append(h_data.GetMaximum());
    if not all_maxima:
        print(f" --> Skipping plot '{var_name}' (no histograms to draw)");
        return None;
    _pred_mode = (overlay_hist is not None) or (overlay_hist2 is not None) or (bernstein_fits is not None)
    _no_sig    = not signal_names
    # Highest VISIBLE yield across data and the stacked BKG MC — the reference for the y-axis top.
    # Blinded bins are excluded: h_data is the FULL unblinded data (the blinded clone is made
    # later), so a tall bin inside the SR blind window would otherwise set the scale while being
    # hidden, leaving the visible data looking tiny with huge headroom (VR has no blinding).
    # The stacked MC total is not the max of the individual components, so it is summed per bin
    # here (h_bkg_total is only built further below, after the frame is created).
    _vis_max = 0.0
    if h_data is not None:
        _bkg_keys = [k for k in ("Top", "ZJets", "WJets", "VV") if k in histos]
        def _visible_max(get_bin_val):
            _vals = [get_bin_val(i) for i in range(1, nbins + 1)
                     if blind_range is None
                     or not (blind_range[0] <= h_data.GetBinCenter(i) <= blind_range[1])]
            return max(_vals) if _vals else 0.0
        _vis_max = _visible_max(lambda i: h_data.GetBinContent(i))
        if _bkg_keys:
            _vis_max = max(_vis_max,
                           _visible_max(lambda i: sum(histos[k].GetBinContent(i) for k in _bkg_keys)))
        if _vis_max <= 0:
            _vis_max = h_data.GetMaximum()

    if pred3_yrange and h_data is not None:
        y_max = 2.7 * _vis_max
        y_min = 0.0
    elif h_data is not None and not logy and not _pred_mode:
        # Plain (non-PRED) plots, any channel: 2x the highest data / stacked-BKG-MC bin.
        y_max = 2.0 * _vis_max
        y_min = 0
    else:
        y_max = (100 if logy else (1.3 if (_pred_mode and _no_sig) else (1.2 if _pred_mode else 1.9))) * max(all_maxima)
        y_min = 0.7 if logy else 0
    frame = ROOT.TH1F("frame", "", nbins, xmin, xmax); frame.SetMaximum(y_max); frame.SetMinimum(y_min);
    if upper_pad_only:
        frame.GetXaxis().SetTitle(xlabel)
        frame.GetXaxis().SetLabelSize(0.057)
        frame.GetXaxis().SetTitleSize(0.066)
        frame.GetXaxis().SetTitleOffset(0.90)
    else:
        frame.GetXaxis().SetTitle("")  # No x-axis title on upper pad
        frame.GetXaxis().SetLabelSize(0)  # No x-axis labels on upper pad
    frame.GetYaxis().SetTitle("Events / bin")
    frame.GetYaxis().SetLabelSize(0.057)
    frame.GetYaxis().SetTitleSize(0.066)
    frame.GetYaxis().SetTitleOffset(0.90)
    frame.Draw("AXIS")
    frame.GetXaxis().SetNdivisions(510); frame.GetYaxis().SetNdivisions(510);
    frame.GetXaxis().SetTickLength(0.03); frame.GetYaxis().SetTickLength(0.03);

    # Stack backgrounds manually: draw cumulative totals from top to bottom.
    # Layer i = sum of components i..N, so drawing largest-first paints the stack correctly;
    # each layer takes the colour of its topmost component (the one it is cloned from).
    # 0l/1l put ttbar directly above VV (W+jets on top instead); 2l keeps Top on top.
    _stack_order = (["WJets", "ZJets", "Top", "VV"] if channel in ("0l", "1l")
                    else ["Top", "ZJets", "WJets", "VV"])
    _stack_present = [k for k in _stack_order if k in histos]

    h_bkg_total = None
    has_zjets = "ZJets" in histos
    has_wjets = "WJets" in histos
    has_top = "Top" in histos
    has_vv = "VV" in histos

    if _stack_present and not bernstein_fits:
        _stack_layers = []
        for _si in range(len(_stack_present)):
            _hl = histos[_stack_present[_si]].Clone("h_bkg_total" if _si == 0 else f"h_stack_{_si}")
            for _sk in _stack_present[_si + 1:]:
                _hl.Add(histos[_sk])
            _stack_layers.append(_hl)
        h_bkg_total = _stack_layers[0]   # full sum of all present backgrounds
        for _hl in _stack_layers:
            _hl.Draw("HIST SAME")

        # Draw MC stat uncertainty as manual TLines — one vertical line per bin.
        # Errors: sqrt(sum w^2) per bin, propagated in quadrature through Clone+Add (weighted, not sqrt(N)).
        # Manual drawing avoids the horizontal step outline that "E" always adds.
        if h_bkg_total:
            mc_unc_lines = []
            for ibin in range(1, h_bkg_total.GetNbinsX() + 1):
                xc  = h_bkg_total.GetBinCenter(ibin)
                yc  = h_bkg_total.GetBinContent(ibin)
                err = h_bkg_total.GetBinError(ibin)
                if err <= 0: continue
                ln = ROOT.TLine(xc, yc - err, xc, yc + err)
                ln.SetLineColor(ROOT.kGray + 1)
                ln.SetLineWidth(2)
                ln.Draw("SAME")
                mc_unc_lines.append(ln)  # keep references alive until canvas is saved

    # Draw CR data template normalized to SR data total — commented out
    _h_cr_draw = None
    # if cr_template_hist is not None and n_data_events > 0:
    #     _h_cr_draw = cr_template_hist.Clone("h_cr_template_draw")
    #     _h_cr_draw.SetDirectory(0)
    #     cr_integral = _h_cr_draw.Integral()
    #     if cr_integral > 0:
    #         _h_cr_draw.Scale(n_data_events / cr_integral)
    #     _h_cr_draw.SetLineColor(ROOT.kGray + 2)
    #     _h_cr_draw.SetLineWidth(2)
    #     _h_cr_draw.SetFillStyle(0)
    #     _h_cr_draw.SetLineStyle(9)
    #     _h_cr_draw.Draw("HIST SAME")

    # Draw MC_SR/MC_CR transfer factor normalized to SR data total — commented out
    _h_tf_draw = None
    _tf_unc_lines = []
    # if tf_hist is not None and n_data_events > 0:
    #     _h_tf_draw = tf_hist.Clone("h_tf_draw")
    #     _h_tf_draw.SetDirectory(0)
    #     tf_integral = _h_tf_draw.Integral()
    #     if tf_integral > 0:
    #         _h_tf_draw.Scale(n_data_events / tf_integral)
    #     _h_tf_draw.SetLineColor(ROOT.kYellow + 1)
    #     _h_tf_draw.SetLineWidth(2)
    #     _h_tf_draw.SetFillStyle(0)
    #     _h_tf_draw.SetLineStyle(1)
    #     _h_tf_draw.Draw("HIST SAME")
    #     for ibin in range(1, _h_tf_draw.GetNbinsX() + 1):
    #         xc  = _h_tf_draw.GetBinCenter(ibin)
    #         yc  = _h_tf_draw.GetBinContent(ibin)
    #         err = _h_tf_draw.GetBinError(ibin)
    #         if err <= 0: continue
    #         ln = ROOT.TLine(xc, yc - err, xc, yc + err)
    #         ln.SetLineColor(ROOT.kYellow + 1); ln.SetLineWidth(2); ln.Draw("SAME")
    #         _tf_unc_lines.append(ln)

    # Draw Term1-only prediction (Pred1) first so Pred2 (blue) renders on top
    _h_ov2_draw = None
    _pred1_unc_lines = []
    if overlay_hist2 is not None:
        _h_ov2_draw = overlay_hist2['hist'].Clone("h_ov2_draw")
        _h_ov2_draw.SetDirectory(0)
        _h_ov2_draw.SetLineColor(ROOT.kGreen + 2)
        _h_ov2_draw.SetLineWidth(2)
        _h_ov2_draw.SetFillStyle(0)
        _h_ov2_draw.SetLineStyle(2)
        _h_ov2_draw.Draw("HIST SAME")
        for ibin in range(1, _h_ov2_draw.GetNbinsX() + 1):
            xc  = _h_ov2_draw.GetBinCenter(ibin)
            yc  = _h_ov2_draw.GetBinContent(ibin)
            err = _h_ov2_draw.GetBinError(ibin)
            if err <= 0: continue
            ln = ROOT.TLine(xc, yc - err, xc, yc + err)
            ln.SetLineColor(ROOT.kGreen + 2); ln.SetLineWidth(2); ln.Draw("SAME")
            _pred1_unc_lines.append(ln)

    # Draw Pred2 (blue) last among predictions so it sits on top
    _pred_unc_lines = []
    _h_ov_draw = None
    if overlay_hist is not None:
        _h_ov_draw = overlay_hist['hist'].Clone("h_ov_draw")
        _h_ov_draw.SetDirectory(0)
        ov_col = overlay_hist.get('color', ROOT.kBlue)
        _h_ov_draw.SetLineColor(ov_col); _h_ov_draw.SetLineWidth(2); _h_ov_draw.SetFillStyle(0); _h_ov_draw.SetLineStyle(1)
        _h_ov_draw.Draw("HIST SAME")
        for ibin in range(1, _h_ov_draw.GetNbinsX() + 1):
            xc  = _h_ov_draw.GetBinCenter(ibin)
            yc  = _h_ov_draw.GetBinContent(ibin)
            err = _h_ov_draw.GetBinError(ibin)
            if err <= 0: continue
            ln = ROOT.TLine(xc, yc - err, xc, yc + err)
            ln.SetLineColor(ov_col); ln.SetLineWidth(2); ln.Draw("SAME")
            _pred_unc_lines.append(ln)

    # Draw signal on top (histogram style with horizontal lines + vertical error bars).
    # Nominal yields are drawn as-is: bins whose yield is below the frame's y_min simply fall
    # outside the frame and are clipped away (pad1.kClipFrame) — genuinely invisible, rather
    # than flattened onto the y_min baseline. Per-bin error-bar TLines are skipped for those
    # sub-y_min bins (TLine isn't reliably clipped, so it would otherwise bleed into pad2).
    _sig_unc_lines = []
    if "signal_total" in histos:
        _h_sig = histos["signal_total"]
        _sig_col = _h_sig.GetLineColor()
        _h_sig.SetMarkerSize(0)
        _h_sig.SetFillStyle(0)
        _h_sig.SetLineWidth(2)
        _h_sig.SetLineColor(_sig_col)
        _h_sig.Draw("HIST ][ SAME")  # nominal yields; frame clips bins below y_min
        for _ib in range(1, _h_sig.GetNbinsX() + 1):
            _xc = _h_sig.GetBinCenter(_ib)
            _yc = _h_sig.GetBinContent(_ib)
            _err = _h_sig.GetBinError(_ib)
            if (_yc == 0 and _err == 0) or _yc < y_min:
                continue  # empty, or below the frame → no error bar (keep it invisible)
            _ln = ROOT.TLine(_xc, max(_yc - _err, y_min), _xc, _yc + _err)
            _ln.SetLineColor(_sig_col); _ln.SetLineWidth(2); _ln.Draw("SAME")
            _sig_unc_lines.append(_ln)

    # Draw signal median line (50/50 split) on upper pad — commented out
    sig_median_line = None
    # if "signal_total" in histos:
    #     h_sig_med = histos["signal_total"]
    #     total_sig_integral = h_sig_med.Integral()
    #     if total_sig_integral > 0:
    #         cumsum = 0.0
    #         median_edge = None
    #         for ibin in range(1, h_sig_med.GetNbinsX() + 1):
    #             cumsum += h_sig_med.GetBinContent(ibin)
    #             if cumsum >= 0.5 * total_sig_integral:
    #                 median_edge = h_sig_med.GetBinLowEdge(ibin + 1)
    #                 break
    #         if median_edge is not None:
    #             ymin_pad1 = frame.GetMinimum()
    #             ymax_pad1 = frame.GetMaximum()
    #             y_quarter = ymin_pad1 + 0.25 * (ymax_pad1 - ymin_pad1)
    #             sig_median_line = ROOT.TLine(median_edge, ymin_pad1, median_edge, y_quarter)
    #             sig_median_line.SetLineColor(ROOT.kRed)
    #             sig_median_line.SetLineWidth(1)
    #             sig_median_line.SetLineStyle(1)
    #             sig_median_line.Draw("SAME")
    #             sig_median_label = ROOT.TLatex()
    #             sig_median_label.SetTextFont(42)
    #             sig_median_label.SetTextSize(0.040)
    #             sig_median_label.SetTextColor(ROOT.kRed)
    #             sig_median_label.SetTextAlign(31)
    #             sig_median_label.DrawLatex(median_edge, y_quarter, "eff 50%")

    # --- Bernstein polynomial fits (PRED3 mode) ---
    _bern_drawn = []  # list of (TH1F_clone, chi2, ndf, order) kept alive for ROOT
    _bern_colors = [ROOT.kOrange+1, ROOT.kRed+1, ROOT.kAzure-4, ROOT.kGreen+2, ROOT.kViolet+1]
    if bernstein_fits:
        for _bi, ((h_b, chi2_b, ndf_b, ord_b), col_b) in enumerate(zip(bernstein_fits, _bern_colors)):
            if h_b is None:
                continue
            _is_winner = (bernstein_winner_idx is not None and _bi == bernstein_winner_idx)
            hd = h_b.Clone(f"_bdraw_{ord_b}")
            hd.SetDirectory(0)
            hd.SetLineColor(col_b)
            hd.SetLineWidth(2)
            hd.SetLineStyle(1 if _is_winner else 2)
            hd.SetFillStyle(0); hd.SetMarkerSize(0)
            hd.Draw("HIST SAME")
            _bern_drawn.append((hd, chi2_b, ndf_b, ord_b, _is_winner))

    # Calculate significance FIRST using full unblinded data
    sig_text_data = None  # Initialize for later text drawing
    plot_info["sig_str"] = "no signif."
    if h_data is not None and "signal_total" in histos:
        B = h_data; S = histos["signal_total"]  # Use FULL unblinded data for significance
        B_integral = B.Integral(); S_integral = S.Integral()
        # Full-range quad sig: sqrt(Σ s_i²/d_i) over ALL histogram bins — not restricted by sig_xrange.
        # 0-3 bin max metrics use sig_xrange (e.g. blind window only); this captures the full histogram.
        _quad_sig_full = 0.0
        for _ib_f in range(1, nbins + 1):
            _s_f = S.GetBinContent(_ib_f) / NORM
            _d_f = B.GetBinContent(_ib_f)
            if _d_f > 0: _quad_sig_full += (_s_f / math.sqrt(_d_f)) ** 2
        _quad_sig_full = math.sqrt(_quad_sig_full)
        plot_info["quad_sig_full"] = _quad_sig_full
        metrics = compute_significance_metrics(B, S, NORM, xmin, xmax, Nbins=3, sig_xrange=sig_xrange)
        if metrics is not None:
            opt1 = metrics["opt_1bin"]; opt2 = metrics["opt_2bin"]; opt3 = metrics["opt_3bin"]
            initial = metrics["initial"]
            # Collect compact significance info with bin boundaries
            sig_str = f"[{opt1['left_edge']:.0f}-{opt1['right_edge']:.0f}]"
            if opt2: sig_str += f" [{opt2['left_edge']:.0f},{opt2['mid1_edge']:.0f},{opt2['right_edge']:.0f}]"
            if opt3: sig_str += f" [{opt3['left_edge']:.0f},{opt3['mid1_edge']:.0f},{opt3['mid2_edge']:.0f},{opt3['right_edge']:.0f}]"
            plot_info["sig_str"] = sig_str
            # Print optimal bin edges with full precision (useful for [0,1]-range variables)
            print(f"  [OptBins] 1-bin: [{opt1['left_edge']:.4f}, {opt1['right_edge']:.4f}]  sig={opt1['sig']:.4f} ({opt1['gain']:+.0f}%)")
            if opt2: print(f"  [OptBins] 2-bin: [{opt2['left_edge']:.4f} | {opt2['mid1_edge']:.4f} | {opt2['right_edge']:.4f}]  sig={opt2['sig']:.4f} ({opt2['gain']:+.0f}%)")
            if opt3: print(f"  [OptBins] 3-bin: [{opt3['left_edge']:.4f} | {opt3['mid1_edge']:.4f} | {opt3['mid2_edge']:.4f} | {opt3['right_edge']:.4f}]  sig={opt3['sig']:.4f} ({opt3['gain']:+.0f}%)")
            # Also store significance values for summary table
            plot_info["sig_0bin"] = initial
            plot_info["sig_1bin"] = opt1['sig']
            plot_info["sig_2bin"] = opt2['sig'] if opt2 else 0.0
            plot_info["sig_3bin"] = opt3['sig'] if opt3 else 0.0
            # Store significance values for later text and line drawing (in lower pad)
            sig_text_data = {"initial": initial, "opt1": opt1, "opt2": opt2, "opt3": opt3, "gain_ref": metrics.get("gain_ref", 0.0), "nbins": nbins, "quad_sig_full": _quad_sig_full}

    # NOW create blinded data histogram for DRAWING only (significance already calculated above)
    h_data_draw = h_data
    plot_info["blind_str"] = ""
    if h_data and blind_range:
        blind_lo, blind_hi = blind_range
        h_data_draw = h_data.Clone("h_data_blinded")
        for ibin in range(1, h_data_draw.GetNbinsX() + 1):
            bin_center = h_data_draw.GetBinCenter(ibin)
            if blind_lo <= bin_center <= blind_hi:
                h_data_draw.SetBinContent(ibin, 0)
                h_data_draw.SetBinError(ibin, 0)
        # SetBinError() above silently reactivates Sumw2 on the WHOLE histogram (all bins,
        # not just the blinded ones) — which makes ROOT fall back to symmetric sqrt(N) errors
        # everywhere, undoing the Garwood/Poisson option. Re-assert it after blinding.
        h_data_draw.Sumw2(ROOT.kFALSE)
        h_data_draw.SetBinErrorOption(ROOT.TH1.kPoisson)
        plot_info["blind_str"] = f", blind [{blind_lo:.0f}-{blind_hi:.0f}]"

    # Draw markers+Garwood error bars as a graph of ONLY the filled bins, so zero-content bins
    # get no marker and no (~1.84 upper) Poisson error bar. Building a TGraphAsymmErrors keeps the
    # asymmetric Garwood errors on filled bins while omitting empties entirely.
    from array import array as _dp_arr
    _pt_graphs = []  # keep refs alive until SaveAs
    def _draw_points_no_empty(h, color):
        xs, ys, elo, ehi = [], [], [], []
        for _ib in range(1, h.GetNbinsX() + 1):
            _c = h.GetBinContent(_ib)
            if _c == 0:
                continue
            xs.append(h.GetBinCenter(_ib)); ys.append(_c)
            elo.append(h.GetBinErrorLow(_ib)); ehi.append(h.GetBinErrorUp(_ib))
        if not xs:
            return None
        _n = len(xs); _z = _dp_arr('d', [0.0] * _n)
        g = ROOT.TGraphAsymmErrors(_n, _dp_arr('d', xs), _dp_arr('d', ys), _z, _z,
                                    _dp_arr('d', elo), _dp_arr('d', ehi))
        g.SetMarkerStyle(h.GetMarkerStyle()); g.SetMarkerSize(h.GetMarkerSize())
        g.SetMarkerColor(color); g.SetLineColor(color); g.SetLineWidth(h.GetLineWidth())
        g.Draw("P Z SAME")  # Z: no end-caps (matches gStyle.SetEndErrorSize(0))
        _pt_graphs.append(g)
        return g
    if h_data_draw: _draw_points_no_empty(h_data_draw, h_data_draw.GetLineColor())

    # Pseudodata in blinded region: Bernstein best-fit as Poisson template (fallback: Poly1)
    h_pseudo = None; h_poly_pred = None; _pseudo_fit_range = (0.0, 1.0); _pseudo_used_bern = False
    if pseudodata_seed is not None and blind_range is not None and h_data is not None:
        _ps_blind_lo = blind_range[0]
        _bern_template = None
        if bernstein_fits is not None and bernstein_winner_idx is not None:
            _bern_template = bernstein_fits[bernstein_winner_idx][0]
        _ps_blind_hi = blind_range[1] if blind_range is not None else None
        h_pseudo, h_poly_pred, _pf_lo, _pf_hi = _make_pseudodata_hist(h_data, pseudodata_seed,
                                                                        fit_hi=_ps_blind_lo, sr_lo=_ps_blind_lo,
                                                                        sr_hi=_ps_blind_hi,
                                                                        template_hist=_bern_template)
        # _pf_lo == -1 is the sentinel that the Bernstein path was taken (not Poly1 fallback)
        _pseudo_used_bern = (_pf_lo is not None and _pf_lo < 0)
        if _pf_lo is not None and _pf_lo >= 0:
            _pseudo_fit_range = (_pf_lo, _pf_hi)
        if h_pseudo is not None:
            h_pseudo.SetMarkerStyle(h_data.GetMarkerStyle())
            h_pseudo.SetMarkerSize(h_data.GetMarkerSize())
            h_pseudo.SetMarkerColor(4)
            h_pseudo.SetLineColor(4)
            h_pseudo.SetLineWidth(h_data.GetLineWidth())
            _draw_points_no_empty(h_pseudo, 4)  # filled bins only — no error bar on =0 pseudodata bins

    display_region = region_label if region_label else args.region
    #lt3 = ROOT.TLatex(); lt3.SetNDC(); lt3.SetTextSize(0.030); lt3.SetTextColor(1);    lt3.DrawLatex(0.5,0.62, f"Signal scaled by #times{NORM:.f} to the data");

    # Compute lepton flavor/charge fractions for data
    data_charge_line = ""
    if h_data and "lep1_pdgId" in events["data_2018"].fields:
        sel_data = selection(events["data_2018"])
        pdgId_data = ak.to_numpy(events["data_2018"]["lep1_pdgId"][sel_data])
        n_total = len(pdgId_data)
        if n_total > 0:
            n_ep = np.sum(pdgId_data == -11); n_em = np.sum(pdgId_data == 11); n_mup = np.sum(pdgId_data == -13); n_mum = np.sum(pdgId_data == 13);
            pct_ep = 100.0 * n_ep / n_total; pct_em = 100.0 * n_em / n_total; pct_mup = 100.0 * n_mup / n_total; pct_mum = 100.0 * n_mum / n_total;
            data_charge_line = f"#mu^{{+}}:{pct_mup:.0f}%, #mu^{{-}}:{pct_mum:.0f}%, \ne^{{+}}:{pct_ep:.0f}%, e^{{-}}:{pct_em:.0f}%"

    # Compute lepton flavor/charge fractions for signal (only for channels with leptons)
    signal_charge_line = ""
    if len(signal_names) >= 2 and signal_names[0] in events and signal_names[1] in events and "lep1_pdgId" in events[signal_names[0]].fields:
        sel_s1 = selection(events[signal_names[0]])
        sel_s2 = selection(events[signal_names[1]])
        pdgId_s1 = ak.to_numpy(events[signal_names[0]]["lep1_pdgId"][sel_s1])
        pdgId_s2 = ak.to_numpy(events[signal_names[1]]["lep1_pdgId"][sel_s2])
        pdgId_sig = np.concatenate([pdgId_s1, pdgId_s2])
        n_total_sig = len(pdgId_sig)
        if n_total_sig > 0:
            n_ep = np.sum(pdgId_sig == -11); n_em = np.sum(pdgId_sig == 11); n_mup = np.sum(pdgId_sig == -13); n_mum = np.sum(pdgId_sig == 13);
            pct_ep = 100.0 * n_ep / n_total_sig; pct_em = 100.0 * n_em / n_total_sig; pct_mup = 100.0 * n_mup / n_total_sig; pct_mum = 100.0 * n_mum / n_total_sig;
            signal_charge_line = f"#mu^{{+}}:{pct_mup:.0f}%, #mu^{{-}}:{pct_mum:.0f}%, \ne^{{+}}:{pct_ep:.0f}%, e^{{-}}:{pct_em:.0f}%"

    # Format normalization with 2 significant digits, no power of 10
    def format_norm(n):
        if n == 0: return "0"
        # Round to 2 significant digits
        from math import log10, floor
        if n >= 1:
            digits = floor(log10(n))
            rounded = round(n, -digits + 1)  # 2 sig figs
            if rounded >= 100:
                return f"{int(rounded)}"
            elif rounded >= 10:
                return f"{rounded:.0f}"
            else:
                return f"{rounded:.1f}"
        else:
            return f"{n:.2g}"

    # Calculate total background and percentages
    n_total_bkg = n_zjets_events + n_wjets_events + n_top_events + n_vv_events
    pct_zjets = 100.0 * n_zjets_events / n_total_bkg if n_total_bkg > 0 else 0
    pct_wjets = 100.0 * n_wjets_events / n_total_bkg if n_total_bkg > 0 else 0
    pct_top = 100.0 * n_top_events / n_total_bkg if n_total_bkg > 0 else 0
    pct_vv = 100.0 * n_vv_events / n_total_bkg if n_total_bkg > 0 else 0

    # Channel-dependent V+jets legend labels
    zjets_legend_labels = {"Z+jets": "Z(#nu#nu)+jets", "DY": "DY(ll)+jets"}
    wjets_legend_labels = {"W+jets": "W(ll#nu)+jets"}
    zjets_legend_text = zjets_legend_labels.get(zjets_label, zjets_label) if zjets_label else ""
    wjets_legend_text = wjets_legend_labels.get(wjets_label, wjets_label) if wjets_label else ""

    leg = ROOT.TLegend(0.48, 0.5, 0.80, 0.9);
    if h_data:
        leg.AddEntry(h_data, f"Data (2018) {n_data_events:.1f} events", "pe");
    if h_pseudo is not None:
        if _pseudo_used_bern:
            _pleg = "PseudoData (Bern. fit)"
        else:
            _pleg = "PseudoData (Poly1 fit %.2f-%.2f)" % (_pseudo_fit_range[0], _pseudo_fit_range[1])
        leg.AddEntry(h_pseudo, _pleg, "pe");
    if not bernstein_fits:
        if "Top"   in histos: leg.AddEntry(histos["Top"]  , f"t#bar{{t}}, single-t  {n_top_events:.1f} #bf{{({pct_top:.0f}%)}}", "f");
        if "ZJets" in histos: leg.AddEntry(histos["ZJets"], f"{zjets_legend_text}  {n_zjets_events:.1f} #bf{{({pct_zjets:.0f}%)}}", "f");
        if "WJets" in histos: leg.AddEntry(histos["WJets"], f"{wjets_legend_text}  {n_wjets_events:.1f} #bf{{({pct_wjets:.0f}%)}}", "f");
        if "VV"    in histos: leg.AddEntry(histos["VV"]   , f"WW,WZ,ZZ  {n_vv_events:.1f} #bf{{({pct_vv:.0f}%)}}", "f");
    sig_legend = signal_legend_label if signal_legend_label else "WH#rightarrowll#nu(gg)"
    if "signal_total" in histos: leg.AddEntry(histos["signal_total"], f"{sig_legend}  {n_signal_events:.1f} #times #bf{{{NORM:.0f}}}", "l");
    if _h_ov2_draw is not None: leg.AddEntry(_h_ov2_draw, overlay_hist2.get('label', 'Pred1'), "l");
    if _h_ov_draw  is not None: leg.AddEntry(_h_ov_draw,  overlay_hist.get('label',  'Pred2'), "l");
    if _h_cr_draw  is not None: leg.AddEntry(_h_cr_draw,  "D_{CR} (norm. to data)", "l");
    if _h_tf_draw  is not None: leg.AddEntry(_h_tf_draw,  "MC_{SR}/MC_{CR} (norm. to data)", "l");
    # AIC weights: AIC_k = chi2_total + 2*(n+1);  w_k = exp(-ΔAIC/2) / Σ exp(-ΔAIC/2)
    _aic_raw = {ord_b: chi2_b + 2.0 * (ord_b + 1)
                for (_, chi2_b, ndf_b, ord_b, _) in _bern_drawn
                if chi2_b is not None and ndf_b is not None and ndf_b > 0}
    if _aic_raw:
        _aic_min = min(_aic_raw.values())
        _aic_exp = {o: float(np.exp(-0.5 * (v - _aic_min))) for o, v in _aic_raw.items()}
        _aic_tot = sum(_aic_exp.values())
        _aic_w   = {o: 100.0 * v / _aic_tot for o, v in _aic_exp.items()}
    else:
        _aic_w = {}
    for (hd, chi2_b, ndf_b, ord_b, _is_winner) in _bern_drawn:
        chi2_str = f"{chi2_b/ndf_b:.2f}" if (ndf_b is not None and ndf_b > 0) else "N/A"
        _w_str   = f"{_aic_w[ord_b]:.0f}%" if ord_b in _aic_w else ""
        _lbl = f"Poly{ord_b} ({_w_str}) #chi^{{2}}/ndf={chi2_str}"
        if _is_winner:
            _lbl = f"#bf{{{_lbl}}}"
        leg.AddEntry(hd, _lbl, "l")

    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.053); leg.Draw();
    if bernstein_fits:
        _fr_lo, _fr_hi = bernstein_fit_range
        lt_fitrange = ROOT.TLatex(); lt_fitrange.SetNDC(); lt_fitrange.SetTextFont(42)
        lt_fitrange.SetTextSize(0.053); lt_fitrange.SetTextColor(ROOT.kGray+2)
        # Placed just below the polynomial legend box (which spans y=[0.5,0.9], x=[0.5,0.82])
        lt_fitrange.DrawLatex(0.48, 0.45, f"Fit range: [{_fr_lo:.2f}, {_fr_hi:.2f}]")

    # Redraw frame axis on top of histograms
    frame.Draw("AXIS SAME")

    # Draw significance text — suppressed when overlay_hist present
    if sig_text_data is not None and overlay_hist is None:
        _gain_ref = sig_text_data.get("gain_ref", 0.0)
        opt1 = sig_text_data["opt1"]; opt2 = sig_text_data["opt2"]; opt3 = sig_text_data["opt3"]
        lt0 = ROOT.TLatex(); lt0.SetTextFont(42); lt0.SetNDC(); lt0.SetTextSize(0.044); lt0.SetTextColor(1); lt0.DrawLatex(0.17, 0.76, "Significance eval. S/#sqrt{D}")
        lt1 = ROOT.TLatex(); lt1.SetTextFont(42); lt1.SetNDC(); lt1.SetTextSize(0.044); lt1.SetTextColor(1); lt1.DrawLatex(0.17, 0.70, "0 full range: #bf{%.3f}" % _gain_ref)
        lt2a = ROOT.TLatex(); lt2a.SetTextFont(42); lt2a.SetNDC(); lt2a.SetTextSize(0.044); lt2a.SetTextColor(1); lt2a.DrawLatex(0.17, 0.64, "1 bin max: #color[616]{#bf{%.3f} (+%.0f%%)}" % (opt1["sig"], opt1["gain"]))
        if opt2: lt3 = ROOT.TLatex(); lt3.SetTextFont(42); lt3.SetNDC(); lt3.SetTextSize(0.044); lt3.SetTextColor(1); lt3.DrawLatex(0.17, 0.58, "2 bin max: #color[418]{#bf{%.3f} (+%.0f%%)}" % (opt2["sig"], opt2["gain"]))
        if opt3: lt4 = ROOT.TLatex(); lt4.SetTextFont(42); lt4.SetNDC(); lt4.SetTextSize(0.044); lt4.SetTextColor(1); lt4.DrawLatex(0.17, 0.52, "3 bin max: #color[600]{#bf{%.3f} (+%.0f%%)}" % (opt3["sig"], opt3["gain"]))
        #if opt4: lt5 = ROOT.TLatex(); lt5.SetTextFont(42); lt5.SetNDC(); lt5.SetTextSize(0.044); lt5.SetTextColor(1); lt5.DrawLatex(0.17, 0.46, "4 bin max: #color[95]{#bf{%.3f} (%.0f%%)}" % (opt4["sig"], opt4["gain"]))
        _lt_nbin = ROOT.TLatex(); _lt_nbin.SetTextFont(42); _lt_nbin.SetNDC(); _lt_nbin.SetTextSize(0.044); _lt_nbin.SetTextColor(1)
        _lt_nbin.DrawLatex(0.17, 0.46, "%d bins full range: #bf{%.3f}" % (sig_text_data["nbins"], sig_text_data.get("quad_sig_full", sig_text_data["initial"])))

    # PRED overlay mode: the 0-3-bin significance block above is suppressed — draw a single
    # full-range S/sqrt(D) line instead (per-panel significance for the montage).
    if overlay_hist is not None and plot_info.get("quad_sig_full", 0.0) > 0:
        _lt_fsig = ROOT.TLatex(); _lt_fsig.SetTextFont(42); _lt_fsig.SetNDC(); _lt_fsig.SetTextSize(0.044); _lt_fsig.SetTextColor(1)
        _lt_fsig.DrawLatex(0.17, 0.52, "S/#sqrt{D} full range (%d bins): #bf{%.3f}" % (nbins, plot_info["quad_sig_full"]))

    # Draw Data/MC and Data/Pred ratio text in upper pad. When the significance block is drawn
    # it ends with "N bins full range" at y=0.46, so sit one 0.06 line-slot below it (0.40) —
    # directly under that text, with no overlap.
    _ratio_y0 = 0.40 if (sig_text_data is not None and overlay_hist is None) else 0.46
    data_mc_ratio = None; data_pred2_ratio = None
    if LOAD_BKG_MC and h_data is not None and h_bkg_total is not None:
        # Exclude blinded bins from BOTH integrals — keeps the printed ratio consistent
        # with the visible points and the lower-pad Data/MC markers (blind bins hidden there).
        data_integral = 0.0; mc_integral = 0.0
        for _ib in range(1, h_data.GetNbinsX() + 1):
            _bc = h_data.GetBinCenter(_ib)
            if blind_range is not None and blind_range[0] <= _bc <= blind_range[1]:
                continue
            data_integral += h_data.GetBinContent(_ib)
            mc_integral   += h_bkg_total.GetBinContent(_ib)
        if mc_integral > 0:
            data_mc_ratio = data_integral / mc_integral
            lt_ratio = ROOT.TLatex(); lt_ratio.SetTextFont(42); lt_ratio.SetNDC(); lt_ratio.SetTextSize(0.044); lt_ratio.SetTextColor(1)
            lt_ratio.DrawLatex(0.17, _ratio_y0, "Data/MC = #bf{%.2f}" % data_mc_ratio)
    # Pre-compute data and pred integrals over the full Pred2 range (all bins, since Pred2 now covers all j)
    _data_integral_full = h_data.Integral() if h_data is not None else 0.0
    if _h_ov2_draw is not None and h_data is not None:
        pred1_integral = 0.0; data_in_pred1_bins = 0.0
        for _ib in range(1, _h_ov2_draw.GetNbinsX() + 1):
            _p1 = _h_ov2_draw.GetBinContent(_ib)
            _bc1 = _h_ov2_draw.GetBinCenter(_ib)
            if _p1 <= 0: continue
            if blind_range is not None and blind_range[0] <= _bc1 <= blind_range[1]:
                continue
            pred1_integral     += _p1
            data_in_pred1_bins += h_data.GetBinContent(_ib)
        if pred1_integral > 0:
            data_pred1_ratio = data_in_pred1_bins / pred1_integral
            lt_ratio_pred1 = ROOT.TLatex(); lt_ratio_pred1.SetTextFont(42); lt_ratio_pred1.SetNDC(); lt_ratio_pred1.SetTextSize(0.044); lt_ratio_pred1.SetTextColor(ROOT.kGreen + 2)
            lt_ratio_pred1.DrawLatex(0.17, _ratio_y0 - 0.06, "Data/Pred1 = #bf{%.2f}" % data_pred1_ratio)
    if _h_ov_draw is not None and h_data is not None:
        # Accumulate pred2 and matching data only over bins where pred2 > 0
        # (first G+J bins are zeroed; including them in the integral would inflate the ratio)
        pred2_integral = 0.0; data_in_pred2_bins = 0.0
        _chi2 = 0.0; _ndf = 0
        for _ib in range(1, _h_ov_draw.GetNbinsX() + 1):
            _pred = _h_ov_draw.GetBinContent(_ib)
            _obs  = h_data.GetBinContent(_ib)
            _ep   = _h_ov_draw.GetBinError(_ib)
            _bc   = _h_ov_draw.GetBinCenter(_ib)
            if _pred <= 0: continue
            if blind_range is not None and blind_range[0] <= _bc <= blind_range[1]:
                continue
            pred2_integral     += _pred
            data_in_pred2_bins += _obs
            _chi2 += (_obs - _pred)**2 / (_obs + _ep**2) if (_obs + _ep**2) > 0 else 0.0
            _ndf  += 1
        if pred2_integral > 0:
            data_pred2_ratio = data_in_pred2_bins / pred2_integral
            _chi2_ndf = _chi2 / _ndf if _ndf > 0 else 0.0
            lt_ratio_pred = ROOT.TLatex(); lt_ratio_pred.SetTextFont(42); lt_ratio_pred.SetNDC(); lt_ratio_pred.SetTextSize(0.044); lt_ratio_pred.SetTextColor(ROOT.kBlue)
            lt_ratio_pred.DrawLatex(0.17, 0.34, "Data/Pred2 = #bf{%.2f},  #chi^{2}/ndf = #bf{%.2f}" % (data_pred2_ratio, _chi2_ndf))
    if pred_params_label:
        lt_params = ROOT.TLatex(); lt_params.SetTextFont(42); lt_params.SetNDC(); lt_params.SetTextSize(0.044); lt_params.SetTextColor(ROOT.kBlack)
        lt_params.DrawLatex(0.17, 0.28, pred_params_label)
    if pred_extra_label:
        lt_extra = ROOT.TLatex(); lt_extra.SetTextFont(42); lt_extra.SetNDC(); lt_extra.SetTextSize(0.044); lt_extra.SetTextColor(ROOT.kBlack)
        lt_extra.DrawLatex(0.17, 0.22, pred_extra_label)

    # Draw region label with channel
    lt_region = ROOT.TLatex(); lt_region.SetNDC(); lt_region.SetTextSize(0.055); lt_region.SetTextColor(1); lt_region.DrawLatex(0.17, 0.82, f"{channel} {display_region}");

    cms_bold = ROOT.TLatex(); cms_bold.SetNDC(); cms_bold.SetTextFont(62); cms_bold.SetTextSize(0.074); cms_bold.DrawLatex(0.135, 0.94, "CMS");
    # "Preliminary" and the lumi are drawn as two TLatex objects (they used to be one string
    # padded with spaces) so the lumi can be positioned independently. The lumi is right-aligned:
    # the frame's right edge is at 1-0.035=0.965, and 0.985 sits 2% to the right of it.
    Prel_Lumi = ROOT.TLatex(); Prel_Lumi.SetNDC(); Prel_Lumi.SetTextFont(42); Prel_Lumi.SetTextSize(0.057); Prel_Lumi.DrawLatex(0.25, 0.94, "#it{Preliminary}");
    Lumi_txt = ROOT.TLatex(); Lumi_txt.SetNDC(); Lumi_txt.SetTextFont(42); Lumi_txt.SetTextSize(0.057); Lumi_txt.SetTextAlign(31); Lumi_txt.DrawLatex(0.965, 0.94, "59.7 fb^{-1} (13 TeV)");

    # Consolidated single-line output (table row format)
    blind_col = plot_info['blind_str'].replace(", blind ", "").replace("[", "").replace("]", "") if plot_info['blind_str'] else "-"
    var_name_display = var_name.replace("3.141592653589793", "pi").replace("2*pi", "2pi")
    if len(var_name_display) > 35: var_name_display = var_name_display[:32] + "..."
    print(f" {plot_index:>3d} {var_name_display:35s} | {plot_info['data']:>8.0f} {plot_info['signal']:>7.1f} {plot_info['bkg']:>10.0f} | {blind_col:10s}")

    # Draw lower pad with S/sqrt(D) histogram and Data/MC ratio (skipped in upper_pad_only mode)
    if not upper_pad_only:
        pad2.cd(); pad2.SetTickx(1); pad2.SetTicky(1)
        # PRED modes (Bernstein fit and/or PRED1/PRED2 overlay prediction) show pulls,
        # (data-fit)/sigma_stat, instead of a ratio — sigma_stat is the data's own
        # symmetrized Garwood/Poisson error (h_data.GetBinError under kPoisson option).
        _is_pull_mode = (bernstein_fits is not None) or (overlay_hist is not None) or (overlay_hist2 is not None)
        if ratio_yrange:
            _ratio_ymin, _ratio_ymax = ratio_yrange
        elif _is_pull_mode:
            _ratio_ymin, _ratio_ymax = (-4.4, 4.4)
        else:
            _ratio_ymin, _ratio_ymax = (0.2, 2.0)
        frame2 = ROOT.TH1F("frame2", "", nbins, xmin, xmax); frame2.SetMinimum(_ratio_ymin); frame2.SetMaximum(_ratio_ymax)
        frame2.GetXaxis().SetTitle(xlabel); frame2.GetXaxis().SetLabelSize(0.101); frame2.GetXaxis().SetTitleSize(0.128)
        # X tick length is a FRACTION OF THE PAD HEIGHT, so the same number is longer in the short
        # lower pad: 0.08*0.36 = 0.029 vs the upper pad's 0.03*0.64 = 0.019. Use 0.04 here
        # (0.04*0.36 = 0.014) so the lower-pad ticks — including the mirrored ones on the top edge
        # of the frame — are visibly shorter than the upper pad's.
        frame2.GetXaxis().SetTitleOffset(0.75); frame2.GetXaxis().SetNdivisions(510); frame2.GetXaxis().SetTickLength(0.04)
        if _bin_labels:
            for _i, _lbl in enumerate(_bin_labels, 1):
                frame2.GetXaxis().SetBinLabel(_i, _lbl)
            frame2.GetXaxis().SetLabelSize(0.14)
        frame2.GetYaxis().SetTitle("Pull" if _is_pull_mode else ""); frame2.GetYaxis().SetTitleOffset(0.52)
        if _is_pull_mode:
            # ROOT's automatic Ndivisions can't guarantee exact ±1..±4 labels with 0.5-spaced
            # ticks on an asymmetric (-4.4, 4.4) range, so draw the Y-axis annotation manually.
            frame2.GetYaxis().SetLabelSize(0); frame2.GetYaxis().SetTickLength(0)
        else:
            frame2.GetYaxis().SetLabelSize(0.101); frame2.GetYaxis().SetNdivisions(502); frame2.GetYaxis().SetTickLength(0.03)
        frame2.GetYaxis().SetTitleSize(0.118)
        frame2.Draw("AXIS")
        _pull_axis_objs = []  # keep refs so ROOT doesn't GC them
        if _is_pull_mode:
            _tick_len_major = 0.030 * (xmax - xmin)
            _tick_len_minor = 0.015 * (xmax - xmin)
            _y = -4.0
            while _y <= 4.0 + 1e-9:
                _is_major = abs(_y - round(_y)) < 1e-9 and round(_y) != 0
                _tlen = _tick_len_major if _is_major else _tick_len_minor
                _tk = ROOT.TLine(xmin, _y, xmin + _tlen, _y)
                _tk.SetLineColor(ROOT.kBlack); _tk.SetLineWidth(1); _tk.Draw("SAME")
                _pull_axis_objs.append(_tk)
                # Mirror tick on the right edge (Y auto-ticks are off, so add it manually)
                _tk_r = ROOT.TLine(xmax - _tlen, _y, xmax, _y)
                _tk_r.SetLineColor(ROOT.kBlack); _tk_r.SetLineWidth(1); _tk_r.Draw("SAME")
                _pull_axis_objs.append(_tk_r)
                if _is_major:
                    _lbl = ROOT.TLatex(xmin - 0.012 * (xmax - xmin), _y, f"{int(round(_y))}")
                    _lbl.SetTextSize(0.095); _lbl.SetTextAlign(32); _lbl.SetTextFont(42)
                    _lbl.Draw("SAME")
                    _pull_axis_objs.append(_lbl)
                _y += 0.5
        # Draw horizontal reference line at pull=0 (ratio=1 in ratio mode) — first, so it's behind everything
        _ref_y = 0.0 if _is_pull_mode else 1.0
        line_unity = ROOT.TLine(xmin, _ref_y, xmax, _ref_y)
        line_unity.SetLineColor(ROOT.kGray+1); line_unity.SetLineStyle(1); line_unity.SetLineWidth(1); line_unity.Draw("SAME")

        # Draw optimization vertical lines — suppressed in PRED mode (overlay_hist present)
        # 1-bin: 2 lines (tallest), 2-bin: 3 lines, 3-bin: 4 lines, 4-bin: 5 lines (shortest)
        # Draw 1b first, 4b last → 4b lines sit on top and are always visible
        _ylo = _ratio_ymin; _lb_yrange = _ratio_ymax - _ratio_ymin
        _yhi_1b = _ylo + 0.20 * _lb_yrange; _yhi_2b = _ylo + 0.15 * _lb_yrange
        _yhi_3b = _ylo + 0.11 * _lb_yrange; _yhi_4b = _ylo + 0.07 * _lb_yrange
        def _ln(x, yhi, col, lw=2):
            l = ROOT.TLine(x, _ylo, x, yhi); l.SetLineColor(col); l.SetLineStyle(1); l.SetLineWidth(lw); l.Draw("SAME"); return l
        if sig_text_data is not None and overlay_hist is None:
            opt1 = sig_text_data["opt1"]; opt2 = sig_text_data["opt2"]; opt3 = sig_text_data["opt3"]
            _lines_lb = []  # keep references so ROOT doesn't GC them
            # 1-bin (magenta) — 2 lines, drawn first (tallest, underneath)
            for _x in (opt1["left_edge"], opt1["right_edge"]):
                _lines_lb.append(_ln(_x, _yhi_1b, ROOT.kMagenta, 2))
            # 2-bin (green) — 3 lines
            if opt2:
                for _x in (opt2["left_edge"], opt2["mid1_edge"], opt2["right_edge"]):
                    _lines_lb.append(_ln(_x, _yhi_2b, ROOT.kGreen+2, 2))
            # 3-bin (blue) — 4 lines
            if opt3:
                for _x in (opt3["left_edge"], opt3["mid1_edge"], opt3["mid2_edge"], opt3["right_edge"]):
                    _lines_lb.append(_ln(_x, _yhi_3b, ROOT.kBlue, 2))
            ## 4-bin (color 95) — 5 lines, drawn last (shortest, on top and visible)
            #if opt4:
            #    for _x in (opt4["left_edge"], opt4["mid1_edge"], opt4["mid2_edge"], opt4["mid3_edge"], opt4["right_edge"]):
            #        _lines_lb.append(_ln(_x, _yhi_4b, 95, 2))

        # Draw Data/MC ratio using full unblinded data (same as significance)
        def _apply_blind_to_ratio(h, denom_h=None):
            """Zero blinded bins; if pseudodata and denominator are available substitute pseudo/denom."""
            if blind_range is None:
                return
            blo, bhi = blind_range
            for ib in range(1, h.GetNbinsX() + 1):
                if blo <= h.GetBinCenter(ib) <= bhi:
                    if h_pseudo is not None and denom_h is not None:
                        dv = denom_h.GetBinContent(ib)
                        if dv > 0:
                            pv = h_pseudo.GetBinContent(ib)
                            pe = h_pseudo.GetBinError(ib)
                            h.SetBinContent(ib, pv / dv)
                            h.SetBinError(ib, pe / dv)
                            continue
                    h.SetBinContent(ib, 0)
                    h.SetBinError(ib, 0)

        from array import array as _pyarray

        def _make_pull_graph(xs, ys, color):
            """Build a TGraphErrors (marker error fixed at ±1) from explicit valid points only —
            never includes a placeholder/masked bin, so no fake pull~0 marker is ever drawn."""
            if not xs:
                return None
            n = len(xs)
            g = ROOT.TGraphErrors(n, _pyarray('d', xs), _pyarray('d', ys),
                                   _pyarray('d', [0.0]*n), _pyarray('d', [1.0]*n))
            g.SetMarkerStyle(h_data.GetMarkerStyle()); g.SetMarkerSize(h_data.GetMarkerSize())
            g.SetLineWidth(h_data.GetLineWidth())
            g.SetMarkerColor(color); g.SetLineColor(color)
            g.Draw("P Z SAME")  # "Z": no perpendicular end-caps on the error bars (matches upper-pad data style)
            return g

        def _pull_graph_vs_pred(pred_h, color):
            """Pull = (data-pred)/sigma_stat vs a single prediction histogram (PRED1/PRED2
            overlays). Unblinded bins use real data; blinded bins use pseudodata instead
            (never real data there) — only genuinely valid points are included."""
            xs, ys = [], []
            for ib in range(1, pred_h.GetNbinsX() + 1):
                bc = h_data.GetBinCenter(ib)
                pred_v = pred_h.GetBinContent(ib)
                if pred_v <= 0:
                    continue
                is_blind = (blind_range is not None and blind_range[0] <= bc <= blind_range[1])
                if is_blind:
                    if h_pseudo is None:
                        continue
                    val, err = h_pseudo.GetBinContent(ib), h_pseudo.GetBinError(ib)
                else:
                    val, err = h_data.GetBinContent(ib), h_data.GetBinError(ib)
                if err <= 0:
                    continue
                xs.append(bc); ys.append((val - pred_v) / err)
            return _make_pull_graph(xs, ys, color)

        # Data vs total-MC comparison in grey.
        #   pull mode  → (data-MC)/sigma_stat, on the SAME pull axis as the Pred pulls.
        #     (drawing the raw data/MC ratio (~1) on a [-4.4,4.4] pull axis collapses every
        #      marker onto y~=1, which is what made these markers look wrong.)
        #   ratio mode → data/MC ratio markers around unity.
        h_ratio = None
        # In pull mode the grey Data-vs-MC pull is redundant with the prediction pull and just
        # clutters the pad — only draw the Data/MC series in ratio (non-pull) mode.
        if LOAD_BKG_MC and h_data is not None and h_bkg_total is not None and not _is_pull_mode:
            h_ratio = h_data.Clone("h_ratio")
            h_ratio.Divide(h_bkg_total)
            _apply_blind_to_ratio(h_ratio, denom_h=h_bkg_total)
            h_ratio.SetMarkerStyle(h_data.GetMarkerStyle())
            h_ratio.SetMarkerSize(h_data.GetMarkerSize())
            h_ratio.SetMarkerColor(ROOT.kBlack)
            h_ratio.SetLineColor(ROOT.kBlack)
            h_ratio.Draw("P E X0 SAME")

        # Pseudodata pull in lower pad [0.9, 1.0] — blue markers, vs poly prediction
        h_ratio_pseudo = None
        if h_pseudo is not None and h_poly_pred is not None:
            _xs_ps, _ys_ps = [], []
            for ib in range(1, h_poly_pred.GetNbinsX() + 1):
                pred_v = h_poly_pred.GetBinContent(ib)
                pe = h_pseudo.GetBinError(ib)
                if pred_v > 0 and pe > 0:
                    _xs_ps.append(h_pseudo.GetBinCenter(ib))
                    _ys_ps.append((h_pseudo.GetBinContent(ib) - pred_v) / pe)
            h_ratio_pseudo = _make_pull_graph(_xs_ps, _ys_ps, 4)

        # Significance metric (S/sqrt(D), used for sig_0bin/reporting) is computed exactly as
        # before, unchanged. The DRAWN lower-pad curve is a separate display quantity:
        # Signal/sigma_stat, where sigma_stat is data's own symmetrized Garwood/Poisson error
        # (same convention as the pull panels) — well-defined even for zero-data bins.
        h_sig_sqrt_data = ROOT.TH1F("h_sig_sqrt_data", "", nbins, xmin, xmax)
        scale_factor = 1.0
        quad_sig = 0.0  # quadratic sum sqrt(sum_i sig_i^2) — S/sqrt(D) significance metric
        if h_data is not None and "signal_total" in histos:
            h_signal = histos["signal_total"]
            max_sig_val = 0.0
            sig_sq_sum = 0.0
            for ibin in range(1, nbins + 1):
                s = h_signal.GetBinContent(ibin) / NORM  # Unscale signal
                d = h_data.GetBinContent(ibin)           # Always real data — matches sig_0bin
                if d > 0:
                    sig_sq_sum += (s / math.sqrt(d)) ** 2
                sigma_stat = h_data.GetBinError(ibin)    # Garwood/Poisson symmetrized stat unc
                if sigma_stat > 0:
                    draw_val = s / sigma_stat
                    h_sig_sqrt_data.SetBinContent(ibin, draw_val)
                    if draw_val > max_sig_val:
                        max_sig_val = draw_val
            quad_sig = math.sqrt(sig_sq_sum)
            # quad_sig now equals sig_0bin (both use real data in all bins).
            # Overwrite to keep them in sync in case of floating-point edge cases.
            if "sig_0bin" in plot_info:
                plot_info["sig_0bin"] = quad_sig
            plot_info["max_sig_val_raw"] = max_sig_val  # unscaled, for cross-panel common scaling
            # Scale histogram: either a caller-supplied common factor (e.g. shared across SR1-4,
            # anchored so SR4's max lands at a fixed target), or the usual self-scale to 1.65.
            if sig_scale_factor is not None:
                scale_factor = sig_scale_factor
                h_sig_sqrt_data.Scale(scale_factor)
            elif max_sig_val > 0:
                scale_factor = 1.65 / max_sig_val
                h_sig_sqrt_data.Scale(scale_factor)
            h_sig_sqrt_data.SetLineColor(ROOT.kRed); h_sig_sqrt_data.SetLineWidth(2); h_sig_sqrt_data.SetLineStyle(2)
            h_sig_sqrt_data.Draw("HIST ][ SAME")  # No vertical lines between bins, dashed

        # Pred_Term1 pull in green — Term1 only prediction (no alpha), drawn first
        h_ratio_pred2 = None
        if _h_ov2_draw is not None and h_data is not None:
            h_ratio_pred2 = _pull_graph_vs_pred(_h_ov2_draw, ROOT.kGreen + 2)

        # Pred2 pull in black — drawn last so it sits on top of all other markers
        h_ratio_pred = None
        if _h_ov_draw is not None and h_data is not None:
            h_ratio_pred = _pull_graph_vs_pred(_h_ov_draw, ROOT.kBlack)

        # Pull vs BestBern fit in PRED3 mode — only within fit range [0.1, 0.9]
        h_ratio_bern = None
        h_ratio_bern_blind = None
        _best_bern_ord_label = ""
        if bernstein_fits and h_data is not None and _bern_drawn:
            _best_bern_h = None
            if bernstein_winner_idx is not None:
                # Use F-test winner
                for (hd_b, chi2_b, ndf_b, ord_b, _iw) in _bern_drawn:
                    if _iw:
                        _best_bern_h = hd_b
                        _best_bern_ord_label = f"Bern. n={ord_b}"
                        break
            if _best_bern_h is None:
                # Fallback: lowest chi2/ndf
                _best_chi2ndf = float('inf')
                for (hd_b, chi2_b, ndf_b, ord_b, _iw) in _bern_drawn:
                    if hd_b is None or ndf_b is None or ndf_b <= 0:
                        continue
                    _r = chi2_b / ndf_b
                    if _r < _best_chi2ndf:
                        _best_chi2ndf = _r
                        _best_bern_h = hd_b
                        _best_bern_ord_label = f"Bern. n={ord_b}"
            if _best_bern_h is not None:
                _bfr_lo, _bfr_hi = bernstein_fit_range
                _xs_bk, _ys_bk = [], []   # black: real data, unblinded, within the fit range
                _xs_bl, _ys_bl = [], []   # blue: pseudodata, blinded bins only
                for _ib in range(1, h_data.GetNbinsX() + 1):
                    _bc = h_data.GetBinCenter(_ib)
                    _pred_val = _best_bern_h.GetBinContent(_ib)
                    _is_blind_bin = (blind_range is not None and blind_range[0] <= _bc <= blind_range[1])
                    if _is_blind_bin:
                        # Blind bins sit beyond the fit range (bernstein_fit_range's upper edge is
                        # the blind boundary itself) by construction — the fit curve extrapolates
                        # into them, so pseudodata is NOT restricted by _bfr_hi here. pred_val==0
                        # (background extrapolates to ~0 in the far tail) is a legitimate value —
                        # the pull is still well-defined via sigma_stat from the pseudodata itself.
                        if h_pseudo is None or _pred_val < 0:
                            continue
                        _pe = h_pseudo.GetBinError(_ib)
                        if _pe <= 0:
                            continue
                        _xs_bl.append(_bc); _ys_bl.append((h_pseudo.GetBinContent(_ib) - _pred_val) / _pe)
                    elif _pred_val >= 0 and _bfr_lo <= _bc <= _bfr_hi:
                        _data_err = h_data.GetBinError(_ib)
                        if _data_err <= 0:
                            continue
                        _xs_bk.append(_bc); _ys_bk.append((h_data.GetBinContent(_ib) - _pred_val) / _data_err)
                h_ratio_bern = _make_pull_graph(_xs_bk, _ys_bk, ROOT.kBlack)
                h_ratio_bern_blind = _make_pull_graph(_xs_bl, _ys_bl, ROOT.kBlue)

        # Add legend for lower pad — 2 columns: pull | significance. The blue pseudodata-pull
        # series (blind bins only) is drawn but intentionally has no legend entry.
        leg2 = ROOT.TLegend(0.17, 0.81, 0.95, 0.97)
        leg2.SetNColumns(2)
        leg2.SetColumnSeparation(0.05)
        # Column 1: pull histogram (left) — black, unblinded bins only. "p" (no "e"): the tiny
        # legend icon shouldn't draw its own little error-bar-with-caps glyph.
        if h_ratio_bern  is not None: leg2.AddEntry(h_ratio_bern,  "(Data-BestFit)/#sigma_{stat}", "pe")
        elif h_ratio_pred is not None: leg2.AddEntry(h_ratio_pred,  "Pull (Pred2)", "pe")
        elif h_ratio_pred2 is not None: leg2.AddEntry(h_ratio_pred2, "Pull (Pred1)", "pe")
        elif h_ratio      is not None: leg2.AddEntry(h_ratio,       "Data/MC",    "pe")
        # Column 2: significance (right)
        if h_data is not None and "signal_total" in histos:
            leg2.AddEntry(h_sig_sqrt_data, f"(S/#sigma_{{stat}})#times#bf{{{scale_factor:.0f}}}", "l")
        leg2.SetBorderSize(0); leg2.SetFillStyle(0); leg2.SetTextSize(0.0903)
        leg2.Draw()

        # Caption: unblinded-integral ratios, same values as the upper-pad text (consistent
        # by construction — both exclude blinded bins).
        _cap_parts = []
        # Data/MC intentionally omitted here — already shown as black text in the upper pad.
        if data_pred2_ratio is not None: _cap_parts.append("Data/Pred2 = #bf{%.2f}" % data_pred2_ratio)
        if _cap_parts:
            lt_cap2 = ROOT.TLatex(); lt_cap2.SetTextFont(42); lt_cap2.SetNDC()
            lt_cap2.SetTextSize(0.075); lt_cap2.SetTextColor(ROOT.kGray + 2)
            lt_cap2.DrawLatex(0.185, 0.055, "   ".join(_cap_parts))

    if blinded_label and blind_range is not None:
        pad1.cd()
        lt_blind = ROOT.TLatex(); lt_blind.SetTextFont(62)
        _blind_size  = 0.042 if bernstein_fits else 0.054
        _blind_color = 4 if bernstein_fits else ROOT.kBlue
        lt_blind.SetTextSize(_blind_size); lt_blind.SetTextColor(_blind_color)
        if upper_pad_only:
            lt_blind.SetTextAngle(90)
            lt_blind.SetTextAlign(12)
            lt_blind.SetTextColor(ROOT.kRed)
            blind_x = 0.5 * (blind_range[0] + xmax)  # moved 1% right (was -0.01*(xmax-xmin))
            blind_y = frame.GetMinimum() + 0.18 * (frame.GetMaximum() - frame.GetMinimum())
        else:
            lt_blind.SetTextAlign(22)
            blind_x = 0.5 * (blind_range[0] + xmax) - 0.02 * (xmax - xmin)  # moved 1% right (was -0.03)
            blind_y = frame.GetMinimum() + 0.06 * (frame.GetMaximum() - frame.GetMinimum())
        lt_blind.DrawLatex(blind_x, blind_y, blinded_label)
        # Red dashed vertical line at the blind boundary (PRED mode only)
        if upper_pad_only:
            _vl_x = blind_range[0]
            _vl_ylo = frame.GetMinimum()
            _vl_yhi = _vl_ylo + 0.5 * (frame.GetMaximum() - _vl_ylo)
            vline_blind = ROOT.TLine(_vl_x, _vl_ylo, _vl_x, _vl_yhi)
            vline_blind.SetLineColor(ROOT.kRed); vline_blind.SetLineStyle(2); vline_blind.SetLineWidth(2); vline_blind.Draw()

    # Draw optimal bin edge lines in upper pad (PRED mode — replaces lower-pad lines)
    if upper_pad_only and sig_text_data is not None and overlay_hist is None:
        pad1.cd()
        _p1_ylo = frame.GetMinimum()
        _yrange = frame.GetMaximum() - _p1_ylo
        _p1_yhi_1b = _p1_ylo + 0.13 * _yrange  # 1-bin: tallest
        _p1_yhi_2b = _p1_ylo + 0.10 * _yrange
        _p1_yhi_3b = _p1_ylo + 0.08 * _yrange
        _opt1 = sig_text_data["opt1"]; _opt2 = sig_text_data["opt2"]; _opt3 = sig_text_data["opt3"]
        def _p1ln(x, yhi, col, lw=2):
            l = ROOT.TLine(x, _p1_ylo, x, yhi); l.SetLineColor(col); l.SetLineStyle(1); l.SetLineWidth(lw); l.Draw("SAME"); return l
        _p1_lines = []
        # Draw 1b first, 3b last → 3b sits on top and is always visible
        for _x in (_opt1["left_edge"], _opt1["right_edge"]):
            _p1_lines.append(_p1ln(_x, _p1_yhi_1b, ROOT.kMagenta, 3))
        if _opt2:
            for _x in (_opt2["left_edge"], _opt2["mid1_edge"], _opt2["right_edge"]):
                _p1_lines.append(_p1ln(_x, _p1_yhi_2b, ROOT.kGreen+2, 2))
        if _opt3:
            for _x in (_opt3["left_edge"], _opt3["mid1_edge"], _opt3["mid2_edge"], _opt3["right_edge"]):
                _p1_lines.append(_p1ln(_x, _p1_yhi_3b, ROOT.kBlue, 2))
        ## 4-bin (color 95) — commented out
        #if _opt4:
        #    for _x in (_opt4["left_edge"], _opt4["mid1_edge"], _opt4["mid2_edge"], _opt4["mid3_edge"], _opt4["right_edge"]):
        #        _p1_lines.append(_p1ln(_x, _p1_yhi_4b, 95, 2))

    c.cd()
    c.SaveAs(f"{output_prefix}.png");
    c.Update(); c.Draw(); c.Close();

    return plot_info  # Return info for table collection


def plot_variable_2D(events, x_var, y_var, selection, bins_x, bins_y, xlabel, ylabel, output_prefix, region_label=None, genEventSumw=None, signal_names=None, cdf_flat_x=False):
    """
    Plot 2D histogram with data as color map and signal as contour lines.
    Based on construct_2Dplot from Makeplots_gKK_Intime_Compile_plot.py

    Args:
        genEventSumw: dict with signal sample names as keys for proper MC normalization
        signal_names: list of signal sample names (e.g., ["WminusH", "WplusH"] for 1l)
    """
    if signal_names is None:
        signal_names = ["WminusH", "WplusH"]  # Default for backward compatibility
    nbins_x, xmin, xmax = bins_x
    nbins_y, ymin, ymax = bins_y

    # Contour levels: (fraction of max, color)
    # Colors: 89=kDeepSea+9, 92=kDeepSea+12, 96=kDeepSea+16, 99=kDeepSea+19
    ContourLevel = [(0.2, 89), (0.4, 92), (0.6, 96), (0.8, 99)]

    # Create 2D histograms
    h_data = ROOT.TH2F("h_data_2D", f";{xlabel};{ylabel}", nbins_x, xmin, xmax, nbins_y, ymin, ymax)
    h_signal = ROOT.TH2F("h_signal_2D", f";{xlabel};{ylabel}", nbins_x, xmin, xmax, nbins_y, ymin, ymax)

    # Evaluate variables (handle computed expressions)
    def eval_var(events_sample, var_name, sel):
        """Evaluate a variable, handling computed expressions."""
        if var_name in events_sample.fields:
            return ak.to_numpy(events_sample[var_name][sel])
        else:
            # Try to evaluate as expression
            local_dict = {field: events_sample[field][sel] for field in events_sample.fields}
            local_dict.update({'np': np, 'ak': ak, 'PI': np.pi, 'pi': np.pi, 'abs': np.abs, 'sqrt': np.sqrt})
            try:
                result = eval(var_name, {"__builtins__": {}}, local_dict)
                return ak.to_numpy(result)
            except Exception as e:
                print(f"  Warning: Could not evaluate {var_name}: {e}")
                return None

    # Build CDF map from data X values (used if cdf_flat_x=True)
    _cdf_sorted = None
    if cdf_flat_x and "data_2018" in events:
        sel_data_cdf = selection(events["data_2018"])
        x_cdf_raw = eval_var(events["data_2018"], x_var, sel_data_cdf)
        if x_cdf_raw is not None and len(x_cdf_raw) > 1:
            _cdf_sorted = np.sort(x_cdf_raw.astype(np.float64))

    def apply_cdf(x_arr):
        if _cdf_sorted is None:
            return x_arr
        return np.searchsorted(_cdf_sorted, x_arr.astype(np.float64)) / len(_cdf_sorted)

    # Fill data histogram
    if "data_2018" in events:
        sel_data = selection(events["data_2018"])
        x_vals = eval_var(events["data_2018"], x_var, sel_data)
        y_vals = eval_var(events["data_2018"], y_var, sel_data)
        if x_vals is not None and y_vals is not None:
            x_vals = apply_cdf(x_vals)
            for x, y in zip(x_vals, y_vals):
                h_data.Fill(x, y)

    # Fill signal histogram (with weights)
    for sample in signal_names:
        if sample not in events:
            continue
        W = events[sample]
        sel = selection(W)
        x_vals = eval_var(W, x_var, sel)
        y_vals = eval_var(W, y_var, sel)
        if x_vals is None or y_vals is None:
            continue
        x_vals = apply_cdf(x_vals)

        # Get weights
        lumi = ak.to_numpy(W["lumiwgt"][sel])
        pu = ak.to_numpy(W["puWeight"][sel])
        genWeight = ak.to_numpy(W["genWeight"][sel])
        # Use genEventSumw from Runs tree for proper normalization (if available)
        sum_gen_wgt = genEventSumw[sample] if (genEventSumw and sample in genEventSumw and genEventSumw[sample] > 0) else ak.sum(W["genWeight"])

        l1PreFiring = ak.to_numpy(safe_get(W, "l1PreFiringWeight", 1.0)[sel])
        muEffWeight = ak.to_numpy(safe_get(W, "muEffWeight", 1.0)[sel])
        elEffWeight = ak.to_numpy(safe_get(W, "elEffWeight", 1.0)[sel])
        pileupJetIdWeight = ak.to_numpy(safe_get(W, "pileupJetIdWeight", 1.0)[sel])

        weights = genWeight * xsec_x_BR[sample] * lumi * (l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight) / sum_gen_wgt

        for x, y, w in zip(x_vals, y_vals, weights):
            h_signal.Fill(x, y, w)

    # Smooth signal histogram and create contours
    h_signal.SetStats(0)
    h_signal.Smooth()

    maxZ = h_signal.GetMaximum()
    h_contours = []
    for frac, color in ContourLevel:
        hc = h_signal.Clone(f"hc_{frac}")
        hc.SetContour(1)
        hc.SetContourLevel(0, maxZ * frac)
        hc.SetLineColor(color)
        hc.SetLineWidth(3)
        h_contours.append((hc, frac, color))

    # Create canvas
    c = ROOT.TCanvas("c2D", "c2D", 500, 500)
    c.SetLeftMargin(0.12); c.SetRightMargin(0.14); c.SetTopMargin(0.09); c.SetBottomMargin(0.10)
    c.SetTickx(1); c.SetTicky(1)

    # Set color palette (kViridis = 112, same as mplhep CMS style)
    ROOT.gStyle.SetPalette(112)

    # Draw data as color map
    h_data.SetStats(0); h_data.Draw("COLZ")
    for ax, off in [(h_data.GetXaxis(), 0.9), (h_data.GetYaxis(), 1.3), (h_data.GetZaxis(), 1.0)]:
        ax.SetLabelSize(0.035); ax.SetTitleSize(0.042); ax.SetTitleOffset(off)
    h_data.GetZaxis().SetTitle("Data events")

    # Draw signal contours on top
    for hc, frac, color in h_contours:
        hc.Draw("CONT3 SAME")

    # Draw CMS label (left)
    cms = ROOT.TLatex(); cms.SetNDC(); cms.SetTextFont(62); cms.SetTextSize(0.047)
    cms.DrawLatex(0.12, 0.955, "CMS")

    # Draw lumi label (right)
    lumi_label = ROOT.TLatex(); lumi_label.SetNDC(); lumi_label.SetTextFont(42); lumi_label.SetTextSize(0.034)
    lumi_label.SetTextAlign(31); lumi_label.DrawLatex(0.86, 0.958, "59.7 fb^{-1} (13 TeV)")

    # Draw region label centered at top, outside frame
    display_region = region_label if region_label else ""
    region_text = ROOT.TLatex(); region_text.SetNDC(); region_text.SetTextFont(42)
    region_text.SetTextSize(0.036); region_text.SetTextAlign(22)
    region_text.DrawLatex(0.49, 0.972, f"{display_region} {channel}")

    # Draw contour legend + signal count in bottom-left of plot area
    n_signal = h_signal.Integral()
    leg_y = 0.37
    sig_count_lt = ROOT.TLatex(); sig_count_lt.SetNDC(); sig_count_lt.SetTextFont(42)
    sig_count_lt.SetTextSize(0.032); sig_count_lt.SetTextColor(ROOT.kYellow+1)
    sig_count_lt.DrawLatex(0.14, leg_y, f"Signal: {n_signal:.1f} events")
    leg_y -= 0.040
    for hc, frac, color in h_contours:
        leg_line = ROOT.TLatex(); leg_line.SetNDC(); leg_line.SetTextFont(42); leg_line.SetTextSize(0.032)
        leg_line.SetTextColor(color); leg_line.DrawLatex(0.14, leg_y, f"Signal {frac*100:.0f}% of max")
        leg_y -= 0.038

    c.SaveAs(f"{output_prefix}.png")
    print(f" --> 2D plot: {output_prefix}.png (Data={h_data.GetEntries():.0f}, Signal={n_signal:.1f})")
    c.Update(); c.Draw(); c.Close()


def plot_variable_2D_significance(events, x_var, y_var, selection, bins_x, bins_y, xlabel, ylabel,
                                   output_prefix, region_label=None, genEventSumw=None, signal_names=None,
                                   cdf_flat_x=False):
    """
    Plot 2D significance map: S/sqrt(D + eps) per bin, where eps=1e-9 (consistent with
    compute_significance_metrics). Data fills the denominator; signal fills the numerator.
    Signal contours are overlaid at the same fractional levels as plot_variable_2D.
    """
    eps = 1e-9
    if signal_names is None:
        signal_names = []
    nbins_x, xmin, xmax = bins_x
    nbins_y, ymin, ymax = bins_y
    ContourLevel = [(0.2, 89), (0.4, 92), (0.6, 96), (0.8, 99)]

    h_data   = ROOT.TH2F("h_signif_data",   "", nbins_x, xmin, xmax, nbins_y, ymin, ymax)
    h_signal = ROOT.TH2F("h_signif_signal", "", nbins_x, xmin, xmax, nbins_y, ymin, ymax)

    def eval_var(ev_sample, var_name, sel):
        if var_name in ev_sample.fields:
            return ak.to_numpy(ev_sample[var_name][sel])
        local_dict = {f: ev_sample[f][sel] for f in ev_sample.fields}
        local_dict.update({'np': np, 'ak': ak, 'abs': np.abs, 'sqrt': np.sqrt})
        try:
            return ak.to_numpy(eval(var_name, {"__builtins__": {}}, local_dict))
        except Exception as e:
            print(f"  Warning [signif 2D]: could not evaluate {var_name}: {e}"); return None

    # Build CDF map for x-axis if requested
    _cdf_sorted = None
    if cdf_flat_x and "data_2018" in events:
        _sel = selection(events["data_2018"])
        _xr  = eval_var(events["data_2018"], x_var, _sel)
        if _xr is not None and len(_xr) > 1:
            _cdf_sorted = np.sort(_xr.astype(np.float64))

    def apply_cdf(arr):
        if _cdf_sorted is None: return arr
        return np.searchsorted(_cdf_sorted, arr.astype(np.float64)) / len(_cdf_sorted)

    # Fill data
    if "data_2018" in events:
        sel_d = selection(events["data_2018"])
        xv = eval_var(events["data_2018"], x_var, sel_d)
        yv = eval_var(events["data_2018"], y_var, sel_d)
        if xv is not None and yv is not None:
            xv = apply_cdf(xv)
            for x, y in zip(xv, yv):
                h_data.Fill(x, y)

    # Fill signal (weighted)
    n_signal = 0.0
    for sample in signal_names:
        if sample not in events: continue
        W   = events[sample]
        sel = selection(W)
        xv  = eval_var(W, x_var, sel)
        yv  = eval_var(W, y_var, sel)
        if xv is None or yv is None: continue
        xv  = apply_cdf(xv)
        lumi       = ak.to_numpy(W["lumiwgt"][sel])
        pu         = ak.to_numpy(W["puWeight"][sel])
        genWeight  = ak.to_numpy(W["genWeight"][sel])
        sum_gen_wgt = genEventSumw[sample] if (genEventSumw and sample in genEventSumw and genEventSumw[sample] > 0) else ak.sum(W["genWeight"])
        l1PreFiring       = ak.to_numpy(safe_get(W, "l1PreFiringWeight", 1.0)[sel])
        muEffWeight       = ak.to_numpy(safe_get(W, "muEffWeight", 1.0)[sel])
        elEffWeight       = ak.to_numpy(safe_get(W, "elEffWeight", 1.0)[sel])
        pileupJetIdWeight = ak.to_numpy(safe_get(W, "pileupJetIdWeight", 1.0)[sel])
        weights = genWeight * xsec_x_BR[sample] * lumi * (l1PreFiring * pu * muEffWeight * elEffWeight * pileupJetIdWeight) / sum_gen_wgt
        for x, y, w in zip(xv, yv, weights):
            h_signal.Fill(x, y, w)
        n_signal += h_signal.Integral()

    # Build per-bin significance histogram
    h_signif = ROOT.TH2F("h_signif_2D", f";{xlabel};{ylabel}",
                          nbins_x, xmin, xmax, nbins_y, ymin, ymax)
    max_signif = 0.0
    for ix in range(1, nbins_x + 1):
        for iy in range(1, nbins_y + 1):
            d = h_data.GetBinContent(ix, iy)
            s = h_signal.GetBinContent(ix, iy)
            sig = s / (d + eps) ** 0.5 if s > 0 else 0.0
            h_signif.SetBinContent(ix, iy, sig)
            if sig > max_signif: max_signif = sig

    # Canvas
    c = ROOT.TCanvas("c2D_signif", "c2D_signif", 500, 500)
    c.SetLeftMargin(0.12); c.SetRightMargin(0.15); c.SetTopMargin(0.09); c.SetBottomMargin(0.10)
    c.SetTickx(1); c.SetTicky(1)
    ROOT.gStyle.SetPalette(55)  # kTemperatureMap: blue→red (distinguishable from data plot)

    h_signif.SetStats(0); h_signif.Draw("COLZ")
    for ax, off in [(h_signif.GetXaxis(), 0.9), (h_signif.GetYaxis(), 1.3), (h_signif.GetZaxis(), 1.1)]:
        ax.SetLabelSize(0.035); ax.SetTitleSize(0.042); ax.SetTitleOffset(off)
    h_signif.GetZaxis().SetTitle("S/#sqrt{D+#varepsilon}")

    # CMS label
    cms = ROOT.TLatex(); cms.SetNDC(); cms.SetTextFont(62); cms.SetTextSize(0.047)
    cms.DrawLatex(0.12, 0.955, "CMS")
    lumi_label = ROOT.TLatex(); lumi_label.SetNDC(); lumi_label.SetTextFont(42); lumi_label.SetTextSize(0.034)
    lumi_label.SetTextAlign(31); lumi_label.DrawLatex(0.85, 0.958, "59.7 fb^{-1} (13 TeV)")

    # Region label centered at top
    display_region = region_label if region_label else ""
    rt = ROOT.TLatex(); rt.SetNDC(); rt.SetTextFont(42); rt.SetTextSize(0.036); rt.SetTextAlign(22)
    rt.DrawLatex(0.49, 0.972, f"{display_region} {channel}")

    # Signal count
    sc = ROOT.TLatex(); sc.SetNDC(); sc.SetTextFont(42); sc.SetTextSize(0.032); sc.SetTextColor(ROOT.kYellow+1)
    sc.DrawLatex(0.14, 0.37, f"Signal: {n_signal:.1f} events")

    c.SaveAs(f"{output_prefix}.png")
    print(f" --> 2D signif plot: {output_prefix}.png (max S/sqrt(D+eps)={max_signif:.4f}, Signal={n_signal:.1f})")
    c.Update(); c.Draw(); c.Close()


def make_roc_rootstyle(events, var_names, var_labels, selection, channel, signal_names, output_prefix):
    """
    ROC comparison in ROC.py style: ROOT TGraph, AUC in legend, CMS Preliminary.
    Signal = combined signal_names (unweighted counts).  Background = data_2018.
    One curve per (var_name, var_label) pair.
    """
    from array import array as carray
    colors = [ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kOrange+7, ROOT.kMagenta+1]

    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas(f"c_roc_{channel}_{output_prefix}", "ROC", 600, 600)
    c.SetLeftMargin(0.12); c.SetRightMargin(0.04)
    c.SetTopMargin(0.08);  c.SetBottomMargin(0.12)

    legend = ROOT.TLegend(0.18, 0.65, 0.62, 0.83)
    legend.SetBorderSize(0); legend.SetTextFont(42)
    legend.SetTextSize(0.035); legend.SetFillStyle(0)

    graphs = []
    nbins, vmin, vmax = 500, 0.0, 1.0

    for i, (var_name, var_label) in enumerate(zip(var_names, var_labels)):
        # --- signal ---
        sig_list = []
        for sname in signal_names:
            if sname not in events or var_name not in events[sname].fields:
                continue
            mask = selection(events[sname])
            vals = ak.to_numpy(events[sname][var_name][mask]).astype(np.float64)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            sig_list.append(vals)
        if not sig_list:
            print(f"  [ROC] no signal for {var_name}, skipping"); continue
        sig_vals = np.concatenate(sig_list)

        # --- background (data) ---
        if "data_2018" not in events or var_name not in events["data_2018"].fields:
            print(f"  [ROC] no data for {var_name}, skipping"); continue
        mask_d = selection(events["data_2018"])
        bkg_vals = ak.to_numpy(events["data_2018"][var_name][mask_d]).astype(np.float64)
        bkg_vals = bkg_vals[np.isfinite(bkg_vals) & (bkg_vals >= 0)]

        sig_hist, _ = np.histogram(sig_vals, bins=nbins, range=(vmin, vmax))
        bkg_hist, _ = np.histogram(bkg_vals, bins=nbins, range=(vmin, vmax))
        s_total = sig_hist.sum(); b_total = bkg_hist.sum()
        if s_total == 0 or b_total == 0:
            print(f"  [ROC] zero entries for {var_name}, skipping"); continue

        # cumulative from high score → low (signal-like end first)
        sig_eff_pts, bkg_eff_pts = [], []
        s_cum = b_cum = 0.0
        for s, b in zip(reversed(sig_hist), reversed(bkg_hist)):
            s_cum += s; b_cum += b
            sig_eff_pts.append(s_cum / s_total)
            bkg_eff_pts.append(b_cum / b_total)

        auc = sum(
            0.5 * (sig_eff_pts[j] + sig_eff_pts[j-1]) * (bkg_eff_pts[j] - bkg_eff_pts[j-1])
            for j in range(1, len(sig_eff_pts))
        )

        g = ROOT.TGraph(len(sig_eff_pts),
                        carray('f', bkg_eff_pts),
                        carray('f', sig_eff_pts))
        g.SetLineWidth(3)
        g.SetLineColor(colors[i % len(colors)])
        g.SetName(var_name)
        g.Draw("AL" if i == 0 else "L SAME")
        legend.AddEntry(g, f"{var_label},  AUC = {auc:.3f}", "l")
        graphs.append(g)

    if not graphs:
        print(f"  [ROC] nothing to draw for {output_prefix}"); return

    graphs[0].GetXaxis().SetLimits(0, 1)
    graphs[0].SetMinimum(0); graphs[0].SetMaximum(1)
    graphs[0].GetXaxis().SetTitle("Background Efficiency")
    graphs[0].GetYaxis().SetTitle("Signal Efficiency")
    graphs[0].SetTitle("")

    cms = ROOT.TLatex(); cms.SetNDC(); cms.SetTextFont(61); cms.SetTextSize(0.048)
    cms.DrawLatex(0.14, 0.93, "CMS")
    prelim = ROOT.TLatex(); prelim.SetNDC(); prelim.SetTextFont(52); prelim.SetTextSize(0.040)
    prelim.DrawLatex(0.26, 0.93, "Preliminary")
    energy = ROOT.TLatex(); energy.SetNDC(); energy.SetTextFont(42); energy.SetTextSize(0.038)
    energy.DrawLatex(0.74, 0.93, "59.7 fb^{-1} (13 TeV)")
    ch_lbl = ROOT.TLatex(); ch_lbl.SetNDC(); ch_lbl.SetTextFont(42); ch_lbl.SetTextSize(0.038)
    ch_lbl.DrawLatex(0.14, 0.87, f"Preselection  {channel}")

    legend.Draw(); c.Update()
    c.SaveAs(f"{output_prefix}.png"); c.SaveAs(f"{output_prefix}.pdf")
    print(f" --> ROC plot: {output_prefix}.png")
    os.system(f"display {output_prefix}.png &")


def plot_roc(events, var_names, var_labels, selection, output_prefix, signal_names=None,
             bkg_name="data_2018", channel=None, region_label=None, higher_is_signal=None, signal_norm=None,
             mc_bkg_names=None, disc_type=None, x_label="Signal Efficiency", save_npz=None):
    """
    Plot ROC curves comparing multiple discriminating variables using ROOT.

    Args:
        events: dict of event arrays (signal and background samples)
        var_names: list of variable names to compare (e.g., ["BDT_ParT", "BDT_KIN"])
        var_labels: list of labels for legend (e.g., ["ParT BDT", "KIN BDT"])
        selection: selection function to apply
        disc_type: discriminator type (e.g., "ParT", "KIN") - shown in region label, stripped from legend
        output_prefix: output file name prefix
        signal_names: list of signal sample names (will be combined)
        bkg_name: background sample name (default: "data_2018")
        channel: channel label for plot title
        region_label: region label for plot title
        higher_is_signal: list of bools, True if higher values = more signal-like (default: all True)
        signal_norm: normalization factor for signal (NORM = data_integral/signal_integral). If None, computed automatically.
        mc_bkg_names: list of MC background sample names to combine for MC ROC (dashed lines). If None, no MC ROC shown.

    Returns:
        dict with AUC values and optimal cut info for each variable
    """
    if signal_names is None:
        signal_names = ["WminusH", "WplusH"]
    if higher_is_signal is None:
        higher_is_signal = [True] * len(var_names)

    # ROOT colors and markers (extended for more curves)
    root_colors = [
        ROOT.kRed+1, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta+1, ROOT.kOrange+1,
        ROOT.kCyan+1, ROOT.kViolet+1, ROOT.kTeal+1, ROOT.kPink+1, ROOT.kAzure+1
    ]
    root_markers = [20, 21, 22, 23, 29, 33, 34, 24, 25, 26]  # Various marker styles

    results = {}
    results_mc = {}  # For MC-based ROC curves
    graphs = []  # Keep graphs alive
    opt_markers = []  # Keep optimal point markers alive

    def compute_roc_for_bkg(var_name, sig_vals, bkg_vals, higher_sig, signal_norm_val):
        """Helper to compute ROC curve for given signal and background arrays."""
        if len(sig_vals) == 0 or len(bkg_vals) == 0:
            return None

        n_sig = len(sig_vals)
        n_bkg = len(bkg_vals)

        sig_sorted = np.sort(sig_vals)
        bkg_sorted = np.sort(bkg_vals)

        all_vals = np.concatenate([sig_vals, bkg_vals])
        n_thresh = min(500, len(np.unique(all_vals)))
        thresholds = np.percentile(all_vals, np.linspace(0, 100, n_thresh))
        thresholds = np.unique(thresholds)
        thresholds = np.sort(thresholds)

        if not higher_sig:
            thresholds = thresholds[::-1]

        norm = signal_norm_val if signal_norm_val is not None else (n_bkg / n_sig)

        if higher_sig:
            sig_pass_arr = n_sig - np.searchsorted(sig_sorted, thresholds, side='left')
            bkg_pass_arr = n_bkg - np.searchsorted(bkg_sorted, thresholds, side='left')
        else:
            sig_pass_arr = np.searchsorted(sig_sorted, thresholds, side='right')
            bkg_pass_arr = np.searchsorted(bkg_sorted, thresholds, side='right')

        sig_eff = sig_pass_arr / n_sig
        bkg_eff = bkg_pass_arr / n_bkg

        with np.errstate(divide='ignore', invalid='ignore'):
            sig_normalized = sig_pass_arr / norm
            significance = np.where(bkg_pass_arr > 0, sig_normalized / np.sqrt(bkg_pass_arr), 0)

        opt_idx = np.argmax(significance)

        sorted_idx = np.argsort(sig_eff)
        sig_eff_sorted = sig_eff[sorted_idx]
        bkg_eff_sorted = bkg_eff[sorted_idx]
        auc = 1 - np.trapz(bkg_eff_sorted, sig_eff_sorted)

        return {
            'auc': auc,
            'opt_cut': thresholds[opt_idx],
            'opt_sig_eff': sig_eff[opt_idx],
            'opt_bkg_eff': bkg_eff[opt_idx],
            'opt_significance': significance[opt_idx],
            'sig_eff': sig_eff,
            'bkg_eff': bkg_eff,
        }

    for i, (var_name, var_label, higher_sig) in enumerate(zip(var_names, var_labels, higher_is_signal)):
        # Collect signal values (combine all signal samples)
        sig_vals = []
        for sig_name in signal_names:
            if sig_name not in events:
                continue
            if var_name not in events[sig_name].fields:
                print(f"  Warning: {var_name} not in {sig_name}")
                continue
            sel_mask = selection(events[sig_name])
            vals = ak.to_numpy(events[sig_name][var_name][sel_mask])
            # Filter out invalid values (e.g., BDT/MLP/DNN < 0 means failed precut)
            vals = vals[np.isfinite(vals)]
            if var_name.startswith("BDT_") or var_name.startswith("MLP_") or var_name.startswith("DNN_") or var_name in ["kinBDT", "flavBDT"]:
                vals = vals[vals >= 0]
            sig_vals.append(vals)

        if not sig_vals:
            print(f"  Warning: No signal data for {var_name}")
            continue
        sig_vals = np.concatenate(sig_vals)

        # Collect Data background values
        data_bkg_vals = None
        if bkg_name in events and var_name in events[bkg_name].fields:
            sel_mask = selection(events[bkg_name])
            data_bkg_vals = ak.to_numpy(events[bkg_name][var_name][sel_mask])
            data_bkg_vals = data_bkg_vals[np.isfinite(data_bkg_vals)]
            if var_name.startswith("BDT_") or var_name.startswith("MLP_") or var_name.startswith("DNN_") or var_name in ["kinBDT", "flavBDT"]:
                data_bkg_vals = data_bkg_vals[data_bkg_vals >= 0]

        # Collect MC background values (combine all MC samples)
        mc_bkg_vals = None
        if mc_bkg_names:
            mc_vals_list = []
            for mc_name in mc_bkg_names:
                if mc_name not in events:
                    continue
                if var_name not in events[mc_name].fields:
                    continue
                sel_mask = selection(events[mc_name])
                vals = ak.to_numpy(events[mc_name][var_name][sel_mask])
                vals = vals[np.isfinite(vals)]
                if var_name.startswith("BDT_") or var_name.startswith("MLP_") or var_name.startswith("DNN_") or var_name in ["kinBDT", "flavBDT"]:
                    vals = vals[vals >= 0]
                if len(vals) > 0:
                    mc_vals_list.append(vals)
            if mc_vals_list:
                mc_bkg_vals = np.concatenate(mc_vals_list)

        # Compute ROC for Data background
        if data_bkg_vals is not None and len(data_bkg_vals) > 0:
            roc_data = compute_roc_for_bkg(var_name, sig_vals, data_bkg_vals, higher_sig, signal_norm)
            if roc_data:
                roc_data['color'] = root_colors[i % len(root_colors)]
                roc_data['marker'] = root_markers[i % len(root_markers)]
                roc_data['label'] = var_label
                results[var_name] = roc_data

        # Compute ROC for MC background
        if mc_bkg_vals is not None and len(mc_bkg_vals) > 0:
            roc_mc = compute_roc_for_bkg(var_name, sig_vals, mc_bkg_vals, higher_sig, signal_norm)
            if roc_mc:
                roc_mc['color'] = root_colors[i % len(root_colors)]
                roc_mc['marker'] = root_markers[i % len(root_markers)]
                roc_mc['label'] = var_label
                results_mc[var_name] = roc_mc

    # Optional: dump raw ROC arrays to .npz for later cross-channel combination
    # (see combine_roc_npz / mode "ROCX") — avoids re-running full event loading.
    if save_npz:
        _npz_kwargs = {}
        for var_name, r in results.items():
            _npz_kwargs[f"{var_name}__data_sig_eff"] = r["sig_eff"]
            _npz_kwargs[f"{var_name}__data_bkg_eff"] = r["bkg_eff"]
            _npz_kwargs[f"{var_name}__data_auc"] = r["auc"]
        for var_name, r in results_mc.items():
            _npz_kwargs[f"{var_name}__mc_sig_eff"] = r["sig_eff"]
            _npz_kwargs[f"{var_name}__mc_bkg_eff"] = r["bkg_eff"]
            _npz_kwargs[f"{var_name}__mc_auc"] = r["auc"]
        if _npz_kwargs:
            np.savez(save_npz, **_npz_kwargs)
            print(f"  --> Saved ROC arrays: {save_npz}")

    # Create ROOT canvas (490x490 to match other plots)
    c = ROOT.TCanvas("c_roc", "c_roc", 490, 490)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.03)
    c.SetTopMargin(0.06)
    c.SetBottomMargin(0.10)
    c.SetLogy(1)
    # Create frame for axes (x: 0–1, y: 0.01–1 log scale)
    frame = c.DrawFrame(0, 0.01, 1, 1)
    frame.GetXaxis().SetTitle(x_label)
    frame.GetYaxis().SetTitle("Background Efficiency")
    frame.GetXaxis().SetTitleSize(0.050)
    frame.GetYaxis().SetTitleSize(0.050)
    frame.GetXaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetLabelSize(0.042)
    frame.GetXaxis().SetTitleOffset(0.85)
    frame.GetYaxis().SetTitleOffset(1.0)

    # Draw random classifier curve (diagonal: sig_eff = bkg_eff)
    n_rand = 100
    x_rand = np.linspace(0, 1, n_rand)
    y_rand = x_rand.copy()  # Random: bkg_eff = sig_eff
    rand_curve = ROOT.TGraph(n_rand, x_rand.astype('float64'), y_rand.astype('float64'))
    rand_curve.SetLineColor(ROOT.kGray+1)
    rand_curve.SetLineStyle(2)
    rand_curve.SetLineWidth(2)
    rand_curve.Draw("L SAME")
    graphs.append(rand_curve)

    # Draw ROC curves (Data - bold solid lines)
    for var_name, r in results.items():
        n_points = len(r['sig_eff'])
        g = ROOT.TGraph(n_points, r['sig_eff'].astype('float64'), r['bkg_eff'].astype('float64'))
        g.SetLineColor(r['color'])
        g.SetLineWidth(3)  # Bold for Data
        g.SetLineStyle(1)  # Solid
        g.Draw("L SAME")
        graphs.append(g)

        # Draw optimal point marker (filled circle, small)
        g_opt = ROOT.TGraph(1, np.array([r['opt_sig_eff']]), np.array([r['opt_bkg_eff']]))
        g_opt.SetMarkerStyle(20)  # Filled circle
        g_opt.SetMarkerSize(0.8)
        g_opt.SetMarkerColor(r['color'])
        g_opt.Draw("P SAME")
        opt_markers.append(g_opt)

    # Draw MC ROC curves (transparent solid lines)
    for var_name, r in results_mc.items():
        n_points = len(r['sig_eff'])
        g = ROOT.TGraph(n_points, r['sig_eff'].astype('float64'), r['bkg_eff'].astype('float64'))
        g.SetLineColorAlpha(r['color'], 0.4)  # Transparent for MC
        g.SetLineWidth(2)
        g.SetLineStyle(1)  # Solid (same as Data, distinguished by transparency)
        g.Draw("L SAME")
        graphs.append(g)

        # Draw optimal point marker (open circle, small)
        g_opt = ROOT.TGraph(1, np.array([r['opt_sig_eff']]), np.array([r['opt_bkg_eff']]))
        g_opt.SetMarkerStyle(24)  # Open circle
        g_opt.SetMarkerSize(0.8)
        g_opt.SetMarkerColor(r['color'])
        g_opt.Draw("P SAME")
        opt_markers.append(g_opt)

    # CMS label (matching other plots)
    cms = ROOT.TLatex()
    cms.SetNDC()
    cms.SetTextFont(62)
    cms.SetTextSize(0.055)
    cms.DrawLatex(0.12, 0.95, "CMS")

    prel_lumi = ROOT.TLatex()
    prel_lumi.SetNDC()
    prel_lumi.SetTextFont(42)
    prel_lumi.SetTextSize(0.042)
    prel_lumi.DrawLatex(0.24, 0.95, "#it{Preliminary}                             59.7 fb^{-1} (13 TeV)")

    # Region label (just below upper frame bound) - include disc_type if provided
    if channel or region_label or disc_type:
        parts = []
        if channel: parts.append(channel)
        if region_label: parts.append(region_label)
        if disc_type: parts.append(disc_type)
        label_text = " ".join(parts)
        region = ROOT.TLatex()
        region.SetNDC()
        region.SetTextFont(42)
        region.SetTextSize(0.045)
        region.DrawLatex(0.15, 0.90, label_text)

    # Legend setup
    has_mc = len(results_mc) > 0
    n_entries = len(results) + 1  # discriminators + random classifier

    # Helper to strip disc_type from label if present
    def strip_disc_type(label):
        if disc_type and disc_type in label:
            return label.replace(f" {disc_type}", "").replace(f"{disc_type} ", "").replace(disc_type, "").strip()
        return label

    # Main legend: discriminators (top left)
    leg_top = 0.89
    leg_bottom = leg_top - n_entries * 0.050
    leg = ROOT.TLegend(0.12, leg_bottom, 0.58, leg_top)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetTextFont(42)

    # Discriminator entries (colored lines, no Data/MC distinction)
    for i, (var_name, r) in enumerate(results.items()):
        g_leg = ROOT.TGraph(1)
        g_leg.SetLineColor(r['color'])
        g_leg.SetLineWidth(2)
        g_leg.SetLineStyle(1)  # Solid
        g_leg.SetMarkerStyle(r['marker'])
        g_leg.SetMarkerColor(r['color'])
        g_leg.SetMarkerSize(1.0)
        graphs.append(g_leg)
        clean_label = strip_disc_type(r['label'])
        # Show both AUCs if MC available
        if has_mc and var_name in results_mc:
            legend_text = f"{clean_label} {r['auc']:.3f}; {results_mc[var_name]['auc']:.3f}"
        else:
            legend_text = f"{clean_label} AUC={r['auc']:.3f}"
        leg.AddEntry(g_leg, legend_text, "lp")

    # Add random classifier to legend at bottom
    g_rand_leg = ROOT.TGraph(1)
    g_rand_leg.SetLineColor(ROOT.kGray+1)
    g_rand_leg.SetLineStyle(2)
    g_rand_leg.SetLineWidth(2)
    graphs.append(g_rand_leg)
    leg.AddEntry(g_rand_leg, "Random", "l")
    leg.Draw()

    # Optimal cut annotations: bottom right, right-aligned
    y_pos = 0.14 + len(results) * 0.046
    for i, (var_name, r) in enumerate(results.items()):
        cut_text = ROOT.TLatex()
        cut_text.SetNDC()
        cut_text.SetTextFont(42)
        cut_text.SetTextSize(0.036)
        cut_text.SetTextAlign(31)  # right-align
        cut_text.SetTextColor(r['color'])
        cut_text.DrawLatex(0.97, y_pos, f"cut > {r['opt_cut']:.2f}: #varepsilon_{{S}}={r['opt_sig_eff']:.2f}, #varepsilon_{{B}}={r['opt_bkg_eff']:.3f}")
        y_pos -= 0.046

    c.Update()
    c.SaveAs(f"{output_prefix}.png")
    print(f" --> ROC plot: {output_prefix}.png")
    for var_name, r in results.items():
        print(f"     {var_name} (Data): AUC={r['auc']:.4f}, optimal cut>{r['opt_cut']:.2f} (eff_S={r['opt_sig_eff']:.3f}, eff_B={r['opt_bkg_eff']:.4f})")
    for var_name, r in results_mc.items():
        print(f"     {var_name} (MC):   AUC={r['auc']:.4f}, optimal cut>{r['opt_cut']:.2f} (eff_S={r['opt_sig_eff']:.3f}, eff_B={r['opt_bkg_eff']:.4f})")

    # Clean up results dict (remove arrays to save memory)
    for var_name in results:
        del results[var_name]['sig_eff']
        del results[var_name]['bkg_eff']
    for var_name in results_mc:
        del results_mc[var_name]['sig_eff']
        del results_mc[var_name]['bkg_eff']

    return results, results_mc


def combine_roc_npz(channels=("0l", "1l", "2l"), var_name="MLPf_X", var_label="X-score (MLPf)",
                     npz_prefix="roc_MLPfX", output_prefix="roc_MLPfX_combined",
                     x_label="X-score signal efficiency"):
    """
    Overlay per-channel ROC curves (Data solid, MC transparent) saved by plot_roc(save_npz=...)
    into a single canvas — one color per channel. Needs no event loading, just reads the .npz
    files produced by prior `python Root_plot.py <ch> ROC --loadmc` runs.
    """
    root_colors = [ROOT.kRed+1, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta+1]
    graphs = []
    data_curves = {}
    mc_curves = {}

    for i, ch in enumerate(channels):
        fname = f"{npz_prefix}_{ch}.npz"
        if not os.path.exists(fname):
            print(f"  Warning: {fname} not found, skipping {ch}"); continue
        z = np.load(fname)
        color = root_colors[i % len(root_colors)]
        if f"{var_name}__data_sig_eff" in z:
            data_curves[ch] = {
                "sig_eff": z[f"{var_name}__data_sig_eff"], "bkg_eff": z[f"{var_name}__data_bkg_eff"],
                "auc": float(z[f"{var_name}__data_auc"]), "color": color,
            }
        if f"{var_name}__mc_sig_eff" in z:
            mc_curves[ch] = {
                "sig_eff": z[f"{var_name}__mc_sig_eff"], "bkg_eff": z[f"{var_name}__mc_bkg_eff"],
                "auc": float(z[f"{var_name}__mc_auc"]), "color": color,
            }

    if not data_curves and not mc_curves:
        print(f"  --> No ROC .npz files found for prefix '{npz_prefix}', nothing to combine.")
        return

    c = ROOT.TCanvas("c_roc_combined", "c_roc_combined", 490, 490)
    c.SetLeftMargin(0.12); c.SetRightMargin(0.03); c.SetTopMargin(0.06); c.SetBottomMargin(0.10)
    c.SetLogy(1)
    frame = c.DrawFrame(0, 0.01, 1, 1)
    frame.GetXaxis().SetTitle(x_label)
    frame.GetYaxis().SetTitle("Background Efficiency")
    frame.GetXaxis().SetTitleSize(0.050); frame.GetYaxis().SetTitleSize(0.050)
    frame.GetXaxis().SetLabelSize(0.042); frame.GetYaxis().SetLabelSize(0.042)
    frame.GetXaxis().SetTitleOffset(0.85); frame.GetYaxis().SetTitleOffset(1.0)

    n_rand = 100
    x_rand = np.linspace(0, 1, n_rand)
    rand_curve = ROOT.TGraph(n_rand, x_rand.astype('float64'), x_rand.astype('float64'))
    rand_curve.SetLineColor(ROOT.kGray+1); rand_curve.SetLineStyle(2); rand_curve.SetLineWidth(2)
    rand_curve.Draw("L SAME")
    graphs.append(rand_curve)

    for ch, r in data_curves.items():
        g = ROOT.TGraph(len(r["sig_eff"]), r["sig_eff"].astype('float64'), r["bkg_eff"].astype('float64'))
        g.SetLineColor(r["color"]); g.SetLineWidth(3); g.SetLineStyle(1)
        g.Draw("L SAME"); graphs.append(g)

    for ch, r in mc_curves.items():
        g = ROOT.TGraph(len(r["sig_eff"]), r["sig_eff"].astype('float64'), r["bkg_eff"].astype('float64'))
        g.SetLineColorAlpha(r["color"], 0.4); g.SetLineWidth(2); g.SetLineStyle(1)
        g.Draw("L SAME"); graphs.append(g)

    cms = ROOT.TLatex(); cms.SetNDC(); cms.SetTextFont(62); cms.SetTextSize(0.055)
    cms.DrawLatex(0.12, 0.95, "CMS")
    prel_lumi = ROOT.TLatex(); prel_lumi.SetNDC(); prel_lumi.SetTextFont(42); prel_lumi.SetTextSize(0.042)
    prel_lumi.DrawLatex(0.24, 0.95, "#it{Preliminary}                             59.7 fb^{-1} (13 TeV)")
    region = ROOT.TLatex(); region.SetNDC(); region.SetTextFont(42); region.SetTextSize(0.045)
    region.DrawLatex(0.15, 0.90, var_label)

    n_entries = len(set(list(data_curves.keys()) + list(mc_curves.keys()))) + 1
    leg = ROOT.TLegend(0.45, 0.89 - n_entries * 0.055, 0.90, 0.89)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035); leg.SetTextFont(42)
    for ch in channels:
        if ch not in data_curves and ch not in mc_curves:
            continue
        color = data_curves.get(ch, mc_curves.get(ch))["color"]
        g_leg = ROOT.TGraph(1); g_leg.SetLineColor(color); g_leg.SetLineWidth(3)
        auc_parts = []
        if ch in data_curves: auc_parts.append(f"Data AUC={data_curves[ch]['auc']:.3f}")
        if ch in mc_curves:   auc_parts.append(f"MC AUC={mc_curves[ch]['auc']:.3f}")
        leg.AddEntry(g_leg, f"{ch}: {', '.join(auc_parts)}", "l")
    leg.AddEntry(rand_curve, "Random", "l")
    leg.Draw()

    c.SaveAs(f"{output_prefix}.png")
    c.SaveAs(f"{output_prefix}.pdf")
    print(f" --> Combined ROC plot: {output_prefix}.png")
    os.system(f"display {output_prefix}.png &")


def plot_mass_sculpting(events, mass_var, scan_var, cut_values, selection, output_prefix, channel=None, use_eff_cuts=False, nbins=15, xmin=30, xmax=230):
    """Mass sculpting study: m_j distributions for different cuts. Two pads: Data (top), MC (bottom).
    If use_eff_cuts=True, cut_values are interpreted as target Data efficiencies (e.g., [1, 0.3, 0.1, 0.03]).
    mass_var can be a column name or an expression using column names (e.g., '1-abs(ak15_sdmass-135)/135')."""
    root_colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen+2, ROOT.kOrange+1, ROOT.kRed+2]

    def eval_var(sample, var_str):
        """Evaluate a variable: either a field name or an expression."""
        if var_str in sample.fields:
            return sample[var_str]
        # Try to evaluate as expression using sample fields
        local_vars = {f: sample[f] for f in sample.fields}
        local_vars['abs'] = ak.abs if hasattr(ak, 'abs') else np.abs
        try:
            return eval(var_str, {"__builtins__": {}, "abs": abs}, local_vars)
        except Exception as e:
            print(f"  Warning: Could not evaluate '{var_str}': {e}")
            return None

    def get_sample_data(sample_name):
        if sample_name not in events: return np.array([]), np.array([])
        sample = events[sample_name]
        if scan_var not in sample.fields: return np.array([]), np.array([])
        mask = selection(sample)
        m_arr = eval_var(sample, mass_var)
        if m_arr is None: return np.array([]), np.array([])
        m_arr = m_arr[mask]; s_arr = sample[scan_var][mask]
        return (ak.to_numpy(ak.flatten(m_arr)) if m_arr.ndim > 1 else ak.to_numpy(m_arr),
                ak.to_numpy(ak.flatten(s_arr)) if s_arr.ndim > 1 else ak.to_numpy(s_arr))

    mass_data, scan_data = get_sample_data("data_2018")
    mc_samples = (["TTTo2L2Nu", "TTToSemiLeptonic", "ST_t-channel_top", "ST_t-channel_antitop", "ST_tW_top", "ST_tW_antitop", "ST_s-channel"] +
                  ["WWTo1L1Nu2Q", "WWTo2L2Nu", "WZTo1L1Nu2Q", "WZTo1L3Nu", "WZTo3LNu", "WZTo2Q2L", "WZTo2Q2Nu", "ZZTo2Q2L", "ZZTo2L2Nu", "ZZTo2Q2Nu", "ZZTo4L"] +
                  [k for k in events.keys() if "DYJets" in k or "ZJets_NuNu" in k or "WJetsToLNu" in k])
    mass_mc_list, scan_mc_list = [], []
    for mc in mc_samples:
        m, s = get_sample_data(mc)
        if len(m) > 0: mass_mc_list.append(m); scan_mc_list.append(s)
    mass_mc = np.concatenate(mass_mc_list) if mass_mc_list else np.array([])
    scan_mc = np.concatenate(scan_mc_list) if scan_mc_list else np.array([])

    # Convert efficiency targets to actual cut values - separately for Data and MC
    eff_labels = None
    if use_eff_cuts:
        # Check for empty arrays before computing percentiles
        if len(scan_data) == 0:
            print(f"  Warning: No data events for {scan_var} - skipping sculpting plot")
            return None
        if len(scan_mc) == 0:
            print(f"  Warning: No MC events for {scan_var} - skipping sculpting plot")
            return None
        eff_labels = cut_values[:]  # Store original efficiencies for legend
        cut_values_data = [np.percentile(scan_data, 100*(1-eff)) if eff < 1 else 0 for eff in eff_labels]
        cut_values_mc = [np.percentile(scan_mc, 100*(1-eff)) if eff < 1 else 0 for eff in eff_labels]
        print(f"  Data eff cuts: {dict(zip(eff_labels, [f'{c:.3f}' for c in cut_values_data]))}")
        print(f"  MC eff cuts:   {dict(zip(eff_labels, [f'{c:.3f}' for c in cut_values_mc]))}")
    else:
        cut_values_data = cut_values
        cut_values_mc = cut_values

    c = ROOT.TCanvas("c_sculpt", "c_sculpt", 490, 490); c.cd()
    pad1 = ROOT.TPad("pad1_sculpt", "pad1_sculpt", 0, 0.5, 1, 1.0)
    pad1.SetLeftMargin(0.1); pad1.SetRightMargin(0.025); pad1.SetTopMargin(0.085); pad1.SetBottomMargin(0.0); pad1.Draw()
    pad2 = ROOT.TPad("pad2_sculpt", "pad2_sculpt", 0, 0.0, 1, 0.5)
    pad2.SetLeftMargin(0.1); pad2.SetRightMargin(0.025); pad2.SetTopMargin(0.0); pad2.SetBottomMargin(0.16); pad2.Draw()

    def make_histos(mass_np, scan_np, prefix, cuts):
        histos, y_max, y_min = [], 0, 1e9
        for i, cut_val in enumerate(cuts):
            h = ROOT.TH1F(f"h_{prefix}_{i}", "", nbins, xmin, xmax)
            if len(mass_np) > 0:
                for m in mass_np[scan_np > cut_val]: h.Fill(m)
            n_ev = int(h.GetEntries())
            if h.Integral() > 0: h.Scale(1.0 / h.Integral())
            col = root_colors[i % len(root_colors)]
            h.SetLineColor(col); h.SetLineWidth(2); h.SetMarkerColor(col); h.SetMarkerStyle(1); h.SetMarkerSize(0)
            h.n_events = n_ev; h.cut_val = cut_val
            h.eff_label = eff_labels[i] if eff_labels else None  # Store efficiency label
            histos.append(h)
            if h.GetMaximum() > y_max: y_max = h.GetMaximum()
            if h.GetMinimum(0) < y_min and h.GetMinimum(0) > 0: y_min = h.GetMinimum(0)  # min excluding 0
        return histos, y_max, y_min

    # --- Data pad ---
    pad1.cd(); pad1.SetTickx(1); pad1.SetTicky(1)
    histos_data, y_max_data, y_min_data = make_histos(mass_data, scan_data, "data", cut_values_data)
    frame1 = ROOT.TH1F("frame1_sculpt", "", nbins, xmin, xmax); frame1.SetMinimum(y_min_data * 0.9); frame1.SetMaximum(y_max_data * 1.1)
    frame1.GetXaxis().SetTitle(""); frame1.GetXaxis().SetLabelSize(0)
    frame1.GetYaxis().SetTitle("Normalized to unity"); frame1.GetYaxis().SetLabelSize(0.063); frame1.GetYaxis().SetTitleSize(0.068); frame1.GetYaxis().SetTitleOffset(0.80)
    frame1.GetXaxis().SetNdivisions(510); frame1.GetYaxis().SetNdivisions(510); frame1.Draw("AXIS")
    for h in histos_data: h.Draw("HIST E SAME")
    scan_var_short = scan_var.replace("ak15_ParTMDV2_", "ParT_")
    leg1 = ROOT.TLegend(0.15, 0.02, 0.70, 0.34); leg1.SetHeader(f"                      #bf{{Data}}  {scan_var_short} cuts")
    leg1.SetNColumns(2); leg1.SetColumnSeparation(0.1); leg1.SetBorderSize(0); leg1.SetFillStyle(0); leg1.SetTextSize(0.065)
    for h in histos_data:
        lbl = f"eff={h.eff_label:.0%} (>{h.cut_val:.2f})" if h.eff_label else f"> {h.cut_val}  (N={h.n_events})"
        leg1.AddEntry(h, lbl, "l")
    leg1.Draw()
    cms = ROOT.TLatex(); cms.SetNDC(); cms.SetTextFont(62); cms.SetTextSize(0.09); cms.DrawLatex(0.12, 0.925, "CMS")
    prel = ROOT.TLatex(); prel.SetNDC(); prel.SetTextFont(42); prel.SetTextSize(0.07)
    prel.DrawLatex(0.22, 0.93, f"#it{{Preliminary}}              {channel} channel              59.7 fb^{{-1}} (13 TeV)")

    # --- MC pad (use same y-range as Data) ---
    pad2.cd(); pad2.SetTickx(1); pad2.SetTicky(1)
    histos_mc, y_max_mc, y_min_mc = make_histos(mass_mc, scan_mc, "mc", cut_values_mc)
    frame2 = ROOT.TH1F("frame2_sculpt", "", nbins, xmin, xmax); frame2.SetMinimum(y_min_data * 0.9); frame2.SetMaximum(y_max_data * 1.1)
    frame2.GetXaxis().SetTitle("m_{j} (SD mass) [GeV]"); frame2.GetXaxis().SetLabelSize(0.058); frame2.GetXaxis().SetTitleSize(0.068); frame2.GetXaxis().SetTitleOffset(0.95)
    frame2.GetYaxis().SetTitle("Normalized to unity"); frame2.GetYaxis().SetLabelSize(0.058); frame2.GetYaxis().SetTitleSize(0.068); frame2.GetYaxis().SetTitleOffset(0.80)
    frame2.GetXaxis().SetNdivisions(510); frame2.GetYaxis().SetNdivisions(510); frame2.Draw("AXIS")
    for h in histos_mc: h.Draw("HIST E SAME")
    leg2 = ROOT.TLegend(0.15, 0.20, 0.70, 0.52); leg2.SetHeader(f"                      #bf{{Total MC}}  {scan_var_short} cuts")
    leg2.SetNColumns(2); leg2.SetColumnSeparation(0.1); leg2.SetBorderSize(0); leg2.SetFillStyle(0); leg2.SetTextSize(0.065)
    for h in histos_mc:
        lbl = f"eff={h.eff_label:.0%} (>{h.cut_val:.2f})" if h.eff_label else f"> {h.cut_val}  (N={h.n_events})"
        leg2.AddEntry(h, lbl, "l")
    leg2.Draw()

    output_file = f"{output_prefix}_sculpting_{scan_var.replace('ak15_ParTMDV2_', '')}_{channel}.png"
    c.SaveAs(output_file); print(f" --> Mass sculpting plot: {output_file}"); c.Close()
    return output_file


def quantile_matched_vr_edges(data_x, sr_edges):
    """Build VR bin edges (descending from sr_edges[0]) so that VR bin i holds the SAME number
    of data events as SR bin i (±1 for parity). VR bins stack downward from sr_edges[0]:
    the VR bin adjacent to sr_edges[0] matches the SR bin adjacent to sr_edges[-1], etc., so
    the VR_i <-> SR_i index mapping (bottom→top of X) is preserved.

    data_x: 1D numpy array of MLPf_X for data under the base selection.
    sr_edges: SR boundaries, e.g. [0.85, 0.97, 0.99, 0.997, 1.0].
    Returns ascending VR edge list of the same length as sr_edges.
    """
    sr0 = sr_edges[0]
    # Data count per SR bin: (lo, hi]
    _x = np.asarray(data_x, dtype=np.float64)
    _x = _x[np.isfinite(_x)]
    sr_counts = [int(np.sum((_x > sr_edges[i]) & (_x <= sr_edges[i + 1])))
                 for i in range(len(sr_edges) - 1)]
    # Valid data below sr0, sorted descending (highest X = closest to sr0 first)
    below = np.sort(_x[(_x >= 0) & (_x <= sr0)])[::-1]
    n_below = len(below)
    edges_desc = [sr0]
    cum = 0
    # Fill VR bins top-down (adjacent to sr0 first) using SR counts in reverse order,
    # so the topmost VR bin gets the last SR bin's count → VR_i matches SR_i after sorting.
    for c in reversed(sr_counts):
        cum += c
        if cum < n_below:
            edges_desc.append(float(below[cum]))   # exactly `c` events in (edge, prev]
        else:
            edges_desc.append(0.0)                 # ran out of events → include all remaining
    return sorted(edges_desc)


def compute_pred2_plots(events, selection_base, channel, genEventSumw,
                        xsec_zjets, xsec_wjets,
                        zjets_samples, wjets_samples,
                        signal_names, signal_legend_label,
                        zjets_label, wjets_label,
                        n_i=50, x_quantile_edges=None, I=3, J=5, G=4, no_display=False, pred_histos=None, no_pred_overlay=False, cdf_flat_m=False, skip_bern_fit=False, ratio_yrange=None, upper_pad_only=False, no_pseudodata=False, tag="", write_limit_output=True, blind_from_idx=5, no_signal_plot=False, fname_suffix="",
                        plot_var="MLPf_Mscore", plot_bins=(50, 0., 1), plot_xlabel=None, abs_blind_range=None, sig_xrange="auto", bern_fit_lo=0.0, bern_fit_hi=None, bern_orders=(2, 3, 4, 5, 6), val_transform=None, bern_exclude_range=None, zero_edge_bins=None, pred3_yrange=False, common_sig_scale_target=None, no_pred1_overlay=False):
    """
    PRED2: For each valid SR X bin (1-indexed i > I+2), plot MLPf_Mscore
    data+MC stack with the prediction histogram overlaid.
    High-M region is blinded (blind_range=(0.75,1.0), 5 bins) with a bold red BLINDED label.
    """
    import ROOT as _ROOT

    data_key = "data_2018"
    if data_key not in events:
        print("  [PRED2] data_2018 not loaded"); return []
    d_ev = events[data_key]
    for br in ("MLPf_X",):
        if br not in d_ev.fields:
            print(f"  [PRED2] branch {br} missing"); return []
    # For plot_var: only check if it's a plain branch name, not an expression
    if plot_var.isidentifier() and plot_var not in d_ev.fields:
        print(f"  [PRED2] branch {plot_var} missing"); return []

    # Use in-memory histograms if passed directly (PRED12 mode), else read from file
    f_pred = None
    pred_fname = f"pred_{channel}_I{I}_J{J}_G{G}.root"
    if no_pred_overlay:
        print(f"  [PRED3] no-prediction mode - skipping pred file")
    elif pred_histos is not None:
        print(f"  [PRED2] using in-memory prediction histograms from PRED1")
    else:
        f_pred = _ROOT.TFile(pred_fname, "READ")
        if not f_pred or f_pred.IsZombie():
            print(f"  [PRED2] {pred_fname} not found — run PRED1 first"); return []
        print(f"  [PRED2] reading prediction from {pred_fname}")

    # Recompute X edges from data at base Preselection (same as PRED1)
    S_data = selection_base(d_ev)
    X_data = ak.to_numpy(d_ev["MLPf_X"][S_data])
    if x_quantile_edges is not None:
        # Explicit quantile boundaries provided as fractions in [0, 1]
        _qe_pct = [q * 100.0 for q in x_quantile_edges]
        X_edges = np.unique(np.percentile(X_data, _qe_pct))
        print(f"  [PRED2] using {len(X_edges)-1} custom quantile bins from x_quantile_edges")
    else:
        X_edges = np.unique(np.percentile(X_data, np.linspace(0, 100, n_i + 1)))
    n_i_eff = len(X_edges) - 1
    print(f"  [PRED2] {n_i_eff} X bins, I={I}, G={G} (explicit-edge mode plots all bins; "
          f"quantile mode needs CR depth i-1-G-I>=0)")

    # Human-readable bin label for on-canvas titles/console prints only — SR/VR distinguished
    # by `tag` (e.g. "SR3"->"SR", "VR32"->"VR"). Internal dict keys (xbin_key, _xk) stay
    # "XbinNN": they're read by montage_pred2.sh and the PRED1->PRED2 file handoff.
    _xbin_prefix = "SR" if tag.startswith("SR") else ("VR" if tag.startswith("VR") else "Xbin")

    saved = []
    chi2_table = []   # [(quantile_label, chi2_ndf, ndf, data/pred2), ...]
    _summary   = []   # collected per-panel for summary ROOT/txt output
    _panel_sigs = []  # sig_0bin per panel for quadratic combination
    _panel_sigs_full = []  # (xbin_key, full-range S/sqrt(D)) per panel
    _pred3_table = []  # per-bin summary rows for PRED3 end-of-run table
    _pred3_aic_all = {}  # Xbin key -> [best, 2nd, 3rd] polynomial orders by AIC
    _chi2ndf_by_xbin = {}  # Xbin key -> AIC-winner chi2/ndf
    _aic_ord_by_xbin = {}  # Xbin key -> AIC-winner poly order
    _ndf_by_xbin     = {}  # Xbin key -> AIC-winner ndf
    _limit_data_obs  = {}  # Xbin key -> TH1F (blinded data)
    _limit_signal    = {}  # Xbin key -> TH1F (weighted signal, truncated)
    _limit_bkg_bern  = {}  # Xbin key -> {order: TH1F}

    # Shared S/sigma_stat display scale across SR1..SRn (common_sig_scale_target set): probe the
    # reference (last, highest-X) SR bin's raw max(S/sigma_stat) — independent of the main loop
    # below (which can't be reordered: plot_idx there drives blinding) — then derive one factor
    # so that bin's curve peaks at the target, applied identically to every SR panel.
    _common_sig_scale = None
    if common_sig_scale_target is not None and tag.startswith("SR") and n_i_eff >= 1:
        _ref_i1 = n_i_eff
        _ref_lo, _ref_hi = float(X_edges[_ref_i1 - 1]), float(X_edges[_ref_i1])
        _ref_sel = lambda e, lo=_ref_lo, hi=_ref_hi: (
            selection_base(e) & (e["MLPf_X"] > lo) & (e["MLPf_X"] <= hi)
        )

        def _probe_vals(ev_dict, sample, sel_fn):
            _ev = ev_dict[sample]
            _mask = sel_fn(_ev)
            if plot_var in _ev.fields:
                _v = ak.to_numpy(_ev[plot_var][_mask])
            else:
                _loc = {f: _ev[f][_mask] for f in _ev.fields}
                _loc.update({'np': np, 'ak': ak, 'abs': np.abs, 'sqrt': np.sqrt})
                _v = ak.to_numpy(eval(plot_var, {"__builtins__": {}}, _loc))
            if val_transform is not None:
                _v = val_transform(np.asarray(_v, dtype=np.float64))
            return _v, _mask

        _pb_nb, _pb_lo, _pb_hi = plot_bins
        _max_raw = 0.0
        if data_key in events:
            _dv, _ = _probe_vals(events, data_key, _ref_sel)
            _h_probe_data = _ROOT.TH1F("h_probe_data", "", _pb_nb, _pb_lo, _pb_hi)
            _h_probe_data.SetDirectory(0)
            for _x in _dv:
                _h_probe_data.Fill(float(_x))
            UnderOverFlow1D(_h_probe_data)
            _h_probe_data.Sumw2(_ROOT.kFALSE)
            _h_probe_data.SetBinErrorOption(_ROOT.TH1.kPoisson)

            _h_probe_sig = _ROOT.TH1F("h_probe_sig", "", _pb_nb, _pb_lo, _pb_hi)
            _h_probe_sig.SetDirectory(0)
            for _sname in (signal_names or []):
                if _sname not in events or xsec_x_BR.get(_sname) is None:
                    continue
                _sv, _smask = _probe_vals(events, _sname, _ref_sel)
                _ev = events[_sname]
                _sumw = genEventSumw.get(_sname, 1.0) if (genEventSumw and genEventSumw.get(_sname, 0) > 0) else float(ak.sum(_ev["genWeight"]))
                _lumi   = ak.to_numpy(_ev["lumiwgt"][_smask])
                _pu     = ak.to_numpy(_ev["puWeight"][_smask])
                _l1pf   = ak.to_numpy(safe_get(_ev, "l1PreFiringWeight", 1.0)[_smask])
                _muE    = ak.to_numpy(safe_get(_ev, "muEffWeight", 1.0)[_smask])
                _elE    = ak.to_numpy(safe_get(_ev, "elEffWeight", 1.0)[_smask])
                _pujid  = ak.to_numpy(safe_get(_ev, "pileupJetIdWeight", 1.0)[_smask])
                _hem    = ak.to_numpy(get_channel_mc_weight(_ev, _smask, channel))
                _gw     = ak.to_numpy(_ev["genWeight"][_smask])
                _wt = _gw * xsec_x_BR[_sname] * _lumi * (_l1pf * _pu * _muE * _elE * _pujid * _hem) / _sumw
                for _x, _w in zip(_sv, _wt):
                    _h_probe_sig.Fill(float(_x), float(_w))
            UnderOverFlow1D(_h_probe_sig)

            for _ib in range(1, _pb_nb + 1):
                _sigma = _h_probe_data.GetBinError(_ib)
                if _sigma > 0:
                    _dv2 = _h_probe_sig.GetBinContent(_ib) / _sigma
                    if _dv2 > _max_raw:
                        _max_raw = _dv2
        if _max_raw > 0:
            _common_sig_scale = common_sig_scale_target / _max_raw
            print(f"  [PRED3] Common S/sigma_stat scale (anchored on {_xbin_prefix}{_ref_i1}, target={common_sig_scale_target}): x{_common_sig_scale:.3g}")

    plot_idx = 0
    for i_1 in range(1, n_i_eff + 1):      # 1-indexed
        _pred3_row = None  # reset each iteration
        if x_quantile_edges is not None:
            # All bins in the explicit list are plotted — no CR-depth check needed
            q_lo = x_quantile_edges[i_1 - 1] * 100.0
            q_hi = x_quantile_edges[i_1]      * 100.0
        else:
            if i_1 - 1 - G - I < 0:            # require full I-bin CR with G-bin gap
                continue
            if i_1 <= n_i_eff - 12:            # keep top 12 quantile bins
                continue
            q_lo = (i_1 - 1) * 100.0 / n_i_eff
            q_hi =  i_1      * 100.0 / n_i_eff

        x_lo = float(X_edges[i_1 - 1])
        x_hi = float(X_edges[i_1])

        sel_xbin = lambda e, lo=x_lo, hi=x_hi: (
            selection_base(e)
            & (e["MLPf_X"] > lo)
            & (e["MLPf_X"] <= hi)
        )

        # --- Bernstein fits on data (PRED3 only) ---
        # Toggle: set True to fit/plot M-score², False for plain M-score
        _pred3_mscore_pow = 1  # power of M-score transform: 1=off, 2=M², 3=M³, etc.

        # First 5 X-bins (low X, < top-14% quantile): fit [0.0, 1.0]
        # Remaining 7 X-bins (high X): fit up to blind boundary
        _bern_fit_lo = bern_fit_lo
        _blind_boundary = abs_blind_range[0] if abs_blind_range is not None else 0.94
        _bern_fit_hi = 1.0 if plot_idx < blind_from_idx else _blind_boundary
        if bern_fit_hi is not None:
            _bern_fit_hi = bern_fit_hi
        bernstein_fits = None
        _bern_winner_idx = None
        _pseudo_seed = None
        if no_pred_overlay and not skip_bern_fit:
            _S_xb = sel_xbin(d_ev)
            if plot_var in d_ev.fields:
                _M_bfit = ak.to_numpy(d_ev[plot_var][_S_xb])
            else:
                _loc = {f: d_ev[f][_S_xb] for f in d_ev.fields}
                _loc.update({'np': np, 'ak': ak, 'abs': np.abs, 'sqrt': np.sqrt})
                _M_bfit = ak.to_numpy(eval(plot_var, {"__builtins__": {}}, _loc))
            if val_transform is not None:
                _M_bfit = val_transform(np.asarray(_M_bfit, dtype=float))
            elif _pred3_mscore_pow > 1:
                _M_bfit = _M_bfit ** _pred3_mscore_pow
            _h_bfit = _ROOT.TH1F(f"h_bfit_{i_1}", "", plot_bins[0], plot_bins[1], plot_bins[2])
            _h_bfit.SetDirectory(0)
            _h_bfit.Sumw2()
            for _v in _M_bfit:
                _h_bfit.Fill(float(_v))
            _n_data_bin = int(_h_bfit.GetEntries())
            bernstein_fits = []
            _bern_excl_lo = bern_exclude_range[0] if bern_exclude_range is not None else None
            _bern_excl_hi = bern_exclude_range[1] if bern_exclude_range is not None else None
            for _ord in bern_orders:
                _hb, _chi2, _ndf = fit_bernstein_to_hist(_h_bfit, _ord, _bern_fit_lo, _bern_fit_hi,
                                                          fit_exclude_lo=_bern_excl_lo, fit_exclude_hi=_bern_excl_hi)
                bernstein_fits.append((_hb, _chi2, _ndf, _ord))
            # --- Order selection: backward elimination from global chi2/ndf minimum ---
            # 1. Find k* = argmin(chi2/ndf) — genuine optimum, not trivially n_max.
            # 2. Walk backward from k*: test F(k*-1 -> k*). If p<0.05 and chi2/ndf does not
            #    worsen, k* is justified -> select it. Otherwise step down and repeat.
            # AIC = chi2 + 2*(order+1) is printed alongside for comparison.
            _bern_winner_idx  = 0
            _bern_aic_idx     = 0
            _ftest_winner_idx = 0
            try:
                from scipy.stats import f as _f_dist
                _v = [(i, _hb, _c, _n, _o)
                      for i, (_hb, _c, _n, _o) in enumerate(bernstein_fits)
                      if _hb is not None and _c is not None and _n and _n > 0]
                if _v:
                    _c2_scan = "  ".join(f"n{_o_i}={_c_i/_n_i:.2f}" for _, _, _c_i, _n_i, _o_i in _v)
                    # print(f"  [PRED3] Xbin{i_1:02d}  {_n_data_bin} evts | {_c2_scan}")  # in summary table
                    _aic_best     = min(_v, key=lambda x: x[2] + 2.0*(x[4]+1))
                    _bern_aic_idx = _aic_best[0]
                    # print(f"  [PRED3] Xbin{i_1:02d} AIC selects: n={_aic_best[4]}")  # in summary table
                    _aic_ranked = sorted(_v, key=lambda x: x[2] + 2.0*(x[4]+1))
                    _pred3_aic_all[f"{_xbin_prefix}{i_1}"] = [x[4] for x in _aic_ranked[:3]]
                    _gmin = min(_v, key=lambda x: x[2]/x[3])
                    _cur  = _gmin[0]
                    # print(f"  [PRED3] Xbin{i_1:02d} Global min chi2/ndf: n={_gmin[4]} ({_gmin[2]/_gmin[3]:.3f}) -> backward elimination")
                    while _cur > 0:
                        _ce = next((x for x in _v if x[0] == _cur),     None)
                        _pe = next((x for x in _v if x[0] == _cur - 1), None)
                        if _ce is None or _pe is None:
                            _cur -= 1; continue
                        _F  = (_pe[2] - _ce[2]) / (_ce[2] / _ce[3])
                        _p  = float(_f_dist.sf(_F, 1, _ce[3]))
                        _ok = (_p < 0.05) and (_ce[2]/_ce[3] <= _pe[2]/_pe[3])
                        # print(f"  [PRED3] Xbin{i_1:02d} BackElim n={_pe[4]}->n={_ce[4]}: "
                        #       f"F={_F:.2f} p={_p:.3f} {'JUSTIFIED' if _ok else 'NOT JUSTIFIED -> step down'}")
                        if _ok:
                            break
                        _cur -= 1
                    _ftest_winner_idx = _cur
                    _bern_winner_idx  = _ftest_winner_idx  # F-test (p<0.05) drives bold + ratio panel
                    _back_elim_ord = bernstein_fits[_ftest_winner_idx][3] if _ftest_winner_idx < len(bernstein_fits) else '?'
                    print(f"  [PRED3] {_xbin_prefix}{i_1} F-test winner: n={_back_elim_ord}  |  AIC winner: n={bernstein_fits[_bern_aic_idx][3]}")
            except ImportError:
                print("  [PRED3] scipy not available -- falling back to n=2")
            # Collect stats for the end-of-run summary table.
            # n = F-test winner order;  m = AIC (chi2+2k) winner order.
            # Both chi2/ndf values are stored; ndf is n's (F-test) for the table column.
            # Data/Fit uses the AIC (m) histogram since m drives the bold curve.
            _w_idx = _ftest_winner_idx
            _w_hb, _w_chi2, _w_ndf, _w_ord = bernstein_fits[_w_idx]
            _w_c2ndf = (_w_chi2 / _w_ndf) if (_w_ndf and _w_ndf > 0 and _w_chi2 is not None) else float('nan')
            _aic_hb, _aic_chi2, _aic_ndf_v, _aic_ord = (bernstein_fits[_bern_aic_idx] if _bern_aic_idx is not None
                                                         else (None, None, None, _w_ord))
            _aic_c2ndf = (_aic_chi2 / _aic_ndf_v) if (_aic_ndf_v and _aic_ndf_v > 0 and _aic_chi2 is not None) else float('nan')
            _chi2ndf_by_xbin[f"{i_1:02d}"] = _aic_c2ndf
            _aic_ord_by_xbin[f"{i_1:02d}"] = _aic_ord
            _ndf_by_xbin[f"{i_1:02d}"]     = int(_aic_ndf_v) if (_aic_ndf_v is not None and _aic_ndf_v > 0) else 0
            _lo_bin = _h_bfit.FindBin(_bern_fit_lo + 1e-9)
            _hi_bin = _h_bfit.FindBin(_bern_fit_hi - 1e-9)
            _data_int3 = _h_bfit.Integral(_lo_bin, _hi_bin)
            _pred_int3 = (_aic_hb.Integral(_lo_bin, _hi_bin) if _aic_hb is not None else 0.0)
            _dp3 = _data_int3 / _pred_int3 if _pred_int3 > 0 else float('nan')
            _pred3_row = {'q_lo': q_lo, 'q_hi': q_hi, 'ord': _w_ord, 'aic_ord': _aic_ord,
                          'c2ndf': _w_c2ndf, 'aic_c2ndf': _aic_c2ndf,
                          'ndf': _w_ndf or 0, 'dp': _dp3, 'sig': float('nan'),
                          'n_data': _n_data_bin}

            # Deterministic pseudodata seed: sha256 of Xbin boundaries (stable across processes)
            _pseudo_seed = None
            if plot_idx >= blind_from_idx and not no_pseudodata:
                _seed_str  = f"{round(x_lo, 6):.6f}_{round(x_hi, 6):.6f}"
                _pseudo_seed = int(hashlib.sha256(_seed_str.encode()).hexdigest(), 16) % (2**31)

            # --- Collect histograms for LIMIT_INPUT root file ---
            if write_limit_output:
                _xk = f"Xbin{i_1:02d}"

                # data_obs: clone _h_bfit, zero bins above blind boundary,
                # then fill [0.9, 1.0] with pseudodata for blinded Xbins (6-12)
                _h_data_obs = _h_bfit.Clone(f"data_obs_{_xk}")
                _h_data_obs.SetDirectory(0)
                _blind_bin_lo = _h_data_obs.FindBin(_bern_fit_hi + 1e-9)
                for _b in range(_blind_bin_lo, _h_data_obs.GetNbinsX() + 2):
                    _h_data_obs.SetBinContent(_b, 0)
                    _h_data_obs.SetBinError(_b, 0)
                if _pseudo_seed is not None:
                    _h_ps, _, _, _ = _make_pseudodata_hist(_h_bfit, _pseudo_seed,
                                                            fit_hi=_bern_fit_hi, sr_lo=_bern_fit_hi)
                    if _h_ps is not None:
                        for _b in range(_blind_bin_lo, _h_data_obs.GetNbinsX() + 1):
                            _h_data_obs.SetBinContent(_b, _h_ps.GetBinContent(_b))
                            _h_data_obs.SetBinError(_b,   _h_ps.GetBinError(_b))
                _limit_data_obs[_xk] = _h_data_obs

                # signal: sum over all signal samples with full event weights in plot_var / plot_bins
                _h_sig = _ROOT.TH1F(f"signal_{_xk}", "", plot_bins[0], float(plot_bins[1]), float(plot_bins[2]))
                _h_sig.SetDirectory(0); _h_sig.Sumw2()
                for _sname in (signal_names or []):
                    if _sname not in events: continue
                    _ev  = events[_sname]
                    _sumw = genEventSumw.get(_sname, 1.0) if genEventSumw else 1.0
                    if _sumw <= 0: continue
                    _xsec = xsec_x_BR.get(_sname, None)
                    if _xsec is None: continue
                    _S_sig = sel_xbin(_ev)
                    try:
                        if plot_var in _ev.fields:
                            _M_sig = ak.to_numpy(_ev[plot_var][_S_sig])
                        else:
                            _sloc = {f: _ev[f][_S_sig] for f in _ev.fields}
                            _sloc.update({'np': np, 'ak': ak, 'abs': np.abs, 'sqrt': np.sqrt})
                            _M_sig = ak.to_numpy(eval(plot_var, {"__builtins__": {}}, _sloc))
                    except Exception:
                        continue
                    _gw_sig = ak.to_numpy(_ev["genWeight"][_S_sig])
                    _lw     = ak.to_numpy(_ev["lumiwgt"][_S_sig])
                    _pu_w   = ak.to_numpy(_ev["puWeight"][_S_sig])
                    _l1pf_w = ak.to_numpy(safe_get(_ev, "l1PreFiringWeight", 1.0)[_S_sig])
                    _muE_w  = ak.to_numpy(safe_get(_ev, "muEffWeight",        1.0)[_S_sig])
                    _elE_w  = ak.to_numpy(safe_get(_ev, "elEffWeight",        1.0)[_S_sig])
                    _pujid_w= ak.to_numpy(safe_get(_ev, "pileupJetIdWeight",  1.0)[_S_sig])
                    _hem_w  = ak.to_numpy(get_channel_mc_weight(_ev, _S_sig, channel))
                    _wt_sig = _gw_sig * _xsec * _lw * _l1pf_w * _pu_w * _muE_w * _elE_w * _pujid_w * _hem_w / _sumw
                    for _mv, _wv in zip(_M_sig, _wt_sig):
                        _h_sig.Fill(float(_mv), float(_wv))
                _limit_signal[_xk] = _h_sig

                # Bernstein fits: clone each order's fitted histogram
                _limit_bkg_bern[_xk] = {}
                for _hb_i, _chi2_i, _ndf_i, _ord_i in bernstein_fits:
                    if _hb_i is not None:
                        _hb_cl = _hb_i.Clone(f"bkg_bern_n{_ord_i}_{_xk}")
                        _hb_cl.SetDirectory(0)
                        _limit_bkg_bern[_xk][_ord_i] = _hb_cl

        # --- Build prediction histogram with propagated uncertainties ---
        xbin_key = f"Xbin{i_1:02d}"
        overlay_hist = None
        overlay_hist2 = None
        h_tf = None
        h_pred_r = None; h_pred1_r = None; h_mc_sr_r = None
        h_mc_cr_r = None; h_data_cr_r = None; h_data_sr_r = None
        _src_ok = False
        if no_pred_overlay:
            pass  # leave all histos None — no prediction drawn
        elif pred_histos is not None:
            _hh = pred_histos.get(xbin_key, {})
            h_pred_r    = _hh.get('h_pred')
            h_pred1_r   = _hh.get('h_pred1')
            h_mc_sr_r   = _hh.get('h_mc_sr')
            h_mc_cr_r   = _hh.get('h_mc_cr')
            h_data_cr_r = _hh.get('h_data_cr')
            h_data_sr_r = _hh.get('h_data_sr')
            _src_ok = h_pred_r is not None
        else:
            d_xbin = f_pred.Get(xbin_key)
            _src_ok = bool(d_xbin)
            if _src_ok:
                h_pred_r    = d_xbin.Get("h_pred")
                h_pred1_r   = d_xbin.Get("h_pred1")
                h_mc_sr_r   = d_xbin.Get("h_mc_sr")
                h_mc_cr_r   = d_xbin.Get("h_mc_cr")
                h_data_cr_r = d_xbin.Get("h_data_cr")
                h_data_sr_r = d_xbin.Get("h_data_sr")
        if _src_ok:
            if h_pred_r and h_mc_sr_r and h_mc_cr_r and h_data_cr_r:
                h_ov = h_pred_r.Clone(f"h_pred_ov_{i_1}")
                h_ov.SetDirectory(0)
                n_j = h_ov.GetNbinsX()

                for j in range(1, n_j + 1):
                    pred  = h_pred_r.GetBinContent(j)
                    d_cr  = h_data_cr_r.GetBinContent(j)
                    mc_sr = h_mc_sr_r.GetBinContent(j)
                    mc_cr = h_mc_cr_r.GetBinContent(j)
                    e_sr  = h_mc_sr_r.GetBinError(j)
                    e_cr  = h_mc_cr_r.GetBinError(j)

                    j0 = j - 1  # 0-indexed
                    if j0 < G + J:
                        # Low bin: Pred2 undefined — leave empty
                        h_ov.SetBinContent(j, 0); h_ov.SetBinError(j, 0); continue

                    if mc_cr <= 0 or mc_sr <= 0 or d_cr <= 0:
                        h_ov.SetBinError(j, 0); continue

                    # Valid bin: alpha window = J bins at [j-G-J, j-G-1] (1-indexed)
                    var_t2_dsr = 0.0; var_t2_pred1w = 0.0
                    if h_data_sr_r is not None and h_pred1_r is not None:
                        w_lo_1 = j0 - G - J + 1  # 1-indexed lower bound
                        w_hi_1 = j0 - G           # 1-indexed upper bound, inclusive
                        d_sr_w  = sum(h_data_sr_r.GetBinContent(k) for k in range(w_lo_1, w_hi_1 + 1))
                        pred1_w = sum(h_pred1_r.GetBinContent(k)   for k in range(w_lo_1, w_hi_1 + 1))
                        sig2_pred1_w = 0.0
                        for k in range(w_lo_1, w_hi_1 + 1):
                            p1k = h_pred1_r.GetBinContent(k)
                            dcrk = h_data_cr_r.GetBinContent(k)
                            mcsrk = h_mc_sr_r.GetBinContent(k); esrk = h_mc_sr_r.GetBinError(k)
                            mccrk = h_mc_cr_r.GetBinContent(k); ecrk = h_mc_cr_r.GetBinError(k)
                            if dcrk > 0 and mcsrk > 0 and mccrk > 0 and p1k > 0:
                                sig2_pred1_w += p1k**2 * (1.0/dcrk + esrk**2/mcsrk**2 + ecrk**2/mccrk**2)
                        if d_sr_w  > 0: var_t2_dsr    = pred**2 / d_sr_w
                        if pred1_w > 0: var_t2_pred1w = pred**2 * sig2_pred1_w / pred1_w**2
                    var = (pred**2 * (1.0/d_cr + e_sr**2/mc_sr**2 + e_cr**2/mc_cr**2)
                           + var_t2_dsr + var_t2_pred1w)
                    h_ov.SetBinError(j, np.sqrt(var))
                overlay_hist = {
                    'hist':  h_ov,
                    'label': "Pred2",
                    'color': _ROOT.kBlue,
                }
                # Build Term1-only (no alpha) prediction with propagated uncertainties
                if h_pred1_r is not None and not no_pred1_overlay:
                    h_ov2 = h_pred1_r.Clone(f"h_pred1_ov_{i_1}")
                    h_ov2.SetDirectory(0)
                    for j in range(1, h_ov2.GetNbinsX() + 1):
                        p1   = h_pred1_r.GetBinContent(j)
                        d_cr = h_data_cr_r.GetBinContent(j)
                        mc_sr = h_mc_sr_r.GetBinContent(j)
                        mc_cr = h_mc_cr_r.GetBinContent(j)
                        e_sr  = h_mc_sr_r.GetBinError(j)
                        e_cr  = h_mc_cr_r.GetBinError(j)
                        if mc_cr <= 0 or mc_sr <= 0 or d_cr <= 0:
                            h_ov2.SetBinError(j, 0); continue
                        var1 = p1**2 * (1.0/d_cr + e_sr**2/mc_sr**2 + e_cr**2/mc_cr**2)
                        h_ov2.SetBinError(j, np.sqrt(var1))
                    overlay_hist2 = {
                        'hist':  h_ov2,
                        'label': "Pred1",
                    }
                # Build per-bin transfer factor MC_SR(j)/MC_CR(j)
                h_tf = h_pred_r.Clone(f"h_tf_{i_1}")
                h_tf.SetDirectory(0)
                for j in range(1, h_tf.GetNbinsX() + 1):
                    mc_sr = h_mc_sr_r.GetBinContent(j)
                    mc_cr = h_mc_cr_r.GetBinContent(j)
                    e_sr_j = h_mc_sr_r.GetBinError(j)
                    e_cr_j = h_mc_cr_r.GetBinError(j)
                    h_tf.SetBinContent(j, mc_sr / mc_cr if mc_cr > 0 else 0)
                    if mc_cr > 0 and mc_sr > 0:
                        rel_err = (e_sr_j / mc_sr)**2 + (e_cr_j / mc_cr)**2
                        h_tf.SetBinError(j, (mc_sr / mc_cr) * rel_err**0.5)
                    else:
                        h_tf.SetBinError(j, 0)
        else:
            src = "in-memory dict" if pred_histos is not None else pred_fname
            if not no_pred_overlay:
                print(f"  [PRED2] WARNING: {xbin_key} not found in {src}")

        # Compute χ²/ndf and Data/Pred2 for the summary table (skip blinded M-bins)
        _is_blinded = (plot_idx >= blind_from_idx)
        _bl_lo_t, _bl_hi_t = abs_blind_range if abs_blind_range is not None else (0.86, 1.0)
        _chi2 = 0.0; _ndf = 0; _d_tot = 0.0; _p_tot = 0.0
        if _src_ok and h_pred_r is not None and h_data_sr_r is not None:
            for _ib in range(1, h_pred_r.GetNbinsX() + 1):
                if (_ib - 1) < G + J:          # skip empty low bins
                    continue
                _bc = h_pred_r.GetBinCenter(_ib)
                if _is_blinded and _bl_lo_t <= _bc <= _bl_hi_t:
                    continue
                _pred = h_pred_r.GetBinContent(_ib)
                _obs  = h_data_sr_r.GetBinContent(_ib)
                _ep   = h_pred_r.GetBinError(_ib)
                _d_tot += _obs; _p_tot += _pred
                if _pred <= 0: continue
                denom = _obs + _ep**2
                if denom > 0:
                    _chi2 += (_obs - _pred)**2 / denom
                _ndf += 1
        _chi2_ndf = _chi2 / _ndf if _ndf > 0 else float('nan')
        _d_over_p = _d_tot / _p_tot if _p_tot > 0 else float('nan')
        chi2_table.append((f"{q_lo:g}-{q_hi:g}%", _chi2_ndf, _ndf, _d_over_p))

        # Collect per-panel histograms for summary output
        _h_mc_s = h_mc_sr_r.Clone(f"h_mc_{xbin_key}") if h_mc_sr_r else None
        _h_p2_s = overlay_hist['hist'].Clone(f"h_p2_{xbin_key}") if overlay_hist else None
        for _hh in [_h_mc_s, _h_p2_s]:
            if _hh: _hh.SetDirectory(0)
        _summary.append({'key': xbin_key, 'q_lo': q_lo, 'q_hi': q_hi,
                         'chi2_ndf': _chi2_ndf, 'ndf': _ndf,
                         'h_mc': _h_mc_s, 'h_pred2': _h_p2_s})

        _msq_tag = f"_Mpow{_pred3_mscore_pow}" if (no_pred_overlay and _pred3_mscore_pow > 1) else ""
        _tag_pfx = f"{tag}_" if tag else ""
        fname  = f"pred2_{_tag_pfx}{channel}_Xbin{i_1:02d}_I{I}_J{J}_G{G}{_msq_tag}"
        # Round to 1 decimal so data-derived VR edges display cleanly (like the SR round values)
        region = f"{_xbin_prefix}{i_1}  [{round(q_lo,1):g}-{round(q_hi,1):g}%]"
        print(f"  [PRED2] {_xbin_prefix}{i_1}  X in [{x_lo:.4f}, {x_hi:.4f}]  ({round(q_lo,1):g}-{round(q_hi,1):g}%)")

        # Last 3 quantile bins are blinded; earlier bins are control region checks
        # BLINDING — comment out to unblind
        # In PRED3 mode data are shown up to M=0.9; only [0.9,1.0] stays blinded
        _blind_lo      = 0.94 if no_pred_overlay else 0.86
        _blind_hi_val  = abs_blind_range if abs_blind_range is not None else (_blind_lo, 1.0)
        _blind_range   = _blind_hi_val if plot_idx >= blind_from_idx else None
        _blinded_label = "BLINDED"     if plot_idx >= blind_from_idx else None

        # Compute RelPur_{CR/SR}(j=last):
        #   (S/B) in j-CR (J bins below last j, gap G) vs (S/B) at j=last,
        #   both within the SR X-range.  Tells whether alpha is biased by signal.
        _sig_extra_label = ""
        _n_j_plot = int(plot_bins[0])   # M-bins in the PRED2 plot
        _mw_plot  = (plot_bins[2] - plot_bins[1]) / _n_j_plot  # M-axis bin width on the plot
        # j-SR: last M-bin (1-indexed = _n_j_plot)
        _j_sr_lo_edge = plot_bins[1] + (_n_j_plot - 1) * _mw_plot
        # j-CR: J bins ending G bins before last (1-indexed: n_j-G-J .. n_j-G-1)
        _j_cr_hi_1 = _n_j_plot - G - 1                        # last bin of j-CR (1-indexed)
        _j_cr_lo_1 = _n_j_plot - G - J                        # first bin of j-CR (1-indexed)
        _j_cr_lo_edge = plot_bins[1] + (_j_cr_lo_1 - 1) * _mw_plot
        _j_cr_hi_edge = plot_bins[1] + _j_cr_hi_1 * _mw_plot
        if _j_cr_lo_1 >= 1 and signal_names and _src_ok:
            # Signal yields: SR X-range, j-SR and j-CR M windows
            _sig_sr_j = 0.0; _sig_cr_j = 0.0
            for _sname in signal_names:
                if _sname not in events: continue
                _ev = events[_sname]
                _sumw = genEventSumw.get(_sname, 1.0) if genEventSumw else 1.0
                if _sumw <= 0: continue
                _base = selection_base(_ev)
                _X = ak.to_numpy(_ev["MLPf_X"][_base])
                _M = ak.to_numpy(_ev["MLPf_Mscore"][_base])
                _gw = ak.to_numpy(_ev["genWeight"][_base])
                _in_xsr = (_X > x_lo) & (_X <= x_hi)
                _sig_sr_j += float(np.sum(_gw[_in_xsr & (_M >= _j_sr_lo_edge)]) / _sumw)
                _sig_cr_j += float(np.sum(_gw[_in_xsr & (_M >= _j_cr_lo_edge) & (_M < _j_cr_hi_edge)]) / _sumw)
            # BKG yields from SR-X MC histogram (h_mc_sr_r covers the SR X-range)
            _bkg_sr_j = h_mc_sr_r.GetBinContent(_n_j_plot) if h_mc_sr_r else 0.0
            _bkg_cr_j = sum(h_mc_sr_r.GetBinContent(b) for b in range(_j_cr_lo_1, _j_cr_hi_1 + 1)) if h_mc_sr_r else 0.0
            _sb_sr = _sig_sr_j / _bkg_sr_j if _bkg_sr_j > 0 else float('nan')
            _sb_cr = _sig_cr_j / _bkg_cr_j if _bkg_cr_j > 0 else float('nan')
            _rel_pur = _sb_cr / _sb_sr if (_sb_sr > 0 and _sb_cr == _sb_cr) else float('nan')
            if _rel_pur == _rel_pur:   # not nan
                _sig_extra_label = f"RelPur_{{CR/SR}}(j={_n_j_plot}) = {_rel_pur:.2f}"

        _pow_sup = {2: "²", 3: "³", 4: "⁴"}.get(_pred3_mscore_pow, f"^{_pred3_mscore_pow}")
        _default_xlabel = f"M-score{_pow_sup} MLP finetuned" if (no_pred_overlay and _pred3_mscore_pow > 1) else "M-score MLP finetuned"
        _mscore_xlabel  = plot_xlabel if plot_xlabel is not None else _default_xlabel
        _mscore_transform = val_transform if val_transform is not None else (
            (lambda x: x**_pred3_mscore_pow) if (no_pred_overlay and _pred3_mscore_pow > 1 and plot_var == "MLPf_Mscore") else None
        )
        _pinfo = plot_variable(
            events=events, var_name=plot_var, selection=sel_xbin,
            bins=plot_bins, xlabel=_mscore_xlabel,
            cdf_flat=cdf_flat_m,
            output_prefix=fname, logy=False,
            region_label=region,
            blind_range=_blind_range,
            blinded_label=_blinded_label,
            genEventSumw=genEventSumw,
            signal_names=([] if no_signal_plot else signal_names),
            zjets_samples=zjets_samples, wjets_samples=wjets_samples,
            xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_label=zjets_label, wjets_label=wjets_label,
            signal_legend_label=signal_legend_label,
            channel=channel, plot_index=plot_idx,
            overlay_hist=overlay_hist,
            cr_template_hist=h_data_cr_r if _src_ok else None,
            tf_hist=h_tf,
            overlay_hist2=overlay_hist2,
            pred_params_label=f"I={I}, J={J}, G={G}" if not no_pred_overlay else None,
            pred_extra_label=_sig_extra_label if _sig_extra_label else None,
            bernstein_fits=bernstein_fits,
            bernstein_fit_range=(_bern_fit_lo, _bern_fit_hi),
            val_transform=_mscore_transform,
            bernstein_winner_idx=_bern_winner_idx,
            pseudodata_seed=_pseudo_seed,
            ratio_yrange=ratio_yrange,
            upper_pad_only=upper_pad_only,
            sig_xrange=(abs_blind_range if sig_xrange == "auto" else sig_xrange),
            zero_edge_bins=zero_edge_bins,
            pred3_yrange=pred3_yrange,
            sig_scale_factor=_common_sig_scale,
        )
        plot_idx += 1
        saved.append(fname + ".png")
        if _pinfo and "sig_0bin" in _pinfo:
            _panel_sigs.append(_pinfo["sig_0bin"])
            if _pred3_row is not None:
                _pred3_row['sig'] = _pinfo["sig_0bin"]
        if _pinfo and _pinfo.get("quad_sig_full", 0.0) > 0:
            _panel_sigs_full.append((xbin_key, _pinfo["quad_sig_full"]))
        if _pred3_row is not None:
            _pred3_table.append(_pred3_row)
        if not no_display:
            os.system(f"display {fname}.png &")

    print(f"  [PRED2] produced {len(saved)} plots")

    # --- Write LIMIT_INPUT_{channel}.root after all X-bins processed ---
    if no_pred_overlay and write_limit_output and _limit_data_obs:
        _limit_fname = f"LIMIT_INPUT_{channel}__12Xbins_50Mbins_2018{fname_suffix}.root"
        _f_limit = _ROOT.TFile(_limit_fname, "RECREATE")
        for _xk in sorted(_limit_data_obs.keys()):
            _d_xk = _f_limit.mkdir(_xk)
            _d_xk.cd()
            _limit_data_obs[_xk].Write()
            if _xk in _limit_signal:
                _limit_signal[_xk].Write()
            if _xk in _limit_bkg_bern:
                for _ord_v in sorted(_limit_bkg_bern[_xk].keys()):
                    _limit_bkg_bern[_xk][_ord_v].Write()
        _f_limit.Close()
        print(f"  [PRED3] Written {_limit_fname}: {len(_limit_data_obs)} X-bins, "
              f"histograms: data_obs + signal + bkg_bern_n{{2..6}} per bin")
        if _pred3_aic_all:
            import json as _json_mod
            _aic_json_fname = f"aic_best3_{channel}{fname_suffix}.json"
            with open(_aic_json_fname, "w") as _fj:
                _json_mod.dump({"channel": channel, "n_best": 3, "bins": _pred3_aic_all}, _fj, indent=2)
            _aic_summary = "  ".join(f"{k}:{v}" for k, v in sorted(_pred3_aic_all.items()))
            print(f"  [PRED3] Written {_aic_json_fname}: top-3 AIC poly orders per bin")
            print(f"  [PRED3] AIC top-3: {_aic_summary}")

    if chi2_table and not no_pred_overlay:
        print(f"\n  {'='*52}")
        print(f"  PRED2 closure summary - channel: {channel}  I={I} J={J} G={G}")
        print(f"  {'='*52}")
        print(f"  {'Quantile':<14}  {'chi2/ndf':>8}  {'ndf':>5}  {'Data/Pred2':>10}")
        print(f"  {'-'*52}")
        for label, c2ndf, ndf, ratio in chi2_table:
            c2str    = f"{c2ndf:.3f}" if not (c2ndf != c2ndf) else "  n/a"
            ratiostr = f"{ratio:.4f}" if not (ratio != ratio) else "     n/a"
            print(f"  {label:<14}  {c2str:>8}  {ndf:>5}  {ratiostr:>10}")
        print(f"  {'='*52}\n")

    _avg_ft_c2 = float('nan'); _avg_aic_c2 = float('nan')
    if no_pred_overlay and _pred3_table:
        _W = 80
        _title = f"  PRED3 summary - channel: {channel}"
        _pad   = (_W - len(_title)) // 2
        print(f"\n  {'-'*_pad}{_title}{'-'*(_W - _pad - len(_title))}")
        print(f"  {'Quantile':<12} {'Data%':>6} {'best Poly. n, m':>16} {'X2/ndf_(n,m)':>13} {'ndf':>4} {'Data/Fit':>9} {'S/sqrt(D)':>8}")
        print(f"  {'-'*_W}")
        for _r in _pred3_table:
            _c2v    = _r['c2ndf']
            _c2_aic = _r.get('aic_c2ndf', float('nan'))
            _c2s    = f"{_c2v:.2f}"    if _c2v    == _c2v    else " n/a"
            _c2a_s  = f"{_c2_aic:.2f}" if _c2_aic == _c2_aic else " n/a"
            _c2_both = f"{_c2s}, {_c2a_s}"
            _ord_s  = f"n={_r['ord']}, m={_r.get('aic_ord', '?')}"
            _dps    = f"{_r['dp']:.2f}"  if _r['dp']  == _r['dp']  else "  n/a"
            _sgs    = f"{_r['sig']:.3f}" if _r['sig'] == _r['sig'] else "   n/a"
            _qlbl   = f"{_r['q_lo']:.1f}-{_r['q_hi']:.1f}%"
            _frac   = f"{_r['q_hi'] - _r['q_lo']:.1f}%"
            _flag   = ""
            if _c2_aic == _c2_aic:
                if _c2_aic > 2.0:
                    _flag = f"  <=== ATTENTION m:{_c2_aic:.2f}>>1"
                elif _c2_aic < 0.5:
                    _flag = f"  <=== ATTENTION m:{_c2_aic:.2f}<<1"
            print(f"  {_qlbl:<12} {_frac:>6} {_ord_s:>16} {_c2_both:>13} {_r['ndf']:>4} {_dps:>9} {_sgs:>8}{_flag}")
        _valid_sigs   = [_r['sig'] for _r in _pred3_table if _r['sig'] == _r['sig']]
        _valid_ft_c2  = [_r['c2ndf'] for _r in _pred3_table if _r['c2ndf'] == _r['c2ndf']]
        _valid_aic_c2 = [_r.get('aic_c2ndf', float('nan')) for _r in _pred3_table
                         if _r.get('aic_c2ndf', float('nan')) == _r.get('aic_c2ndf', float('nan'))]
        _total_s3   = float(np.sqrt(sum(s**2 for s in _valid_sigs))) if _valid_sigs else float('nan')
        _avg_ft_c2  = sum(_valid_ft_c2)  / len(_valid_ft_c2)  if _valid_ft_c2  else float('nan')
        _avg_aic_c2 = sum(_valid_aic_c2) / len(_valid_aic_c2) if _valid_aic_c2 else float('nan')
        _avg_ft_s   = f"{_avg_ft_c2:.2f}"  if _avg_ft_c2  == _avg_ft_c2  else " n/a"
        _avg_aic_s  = f"{_avg_aic_c2:.2f}" if _avg_aic_c2 == _avg_aic_c2 else " n/a"
        print(f"  {'-'*_W}")
        _tot_str = f"{_total_s3:.3f}" if _total_s3 == _total_s3 else "n/a"
        print(f"  {'<chi2/ndf>  F-test | AIC(chi2+2k)':<45} {_avg_ft_s}, {_avg_aic_s}")
        print(f"  {'Quadratic sum S/sqrt(D)':<65} {_tot_str:>8}")
        print(f"  {'-'*_W}\n")

    if _summary and not no_pred_overlay:
        _sfname = f"summary_{channel}_I{I}_J{J}_G{G}"
        with open(f"{_sfname}.txt", 'w') as _f:
            _f.write(f"# PRED12  channel={channel}  I={I} J={J} G={G}\n")
            _f.write(f"{'Panel':<14}  {'chi2/ndf':>10}  {'ndf':>5}\n")
            for s in _summary:
                c2 = s['chi2_ndf']
                _f.write(f"{s['q_lo']:g}-{s['q_hi']:g}%{'':<6}  "
                         f"{'nan' if c2 != c2 else f'{c2:.4f}':>10}  {s['ndf']:>5}\n")
        _froot = ROOT.TFile(f"{_sfname}.root", "RECREATE")
        n_p = len(_summary)
        _hc = ROOT.TH1F("h_chi2ndf", "#chi^{2}/ndf;X quantile bin;#chi^{2}/ndf",
                        n_p, 0.5, n_p + 0.5)
        for ip, s in enumerate(_summary):
            c2 = s['chi2_ndf']
            _hc.SetBinContent(ip + 1, c2 if c2 == c2 else 0)
            _hc.GetXaxis().SetBinLabel(ip + 1, f"{s['q_lo']:g}-{s['q_hi']:g}%")
        _hc.Write()
        for s in _summary:
            _d = _froot.mkdir(s['key']); _d.cd()
            for _hname, _hsrc in [("h_mc_sr", s['h_mc']), ("h_pred2", s['h_pred2'])]:
                if _hsrc:
                    _hh = _hsrc.Clone(_hname); _hh.SetDirectory(_d); _hh.Write()
        _froot.Close()
        print(f"  [PRED2] summary -> {_sfname}.txt  {_sfname}.root")

    total_sig = float(np.sqrt(sum(s**2 for s in _panel_sigs))) if _panel_sigs else 0.0
    if _panel_sigs and not no_pred_overlay:
        print(f"  [PRED2] Total significance (quadratic sum over {len(_panel_sigs)} panels): {total_sig:.3f}")
    if _panel_sigs_full:
        _tot_full = float(np.sqrt(sum(s * s for _, s in _panel_sigs_full)))
        print("  [PRED2] Full-range S/sqrt(D) per X bin:  "
              + "  ".join(f"{k}={s:.3f}" for k, s in _panel_sigs_full))
        print(f"  [PRED2] Full-range total (quadrature, {len(_panel_sigs_full)} bins): {_tot_full:.3f}")

    if _chi2ndf_by_xbin:
        _txt_fname = f"pred3_chi2ndf_{channel}{fname_suffix}.txt"
        with open(_txt_fname, "w") as _tf:
            _tf.write("# Xbin  chi2ndf  poly_order  ndf\n")
            for _k in sorted(_chi2ndf_by_xbin):
                _c2 = _chi2ndf_by_xbin[_k]
                _or = _aic_ord_by_xbin.get(_k, -1)
                _nd = _ndf_by_xbin.get(_k, 0)
                _c2_s = f"{_c2:.4f}" if (_c2 == _c2) else "nan"
                _tf.write(f"{_k}  {_c2_s}  {_or}  {_nd}\n")
        print(f"  [PRED3] Written chi2/ndf + poly order + ndf per Xbin -> {_txt_fname}")

    return saved, total_sig, _avg_ft_c2, _avg_aic_c2


def compute_bkg_prediction(events, selection, channel, genEventSumw,
                           xsec_zjets, xsec_wjets,
                           zjets_samples, wjets_samples,
                           n_i=50, n_j=50, I=3, J=5, G=4,
                           x_quantile_edges=None, m_bins=None, tag=""):
    """
    Two-term background prediction per M-bin j:
      Term1: pred1(j) = D(CR,j) * MC(SR,j) / MC(CR,j)
      Term2: alpha(j) = D(SR,w) / pred1(w),  w = J M-bins [j-G-J, j-G-1]
      Pred2(j) = pred1(j) * alpha(j)  for j >= G+J (0-indexed), else 0
    X = MLPf_X, M = MLPf_Mscore.
    Fine mode (x_quantile_edges=None): n_i data-quantile X bins; SR X bin i uses
      CR = I X bins [i-G-I, i-G-1] (G-bin gap).  M axis: n_j bins in [0,1].
    Coarse mode (x_quantile_edges=[q0..qN], quantile fractions): one prediction per
      target bin; CR spans quantiles [q_lo-(G+I)/n_i, q_lo-G/n_i] — I/G keep their
      fine-bin meaning in units of 1/n_i.  M axis from m_bins=(n, lo, hi).
    Output: pred_{channel}{tag}_I{I}_J{J}_G{G}.root (one dir per X bin) + in-memory dict.
    """
    import ROOT

    # ----- check required branches -----
    data_key = "data_2018"
    if data_key not in events:
        print("  [PRED] data_2018 not loaded — skipping prediction"); return
    d_ev = events[data_key]
    for br in ("MLPf_X", "MLPf_Mscore"):
        if br not in d_ev.fields:
            print(f"  [PRED] branch {br} missing in data — skipping prediction"); return

    all_mc_samples = (list(zjets_samples or []) + list(wjets_samples or [])
                      + ["TTTo2L2Nu", "TTToSemiLeptonic",
                         "ST_t-channel_top", "ST_t-channel_antitop",
                         "ST_tW_top", "ST_tW_antitop", "ST_s-channel",
                         "WWTo1L1Nu2Q", "WWTo2L2Nu", "WZTo1L1Nu2Q", "WZTo1L3Nu",
                         "WZTo3LNu", "WZTo2Q2L", "WZTo2Q2Nu",
                         "ZZTo2Q2L", "ZZTo2L2Nu", "ZZTo2Q2Nu", "ZZTo4L"])
    xsec_all = {**xsec_zjets, **xsec_wjets, **xsec_Top, **xsec_VV}

    # ----- extract data arrays (apply selection) -----
    S_data = selection(d_ev)
    X_data = ak.to_numpy(d_ev["MLPf_X"][S_data])
    M_data = ak.to_numpy(d_ev["MLPf_Mscore"][S_data])

    # M-axis binning: n_j bins in [0,1] unless m_bins=(n, lo, hi) override (coarse mode)
    if m_bins is not None:
        n_j, m_lo, m_hi = int(m_bins[0]), float(m_bins[1]), float(m_bins[2])
    else:
        m_lo, m_hi = 0.0, 1.0
    M_edges = np.linspace(m_lo, m_hi, n_j + 1)

    if x_quantile_edges is None:
        # X bin edges from data quantiles (unique to avoid empty bins)
        X_edges = np.unique(np.percentile(X_data, np.linspace(0, 100, n_i + 1)))
        if len(X_edges) < 3:
            print("  [PRED] too few unique X quantile edges — skipping"); return
        n_i_eff = len(X_edges) - 1
        print(f"  [PRED] X bins: {n_i_eff}  M bins: {n_j}  I: {I}")
    else:
        print(f"  [PRED] coarse mode: {len(x_quantile_edges)-1} target X bins  "
              f"M bins: {n_j} in [{m_lo:g},{m_hi:g}]  I: {I}  G: {G}  (1/{n_i} quantile units)")
    print(f"  [PRED] Data after selection: {len(X_data):,} events")

    # ----- gather MC arrays (weighted) -----
    _mc_X = []; _mc_M = []; _mc_w = []

    for sample in all_mc_samples:
        if sample not in events: continue
        W = events[sample]
        if len(W) == 0: continue
        xsec = xsec_all.get(sample, 0.0)
        if xsec <= 0: continue
        for br in ("MLPf_X", "MLPf_Mscore"):
            if br not in W.fields: continue
        S_mc = selection(W)
        if ak.sum(S_mc) == 0: continue
        sum_gen = genEventSumw.get(sample, 0)
        if sum_gen <= 0:
            sum_gen = float(ak.sum(W["genWeight"]))
        if sum_gen <= 0: continue
        lumi       = ak.to_numpy(W["lumiwgt"][S_mc])
        gw         = ak.to_numpy(W["genWeight"][S_mc])
        pu         = ak.to_numpy(safe_get(W, "puWeight", 1.0)[S_mc])
        l1pf       = ak.to_numpy(safe_get(W, "l1PreFiringWeight", 1.0)[S_mc])
        muEff      = ak.to_numpy(safe_get(W, "muEffWeight", 1.0)[S_mc])
        elEff      = ak.to_numpy(safe_get(W, "elEffWeight", 1.0)[S_mc])
        pujid      = ak.to_numpy(safe_get(W, "pileupJetIdWeight", 1.0)[S_mc])
        hem_w      = get_channel_mc_weight(W, S_mc, channel)
        if hasattr(hem_w, '__len__') and not isinstance(hem_w, np.ndarray):
            hem_w = ak.to_numpy(hem_w)
        if not isinstance(hem_w, np.ndarray):
            hem_w = np.full(int(ak.sum(S_mc)), float(hem_w))
        wt = gw * xsec * lumi * l1pf * pu * muEff * elEff * pujid * hem_w / sum_gen
        _mc_X.append(ak.to_numpy(W["MLPf_X"][S_mc]))
        _mc_M.append(ak.to_numpy(W["MLPf_Mscore"][S_mc]))
        _mc_w.append(np.asarray(wt, dtype=np.float64))

    X_mc_all = np.concatenate(_mc_X) if _mc_X else np.zeros(0)
    M_mc_all = np.concatenate(_mc_M) if _mc_M else np.zeros(0)
    W_mc_all = np.concatenate(_mc_w) if _mc_w else np.zeros(0)
    if W_mc_all.size == 0 or float(np.sum(W_mc_all)) <= 0:
        print("  [PRED] WARNING: no background MC loaded — prediction will be all-zero (run with --loadmc!)")

    def _pred_core(D_cr, MC_cr, MC_sr, D_sr):
        """Term1 x Term2 per M-bin. alpha window: J bins at [j-G-J, j-G-1]; valid j >= G+J."""
        pred1 = np.where(MC_cr > 0, D_cr * MC_sr / MC_cr, 0.0)
        alpha = np.zeros(n_j)
        for j in range(G + J, n_j):
            d_sr_w  = D_sr [j - G - J : j - G].sum()
            pred1_w = pred1[j - G - J : j - G].sum()
            if pred1_w > 0:
                alpha[j] = d_sr_w / pred1_w
        # j < G+J → 0 (insufficient history; bins left empty)
        j_idx = np.arange(n_j)
        pred = np.where(j_idx < G + J, 0.0, np.where(alpha > 0, pred1 * alpha, 0.0))
        return pred1, alpha, pred

    fname = f"pred_{channel}{tag}_I{I}_J{J}_G{G}.root"
    f_out = ROOT.TFile(fname, "RECREATE")
    _inmem = {}
    n_written = 0

    # Keep clones alive detached from file for in-memory handoff to PRED2
    def _detach(h): h.SetDirectory(0); return h

    def _write_bin(key, pred, pred1, D_cr, D_sr, MC_sr, MC_cr, emc_sr, emc_cr):
        d = f_out.mkdir(key)
        d.cd()
        h_pred    = ROOT.TH1D("h_pred",    f"Prediction {key};M score;Events", n_j, m_lo, m_hi)
        h_pred1   = ROOT.TH1D("h_pred1",   f"Pred Term1 {key};M score;Events", n_j, m_lo, m_hi)
        h_data_cr = ROOT.TH1D("h_data_cr", f"Data CR {key};M score;Events",    n_j, m_lo, m_hi)
        h_data_sr = ROOT.TH1D("h_data_sr", f"Data SR {key};M score;Events",    n_j, m_lo, m_hi)
        h_mc_sr   = ROOT.TH1D("h_mc_sr",   f"MC SR {key};M score;Events",      n_j, m_lo, m_hi)
        h_mc_cr   = ROOT.TH1D("h_mc_cr",   f"MC CR {key};M score;Events",      n_j, m_lo, m_hi)
        for j in range(n_j):
            h_pred   .SetBinContent(j + 1, pred   [j])
            h_pred1  .SetBinContent(j + 1, pred1  [j])
            h_data_cr.SetBinContent(j + 1, D_cr   [j])
            h_data_sr.SetBinContent(j + 1, D_sr   [j])
            h_mc_sr  .SetBinContent(j + 1, MC_sr  [j]); h_mc_sr.SetBinError(j + 1, emc_sr[j])
            h_mc_cr  .SetBinContent(j + 1, MC_cr  [j]); h_mc_cr.SetBinError(j + 1, emc_cr[j])
        h_pred.Write(); h_pred1.Write(); h_data_cr.Write(); h_data_sr.Write()
        h_mc_sr.Write(); h_mc_cr.Write()
        _inmem[key] = {
            'h_pred':    _detach(h_pred.Clone()),
            'h_pred1':   _detach(h_pred1.Clone()),
            'h_mc_sr':   _detach(h_mc_sr.Clone()),
            'h_mc_cr':   _detach(h_mc_cr.Clone()),
            'h_data_cr': _detach(h_data_cr.Clone()),
            'h_data_sr': _detach(h_data_sr.Clone()),
        }

    if x_quantile_edges is not None:
        # ── Coarse mode: one prediction per explicit target quantile bin.
        # CR spans quantiles [q_lo-(G+I)/n_i, q_lo-G/n_i]: I fine-bin equivalents wide
        # with a G-bin gap, in units of 1/n_i — I/G keep their fine-mode meaning.
        _q  = [float(v) for v in x_quantile_edges]
        _xq = lambda fq: float(np.percentile(X_data, min(max(fq, 0.0), 1.0) * 100.0))

        def _mhist(x_lo, x_hi, X, M, w=None):
            msk = (X > x_lo) & (X <= x_hi)
            h, _ = np.histogram(M[msk], bins=M_edges, weights=(w[msk] if w is not None else None))
            return h.astype(np.float64)

        for i in range(len(_q) - 1):
            q_lo, q_hi = _q[i], _q[i + 1]
            cr_q_lo, cr_q_hi = q_lo - (G + I) / float(n_i), q_lo - G / float(n_i)
            if cr_q_lo < 0:
                print(f"  [PRED] Xbin{i+1:02d}: CR quantile window below 0 — skipped"); continue
            x_lo, x_hi   = _xq(q_lo),    _xq(q_hi)
            xc_lo, xc_hi = _xq(cr_q_lo), _xq(cr_q_hi)
            D_sr   = _mhist(x_lo,  x_hi,  X_data,   M_data)
            D_cr   = _mhist(xc_lo, xc_hi, X_data,   M_data)
            MC_sr  = _mhist(x_lo,  x_hi,  X_mc_all, M_mc_all, W_mc_all)
            MC_cr  = _mhist(xc_lo, xc_hi, X_mc_all, M_mc_all, W_mc_all)
            emc_sr = np.sqrt(_mhist(x_lo,  x_hi,  X_mc_all, M_mc_all, W_mc_all**2))
            emc_cr = np.sqrt(_mhist(xc_lo, xc_hi, X_mc_all, M_mc_all, W_mc_all**2))
            pred1, alpha, pred = _pred_core(D_cr, MC_cr, MC_sr, D_sr)
            print(f"  [PRED] Xbin{i+1:02d}  X q[{q_lo:g},{q_hi:g}] -> [{x_lo:.4f},{x_hi:.4f}]  "
                  f"CR q[{cr_q_lo:.3f},{cr_q_hi:.3f}]  pred1={pred1.sum():.1f}  pred={pred.sum():.1f}")
            _write_bin(f"Xbin{i+1:02d}", pred, pred1, D_cr, D_sr, MC_sr, MC_cr, emc_sr, emc_cr)
            n_written += 1
    else:
        # ── Fine mode: 2D histograms on the n_i-quantile x n_j grid (original behavior)
        h2_data, _, _ = np.histogram2d(X_data, M_data, bins=[X_edges, M_edges])
        h2_mc  = np.zeros((n_i_eff, n_j), dtype=np.float64)
        h2_mc2 = np.zeros((n_i_eff, n_j), dtype=np.float64)  # sum-of-weights-squared
        if X_mc_all.size:
            h2_mc,  _, _ = np.histogram2d(X_mc_all, M_mc_all, bins=[X_edges, M_edges], weights=W_mc_all)
            h2_mc2, _, _ = np.histogram2d(X_mc_all, M_mc_all, bins=[X_edges, M_edges], weights=W_mc_all**2)

        for i in range(n_i_eff):
            # CR = I X bins separated from SR by G gap bins: [i-G-I, i-G-1]
            cr_lo = i - G - I
            cr_hi = i - G - 1
            if cr_lo < 0 or cr_hi < cr_lo:
                continue
            cr_slice = slice(cr_lo, cr_hi + 1)

            D_cr   = h2_data[cr_slice, :].sum(axis=0)    # shape (n_j,)
            MC_cr  = h2_mc  [cr_slice, :].sum(axis=0)
            MC_sr  = h2_mc  [i,        :]                # shape (n_j,)
            # sum-of-weights-squared for error propagation in PRED2
            emc_cr = np.sqrt(h2_mc2[cr_slice, :].sum(axis=0))
            emc_sr = np.sqrt(h2_mc2[i,        :])
            D_sr   = h2_data[i, :]                       # shape (n_j,) — SR data

            pred1, alpha, pred = _pred_core(D_cr, MC_cr, MC_sr, D_sr)

            if i == n_i_eff - 1:  # diagnostic for top SR bin only
                print(f"  [PRED diag] Xbin{i+1:02d}  CR bins {cr_lo+1}-{cr_hi+1}")
                print(f"  {'Mbin':>5}  {'D_CR':>8}  {'pred1':>10}  {'D_SR':>8}  {'alpha':>8}  {'pred':>8}")
                for j in range(n_j):
                    print(f"  {j+1:>5}  {D_cr[j]:>8.1f}  {pred1[j]:>10.3f}  {D_sr[j]:>8.1f}  {alpha[j]:>8.4f}  {pred[j]:>8.2f}")
                print(f"  pred1 total={pred1.sum():.2f}  pred total={pred.sum():.2f}")

            _write_bin(f"Xbin{i+1:02d}", pred, pred1, D_cr, D_sr, MC_sr, MC_cr, emc_sr, emc_cr)
            n_written += 1

    f_out.Close()
    print(f"  [PRED] wrote {n_written} X bins into {fname}")
    return _inmem


if __name__ == "__main__":
    start_time = time.time();

    # Pre-scan sys.argv for channel and mode keywords in any order.
    # This allows: python Root_plot.py 2l PRED1  OR  python Root_plot.py PRED1 2l
    valid_channels = ["0l", "1l", "2l"]
    valid_modes    = ["ND", "ROC", "PRED1", "PRED2", "PRED12", "PRED3", "PRED3.1", "PRED3.2", "PRED3.3", "PRED", "PRED4", "SCAN4"]

    # ROCX: combine per-channel MLPf_X ROC .npz files (from prior `<ch> ROC --loadmc` runs)
    # into one overlay plot. No event loading needed — exits immediately after.
    if "ROCX" in sys.argv[1:]:
        combine_roc_npz()
        sys.exit(0)

    channel = None
    mode    = None
    _passthrough = []
    for _a in sys.argv[1:]:
        if _a in valid_channels and channel is None:
            channel = _a
        elif _a in valid_modes and mode is None:
            mode = _a
        else:
            _passthrough.append(_a)

    if channel is None:
        print(f"ERROR: Channel required. One of: {', '.join(valid_channels)}")
        print(f"Usage: python Root_plot.py <channel> [ND|ROC|PRED1|PRED2] [-Region <region>]")
        sys.exit(1)

    print(f"Channel: {channel}  Mode: {mode or 'default'}")

    NO_DISPLAY  = mode in ("ND", "ROC", "PRED1", "PRED2", "PRED12", "PRED3", "PRED3.1", "PRED3.2", "PRED", "PRED4", "SCAN4")
    ROC_ONLY    = (mode == "ROC")
    PRED1       = mode in ("PRED1", "PRED12")
    PRED2       = mode in ("PRED2", "PRED12")
    PRED12      = (mode == "PRED12")
    PRED31      = (mode == "PRED3.1")   # like PRED3 but without pseudodata
    PRED32      = (mode == "PRED3.2")   # like PRED3 but plots 1-|mSD-133|/133, blind > 0.85
    PRED33      = (mode == "PRED3.3")   # like PRED3 but plots raw ak15_sdmass, blind [115,155]
    PRED3       = mode in ("PRED3", "PRED3.1", "PRED3.2", "PRED3.3")
    PRED_ONLY   = (mode == "PRED")   # data+signal only, same X bins/blinding as PRED3, no fits
    PRED4       = (mode == "PRED4")  # like PRED3 but plots ak15_SDmass instead of MLPf_Mscore
    SCAN4       = (mode == "SCAN4")  # fast S/sqrt(D) FOM scan over 4 SR X-bin boundaries, no fits
    _ANY_PRED   = PRED1 or PRED2 or PRED3 or PRED_ONLY or PRED4
    if ROC_ONLY: PLOT_ROC = True

    if PRED32:
        _SIG_NORM_DIV = 2.0
    elif PRED33:
        _SIG_NORM_DIV = 2.0
    elif PRED_ONLY or PRED3 or PRED12 or PRED4:
        _SIG_NORM_DIV = 4.0
    else:
        _SIG_NORM_DIV = 1.0
    # Generate 9 SR bins based on binning scheme flag
    if USE_1D_AVG_BINNING:
        BDT_ParT_KIN_bins = make_BDT_AVG_bins()
    else:
        BDT_ParT_KIN_bins = make_BDT_ParT_KIN_bins(channel)

    # Pass only non-keyword args to argparse (-Region etc.)
    sys.argv = [sys.argv[0]] + _passthrough

    parser = argparse.ArgumentParser();
    parser.add_argument("--loadmc", dest="loadmc", action="store_true", default=False,
                        help="Load WJets/Top/VV background MC (memory-heavy; off by default)")
    parser.add_argument("--sigscale", dest="sigscale", type=float, default=10.0,
                        help="Extra divisor applied on top of the default signal norm "
                             "(e.g. 100 = signal drawn 100x smaller). Default: 10 (signal = Data/10)")
    parser.add_argument("--noflow", dest="noflow", action="store_true", default=False,
                        help="Ignore under/overflow: events outside the histogram range are DISCARDED "
                             "instead of being merged into the first/last bin. Adds a '_noflow' "
                             "suffix to the output filenames so both versions can coexist.")
    parser.add_argument("-Region", dest="region", default="Preselection", choices=["Preselection", "SR_light", "SRs_7", "SR", "SR1", "SR2", "SR3", "SR4"], help="Region to run over")
    parser.add_argument("--regions", dest="regions", nargs='+', default=None,
                        choices=["Preselection", "SR_light", "SRs_7", "SR", "SR1", "SR2", "SR3", "SR4"],
                        help="Plot several regions in ONE process (data/MC loaded once, then reused). "
                             "Hugely cheaper than one process per region. E.g. --regions Preselection SR1 SR2 SR3 SR4")
    parser.add_argument("-I", dest="I", type=int, default=3, help="Number of CR X-bins")
    parser.add_argument("-J", dest="J", type=int, default=3, help="Number of alpha M-bins")
    parser.add_argument("-G", dest="G", type=int, default=3, help="Gap bins between CR and SR")
    parser.add_argument("--edges", nargs='+', type=float, metavar="EDGE",
                        default=None, help="Override SR bin edges (all boundaries; 1.0 appended if omitted). E.g. --edges 0.990 0.994 0.997 0.999 1.0")
    parser.add_argument("--p3bins", dest="p3bins", nargs=3, type=float, metavar=("N", "LO", "HI"),
                        default=None,
                        help="PRED3: override Mscore bins, lo, hi (default: 51 0.49 1.0)")
    parser.add_argument("--mscore", dest="mscore", nargs='+', type=float, metavar=("LO", "HI"),
                        default=None,
                        help="PRED3.3: additional Mscore cut applied on top of the nominal X-bins. "
                             "Pass one value (lower bound) or two (range). E.g. --mscore 0.99 or --mscore 0.95 0.99")
    parser.add_argument("--p4var", dest="p4var", default="1-abs(ak15_sdmass - 133)/133",
                        help="PRED4: variable expression to plot in each X-bin (default: 1-abs(ak15_sdmass-133)/133)")
    parser.add_argument("--p4bins", dest="p4bins", nargs=3, type=float, metavar=("N", "LO", "HI"),
                        default=[25, 0, 1],
                        help="PRED4: bins, lo, hi for the plotted variable (default: 25 0 1)")
    parser.add_argument("--p4blind", dest="p4blind", nargs=2, type=float, metavar=("LO", "HI"),
                        default=[0.85, 1.0],
                        help="PRED4: absolute blind range for SR bins (default: 0.85 1.0); pass 0 0 to disable")
    parser.add_argument("--p4xlabel", dest="p4xlabel", default="1-|m_{SD}-133|/133",
                        help="PRED4: x-axis label (default: 1-|m_{SD}-133|/133)")
    args = parser.parse_args()

    if args.loadmc:
        LOAD_BKG_MC = True
    if not _ANY_PRED:
        _SIG_NORM_DIV *= args.sigscale

    # In ROC_ONLY or PRED3-only mode, skip loading background MC for faster startup
    if ROC_ONLY:
        LOAD_BKG_MC = False
    if (PRED3 or PRED4) and not PRED1 and not PRED2 and not PRED12:
        LOAD_BKG_MC = False
        print("  [PRED3] Skipping BKG MC loading (not needed in PRED3 mode)")
    if PRED_ONLY:
        LOAD_BKG_MC = False
        print("  [PRED] Skipping BKG MC loading (data+signal only mode)")
    if PRED12 and not LOAD_BKG_MC:
        LOAD_BKG_MC = True
        print("  [PRED12] Forcing BKG MC loading (required for the Term1 MC transfer factor)")
    if SCAN4:
        LOAD_BKG_MC = False
        print("  [SCAN4] Skipping BKG MC loading (data+signal only mode)")

    _nomc_suffix = "_noMC" if not LOAD_BKG_MC else ""
    _noflow_suffix = "_noflow" if args.noflow else ""

    # Select channel-specific preselection from lookup dictionary (defined at top of file)
    Preselection = PRESELECTION[channel]
    # SR_light: Preselection + Hgg > 0.1 + kinBDT > 0.52
    SR_light = lambda e: Preselection(e) & (e["ak15_ParTMDV2_Hgg"] > 0.1) & (e["kinBDT"] > 0.52)
    region_map = {"Preselection": Preselection, "SR_light": SR_light, "SRs_7": Preselection}
    # SR1..SR4: Preselection + MLPf_X inside the corresponding _PRED_SR_edges bin — the same
    # boundaries that define PRED3's SR X-bins, so "-Region SR3" works like "-Region Preselection".
    # (_XQ_REGIONS is used further below to load MLPf_X and to skip the legacy BDT-binned plots.)
    _XQ_REGIONS = [f"SR{_i}" for _i in range(1, len(_PRED_SR_edges))]
    def _make_sr_selection(lo, hi):
        return lambda e, _lo=lo, _hi=hi: Preselection(e) & (e["MLPf_X"] > _lo) & (e["MLPf_X"] <= _hi)
    for _sr_i in range(1, len(_PRED_SR_edges)):
        region_map[f"SR{_sr_i}"] = _make_sr_selection(_PRED_SR_edges[_sr_i - 1], _PRED_SR_edges[_sr_i])
    if args.region not in region_map: raise ValueError(f"Unknown region {args.region}")
    selection = region_map[args.region]
    # Regions to plot in THIS process: --regions (several, data loaded once) else -Region (one).
    _plot_regions = list(args.regions) if args.regions else [args.region]
    for _r in _plot_regions:
        if _r not in region_map: raise ValueError(f"Unknown region {_r}")
    if args.regions:
        print(f"  [regions] plotting {len(_plot_regions)} regions in one process: {', '.join(_plot_regions)}")

    # collect only real branches from variable list
    if ROC_ONLY:
        # Minimal branches for ROC-only mode: selection cuts + ROC variables
        expressions = []
        print("ROC-only mode: loading minimal branches for faster startup")
    else:
        expressions = []
        for var in variables_to_plot:
            if var.name.isidentifier():
                expressions.append(var.name)
            else:
                tokens = re.findall(r"\b[a-zA-Z_]\w*\b", var.name)
                expressions.extend(tokens)

        # Also collect branches from 2D plot variables
        for var2d in variables_2D_plot:
            for var_name in [var2d["x_name"], var2d["y_name"]]:
                if var_name.isidentifier():
                    expressions.append(var_name)
                else:
                    tokens = re.findall(r"\b[a-zA-Z_]\w*\b", var_name)
                    expressions.extend(tokens)

    # remove duplicates
    expressions = list(set(expressions))
    # add extra branches needed for cuts (common to all channels)
    expressions += ["ak15_pt", "met", "ak15_ParTMDV2_Hgg", "v_pt", "v_eta", "v_phi", "dphi_V_ak15", "ak15_sdmass", "passTrigMu", "passTrigMET", "passTrigEl", "flavBDT", "kinBDT", "ak15_deltaR_sj12", "passmetfilters", "run", "year", "ak15_eta", "ak15_phi", "ak4_1_eta", "ak4_1_phi", "ak15_sj1_pt", "ak15_sj2_pt", "ak15_ParTMDV2_resonanceMassCorr", "BDT_ParT", "BDT_KIN"]
    # KIN BDT input branches needed for correlation plots
    expressions += ["ak4_1_mass", "ak4_1_pt", "n_ak4", "v_mass",
                    "jet_1_mass", "jet_1_pt", "jet_1_eta", "jet_1_phi"]
    # add channel-specific branches
    if channel in ["1l", "2l"]:
        expressions += ["lep1_pt", "lep1_eta", "lep1_phi", "lep1_pdgId", "dphi_lep_met", "met_phi"]
    if channel == "2l":
        expressions += ["lep2_pt", "lep2_eta", "lep2_phi", "lep2_pdgId", "deltaR_ll",
                        "deta_z_ak15", "min_deta_ak15_lep", "min_dr_z_ak4"]
    if channel == "1l":
        expressions += ["deta_lep_ak15", "min_dr_lep_ak4"]
    if channel == "0l":
        expressions += ["met_phi", "dphi_met_tkmet", "min_dphi_V_ak4", "min_dphi_met_jet"]


    # Channel-dependent base paths
    channel_dir = channel.upper()  # 0L, 1L, 2L
    mc_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/mc/merged"
    # Signal and data paths differ by channel
    if channel == "1l":
        signal_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/mc/merged_3"
        data_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/data/merged_3"
    elif channel == "0l":
        signal_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/mc/merged_sig"
        data_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/data/merged"
    elif channel == "2l":
        signal_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/mc/merged"
        data_base = f"/eos/user/z/zkou/hgg/samples/Tree/2018/{channel_dir}/data/merged"

    # Channel-dependent signal files
    if channel == "1l":
        # WH signal: W->lnu, H->gg
        signal_files_1 = [f"{signal_base}/WminusH_HToGG_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8_merged_tree.root"]
        signal_files_2 = [f"{signal_base}/WplusH_HToGG_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8_merged_tree.root"]
        signal_names = ["WminusH", "WplusH"]
    elif channel == "0l":
        # ZH signal: Z->nunu, H->gg (qq->ZH and gg->ZH)
        signal_files_1 = [f"{signal_base}/ZH_HToGG_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8_merged_tree.root"]
        signal_files_2 = [f"{signal_base}/ggZH_HToGG_ZToNuNu_M-125_TuneCP5_13TeV-powheg-pythia8_merged_tree.root"]
        signal_names = ["ZH_ZToNuNu", "ggZH_ZToNuNu"]
    elif channel == "2l":
        # ZH signal: Z->ll, H->gg (both qq->ZH and gg->ZH)
        signal_files_1 = [f"{signal_base}/ZH_HToGG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_merged_tree.root"]
        signal_files_2 = [f"{signal_base}/ggZH_HToGG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_merged_tree.root"]
        signal_names = ["ZH_ZToLL", "ggZH_ZToLL"]

    # Channel-dependent data files
    if channel == "0l":
        # 0L uses MET dataset (in merged/, not merged_3)
        _data_pattern = f"{data_base}/MET_*.root"
    else:
        # 1L and 2L use EGamma + SingleMuon datasets
        _data_pattern = f"{data_base}/*.root"
    data_files = robust_glob(_data_pattern)
    # glob returns [] both when nothing matches AND when the directory listing fails (e.g. a
    # transient AFS/EOS hiccup under many concurrent jobs) — fail loudly rather than much later
    # with an opaque "ak.concatenate of empty sequence" error.
    if not data_files:
        raise RuntimeError(f"No data files matched '{_data_pattern}' (channel {channel}). "
                           f"If the path is correct, this is likely a transient AFS/EOS listing "
                           f"failure — retry, ideally with fewer concurrent jobs.")

    # Channel-dependent V+jets background files (Z+jets and W+jets separately)
    zjets_files = {}
    wjets_files = {}

    if channel == "1l":
        # 1L: W+jets is main V+jets background
        wjets_files = {
            "WJetsToLNu_Pt-100To250": f"{mc_base}/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "WJetsToLNu_Pt-250To400": f"{mc_base}/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "WJetsToLNu_Pt-400To600": f"{mc_base}/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "WJetsToLNu_Pt-600ToInf": f"{mc_base}/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        }
    elif channel == "0l":
        # 0L: Both Z(->nunu)+jets and W(->lnu)+jets backgrounds
        zjets_files = {
            "Z1JetsToNuNu_PtZ-50To150": f"{mc_base}/Z1JetsToNuNu_M-50_LHEFilterPtZ-50To150_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z1JetsToNuNu_PtZ-150To250": f"{mc_base}/Z1JetsToNuNu_M-50_LHEFilterPtZ-150To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z1JetsToNuNu_PtZ-250To400": f"{mc_base}/Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z1JetsToNuNu_PtZ-400ToInf": f"{mc_base}/Z1JetsToNuNu_M-50_LHEFilterPtZ-400ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z2JetsToNuNu_PtZ-50To150": f"{mc_base}/Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z2JetsToNuNu_PtZ-150To250": f"{mc_base}/Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z2JetsToNuNu_PtZ-250To400": f"{mc_base}/Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "Z2JetsToNuNu_PtZ-400ToInf": f"{mc_base}/Z2JetsToNuNu_M-50_LHEFilterPtZ-400ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        }
        wjets_files = {
            "WJetsToLNu_Pt-100To250": f"{mc_base}/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "WJetsToLNu_Pt-250To400": f"{mc_base}/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "WJetsToLNu_Pt-400To600": f"{mc_base}/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "WJetsToLNu_Pt-600ToInf": f"{mc_base}/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        }
    elif channel == "2l":
        # 2L: DY (Z->ll)+jets is main background
        zjets_files = {
            "DYJetsToLL_PtZ-0To50": f"{mc_base}/DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "DYJetsToLL_PtZ-50To100": f"{mc_base}/DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "DYJetsToLL_PtZ-100To250": f"{mc_base}/DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "DYJetsToLL_PtZ-250To400": f"{mc_base}/DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "DYJetsToLL_PtZ-400To650": f"{mc_base}/DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
            "DYJetsToLL_PtZ-650ToInf": f"{mc_base}/DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        }

    # Cross-section dictionaries for Z+jets and W+jets
    xsec_zjets = xsec_ZJets if channel == "0l" else xsec_DY if channel == "2l" else {}
    xsec_wjets = xsec_WJets if channel in ["0l", "1l"] else {}
    # Z+jets label: "ZJets" for 0l (Z->nunu), "DY" for 2l (Z->ll)
    zjets_label = "Z+jets" if channel == "0l" else "DY" if channel == "2l" else ""
    wjets_label = "W+jets" if wjets_files else ""

    # Channel-dependent signal legend label
    signal_legend_labels = {"1l": "WH#rightarrowll#nu(gg)", "0l": "ZH#rightarrow#nu#nu(gg)", "2l": "ZH#rightarrowll(gg)"}
    signal_legend_label = signal_legend_labels.get(channel, "VH#rightarrow(gg)")

    # Top background MC files (channel-dependent)
    if channel == "2l":
        # No Top samples for 2L channel (files don't exist)
        top_files = {}
    else:
        # Full top samples for 0l and 1l channels
        top_files = {
            # TTbar
            "TTTo2L2Nu": f"{mc_base}/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_merged_tree.root",
            "TTToSemiLeptonic": f"{mc_base}/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_merged_tree.root",
            # Single top t-channel
            "ST_t-channel_top": f"{mc_base}/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8_merged_tree.root",
            "ST_t-channel_antitop": f"{mc_base}/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8_merged_tree.root",
            # Single top tW-channel
            "ST_tW_top": f"{mc_base}/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_merged_tree.root",
            "ST_tW_antitop": f"{mc_base}/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_merged_tree.root",
            # Single top s-channel
            "ST_s-channel": f"{mc_base}/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_merged_tree.root",
        }

    # Diboson (VV) background MC files (common to all channels)
    vv_files = {
        # WW
        "WWTo1L1Nu2Q": f"{mc_base}/WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "WWTo2L2Nu": f"{mc_base}/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_merged_tree.root",
        # WZ
        "WZTo1L1Nu2Q": f"{mc_base}/WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "WZTo1L3Nu": f"{mc_base}/WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "WZTo3LNu": f"{mc_base}/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "WZTo2Q2L": f"{mc_base}/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "WZTo2Q2Nu": f"{mc_base}/WZTo2Q2Nu_4f_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_merged_tree.root",
        # ZZ
        "ZZTo2Q2L": f"{mc_base}/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "ZZTo2L2Nu": f"{mc_base}/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_merged_tree.root",
        "ZZTo2Q2Nu": f"{mc_base}/ZZTo2Q2Nu_TuneCP5_13TeV-amcatnloFXFX-pythia8_merged_tree.root",
        "ZZTo4L": f"{mc_base}/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_merged_tree.root",
    }

    # ------- Print samples summary table ----------
    col_width = 50  # Width for sample name column
    print("\n" + "-"*8 + f"[ Samples ({channel}) ]" + "-"*(col_width-12) + "[ x-sec (fb) ]-----")

    # Channel-dependent signal cross-section factors
    # Format: factor1 * sigma * BR(H->gg) where BR(H->gg) = 0.08187
    if channel == "1l":
        # WH: 3 (W->lnu BRs for e/mu/tau) * sigma_WH * BR(H->gg)
        signal_xsec_factors = {"WminusH": (3, 59.83, 0.08187), "WplusH": (3, 94.26, 0.08187)}
    elif channel == "0l":
        # ZH->vv: BR(Z->vv)=0.2 * sigma_ZH * BR(H->gg)
        signal_xsec_factors = {"ZH_ZToNuNu": (0.2, 883.4, 0.08187), "ggZH_ZToNuNu": (0.2, 135.3, 0.08187)}
    elif channel == "2l":
        # ZH->ll: BR(Z->ll)=0.0673 * sigma_ZH * BR(H->gg)
        signal_xsec_factors = {"ZH_ZToLL": (0.0673, 883.4, 0.08187), "ggZH_ZToLL": (0.0673, 135.3, 0.08187)}

    # Print signal samples
    all_signal_files = signal_files_1 + signal_files_2
    for i, fpath in enumerate(all_signal_files):
        if not fpath:
            continue
        fname = os.path.basename(fpath).replace("_merged_tree.root", "")[:col_width]
        sample_name = signal_names[i] if i < len(signal_names) else f"Signal_{i}"
        if sample_name in signal_xsec_factors:
            f1, f2, f3 = signal_xsec_factors[sample_name]
            result = f1 * f2 * f3
            print(f"Signal  {fname:<{col_width}} ({f1}*{f2}*{f3}=) {result:.3f}")
        else:
            xsec = xsec_x_BR.get(sample_name, 0)
            print(f"Signal  {fname:<{col_width}} {xsec:>12.3f}")

    # Data samples (no cross-section)
    for f in data_files:
        fname = os.path.basename(f).replace("_merged_tree.root", "")[:col_width]
        print(f"Data    {fname:<{col_width}} {'--':>12}")

    # V+jets samples (channel-dependent: Z+jets and W+jets separately)
    if LOAD_BKG_MC:
        # Z+jets samples
        for sample, fpath in zjets_files.items():
            fname = os.path.basename(fpath).replace("_merged_tree.root", "")[:col_width]
            xsec = xsec_zjets.get(sample, 0)
            print(f"{zjets_label:<7} {fname:<{col_width}} {xsec:>12.3f}")
        # W+jets samples
        for sample, fpath in wjets_files.items():
            fname = os.path.basename(fpath).replace("_merged_tree.root", "")[:col_width]
            xsec = xsec_wjets.get(sample, 0)
            print(f"{wjets_label:<7} {fname:<{col_width}} {xsec:>12.3f}")

        # Top samples (use full name from top_files)
        for sample, fpath in top_files.items():
            fname = os.path.basename(fpath).replace("_merged_tree.root", "")[:col_width]
            xsec = xsec_Top[sample]
            print(f"Top     {fname:<{col_width}} {xsec:>12.3f}")

        # VV samples (use full name from vv_files)
        for sample, fpath in vv_files.items():
            fname = os.path.basename(fpath).replace("_merged_tree.root", "")[:col_width]
            xsec = xsec_VV[sample]
            print(f"VV      {fname:<{col_width}} {xsec:>12.3f}")

    n_signals = len([f for f in all_signal_files if f])
    n_vjets_total = len(zjets_files) + len(wjets_files)
    print("-"*(col_width + 8 + 13))
    if LOAD_BKG_MC:
        vjets_str = ""
        if zjets_files: vjets_str += f"{len(zjets_files)} {zjets_label}"
        if zjets_files and wjets_files: vjets_str += " + "
        if wjets_files: vjets_str += f"{len(wjets_files)} {wjets_label}"
        print(f"Total: {n_signals} Signal + {len(data_files)} Data + {vjets_str} + {len(top_files)} Top + {len(vv_files)} VV = {n_signals + len(data_files) + n_vjets_total + len(top_files) + len(vv_files)} samples\n")
    else:
        print(f"Total: {n_signals} Signal + {len(data_files)} Data (LOAD_BKG_MC=False, skipping BKG MC samples)\n")

    # ------- Loading data files ----------
    print("------- Loading data files ----------")
    total_start_time = time.time()
    load_start = time.time()

    # Collect ALL branches needed for ALL variables upfront
    all_expressions = set(expressions)
    if not ROC_ONLY:
        for var in variables_to_plot:
            tokens = re.findall(r"\b[a-zA-Z_]\w*\b", var.name)
            all_expressions.update(tokens)

    # Add all weight branches needed for plotting (skip in ROC mode — not needed for ROC curves)
    if not ROC_ONLY:
        weight_branches = ["genWeight", "lumiwgt", "puWeight", "l1PreFiringWeight",
                           "elEffWeight", "muEffWeight", "pileupJetIdWeight",
                           "topptWeight", "topptWeightNNLO", "vptWeightEWK",
                           "wptWeightEWK", "zptWeightEWK", "vhWeightEWK", "vvWeightNNLO",
                           "xsecWeight"]
        all_expressions.update(weight_branches)

    # Add branches needed for selections/cuts (common to all channels)
    selection_branches = ["ak15_pt", "met", "ak15_ParTMDV2_Hgg",
                          "v_pt", "v_eta", "v_phi", "dphi_V_ak15", "ak15_sdmass",
                          "passTrigMu", "passTrigMET", "passTrigEl",
                          "flavBDT", "kinBDT", "ak15_deltaR_sj12", "passmetfilters",
                          "run", "year", "ak15_eta", "ak15_phi", "ak4_1_eta", "ak4_1_phi"]
    all_expressions.update(selection_branches)
    # Add channel-specific branches
    if channel in ["1l", "2l"]:
        all_expressions.update(["lep1_pt", "lep1_eta", "lep1_phi", "lep1_pdgId", "dphi_lep_met", "met_phi"])
    if channel == "2l":
        all_expressions.update(["lep2_pt", "lep2_eta", "lep2_phi", "lep2_pdgId", "deltaR_ll", "v_mass"])
    if channel == "0l":
        all_expressions.update(["met_phi", "dphi_met_tkmet", "min_dphi_V_ak4", "min_dphi_met_jet"])

    # PRED4: extract constituent branch names from the variable expression and ensure they're loaded
    if PRED4:
        for _tok in re.findall(r"\b[a-zA-Z_]\w*\b", args.p4var):
            all_expressions.add(_tok)

    # Add gen-matching branches for Top background debugging (skip in ROC mode and PRED3 — not needed)
    if not ROC_ONLY and not PRED3:
        gen_match_branches = ["dr_ak15_hadtop_b", "dr_ak15_hadtop_wqmax", "dr_ak15_hadtop_wqmin",
                              "dr_ak15_leptop1_b", "dr_ak15_leptop2_b"]
        all_expressions.update(gen_match_branches)

    all_expressions = list(all_expressions)

    # Determine which BDT types are needed based on:
    # 1. Variables to plot that use BDT scores
    # 2. BDT binned plots requested
    # 3. Whether BDT score files actually exist for this channel (per type)
    bdt_types_needed = set()

    # Helper to check if BDT type's scores exist
    def bdt_type_exists(bdt_type):
        bdt_dir = f"BDT_scores_{channel}_{bdt_type}"
        return os.path.isdir(bdt_dir) and len(robust_glob(f"{bdt_dir}/*.root", attempts=3, delay=2.0)) > 0

    # Check which BDT types are requested AND exist
    # Note: use separate if statements (not elif) so combinations like (BDT_KIN*BDT_ParT) detect both
    for var in variables_to_plot:
        if "BDT_ParT"   in var.name and bdt_type_exists("ParT"):   bdt_types_needed.add("ParT")
        if "BDT_PNet"   in var.name and bdt_type_exists("PNet"):   bdt_types_needed.add("PNet")
        if "BDT_KIN"    in var.name and bdt_type_exists("KIN"):    bdt_types_needed.add("KIN")
        if ("BDT_MASS"   in var.name or "BDTf_MASS"   in var.name) and bdt_type_exists("MASS"):   bdt_types_needed.add("MASS")
        if "BDT_Mscore" in var.name and bdt_type_exists("Mscore"): bdt_types_needed.add("Mscore")
    # Also check if BDT binned plots are requested
    if len(variables_binned_plot) > 0:
        if len(BDT_ParT_bins) > 0 and bdt_type_exists("ParT"): bdt_types_needed.add("ParT")
        if len(BDT_KIN_bins) > 0 and bdt_type_exists("KIN"): bdt_types_needed.add("KIN")
        if len(BDT_ParT_KIN_bins) > 0:
            if bdt_type_exists("ParT"): bdt_types_needed.add("ParT")
            if bdt_type_exists("KIN"): bdt_types_needed.add("KIN")
    # ROC plots need BDT scores even when the Var entry is commented out
    if PLOT_ROC:
        if bdt_type_exists("ParT"):   bdt_types_needed.add("ParT")
        if bdt_type_exists("KIN"):    bdt_types_needed.add("KIN")
        if bdt_type_exists("Mscore"): bdt_types_needed.add("Mscore")

    # Determine which MLP types are needed (similar to BDT)
    mlp_types_needed = set()

    # Helper to check if MLP type's scores exist
    def mlp_type_exists(mlp_type):
        mlp_dir = f"MLP_scores_{channel}_{mlp_type}"
        return os.path.isdir(mlp_dir) and len(robust_glob(f"{mlp_dir}/*.root", attempts=3, delay=2.0)) > 0

    # Check which MLP types are requested AND exist (scan both plot lists)
    for var in variables_to_plot + variables_binned_plot:
        if "MLP_ParT"   in var.name and mlp_type_exists("ParT"):   mlp_types_needed.add("ParT")
        if "MLP_PNet"   in var.name and mlp_type_exists("PNet"):   mlp_types_needed.add("PNet")
        if "MLP_KIN"    in var.name and mlp_type_exists("KIN"):    mlp_types_needed.add("KIN")
        if ("MLP_MASS"   in var.name or "MLPf_MASS"   in var.name) and mlp_type_exists("MASS"):   mlp_types_needed.add("MASS")
        if "MLP_Mscore" in var.name and mlp_type_exists("Mscore"): mlp_types_needed.add("Mscore")
        if "MLP_X"      in var.name and mlp_type_exists("X"):      mlp_types_needed.add("X")

    # ROC plots need MLP scores even when the Var entry is commented out
    if PLOT_ROC:
        if mlp_type_exists("ParT"):   mlp_types_needed.add("ParT")
        if mlp_type_exists("KIN"):    mlp_types_needed.add("KIN")
        if mlp_type_exists("Mscore"): mlp_types_needed.add("Mscore")
        if mlp_type_exists("X"):      mlp_types_needed.add("X")

    # PRED3 only uses flat MLPf_X and MLPf_Mscore (detected separately below).
    # Raw BDT and MLP scores are not used in PRED3 fitting, selection, or plotting.
    if PRED3 or PRED4:
        bdt_types_needed.clear()
        mlp_types_needed.clear()

    if bdt_types_needed:
        print(f"BDT types to load: {bdt_types_needed}")
    if mlp_types_needed:
        print(f"MLP types to load: {mlp_types_needed}")

    # Determine which DNN types are needed (similar to MLP/BDT)
    dnn_types_needed = set()

    # Helper to check if DNN type's scores exist
    def dnn_type_exists(dnn_type):
        dnn_dir = f"DNN_scores_{channel}_{dnn_type}"
        return os.path.isdir(dnn_dir) and len(robust_glob(f"{dnn_dir}/*.root", attempts=3, delay=2.0)) > 0

    # Check which DNN types are requested AND exist (scan both plot lists)
    for var in variables_to_plot + variables_binned_plot:
        if "DNN_ParT"   in var.name and dnn_type_exists("ParT"):   dnn_types_needed.add("ParT")
        if "DNN_PNet"   in var.name and dnn_type_exists("PNet"):   dnn_types_needed.add("PNet")
        if "DNN_KIN"    in var.name and dnn_type_exists("KIN"):    dnn_types_needed.add("KIN")
        if ("DNN_MASS"   in var.name or "DNNf_MASS"   in var.name) and dnn_type_exists("MASS"):   dnn_types_needed.add("MASS")
        if "DNN_Mscore" in var.name and dnn_type_exists("Mscore"): dnn_types_needed.add("Mscore")

    # ROC plots need DNN scores even when the Var entry is commented out
    if PLOT_ROC:
        if dnn_type_exists("ParT"):   dnn_types_needed.add("ParT")
        if dnn_type_exists("KIN"):    dnn_types_needed.add("KIN")
        if dnn_type_exists("Mscore"): dnn_types_needed.add("Mscore")

    if dnn_types_needed:
        print(f"DNN types to load: {dnn_types_needed}")
    else:
        print(f"DNN scores: Skipping (no DNN_scores_{channel}_<type>/ directories or no .root files found)")

    # Determine which flat (CDF-transformed) score types are needed
    def flat_score_exists(prefix, score_type):
        d = f"{prefix}_scores_{channel}_{score_type}_FLAT"
        return os.path.isdir(d) and len(robust_glob(f"{d}/*.root", attempts=3, delay=2.0)) > 0

    flat_bdt_types_needed = set()
    flat_mlp_types_needed = set()
    flat_dnn_types_needed = set()
    # Collect all variable name strings to scan: 1D vars + 2D plot x/y names
    _var_names_to_scan = [var.name for var in variables_to_plot + variables_binned_plot]
    for var2d in variables_2D_plot:
        _var_names_to_scan.append(var2d["x_name"])
        _var_names_to_scan.append(var2d["y_name"])
    for _vname in _var_names_to_scan:
        if "BDTf_ParT"   in _vname and flat_score_exists("BDT", "ParT"):   flat_bdt_types_needed.add("ParT")
        if "BDTf_KIN"    in _vname and flat_score_exists("BDT", "KIN"):    flat_bdt_types_needed.add("KIN")
        if "BDTf_PNet"   in _vname and flat_score_exists("BDT", "PNet"):   flat_bdt_types_needed.add("PNet")
        if "BDTf_MASS"   in _vname and flat_score_exists("BDT", "MASS"):   flat_bdt_types_needed.add("MASS")
        if "BDTf_Mscore" in _vname and flat_score_exists("BDT", "Mscore"): flat_bdt_types_needed.add("Mscore")
        if "MLPf_ParT"   in _vname and flat_score_exists("MLP", "ParT"):   flat_mlp_types_needed.add("ParT")
        if "MLPf_KIN"    in _vname and flat_score_exists("MLP", "KIN"):    flat_mlp_types_needed.add("KIN")
        if "MLPf_Mscore" in _vname and flat_score_exists("MLP", "Mscore"): flat_mlp_types_needed.add("Mscore")
        if "MLPf_X"      in _vname and flat_score_exists("MLP", "X"):      flat_mlp_types_needed.add("X")
        if "DNNf_ParT"   in _vname and flat_score_exists("DNN", "ParT"):   flat_dnn_types_needed.add("ParT")
        if "DNNf_KIN"    in _vname and flat_score_exists("DNN", "KIN"):    flat_dnn_types_needed.add("KIN")
        if "DNNf_Mscore" in _vname and flat_score_exists("DNN", "Mscore"): flat_dnn_types_needed.add("Mscore")
    # ROC mode needs flat BDT/MLP scores for the X comparison plot
    if PLOT_ROC:
        if flat_score_exists("BDT", "ParT"):   flat_bdt_types_needed.add("ParT")
        if flat_score_exists("BDT", "KIN"):    flat_bdt_types_needed.add("KIN")
        if flat_score_exists("BDT", "Mscore"): flat_bdt_types_needed.add("Mscore")
        if flat_score_exists("MLP", "ParT"):   flat_mlp_types_needed.add("ParT")
        if flat_score_exists("MLP", "KIN"):    flat_mlp_types_needed.add("KIN")
    # PRED2/PRED3/PRED12/PRED modes always need MLPf_X and MLPf_Mscore regardless
    # of what appears in variables_to_plot (those vars are often commented out).
    # -Region SR1..SR4 likewise cut on MLPf_X, so the branch must be loaded even when no
    # plotted variable references it (e.g. plotting only ak15_sdmass).
    if PRED2 or PRED3 or PRED12 or PRED_ONLY or PRED4 or SCAN4 or any(_r in _XQ_REGIONS for _r in _plot_regions):
        if flat_score_exists("MLP", "X"): flat_mlp_types_needed.add("X")
    if PRED2 or PRED3 or PRED12 or PRED_ONLY or SCAN4:
        if flat_score_exists("MLP", "Mscore"): flat_mlp_types_needed.add("Mscore")
    if PRED4:
        # Load any flat scores referenced in --p4var expression
        for _p4t, _p4prefix in [("Mscore","MLP"),("KIN","MLP"),("ParT","MLP"),("X","MLP"),
                                  ("Mscore","BDT"),("KIN","BDT"),("ParT","BDT"),
                                  ("Mscore","DNN"),("KIN","DNN"),("ParT","DNN")]:
            if f"{_p4prefix}f_{_p4t}" in args.p4var and flat_score_exists(_p4prefix, _p4t):
                {"MLP": flat_mlp_types_needed, "BDT": flat_bdt_types_needed, "DNN": flat_dnn_types_needed}[_p4prefix].add(_p4t)

    if flat_bdt_types_needed: print(f"Flat BDT types to load: {flat_bdt_types_needed}")
    if flat_mlp_types_needed: print(f"Flat MLP types to load: {flat_mlp_types_needed}")
    if flat_dnn_types_needed: print(f"Flat DNN types to load: {flat_dnn_types_needed}")

    # Load signal files (channel-dependent)
    print("Signal MC:")
    events_signal = {}
    genEventSumw_signal = {}

    # Channel-dependent signal labels for printing
    if channel == "1l":
        signal_labels = ["W^-H-->vl(gg)", "W^+H-->lv(gg)"]
    elif channel == "0l":
        signal_labels = ["ZH-->vv(gg)", "ggZH-->vv(gg)"]
    elif channel == "2l":
        signal_labels = ["ZH-->ll(gg)", "ggZH-->ll(gg)"]

    # Collect missing branches from all load calls for consolidated warning
    all_missing_branches = set()

    # Load signal_files_1
    if signal_files_1:
        events_1, info_1, genEventSumw_1, missing_1 = load_select_branches(signal_files_1, all_expressions, bdt_type=None, verbose=True)
        events_signal[signal_names[0]] = events_1
        genEventSumw_signal[signal_names[0]] = genEventSumw_1
        all_missing_branches.update(missing_1)
        for fname, nevt in info_1:
            print(f"  {signal_labels[0]} :  {short_fname(fname)} {nevt:>10,} events")

    # Load signal_files_2 (if exists)
    if signal_files_2:
        events_2, info_2, genEventSumw_2, missing_2 = load_select_branches(signal_files_2, all_expressions, bdt_type=None, verbose=True)
        events_signal[signal_names[1]] = events_2
        genEventSumw_signal[signal_names[1]] = genEventSumw_2
        all_missing_branches.update(missing_2)
        for fname, nevt in info_2:
            print(f"  {signal_labels[1]} :  {short_fname(fname)} {nevt:>10,} events")

    # Load data files with per-file info
    print("Data:")
    events_data, info_data, _, missing_data = load_select_branches(data_files, all_expressions, bdt_type=None, verbose=True)
    all_missing_branches.update(missing_data)
    for fname, nevt in info_data:
        print(f"  {short_fname(fname)} {nevt:>10,} events")
    print(f"  {'Total data:':45} {len(events_data):>10,} events")

    # Helper to get clean sample name from filename (truncate to 38 chars)
    def clean_sample_name(fname, max_len=38):
        base = fname.replace("_merged_tree.root", "").replace("_tree.root", "")
        return base[:max_len] if len(base) > max_len else base

    # Load Z+jets, W+jets, Top, VV background MC files — all samples in parallel
    events_zjets = {}; genEventSumw_zjets = {}
    events_wjets = {}; genEventSumw_wjets = {}
    events_top   = {}; genEventSumw_top   = {}
    events_vv    = {}; genEventSumw_vv    = {}

    if LOAD_BKG_MC:
        # Build ordered task list: (group, name, filepath)
        bkg_tasks = (
            [('zjets', n, f) for n, f in zjets_files.items()] +
            [('wjets', n, f) for n, f in wjets_files.items()] +
            [('top',   n, f) for n, f in top_files.items()]   +
            [('vv',    n, f) for n, f in vv_files.items()]
        )

        def _load_one_bkg(args):
            _group, _name, _fpath = args
            ev, info, sumw, missing = load_select_branches(
                [_fpath], all_expressions, bdt_type=None, verbose=True, warn=False,
                precut=_UPROOT_PRECUT.get(channel, None))
            return _group, _name, ev, sumw, missing, info

        n_workers = min(len(bkg_tasks), 3 if (PRED1 or PRED2 or PRED12 or PRED3 or PRED_ONLY) else 2)
        print(f"  [parallel] loading {len(bkg_tasks)} BKG samples with {n_workers} workers...")
        bkg_results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_load_one_bkg, bkg_tasks):
                _group, _name, ev, sumw, missing, info = result
                bkg_results[_name] = (ev, sumw, missing, info)

        # Unpack in original order and print
        for name in zjets_files:
            ev, sumw, missing, info = bkg_results[name]
            events_zjets[name] = ev; genEventSumw_zjets[name] = sumw
            all_missing_branches.update(missing)
            xsec = xsec_zjets.get(name, 0)
            for fname, nevt in info:
                print(f"{zjets_label:<7} {clean_sample_name(fname):<38}  {nevt:>10,}  {sumw:>14.1f}  {xsec:>10.1f}")
        for name in wjets_files:
            ev, sumw, missing, info = bkg_results[name]
            events_wjets[name] = ev; genEventSumw_wjets[name] = sumw
            all_missing_branches.update(missing)
            xsec = xsec_wjets.get(name, 0)
            for fname, nevt in info:
                print(f"{wjets_label:<7} {clean_sample_name(fname):<38}  {nevt:>10,}  {sumw:>14.1f}  {xsec:>10.1f}")
        for name in top_files:
            ev, sumw, missing, info = bkg_results[name]
            events_top[name] = ev; genEventSumw_top[name] = sumw
            all_missing_branches.update(missing)
            xsec = xsec_Top.get(name, 0)
            for fname, nevt in info:
                print(f"{'Top':<7} {clean_sample_name(fname):<38}  {nevt:>10,}  {sumw:>14.1f}  {xsec:>10.1f}")
        for name in vv_files:
            ev, sumw, missing, info = bkg_results[name]
            events_vv[name] = ev; genEventSumw_vv[name] = sumw
            all_missing_branches.update(missing)
            xsec = xsec_VV.get(name, 0)
            for fname, nevt in info:
                print(f"{'VV':<7} {clean_sample_name(fname):<38}  {nevt:>10,}  {sumw:>14.1f}  {xsec:>10.1f}")

    # Build events dictionary with channel-dependent signal names
    events = {'data_2018': events_data}
    events.update(events_signal)
    # Add Z+jets and W+jets samples to events dict
    events.update(events_zjets)
    events.update(events_wjets)
    # Add Top samples to events dict
    events.update(events_top)
    # Add VV samples to events dict
    events.update(events_vv)

    # Store genEventSumw for proper MC normalization (from Runs tree)
    genEventSumw = {}
    genEventSumw.update(genEventSumw_signal)
    genEventSumw.update(genEventSumw_zjets)
    genEventSumw.update(genEventSumw_wjets)
    genEventSumw.update(genEventSumw_top)
    genEventSumw.update(genEventSumw_vv)
    sumw_str = ", ".join([f"{k}={v:.2f}" for k, v in genEventSumw_signal.items()])
    print(f"  genEventSumw: {sumw_str}")
    # Summary of loaded BKG samples
    n_zjets_loaded = sum(1 for k, v in events_zjets.items() if len(v) > 0)
    n_wjets_loaded = sum(1 for k, v in events_wjets.items() if len(v) > 0)
    n_top_loaded = sum(1 for k, v in events_top.items() if len(v) > 0)
    n_vv_loaded = sum(1 for k, v in events_vv.items() if len(v) > 0)
    bkg_summary = []
    if zjets_files: bkg_summary.append(f"{zjets_label}={n_zjets_loaded}/{len(zjets_files)}")
    if wjets_files: bkg_summary.append(f"{wjets_label}={n_wjets_loaded}/{len(wjets_files)}")
    bkg_summary.append(f"Top={n_top_loaded}/{len(top_files)}")
    bkg_summary.append(f"VV={n_vv_loaded}/{len(vv_files)}")
    print(f"  BKG samples loaded: {', '.join(bkg_summary)}")

    # Print consolidated warning about missing branches (once after all files loaded)
    if all_missing_branches:
        print(f"  Warning: branches not found in samples: {sorted(all_missing_branches)}")

    # =====================================================================
    # Cross-check: compare xsecWeight (from ROOT file) vs xsec/sumGenWgt
    # (hardcoded in this script) for all MC samples
    # =====================================================================
    print("\n" + "="*130)
    print("CROSS-CHECK: xsecWeight (ROOT branch) vs xsec_x_BR/sum_genWeight (hardcoded)")
    print(f"  {'Sample':<42} {'xsecWgt(branch)':>16} {'xsec/sumW(hc)':>16} {'Ratio':>8} | {'xsec_branch(fb)':>16} {'xsec_hc(fb)':>14} {'sumW(Runs)':>16} {'sumW_eff(branch)':>16}")
    print("-"*130)
    # Collect all MC sample dicts: (events_dict, xsec_dict, genEventSumw_dict)
    all_mc = []
    all_mc.append(("Signal", events_signal, xsec_x_BR, genEventSumw_signal))
    if LOAD_BKG_MC:
        all_mc.append(("Z+jets", events_zjets, xsec_zjets, genEventSumw_zjets))
        all_mc.append(("W+jets", events_wjets, xsec_wjets, genEventSumw_wjets))
        all_mc.append(("Top",    events_top,   xsec_Top,   genEventSumw_top))
        all_mc.append(("VV",     events_vv,    xsec_VV,    genEventSumw_vv))
    for group_name, ev_dict, xsec_dict, sumw_dict in all_mc:
        for sample_name, sample_events in ev_dict.items():
            if len(sample_events) == 0:
                continue
            # xsecWeight from ROOT branch (should be constant per sample)
            if "xsecWeight" in sample_events.fields:
                xsec_branch = float(ak.mean(sample_events["xsecWeight"]))
            else:
                xsec_branch = float('nan')
            # Hardcoded value
            xsec_hc = xsec_dict.get(sample_name, float('nan'))
            sumw = sumw_dict.get(sample_name, 0)
            if sumw > 0 and not np.isnan(xsec_hc):
                xsec_over_sumw = xsec_hc / sumw
            else:
                xsec_over_sumw = float('nan')
            # Ratio of the two xsecWeight values
            if not np.isnan(xsec_branch) and not np.isnan(xsec_over_sumw) and xsec_over_sumw != 0:
                ratio = xsec_branch / xsec_over_sumw
            else:
                ratio = float('nan')
            # Effective xsec from branch: xsecWeight * sumW(Runs) = xsec used by ntuple producer
            xsec_eff_branch = xsec_branch * sumw if (not np.isnan(xsec_branch) and sumw > 0) else float('nan')
            # Effective sumW from branch: xsec_hc / xsecWeight = what sumW the producer used
            sumw_eff_branch = xsec_hc / xsec_branch if (not np.isnan(xsec_branch) and xsec_branch != 0 and not np.isnan(xsec_hc)) else float('nan')
            print(f"  {sample_name:<42} {xsec_branch:>16.7g} {xsec_over_sumw:>16.7g} {ratio:>8.4f} | {xsec_eff_branch:>16.2f} {xsec_hc:>14.2f} {sumw:>16.2f} {sumw_eff_branch:>16.2f}")
    print("="*130)
    print("  Columns after '|':  xsec_branch = xsecWeight*sumW(Runs) [effective xsec if sumW from Runs is correct]")
    print("                      xsec_hc     = hardcoded xsec in Root_plot.py")
    print("                      sumW(Runs)  = sum_genWeight from Runs tree of merged file")
    print("                      sumW_eff    = xsec_hc/xsecWeight [what sumW the ntuple producer used]")
    print("  Compare xsec_branch vs xsec_hc to see if xsec values differ.")
    print("  Compare sumW(Runs) vs sumW_eff to see if sum_genWeight differs.\n")

    # Helper: load one score branch for all BKG samples in parallel
    def _load_all_bkg_scores(branch_name, **load_kwargs):
        if not LOAD_BKG_MC:
            return {}
        all_bkg = (list(zjets_files.items()) + list(wjets_files.items()) +
                   list(top_files.items())   + list(vv_files.items()))
        if not all_bkg:
            return {}
        def _task(args, _branch=branch_name, _kw=load_kwargs):
            name, fpath = args
            return name, load_select_branches([fpath], [_branch], precut=_UPROOT_PRECUT.get(channel, None), **_kw)[0]
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(all_bkg), 2 if (PRED1 or PRED2 or PRED12 or PRED3 or PRED_ONLY) else 3)) as executor:
            for name, ev in executor.map(_task, all_bkg):
                results[name] = ev
        return results

    def _load_score_direct(file_list, score_dir, branch_name, n=None):
        """Load a score branch from local score files without re-opening XRootD main files."""
        parts = []
        for f in file_list:
            score_path = os.path.join(score_dir, os.path.basename(f))
            if not os.path.exists(score_path):
                print(f"  Warning: score file not found: {score_path}")
                continue
            try:
                sf = uproot.open(score_path)
                arr = sf["Events"].arrays([branch_name], entry_stop=n)
                print(f"    Loaded {len(arr)} {branch_name} scores from {score_path}")
                parts.append(arr)
            except Exception as e:
                print(f"  Warning: Could not load {branch_name} from {score_path}: {e}")
        return ak.concatenate(parts, axis=0) if parts else ak.Array({branch_name: np.array([], dtype=np.float32)})

    # Load BDT scores separately if needed and merge them
    bdt_events = {}
    for bdt_type in bdt_types_needed:
        bdt_branch = f"BDT_{bdt_type}"
        _bdt_score_dir = f"BDT_scores_{channel}_{bdt_type}"
        print(f"  Loading {bdt_branch} scores from {_bdt_score_dir}/...")
        bdt_events[bdt_type] = {'data_2018': _load_score_direct(data_files, _bdt_score_dir, bdt_branch)}
        if signal_files_1:
            bdt_events[bdt_type][signal_names[0]] = _load_score_direct(signal_files_1, _bdt_score_dir, bdt_branch)
        if signal_files_2:
            bdt_events[bdt_type][signal_names[1]] = _load_score_direct(signal_files_2, _bdt_score_dir, bdt_branch)
        # Load BDT scores for all BKG samples in parallel
        bdt_events[bdt_type].update(
            _load_all_bkg_scores(bdt_branch, bdt_type=bdt_type, channel=channel))
        # Merge BDT scores into main events, then immediately free the temporary arrays
        for sample in events:
            if sample not in bdt_events[bdt_type]:
                continue
            if bdt_branch in bdt_events[bdt_type][sample].fields:
                main_len = len(events[sample])
                bdt_len = len(bdt_events[bdt_type][sample])
                if main_len != bdt_len:
                    print(f"  WARNING: {sample} event count mismatch: main={main_len}, BDT={bdt_len}")
                else:
                    events[sample] = ak.with_field(events[sample], bdt_events[bdt_type][sample][bdt_branch], bdt_branch)
        del bdt_events[bdt_type]
        gc.collect()
    del bdt_events
    gc.collect()

    # Load MLP scores separately if needed and merge them (similar to BDT)
    mlp_events = {}
    for mlp_type in mlp_types_needed:
        mlp_branch = f"MLP_{mlp_type}"
        _mlp_score_dir = f"MLP_scores_{channel}_{mlp_type}"
        print(f"  Loading {mlp_branch} scores from {_mlp_score_dir}/...")
        mlp_events[mlp_type] = {'data_2018': _load_score_direct(data_files, _mlp_score_dir, mlp_branch)}
        if signal_files_1:
            mlp_events[mlp_type][signal_names[0]] = _load_score_direct(signal_files_1, _mlp_score_dir, mlp_branch)
        if signal_files_2:
            mlp_events[mlp_type][signal_names[1]] = _load_score_direct(signal_files_2, _mlp_score_dir, mlp_branch)
        # Load MLP scores for all BKG samples in parallel
        mlp_events[mlp_type].update(
            _load_all_bkg_scores(mlp_branch, bdt_type=mlp_type, channel=channel, mlp_mode=True))
        # Merge MLP scores into main events, then immediately free the temporary arrays
        for sample in events:
            if sample not in mlp_events[mlp_type]:
                continue
            if mlp_branch in mlp_events[mlp_type][sample].fields:
                main_len = len(events[sample])
                mlp_len = len(mlp_events[mlp_type][sample])
                if main_len != mlp_len:
                    print(f"  WARNING: {sample} event count mismatch: main={main_len}, MLP={mlp_len}")
                else:
                    events[sample] = ak.with_field(events[sample], mlp_events[mlp_type][sample][mlp_branch], mlp_branch)
        del mlp_events[mlp_type]
        gc.collect()
    del mlp_events
    gc.collect()

    # Load DNN scores separately if needed and merge them (similar to MLP)
    dnn_events = {}
    for dnn_type in dnn_types_needed:
        dnn_branch = f"DNN_{dnn_type}"
        _dnn_score_dir = f"DNN_scores_{channel}_{dnn_type}"
        print(f"  Loading {dnn_branch} scores from {_dnn_score_dir}/...")
        dnn_events[dnn_type] = {'data_2018': _load_score_direct(data_files, _dnn_score_dir, dnn_branch)}
        if signal_files_1:
            dnn_events[dnn_type][signal_names[0]] = _load_score_direct(signal_files_1, _dnn_score_dir, dnn_branch)
        if signal_files_2:
            dnn_events[dnn_type][signal_names[1]] = _load_score_direct(signal_files_2, _dnn_score_dir, dnn_branch)
        # Load DNN scores for all BKG samples in parallel
        dnn_events[dnn_type].update(
            _load_all_bkg_scores(dnn_branch, bdt_type=dnn_type, channel=channel, dnn_mode=True))
        # Merge DNN scores into main events, then immediately free the temporary arrays
        for sample in events:
            if sample not in dnn_events[dnn_type]:
                continue
            if dnn_branch in dnn_events[dnn_type][sample].fields:
                main_len = len(events[sample])
                dnn_len = len(dnn_events[dnn_type][sample])
                if main_len != dnn_len:
                    print(f"  WARNING: {sample} event count mismatch: main={main_len}, DNN={dnn_len}")
                else:
                    events[sample] = ak.with_field(events[sample], dnn_events[dnn_type][sample][dnn_branch], dnn_branch)
        del dnn_events[dnn_type]
        gc.collect()
    del dnn_events
    gc.collect()

    # Helper to load flat scores and merge under alias BDTf_*/MLPf_*/DNNf_*
    def _load_flat_scores(score_type_set, prefix, mode_kwargs):
        flat_evts = {}
        for stype in score_type_set:
            raw_branch   = f"{prefix}_{stype}"
            alias_branch = f"{prefix}f_{stype}"
            _flat_dir = f"{prefix}_scores_{channel}_{stype}_FLAT"
            print(f"  Loading {alias_branch} scores from {_flat_dir}/...")
            flat_evts[stype] = {'data_2018': _load_score_direct(data_files, _flat_dir, raw_branch)}
            if signal_files_1:
                flat_evts[stype][signal_names[0]] = _load_score_direct(signal_files_1, _flat_dir, raw_branch)
            if signal_files_2:
                flat_evts[stype][signal_names[1]] = _load_score_direct(signal_files_2, _flat_dir, raw_branch)
            # Load flat scores for all BKG samples in parallel
            kwargs = dict(bdt_type=stype, channel=channel, flat=True, **mode_kwargs)
            flat_evts[stype].update(_load_all_bkg_scores(raw_branch, **kwargs))
            # Merge into main events under alias name, then immediately free the temporary arrays
            for sample in events:
                if sample not in flat_evts[stype]: continue
                if raw_branch in flat_evts[stype][sample].fields:
                    if len(events[sample]) == len(flat_evts[stype][sample]):
                        events[sample] = ak.with_field(events[sample], flat_evts[stype][sample][raw_branch], alias_branch)
                    else:
                        print(f"  WARNING: {sample} length mismatch for {alias_branch}")
            del flat_evts[stype]
            gc.collect()

    _load_flat_scores(flat_bdt_types_needed, "BDT", {})
    _load_flat_scores(flat_mlp_types_needed, "MLP", {"mlp_mode": True})
    _load_flat_scores(flat_dnn_types_needed, "DNN", {"dnn_mode": True})

    load_time = time.time() - load_start
    load_mins, load_secs = divmod(load_time, 60)
    print(f"Data loaded in {int(load_mins)}m {int(load_secs)}s")

    # ------- Apply combo score quantile window to selection (2l only, SR regions) ----------
    combo_q_suffix = ""  # appended to output filenames when COMBO_Q_RANGE is active
    if channel == "2l" and args.region not in ("Preselection",) and "COMBO_Q_RANGE" in dir() and "data_2018" in events:
        _d = events["data_2018"]
        if "MLPf_ParT" in _d.fields and "MLPf_KIN" in _d.fields:
            _base_mask = Preselection_2l(_d)
            _combo = (2.8*ak.to_numpy(_d["MLPf_ParT"][_base_mask]) + ak.to_numpy(_d["MLPf_KIN"][_base_mask])) / 3.8
            _q_lo = float(np.percentile(_combo, COMBO_Q_RANGE[0]))
            _q_hi = float(np.percentile(_combo, COMBO_Q_RANGE[1]))
            combo_q_suffix = f"_q{COMBO_Q_RANGE[0]}_{COMBO_Q_RANGE[1]}"
            print(f"  Combo quantile window p{COMBO_Q_RANGE[0]}-p{COMBO_Q_RANGE[1]}: combo in [{_q_lo:.4f}, {_q_hi:.4f}]")
            _base_sel = selection
            selection = lambda e, lo=_q_lo, hi=_q_hi: (
                _base_sel(e) & ((2.8*e["MLPf_ParT"]+e["MLPf_KIN"])/3.8 > lo)
                             & ((2.8*e["MLPf_ParT"]+e["MLPf_KIN"])/3.8 < hi)
            )

    # ------- Pre-compute derived variables ----------
    print("------- Pre-computing variables ----------")
    precompute_start = time.time()

    # Compute met_phi from V and lepton kinematics: V = lepton + MET(neutrino)
    # Therefore: MET = V - lepton => met_phi = atan2(V_py - lep_py, V_px - lep_px)
    # Only for channels with leptons (1l, 2l)
    if channel in ["1l", "2l"]:
        print("  Computing met_phi from V = lepton + MET...")
        for sample, data in events.items():
            v_pt = ak.to_numpy(data["v_pt"])
            v_phi = ak.to_numpy(data["v_phi"])
            lep1_pt = ak.to_numpy(data["lep1_pt"])
            lep1_phi = ak.to_numpy(data["lep1_phi"])

            # MET components from vector subtraction: MET = V - lepton
            met_px = v_pt * np.cos(v_phi) - lep1_pt * np.cos(lep1_phi)
            met_py = v_pt * np.sin(v_phi) - lep1_pt * np.sin(lep1_phi)
            met_phi = np.arctan2(met_py, met_px)

            events[sample] = ak.with_field(events[sample], met_phi, "met_phi")
    else:
        print("  Skipping met_phi computation (no leptons in 0l channel)")

    # Filter by channel to avoid evaluating variables with missing branches (e.g., lep2_* in 1l)
    computed_vars = [var for var in variables_to_plot
                     if not var.name.isidentifier() and (channel in var.channels or var.channels == "")]
    for var in computed_vars:
        var_name = var.name
        # Replace np.pi with literal value for numexpr compatibility
        eval_expr = var_name.replace("np.pi", str(np.pi))
        for sample, data in events.items():
            try:
                local_vars = {field: ak.to_numpy(data[field]) for field in data.fields}
                local_vars['pi'] = np.pi  # Add pi as a variable
                computed_result = numexpr.evaluate(eval_expr, local_dict=local_vars)
                events[sample] = ak.with_field(events[sample], computed_result, var_name)
            except Exception as e:
                print(f" --> Could not evaluate '{var_name}' for {sample}: {e}")

    # Compute BDTf_MASS / MLPf_MASS / DNNf_MASS:
    # Apply CDF of background 1-|MASS-125|/125 to make the mass-proximity variable flat.
    # CDF is built from data (background proxy at preselection level).
    for raw_branch, flat_branch in [("BDT_MASS", "BDTf_MASS"), ("MLP_MASS", "MLPf_MASS"), ("DNN_MASS", "DNNf_MASS")]:
        # Only proceed if this flat branch is actually requested and raw scores are loaded
        flat_vars_requested = any(v.name == flat_branch for v in variables_to_plot)
        if not flat_vars_requested:
            continue
        bkg_sample = "data_2018"
        if bkg_sample not in events or raw_branch not in events[bkg_sample].fields:
            print(f"  {flat_branch}: skipped (no {raw_branch} in {bkg_sample})")
            continue
        # Build CDF from data background distribution of 1-|MASS-125|/125
        raw_bkg = ak.to_numpy(events[bkg_sample][raw_branch]).astype(np.float64)
        valid_bkg = (raw_bkg > 0) & (raw_bkg < 1000)
        derived_bkg = np.clip(1.0 - np.abs(raw_bkg[valid_bkg] - 125.0) / 125.0, 0.0, 1.0)
        counts, bin_edges = np.histogram(derived_bkg, bins=200, range=(0.0, 1.0))
        total = counts.sum()
        if total == 0:
            print(f"  {flat_branch}: skipped (empty background histogram)")
            continue
        cdf_x = np.concatenate([[bin_edges[0]], bin_edges[1:]])
        cdf_y = np.concatenate([[0.0], np.cumsum(counts) / total])
        print(f"  {flat_branch}: CDF from {len(derived_bkg):,} bkg events")
        # Apply CDF to all samples
        for sample, data in events.items():
            if raw_branch not in data.fields:
                continue
            raw = ak.to_numpy(data[raw_branch]).astype(np.float64)
            valid = (raw > 0) & (raw < 1000)
            derived = np.clip(1.0 - np.abs(raw - 125.0) / 125.0, 0.0, 1.0)
            flat_val = np.full(len(raw), -1.0, dtype=np.float32)
            flat_val[valid] = np.interp(derived[valid], cdf_x, cdf_y).astype(np.float32)
            events[sample] = ak.with_field(events[sample], flat_val, flat_branch)

    precompute_time = time.time() - precompute_start
    print(f"Pre-computation done in {precompute_time:.1f}s")

    # ------- Quantile cut on combo score (top-25% data) ----------
    # Compute the 75th-percentile threshold of (2.8*MLPf_ParT+MLPf_KIN)/3.8 on data
    # using only the base preselection (no quantile cut yet), then tighten selection.
    COMBO_QUANTILE_CUT = False   # hardcoded window in Preselection_2l instead
    COMBO_QUANTILE = 0.6
    if COMBO_QUANTILE_CUT and "data_2018" in events:
        _d = events["data_2018"]
        _has_combo = "MLPf_ParT" in _d.fields and "MLPf_KIN" in _d.fields
        if _has_combo:
            _base_mask = Preselection(_d)
            _combo_vals = (2.8 * ak.to_numpy(_d["MLPf_ParT"][_base_mask])
                           + ak.to_numpy(_d["MLPf_KIN"][_base_mask])) / 3.8
            _combo_vals = _combo_vals[np.isfinite(_combo_vals)]
            combo_threshold = float(np.percentile(_combo_vals, COMBO_QUANTILE * 100))
            print(f"  Combo quantile cut: (2.8*MLPf_ParT+MLPf_KIN)/3.8 > {combo_threshold:.4f}  "
                  f"(top {100*(1-COMBO_QUANTILE):.0f}% of data, p{COMBO_QUANTILE*100:.0f})")
            _base_selection = selection
            selection = lambda e, thr=combo_threshold: (
                _base_selection(e) & ((2.8*e["MLPf_ParT"] + e["MLPf_KIN"]) / 3.8 > thr)
            )
        else:
            print("  Warning: MLPf_ParT or MLPf_KIN not loaded — quantile cut skipped")

    # ------- Print Selection Table ----------
    print(f"\n------- Selection: {args.region} ({channel}) ----------")
    precut_str = PRECUT_STR.get(channel, "")
    for cut in precut_str.split(" & "):
        print(f"  {cut}")
    if combo_q_suffix:
        print(f"  combo score quantile window: p{COMBO_Q_RANGE[0]}-p{COMBO_Q_RANGE[1]} (score [{_q_lo:.4f}, {_q_hi:.4f}])")
    print("")

    # ------- Plotting ----------
    # Filter variables by channel: plot only if channel is in "channels" field OR "channels" is empty
    filtered_vars = [v for v in variables_to_plot if channel in v.channels or v.channels == ""]
    saved_files = []

    if ROC_ONLY or _ANY_PRED:
        _skip_reason = "ROC-only" if ROC_ONLY else mode
        print(f"------- {_skip_reason} mode: skipping {len(filtered_vars)} Preselection variable plots ----------")
    else:
        plot_start = time.time()
        # Loop over the requested regions, reusing the already-loaded events (the expensive part).
        for _reg in _plot_regions:
            _reg_sel = region_map[_reg]
            print(f"------- Generating {len(filtered_vars)} plots ({_reg}, {channel} channel) ----------")
            print(f" {'#':>3s} {'Variable':35s} | {'Data':>8s} {'Signal':>7s} {'BKG MC':>10s} | {'Blind':10s}")

            for plot_idx, var in enumerate(filtered_vars, 1):
                if _ANY_PRED and var.name == "MLPf_X":
                    continue
                name = var.name
                # Replace PI digits with 'pi' before removing special characters
                name = name.replace("3.141592653589793", "pi")
                clean_name = re.sub(r"[!@#$%^&*()\=+,./;'\[\]{}\":>< ]", "", name)
                name = clean_name[:100]

                output_file = f"plot_{_reg}_{name}{combo_q_suffix}{_nomc_suffix}{_noflow_suffix}_{channel}"
                # Clean xlabel: remove ^1.0 or ^1 (redundant power of 1)
                xlabel = re.sub(r"\^1(\.0+)?$", "", var.xlabel)
                xlabel = re.sub(r"\^1(\.0+)([^0-9])", r"\2", xlabel)
                # Blinding in SR regions, using the same windows as the matching PRED mode.
                # Order matters: "1-abs(ak15_sdmass-133)/133" also contains "ak15_sdmass", so the
                # transform must be tested BEFORE the raw-mass rule or it would get the GeV window.
                blind = None
                if "SR" in _reg:
                    if "abs(ak15_sdmass" in var.name:
                        blind = (0.85, 1.0)      # PRED3.2 window for 1-|m_SD-133|/133
                    elif "MLPf_Mscore" in var.name:
                        blind = (0.97, 1.0)      # PRED3 window for the M-score
                    elif "ak15_sdmass" in var.name:
                        blind = (115, 155)       # PRED3.3 Higgs mass window [GeV]
                _q_label = f" q{COMBO_Q_RANGE[0]}-{COMBO_Q_RANGE[1]}%" if combo_q_suffix else ""
                plot_variable(events=events, var_name=var.name, selection=_reg_sel, bins=var.bins, xlabel=xlabel, output_prefix=output_file, logy=var.logy, blind_range=blind, genEventSumw=genEventSumw, signal_names=signal_names, zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()), xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets, zjets_label=zjets_label, wjets_label=wjets_label, signal_legend_label=signal_legend_label, channel=channel, plot_index=plot_idx, cdf_flat=var.cdf_flat, region_label=f"{_reg}{_q_label}", zero_edge_bins=("both" if args.noflow else None))

                saved_files.append(output_file)
                if not NO_DISPLAY: os.system(f"display {output_file}.png &")

        plot_time = time.time() - plot_start

    # ------- Correlation with SD mass ----------
    PLOT_CORRELATIONS = False
    if PLOT_CORRELATIONS:
        print(f"\n------- Generating sdmass correlation plot ({args.region}, {channel}) ----------")
        # MC samples list
        mc_samples_for_corr = (["TTTo2L2Nu", "TTToSemiLeptonic", "ST_t-channel_top", "ST_t-channel_antitop", "ST_tW_top", "ST_tW_antitop", "ST_s-channel"] +
                               ["WWTo1L1Nu2Q", "WWTo2L2Nu", "WZTo1L1Nu2Q", "WZTo1L3Nu", "WZTo3LNu", "WZTo2Q2L", "WZTo2Q2Nu", "ZZTo2Q2L", "ZZTo2L2Nu", "ZZTo2Q2Nu", "ZZTo4L"] +
                               [k for k in events.keys() if "DYJets" in k or "ZJets_NuNu" in k or "WJetsToLNu" in k])
        corr_file = compute_sdmass_correlations(events, "data_2018", mc_samples_for_corr, variables_to_plot, selection,
                                                 f"plot_{args.region}_corr_sdmass", channel)
        if corr_file: saved_files.append(corr_file.replace(".png", ""))
        if corr_file and not NO_DISPLAY: os.system(f"display {corr_file} &")

        # KIN variables correlation plot: BDT output scores + all raw input branches
        # Raw branches used in BDT KIN training (power transforms preserve rank order,
        # so rs(branch, mSD) = rs(branch**0.5, mSD) — we check raw branches directly)
        # Current KIN training variables for 0l (updated to match Train_BDT/MLP/DNN.py):
        #   met, ak15_eta, dphi_V_ak15,
        #   ak4_1_phi-v_phi, dphi_met_tkmet, jet_1_eta,
        #   ak15_eta-jet_1_eta, ak15_phi-jet_1_phi, jet_1_phi-v_phi
        #   [ak15_pt and met/ak15_pt removed — caused pT-sdmass sculpting on background]
        # Composite expressions cannot be looked up as branches, so we include
        # the component branches so their individual correlations are visible.
        # ak15_pt is kept here (commented out from training) so its residual correlation
        # can still be monitored in the KIN correlation plot.
        kin_input_branches_0l = [
            # Direct training inputs (simple branches)
            "met", "ak15_eta",
            "dphi_V_ak15", "dphi_met_tkmet", "jet_1_eta",
            # Component branches of composite training variables
            # (ak4_1_phi-v_phi, ak15_eta-jet_1_eta, ak15_phi-jet_1_phi, jet_1_phi-v_phi)
            "ak4_1_phi", "v_phi",
            "ak15_phi", "jet_1_phi",
            # Removed from training but kept for monitoring (pT-sdmass correlation)
            "ak15_pt",
        ]
        kin_input_branches_1l = kin_input_branches_0l + [
            "lep1_pt", "lep1_eta", "lep1_phi",
            "ak4_1_pt", "deta_lep_ak15", "min_dr_lep_ak4",
        ]
        # Remove 0l-only branches not in 1l
        kin_input_branches_1l = [b for b in kin_input_branches_1l if b != "dphi_met_tkmet"]
        kin_input_branches_2l = kin_input_branches_1l + [
            "lep2_pt", "lep2_eta", "lep2_phi",
            "v_mass", "deltaR_ll", "deta_z_ak15", "min_deta_ak15_lep", "min_dr_z_ak4",
        ]
        # Remove 1l-only branches not in 2l
        kin_input_branches_2l = [b for b in kin_input_branches_2l if b not in ["deta_lep_ak15", "min_dr_lep_ak4"]]

        kin_input_map = {"0l": kin_input_branches_0l, "1l": kin_input_branches_1l, "2l": kin_input_branches_2l}
        kin_input_branches = kin_input_map.get(channel, [])

        # BDT/MLP/DNN output scores
        kin_score_names = ["BDT_KIN", "MLP_KIN", "DNN_KIN", "kinBDT"]
        # Combine: scores first, then input branches
        all_kin_vars = [v for v in kin_score_names if any(v in str(getattr(var, 'name', var)) for var in variables_to_plot)]
        all_kin_vars += kin_input_branches
        if all_kin_vars:
            print(f"\n------- Generating KIN sdmass correlation plot ({args.region}, {channel}) ----------")
            print(f"  Including {len(kin_input_branches)} raw KIN input branches + {len(all_kin_vars) - len(kin_input_branches)} score variables")
            corr_file_kin = compute_sdmass_correlations(events, "data_2018", mc_samples_for_corr, all_kin_vars, selection,
                                                         f"plot_{args.region}_corr_sdmass_KIN", channel, max_vars=50)
            if corr_file_kin: saved_files.append(corr_file_kin.replace(".png", ""))
            if corr_file_kin and not NO_DISPLAY: os.system(f"display {corr_file_kin} &")

    # ------- BDT Binned Plots ----------
    # List of BDT bin sets to loop over
    all_BDT_bins = [
        ("BDT_KIN", BDT_KIN_bins),
        ("BDT_ParT", BDT_ParT_bins),
        ("BDT_ParT_KIN", BDT_ParT_KIN_bins),
    ]

    # SR1..SR4 are single X-quantile slices (Preselection-style single-region plots) — they must
    # not trigger the legacy BDT-binned / 9SR machinery, which belongs to the old SR/SRs_7 scheme.
    # (_XQ_REGIONS defined above, next to region_map.)
    if len(variables_binned_plot) > 0 and not ROC_ONLY and args.region not in ["SR_light", "Preselection"] + _XQ_REGIONS:
        for bdt_name, bdt_bins in all_BDT_bins:
            if len(bdt_bins) == 0: continue
            print(f"\n------- Plotting {len(variables_binned_plot)} variables in {len(bdt_bins)} {bdt_name} bins ----------")
            print(f" {'Variable':40s} | {'Data':>8s} {'Signal':>7s} {'BKG MC':>10s} | {'Blind':10s}")

            # Collect significances for summary table (keyed by region label like "SR1a")
            sig_summary_0bin = {}
            sig_summary_1bin = {}
            sig_summary_2bin = {}
            sig_summary_3bin = {}

            for var in variables_binned_plot:
                for file_suffix, display_label, bin_cut in bdt_bins:
                    # Combine preselection with BDT bin cut
                    binned_selection = lambda e, bc=bin_cut: selection(e) & bc(e)

                    # Clean variable name for filename
                    name = var.name.replace("3.141592653589793", "pi")
                    clean_name = re.sub(r"[!@#$%^&*()\-=+,./;'\[\]{}\":>< ]", "", name)[:100]

                    output_file = f"plot_{args.region}_{clean_name}_{file_suffix}{_nomc_suffix}_{channel}"

                    # Clean xlabel: remove ^1.0 or ^1 (redundant power of 1)
                    xlabel = re.sub(r"\^1(\.0+)?$", "", var.xlabel)
                    xlabel = re.sub(r"\^1(\.0+)([^0-9])", r"\2", xlabel)

                    # Blind ak15_sdmass in range 115-160 GeV; blind last 5 bins of 1-|MLP_MASS-125|/125
                    if "ak15_sdmass" in var.name:
                        blind = (115, 160)
                    elif "MLP_MASS" in var.name:
                        blind = (0.85, 1.0)  # last 3 of 20 bins ([0,1] range, bin width 0.05)
                    else:
                        blind = None
                    plot_info = plot_variable(events=events, var_name=var.name, selection=binned_selection, bins=var.bins, xlabel=xlabel, output_prefix=output_file, logy=var.logy, region_label=display_label, blind_range=blind, genEventSumw=genEventSumw, signal_names=signal_names, zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()), xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets, zjets_label=zjets_label, wjets_label=wjets_label, signal_legend_label=signal_legend_label, channel=channel)

                    # Store significance values for summary tables
                    if plot_info:
                        if "sig_0bin" in plot_info: sig_summary_0bin[display_label] = plot_info["sig_0bin"]
                        if "sig_1bin" in plot_info: sig_summary_1bin[display_label] = plot_info["sig_1bin"]
                        if "sig_2bin" in plot_info: sig_summary_2bin[display_label] = plot_info["sig_2bin"]
                        if "sig_3bin" in plot_info: sig_summary_3bin[display_label] = plot_info["sig_3bin"]

                    saved_files.append(output_file)
                    if not NO_DISPLAY: os.system(f"display {output_file}.png &")

            # Print significance summary table (separate from plotting table)
            if len(sig_summary_1bin) > 0 and bdt_name == "BDT_ParT_KIN":
                n_sr = len(sig_summary_1bin)
                print(f"\n-------- Significance S/sqrt(D) for {n_sr} SRs --------")
                print(f" {'SR':20s} | {'0 bin':>7s} | {'1 bin':>7s} | {'2 bins':>7s} | {'3 bins':>7s} |")
                print(f"----------------------+---------+---------+---------+---------+")

                # Calculate quadrature sums for each bin type
                quad_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

                for sr_label in sig_summary_1bin.keys():
                    vals = [
                        sig_summary_0bin.get(sr_label, 0.0),
                        sig_summary_1bin.get(sr_label, 0.0),
                        sig_summary_2bin.get(sr_label, 0.0),
                        sig_summary_3bin.get(sr_label, 0.0),
                    ]
                    for i, v in enumerate(vals):
                        quad_sums[i] += v ** 2
                    print(f" {sr_label:20s} | {vals[0]:>7.3f} | {vals[1]:>7.3f} | {vals[2]:>7.3f} | {vals[3]:>7.3f} |")

                print(f"----------------------+---------+---------+---------+---------+")
                print(f" {'Quad':20s} | {quad_sums[0]**0.5:>7.3f} | {quad_sums[1]**0.5:>7.3f} | {quad_sums[2]**0.5:>7.3f} | {quad_sums[3]**0.5:>7.3f} |")
                print(f"----------------------+---------+---------+---------+---------+")

    # ------- Combined SR Summary Plot (independent of variables_binned_plot) ----------
    if len(BDT_ParT_KIN_bins) > 0 and not ROC_ONLY and args.region not in ["SR_light", "Preselection"] + _XQ_REGIONS:
        n_sr = len(BDT_ParT_KIN_bins)
        print(f"\n------- Generating {n_sr} SR x 12 bins Summary Plot ----------")
        output_9SR = f"plot_combined_{n_sr}SR_{args.region}_{channel}"
        result_9SR = plot_combined_9SR(events, selection, BDT_ParT_KIN_bins, output_9SR, genEventSumw, LOAD_BKG_MC, signal_names)
        output_9SR_file = result_9SR.get("output_file", f"{output_9SR}.png")
        saved_files.append(output_9SR_file.replace(".png", ""))
        if not NO_DISPLAY: os.system(f"display {output_9SR_file} &")

        # Combined SR summary plot for 1-|MLP_MASS-125|/125 (10 bins/SR, blind last 3)
        mlp_mass_avail = any("MLP_MASS" in (f if isinstance(f, str) else "") for f in (events.get(list(events.keys())[0]).fields if events else []))
        if mlp_mass_avail:
            print(f"\n------- Generating {n_sr} SR x 10 bins MLP MASS Summary Plot ----------")
            output_mlp = f"plot_combined_{n_sr}SR_{args.region}_mlpmass_{channel}"
            result_mlp = plot_combined_nSR_mlpmass(events, selection, BDT_ParT_KIN_bins, output_mlp, genEventSumw, LOAD_BKG_MC, signal_names)
            output_mlp_file = result_mlp.get("output_file", f"{output_mlp}.png")
            saved_files.append(output_mlp_file.replace(".png", ""))
            if not NO_DISPLAY: os.system(f"display {output_mlp_file} &")

    # ------- 2D Plots ----------
    if len(variables_2D_plot) > 0 and not ROC_ONLY and not PRED3 and not PRED4:
        import math as _math
        print(f"\n------- Generating {len(variables_2D_plot)} 2D plots + significance maps ({args.region}) ----------")
        _2d_pngs      = []  # data+signal plots
        _2d_sig_pngs  = []  # S/sqrt(D+eps) significance maps
        _kw2d = dict(selection=selection, region_label=args.region,
                     genEventSumw=genEventSumw, signal_names=signal_names)

        for var2d in variables_2D_plot:
            x_clean = var2d.get("x_tag") or re.sub(r"[!@#$%^&*()\-=+,./;'\[\]{}\":>< ]", "", var2d["x_name"])[:50]
            y_clean = var2d.get("y_tag") or re.sub(r"[!@#$%^&*()\-=+,./;'\[\]{}\":>< ]", "", var2d["y_name"])[:50]
            _vkw = dict(events=events, x_var=var2d["x_name"], y_var=var2d["y_name"],
                        bins_x=var2d["bins_x"], bins_y=var2d["bins_y"],
                        xlabel=var2d["xlabel"], ylabel=var2d["ylabel"],
                        cdf_flat_x=var2d.get("cdf_flat_x", False), **_kw2d)

            # Data + signal contours plot (existing)
            output_file = f"plot2D_{args.region}_{x_clean}_vs_{y_clean}_{channel}"
            plot_variable_2D(output_prefix=output_file, **_vkw)
            saved_files.append(output_file)
            _2d_pngs.append(output_file + ".png")

            # S/sqrt(D+eps) significance map (new)
            output_sig_file = f"plot2D_signif_{args.region}_{x_clean}_vs_{y_clean}_{channel}"
            plot_variable_2D_significance(output_prefix=output_sig_file, **_vkw)
            saved_files.append(output_sig_file)
            _2d_sig_pngs.append(output_sig_file + ".png")

        def _make_montage(pngs, out_name):
            n = len(pngs)
            ncols = _math.ceil(_math.sqrt(n * 1.4))
            nrows = _math.ceil(n / ncols)
            os.system(f"montage {' '.join(pngs)} -tile {ncols}x{nrows} -geometry +2+2 -background white {out_name}")
            print(f"  Montage: {out_name}  ({ncols}×{nrows} grid)")
            saved_files.append(out_name.replace(".png", ""))
            os.system(f"display {out_name} &")

        if _2d_pngs:
            _make_montage(_2d_pngs,     f"montage_2D_{args.region}_{channel}.png")
        if _2d_sig_pngs:
            _make_montage(_2d_sig_pngs, f"montage_2D_signif_{args.region}_{channel}.png")

    # ------- ROC Curves ----------
    # Compare discriminating variables (BDT, MLP, DNN) dynamically based on availability
    if PLOT_ROC and not PRED2:
        print(f"\n------- Generating ROC curves ({args.region}, {channel}) ----------")

        def var_available(var_name):
            if "data_2018" not in events or var_name not in events["data_2018"].fields:
                return False
            return any(sn in events and var_name in events[sn].fields for sn in signal_names)

        def first_available(*names):
            for n in names:
                if var_available(n): return n
            return None

        # --- ParT: BDT vs MLP vs DNN vs raw Hgg score (4 curves) ---
        parT_vars, parT_labels = [], []
        v = first_available("BDT_ParT"); (parT_vars.append(v), parT_labels.append("BDT ParT (finetuned)")) if v else None
        v = first_available("MLP_ParT"); (parT_vars.append(v), parT_labels.append("MLP ParT (finetuned)")) if v else None
        v = first_available("DNN_ParT"); (parT_vars.append(v), parT_labels.append("DNN ParT (finetuned)")) if v else None
        v = first_available("ak15_ParTMDV2_Hgg"); (parT_vars.append(v), parT_labels.append("ParT Hgg (raw score)")) if v else None

        if parT_vars:
            print(f"  ParT ROC: {parT_vars}")
            plot_roc(
                events=events, var_names=parT_vars, var_labels=parT_labels,
                selection=selection, output_prefix=f"roc_ParT_comparison_{channel}",
                signal_names=signal_names, bkg_name="data_2018",
                channel=channel, region_label=args.region,
                mc_bkg_names=None, disc_type=None,
                x_label="ParT signal efficiency",
            )
            saved_files.append(f"roc_ParT_comparison_{channel}")
            os.system(f"display roc_ParT_comparison_{channel}.png &")

        # --- KIN: BDT vs MLP vs DNN (3 curves) ---
        kin_vars, kin_labels = [], []
        v = first_available("BDT_KIN"); (kin_vars.append(v), kin_labels.append("BDT KIN")) if v else None
        v = first_available("MLP_KIN"); (kin_vars.append(v), kin_labels.append("MLP KIN")) if v else None
        v = first_available("DNN_KIN"); (kin_vars.append(v), kin_labels.append("DNN KIN")) if v else None

        if kin_vars:
            print(f"  KIN ROC: {kin_vars}")
            plot_roc(
                events=events, var_names=kin_vars, var_labels=kin_labels,
                selection=selection, output_prefix=f"roc_KIN_comparison_{channel}",
                signal_names=signal_names, bkg_name="data_2018",
                channel=channel, region_label=args.region,
                mc_bkg_names=None, disc_type=None,
                x_label="KIN signal efficiency",
            )
            saved_files.append(f"roc_KIN_comparison_{channel}")
            os.system(f"display roc_KIN_comparison_{channel}.png &")

        # --- Mscore: BDT vs MLP vs DNN vs mass-proximity proxy ---
        _mass_proxy_field = "mass_proxy_sdmass135"
        _has_mass_proxy = var_available("ak15_sdmass")
        if _has_mass_proxy:
            for _sname, _ev in events.items():
                if "ak15_sdmass" in _ev.fields:
                    _proxy = 1.0 - np.abs(ak.to_numpy(_ev["ak15_sdmass"]) - 135.0) / 135.0
                    events[_sname] = ak.with_field(events[_sname], ak.Array(_proxy), _mass_proxy_field)

        mscore_vars, mscore_labels = [], []
        v = first_available("BDT_Mscore"); (mscore_vars.append(v), mscore_labels.append("BDT M-score")) if v else None
        v = first_available("MLP_Mscore"); (mscore_vars.append(v), mscore_labels.append("MLP M-score")) if v else None
        v = first_available("DNN_Mscore"); (mscore_vars.append(v), mscore_labels.append("DNN M-score")) if v else None
        if _has_mass_proxy:
            mscore_vars.append(_mass_proxy_field)
            mscore_labels.append("1-|m_{SD}-135|/135")

        if mscore_vars:
            print(f"  Mscore ROC: {mscore_vars}")
            plot_roc(
                events=events, var_names=mscore_vars, var_labels=mscore_labels,
                selection=selection, output_prefix=f"roc_Mscore_comparison_{channel}",
                signal_names=signal_names, bkg_name="data_2018",
                channel=channel, region_label=args.region,
                mc_bkg_names=None, disc_type=None,
                x_label="M-score signal efficiency",
            )
            saved_files.append(f"roc_Mscore_comparison_{channel}")
            os.system(f"display roc_Mscore_comparison_{channel}.png &")

        # --- X: MLP_X vs BDTf_ParT vs BDTf_KIN vs (2.8*MLPf_ParT+MLPf_KIN)/3.8 ---
        _combo_field = "combo_2p8MLPf_ParT_1MLPf_KIN"
        _has_combo = all(first_available(v) for v in ("MLPf_ParT", "MLPf_KIN"))
        if _has_combo:
            for _sname, _ev in events.items():
                if all(f in _ev.fields for f in ("MLPf_ParT", "MLPf_KIN")):
                    _combo = (2.8*ak.to_numpy(_ev["MLPf_ParT"]) + ak.to_numpy(_ev["MLPf_KIN"])) / 3.8
                    events[_sname] = ak.with_field(events[_sname], ak.Array(_combo), _combo_field)

        x_vars, x_labels = [], []
        v = first_available("MLP_X");      (x_vars.append(v), x_labels.append("MLP X (ParT+KIN)"))              if v else None
        v = first_available("BDTf_ParT");  (x_vars.append(v), x_labels.append("BDT ParT (flat)"))               if v else None
        v = first_available("BDTf_KIN");   (x_vars.append(v), x_labels.append("BDT KIN (flat)"))                if v else None
        if _has_combo:
            x_vars.append(_combo_field)
            x_labels.append("(2.8MLPf_{ParT}+MLPf_{KIN})/3.8")

        if x_vars:
            print(f"  X ROC: {x_vars}")
            plot_roc(
                events=events, var_names=x_vars, var_labels=x_labels,
                selection=selection, output_prefix=f"roc_X_comparison_{channel}",
                signal_names=signal_names, bkg_name="data_2018",
                channel=channel, region_label=args.region,
                mc_bkg_names=None, disc_type=None,
                x_label="X signal efficiency",
            )
            saved_files.append(f"roc_X_comparison_{channel}")
            os.system(f"display roc_X_comparison_{channel}.png &")

        # --- MLPf_X alone, dumped to .npz for cross-channel combination (mode ROCX) ---
        v = first_available("MLPf_X")
        if v:
            _mc_bkg_names = None
            if LOAD_BKG_MC:
                _mc_bkg_names = (list(wjets_files.keys()) + list(zjets_files.keys())
                                 + list(top_files.keys()) + list(vv_files.keys()))
            print(f"  MLPf_X ROC: {v} (mc={'yes' if _mc_bkg_names else 'no'})")
            plot_roc(
                events=events, var_names=[v], var_labels=["X-score (MLPf)"],
                selection=selection, output_prefix=f"roc_MLPfX_{channel}",
                signal_names=signal_names, bkg_name="data_2018",
                channel=channel, region_label=args.region,
                mc_bkg_names=_mc_bkg_names, disc_type=None,
                x_label="X-score signal efficiency",
                save_npz=f"roc_MLPfX_{channel}.npz",
            )
            saved_files.append(f"roc_MLPfX_{channel}")
            os.system(f"display roc_MLPfX_{channel}.png &")

    # ------- Mass Sculpting Study ----------
    if PLOT_SCULPTING:
        print(f"\n------- Generating mass sculpting plots ({args.region}, {channel}) ----------")
        USE_EFF_CUTS = True  # True = efficiency-based, False = fixed cut values
        # Loop over all relevant discriminants to compare sculpting behaviour
        sculpt_vars = []
        for sv in ["BDT_KIN", "MLP_KIN", "DNN_KIN", "kinBDT", "BDT_ParT"]:
            if any(sv in str(getattr(v, 'name', v)) for v in variables_to_plot):
                sculpt_vars.append(sv)
        if not sculpt_vars:
            sculpt_vars = ["BDT_KIN"]  # fallback

        for SCULPT_VAR in sculpt_vars:
            # Auto-select cuts: BDTs/MLPs use efficiency-based [1, 0.5, 0.3, 0.1], taggers use tighter
            is_bdt = "BDT" in SCULPT_VAR or "MLP" in SCULPT_VAR or "DNN" in SCULPT_VAR or SCULPT_VAR in ["kinBDT", "flavBDT"]
            eff_cuts = [1, 0.5, 0.3, 0.1] if is_bdt else [1, 0.3, 0.1, 0.03]
            fixed_cuts = [-0.5, 0, 0.3, 0.5] if is_bdt else [0, 0.05, 0.1, 0.25]
            sculpt_file = plot_mass_sculpting(
                events=events,
                mass_var="ak15_sdmass",
                scan_var=SCULPT_VAR,
                cut_values=eff_cuts if USE_EFF_CUTS else fixed_cuts,
                selection=selection,
                output_prefix=f"plot_{args.region}",
                channel=channel,
                use_eff_cuts=USE_EFF_CUTS,
                nbins=25, xmin=50, xmax=300  # sdmass range in GeV
            )
            if sculpt_file:
                saved_files.append(sculpt_file.replace(".png", ""))
                if not NO_DISPLAY: os.system(f"display {sculpt_file} &")

    # ------- Background prediction (Term 1) ----------
    _pred_inmem = None
    if PRED1 and not PRED12:
        print(f"------- Running background prediction (Term 1, I={args.I}) ----------")
        _pred_inmem = compute_bkg_prediction(
            events       = events,
            selection    = Preselection,
            channel      = channel,
            genEventSumw = genEventSumw,
            xsec_zjets   = xsec_zjets,
            xsec_wjets   = xsec_wjets,
            zjets_samples= list(zjets_files.keys()),
            wjets_samples= list(wjets_files.keys()),
            n_i=50, n_j=50, I=args.I, J=args.J, G=args.G,
        )

    # ── Shared SR bin edges: _PRED_SR_edges is defined at module scope (top of file),
    #    shared with the -Region SR1..SR4 selections. ─────────────────────────────

    # ------- PRED12: Pred2 overlay on PRED3-style coarse SR/VR bins ------------
    # Same layout/axes as PRED3: 4 SR + 4 VR quantile X bins, M in 60 bins [0.70, 1.0],
    # 2x4 montage (VR row on top), common S/sigma_stat display scale anchored at 4.2.
    if PRED12:
        print(f"------- PRED12: prediction overlay on PRED3-style SR/VR bins ({channel}) ----------")
        _p12_I, _p12_J, _p12_G = args.I, args.J, args.G
        _SR_edges = list(_PRED_SR_edges)
        # VR bins quantile-matched by data count: VR_i holds the same number of data events
        # as SR_i (not the same X-width), so their absolute yields agree.
        _d_x12 = ak.to_numpy(events["data_2018"]["MLPf_X"][Preselection(events["data_2018"])])
        _VR_edges = quantile_matched_vr_edges(_d_x12, _SR_edges)
        print(f"  [PRED12] quantile-matched VR edges: {[round(e,4) for e in _VR_edges]}")
        _p12_bins = (50, 0.50, 1.0)   # M axis: 50 bins over [0.5, 1.0] (width 0.01)

        _p12_pred_common = dict(
            events=events, selection=Preselection, channel=channel,
            genEventSumw=genEventSumw, xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()),
            n_i=50, I=_p12_I, J=_p12_J, G=_p12_G, m_bins=_p12_bins,
        )
        _pred12_sr = compute_bkg_prediction(x_quantile_edges=_SR_edges, tag="_SR12", **_p12_pred_common)
        _pred12_vr = compute_bkg_prediction(x_quantile_edges=_VR_edges, tag="_VR12", **_p12_pred_common)

        _common12 = dict(
            events=events, selection_base=Preselection, channel=channel,
            genEventSumw=genEventSumw, xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()),
            signal_names=signal_names, signal_legend_label=signal_legend_label,
            zjets_label=zjets_label, wjets_label=wjets_label,
            n_i=50, I=_p12_I, J=_p12_J, G=_p12_G, no_display=True,
            no_pred_overlay=False, cdf_flat_m=False, skip_bern_fit=True,
            no_pseudodata=True, write_limit_output=False,
            plot_bins=_p12_bins, pred3_yrange=True, zero_edge_bins="first",
        )
        sr12_files, _p12_total_sig, _, _ = compute_pred2_plots(
            x_quantile_edges=_SR_edges, tag="SR12", pred_histos=_pred12_sr,
            blind_from_idx=0, abs_blind_range=(0.97, 1.0), no_pred1_overlay=True,
            no_signal_plot=False, common_sig_scale_target=4.2, **_common12)
        vr12_files, _, _, _ = compute_pred2_plots(
            x_quantile_edges=_VR_edges, tag="VR12", pred_histos=_pred12_vr,
            blind_from_idx=999, no_signal_plot=True, no_pred1_overlay=True, **_common12)

        _p12_output    = f"montage_pred12_{channel}.png"
        montage_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "montage_pred2.sh")
        os.system(f"bash {montage_script} {channel} {_p12_I} {_p12_J} {_p12_G} "
                  f"{_p12_total_sig:.3f} {_p12_output} n/a 0 0 SR12")

    if PRED3:
        print(f"------- Generating PRED3 plots (no prediction overlay) ({channel}) ----------")
        _pred3_I, _pred3_J, _pred3_G = args.I, args.J, args.G

        # ── SR bin edges: edit _PRED_SR_edges above ───────────────────────────
        _SR_edges = list(_PRED_SR_edges)
        # Override from --edges E0 E1 ... EN (all boundaries; 1.0 appended if last < 1.0)
        if args.edges is not None:
            _SR_edges = sorted(args.edges)
            if _SR_edges[-1] < 1.0:
                _SR_edges.append(1.0)
            print(f"  [PRED3] --edges override: SR bins = {_SR_edges}")
        if args.edges is not None:
            _inner = _SR_edges[:-1]  # all boundaries except 1.0
            _edges_suffix = "_e" + "_".join(f"{int(round(e*1000)):04d}" for e in _inner)
        else:
            _edges_suffix = ""

        # ── Per-submode variable/binning/blind config ────────────────────────────
        _p32_var    = "1-abs(ak15_sdmass-133)/133"
        _p32_blind  = (0.85, 1.0)
        _p32_xlabel = "1-|m_{SD}-133|/133"

        _p33_var    = "ak15_sdmass"
        _p33_blind  = (115., 155.)   # Higgs window: interior blind [115, 155] GeV (+1 bin on the right, 5 GeV/bin)
        _p33_xlabel = "m_{SD} [GeV]"

        # VR edges quantile-matched by data count: VR_i holds the same number of data events
        # as SR_i (not the same X-width), so their absolute yields agree.
        _d_x3 = ak.to_numpy(events["data_2018"]["MLPf_X"][Preselection(events["data_2018"])])
        _p32_vr_edges = quantile_matched_vr_edges(_d_x3, _SR_edges)
        print(f"  [PRED3] quantile-matched VR edges: {[round(e,4) for e in _p32_vr_edges]}")

        # Variable-specific kwargs merged into _common
        if PRED32:
            _var_kwargs = {
                "plot_var": _p32_var, "plot_bins": (15, 0.40, 1.), "plot_xlabel": _p32_xlabel,
                "bern_fit_lo": 0.40, "sig_xrange": None, "bern_orders": (1, 2, 3),
                "pred3_yrange": True, "zero_edge_bins": "first",
            }
        elif PRED33:
            _var_kwargs = {
                "plot_var": _p33_var, "plot_bins": (28, 70., 210.), "plot_xlabel": _p33_xlabel,
                "bern_fit_lo": 70., "bern_fit_hi": 210., "sig_xrange": None,
                "bern_orders": (2, 3, 4),
                "pred3_yrange": True, "zero_edge_bins": "both",
            }
        else:
            _p3_plot_bins = (int(args.p3bins[0]), args.p3bins[1], args.p3bins[2]) if args.p3bins else (60, 0.70, 1.)
            _p3_bern_fit_lo = args.p3bins[1] + 0.01 if args.p3bins else 0.70
            _var_kwargs = {
                "bern_orders": (2, 3, 4), "plot_bins": _p3_plot_bins, "bern_fit_lo": _p3_bern_fit_lo,
                "pred3_yrange": True, "zero_edge_bins": "first",
            }

        # Optional Mscore slice: restrict events to MLPf_Mscore in [lo, hi] on top of Preselection.
        # Applied as an additional cut; X-bin edges remain the nominal _PRED_SR_edges.
        _mscore_suffix = ""
        _selection_pred3 = Preselection
        if args.mscore is not None:
            _ms_lo = args.mscore[0]
            _ms_hi = args.mscore[1] if len(args.mscore) > 1 else 1.0
            _selection_pred3 = lambda e, _sel=Preselection, _lo=_ms_lo, _hi=_ms_hi: \
                _sel(e) & (e["MLPf_Mscore"] >= _lo) & (e["MLPf_Mscore"] <= _hi)
            _mscore_suffix = f"_ms{int(round(_ms_lo*1000)):04d}"
            if _ms_hi < 1.0:
                _mscore_suffix += f"_{int(round(_ms_hi*1000)):04d}"
            print(f"  [PRED3] --mscore cut: MLPf_Mscore in [{_ms_lo}, {_ms_hi}]")

        _common = dict(
            events=events, selection_base=_selection_pred3, channel=channel,
            genEventSumw=genEventSumw, xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()),
            signal_names=signal_names, signal_legend_label=signal_legend_label,
            zjets_label=zjets_label, wjets_label=wjets_label,
            n_i=50, I=0, J=_pred3_J, G=_pred3_G, no_display=True,
            pred_histos=None, no_pred_overlay=True, cdf_flat_m=False,
            no_pseudodata=PRED31,
            **_var_kwargs,
        )

        # All 6 SR bins blinded; blind window differs per sub-mode
        if PRED32:
            _sr_blind = _p32_blind
        elif PRED33:
            _sr_blind = _p33_blind
        else:
            _sr_blind = (0.97, 1.0)
        _sr_blind_from = 0  # blind applies to all 6 SR X-bins

        # Unique file tags to avoid collision between parallel runs; append Mscore suffix if set
        if PRED32:
            _sr_tag, _vr_tag = f"SR32{_mscore_suffix}", f"VR32{_mscore_suffix}"
        elif PRED33:
            _sr_tag, _vr_tag = f"SR33{_mscore_suffix}", f"VR33{_mscore_suffix}"
        elif PRED31:
            _sr_tag, _vr_tag = f"SR31{_mscore_suffix}", f"VR31{_mscore_suffix}"
        else:
            _sr_tag, _vr_tag = f"SR3{_mscore_suffix}", f"VR3{_mscore_suffix}"

        # SR pass — signal drawn; PRED3.3 excludes blind window from Bernstein fit
        _sr_bern_excl = _p33_blind if PRED33 else None
        _mode_sfx = "_p32" if PRED32 else "_p33" if PRED33 else ""
        sr_files, _pred3_total_sig, _pred3_avg_ft_c2, _pred3_avg_aic_c2 = compute_pred2_plots(
            x_quantile_edges=_SR_edges, tag=_sr_tag, write_limit_output=True,
            blind_from_idx=_sr_blind_from, no_signal_plot=False, fname_suffix=_edges_suffix + _mode_sfx,
            abs_blind_range=_sr_blind, bern_exclude_range=_sr_bern_excl,
            common_sig_scale_target=4.2, **_common)

        # VR pass + montage — skipped in scan mode (--edges provided)
        if args.edges is None:
            _vr_xedges = _p32_vr_edges
            vr_files, _, _, _ = compute_pred2_plots(
                x_quantile_edges=_vr_xedges, tag=_vr_tag, write_limit_output=False,
                blind_from_idx=999, no_signal_plot=True, **_common)

            if PRED33:
                _pred3_output = f"montage_pred3.3{_mscore_suffix}_{channel}.png"
            elif PRED32:
                _pred3_output = f"montage_pred3.2{_mscore_suffix}_{channel}.png"
            elif PRED31:
                _pred3_output = f"montage_pred3.1{_mscore_suffix}_{channel}.png"
            else:
                _pred3_output = f"montage_pred3{_mscore_suffix}_{channel}.png"
            montage_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "montage_pred2.sh")
            _aic_s = f"{_pred3_avg_aic_c2:.2f}" if _pred3_avg_aic_c2 == _pred3_avg_aic_c2 else "n/a"
            _keep  = "1" if PRED31 else "0"
            os.system(f"bash {montage_script} {channel} 0 {_pred3_J} {_pred3_G} "
                      f"{_pred3_total_sig:.3f} {_pred3_output} {_aic_s} 0 {_keep} {_sr_tag}")

    if PRED4:
        _p4_var    = args.p4var
        _p4_n, _p4_lo, _p4_hi = int(args.p4bins[0]), args.p4bins[1], args.p4bins[2]
        _p4_blind  = tuple(args.p4blind) if (args.p4blind[0] != 0 or args.p4blind[1] != 0) else None
        _p4_xlabel = args.p4xlabel
        print(f"------- Generating PRED4 plots ({_p4_var} per X-bin, same quantiles as PRED3) ({channel}) ----------")
        print(f"  variable={_p4_var}  bins=({_p4_n}, {_p4_lo}, {_p4_hi})  blind={_p4_blind}")
        _pred4_J, _pred4_G = args.J, args.G

        # Same SR/VR X-quantile edges as PRED3 (shared _PRED_SR_edges)
        _SR_edges = list(_PRED_SR_edges)
        _VR_edges = [0.0] + [round(1.0 - e, 6) for e in reversed(_SR_edges[:-1])]

        _common4 = dict(
            events=events, selection_base=Preselection, channel=channel,
            genEventSumw=genEventSumw, xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()),
            signal_names=signal_names, signal_legend_label=signal_legend_label,
            zjets_label=zjets_label, wjets_label=wjets_label,
            n_i=50, I=0, J=_pred4_J, G=_pred4_G, no_display=True,
            pred_histos=None, no_pred_overlay=True, cdf_flat_m=False,
            skip_bern_fit=True, no_pseudodata=True, write_limit_output=False,
            plot_var=_p4_var, plot_bins=(_p4_n, _p4_lo, _p4_hi),
            plot_xlabel=_p4_xlabel,
        )

        # SR: blinded if p4_blind set, signal drawn; sig_xrange=None → full range
        sr4_files, _pred4_total_sig, _, _ = compute_pred2_plots(
            x_quantile_edges=_SR_edges, tag="SR4",
            blind_from_idx=(0 if _p4_blind else 999), no_signal_plot=False,
            abs_blind_range=_p4_blind, sig_xrange=None, **_common4)

        # VR: no blinding, no signal
        vr4_files, _, _, _ = compute_pred2_plots(
            x_quantile_edges=_VR_edges, tag="VR4",
            blind_from_idx=999, no_signal_plot=True,
            abs_blind_range=None, **_common4)

        _pred4_output  = f"montage_pred4_{channel}.png"
        montage_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "montage_pred2.sh")
        os.system(f"bash {montage_script} {channel} 0 {_pred4_J} {_pred4_G} "
                  f"{_pred4_total_sig:.3f} {_pred4_output} n/a 0 0 SR4")

    if PRED_ONLY:
        print(f"------- Generating PRED plots (data+signal only, same binning/blinding as PRED3) ({channel}) ----------")
        _pred_I, _pred_J, _pred_G = args.I, args.J, args.G
        _pred_x_edges = [0.77, 0.805, 0.835, 0.865, 0.89, 0.915, 0.935, 0.955, 0.97, 0.982, 0.991, 0.997, 1.0]
        pred_files, _pred_total_sig, _, _ = compute_pred2_plots(
            events=events, selection_base=Preselection, channel=channel,
            genEventSumw=genEventSumw, xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()),
            signal_names=signal_names, signal_legend_label=signal_legend_label,
            zjets_label=zjets_label, wjets_label=wjets_label,
            n_i=50, x_quantile_edges=_pred_x_edges,
            I=_pred_I, J=_pred_J, G=_pred_G, no_display=True,
            pred_histos=None, no_pred_overlay=True, cdf_flat_m=False, skip_bern_fit=True,
            ratio_yrange=(0, 2), upper_pad_only=True,
        )
        if pred_files:
            _pred_output  = f"montage_pred_{channel}.png"
            montage_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "montage_pred2.sh")
            os.system(f"bash {montage_script} {channel} {_pred_I} {_pred_J} {_pred_G} 0.000 {_pred_output} n/a 10")

    if PRED2 and not PRED12:
        print(f"------- Generating PRED2 plots ({channel}) ----------")
        _pred2_I, _pred2_J, _pred2_G = args.I, args.J, args.G
        pred2_files, _pred2_total_sig, _, _ = compute_pred2_plots(
            events=events, selection_base=Preselection, channel=channel,
            genEventSumw=genEventSumw, xsec_zjets=xsec_zjets, xsec_wjets=xsec_wjets,
            zjets_samples=list(zjets_files.keys()), wjets_samples=list(wjets_files.keys()),
            signal_names=signal_names, signal_legend_label=signal_legend_label,
            zjets_label=zjets_label, wjets_label=wjets_label,
            n_i=50, I=_pred2_I, J=_pred2_J, G=_pred2_G, no_display=True,
            pred_histos=_pred_inmem,   # None when standalone PRED2 → falls back to ROOT file
        )
        if pred2_files:
            montage_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "montage_pred2.sh")
            os.system(f"bash {montage_script} {channel} {_pred2_I} {_pred2_J} {_pred2_G} {_pred2_total_sig:.3f}")

    # ------- SCAN4: 3-FOM scan, no Bernstein fits, no Combine ----------------
    if SCAN4:
        import itertools, datetime as _dt

        print(f"------- SCAN4: 3-FOM scan ({channel}) ----------")

        # ── Scan grid — edit these 4 lists to change the scan points ─────────
        _S4_B1 = [0.75, 0.8, 0.85, 0.9]
        _S4_B2 = [0.970, 0.980, 0.985]
        _S4_B3 = [0.99, 0.993, 0.995, 0.996]
        _S4_B4 = [0.996, 0.997, 0.998]
        _S4_MSCORE_LO = 0.9   # SR lower bound on Mscore
        _S4_N_MSCORE  = 50    # histogram bins in [_S4_MSCORE_LO, 1.0]
        _S4_F_SYST    = 0.10  # fractional bkg systematic for FOM3 (10%)
        _S4_RESULTS   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan4_results.txt")
        # ─────────────────────────────────────────────────────────────────────

        _mscore_edges = np.linspace(_S4_MSCORE_LO, 1.0, _S4_N_MSCORE + 1)

        # ── Data arrays after preselection, filtered to Mscore SR ─────────────
        _d    = events["data_2018"]
        _sel  = Preselection(_d)
        _x_d  = ak.to_numpy(_d["MLPf_X"][_sel]).astype(np.float32)
        _m_d  = ak.to_numpy(_d["MLPf_Mscore"][_sel]).astype(np.float32)
        _sr   = _m_d >= _S4_MSCORE_LO
        _x_d, _m_d = _x_d[_sr], _m_d[_sr]
        print(f"  Data events in SR (Mscore >= {_S4_MSCORE_LO}): {len(_x_d):,}")

        # ── Signal arrays: genWeight × lumiwgt × xsec per sample ──────────────
        _x_s_parts, _m_s_parts, _w_s_parts = [], [], []
        for _sn in signal_names:
            if _sn not in events or len(events[_sn]) == 0:
                print(f"  WARNING: signal {_sn} not found, skipping")
                continue
            _s    = events[_sn]
            _sels = Preselection(_s)
            _x_s  = ak.to_numpy(_s["MLPf_X"][_sels]).astype(np.float32)
            _m_s  = ak.to_numpy(_s["MLPf_Mscore"][_sels]).astype(np.float32)
            _gw   = ak.to_numpy(_s["genWeight"][_sels]).astype(np.float64)
            _lw   = ak.to_numpy(_s["lumiwgt"][_sels]).astype(np.float64)
            _xsec = xsec_x_BR.get(_sn, 0.0)
            _w_s  = _gw * _lw * _xsec
            _sr_s = _m_s >= _S4_MSCORE_LO
            _x_s_parts.append(_x_s[_sr_s])
            _m_s_parts.append(_m_s[_sr_s])
            _w_s_parts.append(_w_s[_sr_s])
            print(f"  Signal {_sn}: {_sr_s.sum():,} SR events  sumW={_w_s[_sr_s].sum():.2f}")

        if not _x_s_parts:
            print("  No signal loaded -- cannot compute FOM.")
        else:
            _x_sig = np.concatenate(_x_s_parts)
            _m_sig = np.concatenate(_m_s_parts)
            _w_sig = np.concatenate(_w_s_parts)

            # ── Enumerate valid combos ─────────────────────────────────────────
            _all_combos = [
                (b1, b2, b3, b4)
                for b1, b2, b3, b4 in itertools.product(_S4_B1, _S4_B2, _S4_B3, _S4_B4)
                if b1 < b2 < b3 < b4
            ]
            _n_raw = len(_S4_B1) * len(_S4_B2) * len(_S4_B3) * len(_S4_B4)
            print(f"  Grid: {_n_raw} raw -> {len(_all_combos)} valid (strictly increasing)")
            print(f"  FOM3 bkg systematic: f={_S4_F_SYST:.0%}")

            # ── Scan: compute all 3 FOMs per combo ────────────────────────────
            # FOM1: sqrt(sum S^2/D)                  -- S/sqrt(D) quadrature
            # FOM2: sqrt(2*sum[(S+D)*ln(1+S/D) - S]) -- Asimov likelihood ratio
            # FOM3: sqrt(sum S^2/(D + f^2*D^2))      -- FOM1 + bkg syst uncertainty
            _t0_scan = time.time()
            _rows = []
            for b1, b2, b3, b4 in _all_combos:
                _f1, _f2, _f3 = 0.0, 0.0, 0.0
                for _xlo, _xhi in [(b1,b2),(b2,b3),(b3,b4),(b4,1.0)]:
                    _md = (_x_d >= _xlo) & (_x_d <= _xhi) if _xhi >= 1.0 else (_x_d >= _xlo) & (_x_d < _xhi)
                    _ms = (_x_sig >= _xlo) & (_x_sig <= _xhi) if _xhi >= 1.0 else (_x_sig >= _xlo) & (_x_sig < _xhi)
                    _hd, _ = np.histogram(_m_d[_md],   bins=_mscore_edges)
                    _hs, _ = np.histogram(_m_sig[_ms], bins=_mscore_edges, weights=_w_sig[_ms])
                    _hd    = _hd.astype(np.float64)
                    _nz    = _hd > 0
                    # FOM1: S^2/D
                    _f1 += float(np.sum(_hs[_nz]**2 / _hd[_nz]))
                    # FOM2: Asimov  2*[(S+D)*ln(1+S/D) - S]
                    _ratio = np.where(_nz, _hs / _hd, 0.0)
                    _f2 += float(2.0 * np.sum(np.where(_nz, (_hs + _hd) * np.log1p(_ratio) - _hs, 0.0)))
                    # FOM3: S^2 / (D + f^2*D^2)
                    _f3 += float(np.sum(np.where(_nz, _hs**2 / (_hd + _S4_F_SYST**2 * _hd**2), 0.0)))
                _rows.append((b1, b2, b3, b4, np.sqrt(_f1), np.sqrt(max(_f2, 0.0)), np.sqrt(_f3)))
            print(f"  Scan done in {time.time()-_t0_scan:.2f}s\n")

            # Sort by FOM2 (Asimov — most physically motivated)
            _rows.sort(key=lambda r: r[5], reverse=True)

            # Best values for each FOM
            _best1 = max(_rows, key=lambda r: r[4])[4]
            _best2 = _rows[0][5]   # already sorted by FOM2
            _best3 = max(_rows, key=lambda r: r[6])[6]

            # ── Print header ───────────────────────────────────────────────────
            print("=" * 80)
            print(f"3-FOM comparison  ({_S4_N_MSCORE} Mscore bins in [{_S4_MSCORE_LO},1.0], 4 SR X-bins)")
            print(f"  FOM1 = sqrt(sum S^2/D)                  [S/sqrt(D) quadrature]")
            print(f"  FOM2 = sqrt(2*sum[(S+D)*ln(1+S/D)-S])   [Asimov likelihood ratio]")
            print(f"  FOM3 = sqrt(sum S^2/(D+f^2*D^2)) f={_S4_F_SYST:.0%}  [Asimov + {_S4_F_SYST:.0%} bkg syst]")
            print(f"channel={channel}   {len(_rows)} combos, ranked by FOM2")
            print("=" * 80)
            print(f"{'rank':>4}  {'b1':>7}  {'b2':>7}  {'b3':>7}  {'b4':>7}  {'loss1%':>8}  {'loss2%':>8}  {'loss3%':>8}")
            print("-" * 76)
            for _rk, (b1, b2, b3, b4, _f1, _f2, _f3) in enumerate(_rows, 1):
                if _rk > 20:
                    break
                _l1 = 100.0 * (_best1 - _f1) / _best1
                _l2 = 100.0 * (_best2 - _f2) / _best2
                _l3 = 100.0 * (_best3 - _f3) / _best3
                _mark = " <--" if _rk == 1 else ""
                print(f"{_rk:>4}  {b1:>7.4f}  {b2:>7.4f}  {b3:>7.4f}  {b4:>7.4f}"
                      f"  {_l1:>7.3f}%  {_l2:>7.3f}%  {_l3:>7.3f}%{_mark}")
            if len(_rows) > 20:
                print(f"  ... ({len(_rows) - 20} more combos not shown)")
            print()

            # ── Best per FOM ───────────────────────────────────────────────────
            _best_r1 = max(_rows, key=lambda r: r[4])
            _best_r2 = _rows[0]
            _best_r3 = max(_rows, key=lambda r: r[6])
            print(f"Best FOM1: ({_best_r1[0]:.4f}, {_best_r1[1]:.4f}, {_best_r1[2]:.4f}, {_best_r1[3]:.4f})")
            print(f"Best FOM2: ({_best_r2[0]:.4f}, {_best_r2[1]:.4f}, {_best_r2[2]:.4f}, {_best_r2[3]:.4f})")
            print(f"Best FOM3: ({_best_r3[0]:.4f}, {_best_r3[1]:.4f}, {_best_r3[2]:.4f}, {_best_r3[3]:.4f})")
            print()

            # ── 2D slice for FOM2 best (b1, b2) ───────────────────────────────
            _bb1, _bb2 = _best_r2[0], _best_r2[1]
            _slice = [(b3, b4, f1, f2, f3) for b1, b2, b3, b4, f1, f2, f3 in _rows
                      if b1 == _bb1 and b2 == _bb2]
            if len(_slice) > 1:
                _b3v = sorted(set(r[0] for r in _slice))
                _b4v = sorted(set(r[1] for r in _slice))
                _slice_hdr = "b3 \\ b4"
                print(f"2D slice (FOM2 loss%): b1={_bb1:.4f}  b2={_bb2:.4f}")
                print(f"{_slice_hdr:>10}" + "".join(f"  {v:>9.4f}" for v in _b4v))
                print("-" * (10 + 12 * len(_b4v)))
                for _b3 in _b3v:
                    _rstr = f"  {_b3:>8.4f}"
                    for _b4 in _b4v:
                        _entry = next((r for r in _slice if r[0] == _b3 and r[1] == _b4), None)
                        if _entry is not None:
                            _rstr += f"  {100.0*(_best2-_entry[3])/_best2:>8.3f}%"
                        else:
                            _rstr += "        n/a"
                    print(_rstr)
                print()

            # ── Append best (FOM2) to scan4_results.txt ───────────────────────
            _ts  = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            _hdr = not os.path.exists(_S4_RESULTS)
            with open(_S4_RESULTS, "a") as _rf:
                if _hdr:
                    _rf.write("# scan4 results (3 FOMs, 50 Mscore bins) -- appended across runs\n")
                    _rf.write(f"# {'timestamp':<18}  {'ch':<4}  {'b1':>7}  {'b2':>7}  {'b3':>7}  {'b4':>7}"
                              f"  {'FOM1':>12}  {'FOM2':>12}  {'FOM3':>12}  {'n':>6}\n")
                    _rf.write("#" + "-" * 100 + "\n")
                _rf.write(f"  {_ts:<18}  {channel:<4}"
                          f"  {_best_r2[0]:>7.4f}  {_best_r2[1]:>7.4f}  {_best_r2[2]:>7.4f}  {_best_r2[3]:>7.4f}"
                          f"  {_best_r2[4]:>12.3f}  {_best_r2[5]:>12.3f}  {_best_r2[6]:>12.3f}  {len(_rows):>6}\n")
            print(f"Appended to {_S4_RESULTS}")
            print(f"  Best (FOM2): ({_best_r2[0]:.4f}, {_best_r2[1]:.4f}, {_best_r2[2]:.4f}, {_best_r2[3]:.4f})"
                  f"  FOM1={_best_r2[4]:.1f}  FOM2={_best_r2[5]:.1f}  FOM3={_best_r2[6]:.1f}")

    # ------- Summary ----------
    total_time = time.time() - total_start_time
    total_mins, total_secs = divmod(total_time, 60)
    print(f"------- Done: {len(saved_files)} plots in {int(total_mins)}m {int(total_secs)}s ----------")
