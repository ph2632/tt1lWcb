#!/usr/bin/env python3
"""
PyROOT / CMS-style figures for the S1 boosted-cb tagger.

Reads only summary.json + the per-model eval.npz / cmp_*.npz written by
02_train_tagger.py, so it runs under an LCG view (ROOT) while training runs
in the venv.

Step 3/3 of the S1 tagger pipeline: 01_build_trainset.py -> 02_train_tagger.py
-> 03_render_report.py (or run_s1_tagger.py to drive all three).

S1' was dropped as a carried tagger (substructure added ~0 AUC), so everything
here is S1-only.

Figures (all square canvases, frameless legends, CMS Simulation + lumi):
  roc.png             raw-sum baselines / old Dbc / 3-class Dbc / S1
  overtrain_S1.png    train (filled) vs test (points) + KS and |t-tr|/t
  permimp_S1.png      permutation importance, one bar per input
  score_S1.png        score by W->cb topology + ratio pad (component/total)
  panel_S1.png        2x2 montage of the four; the four singles are KEPT as
                      their own png files and published alongside it

Run (from anywhere, any shell -- it re-execs itself under an LCG view if
PyROOT is not already importable; override the view with $S1_LCG_VIEW):

  python3 03_render_report.py [--config S1_tagger/config.json]
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# LCG view providing PyROOT.  Override with $S1_LCG_VIEW.
LCG_VIEW = os.environ.get(
    "S1_LCG_VIEW", "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt")


def _bootstrap_lcg():
    """Re-exec under the LCG view when PyROOT is missing.

    The login shell here is tcsh but LCG ships a bash setup.sh, so
    'source setup.sh && python3 ...' aborts before running anything.  Rather
    than depend on the caller's shell, sourcing is done in an explicit bash
    subshell -- so plain 'python 03_render_report.py' works from tcsh, bash,
    or the project venv.
    """
    try:
        import ROOT  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_S1_LCG_REEXEC"):        # already tried -- don't loop
        raise SystemExit(
            "[03_render_report] PyROOT still missing after sourcing "
            f"{LCG_VIEW}. Set $S1_LCG_VIEW to a view that provides ROOT.")
    setup = Path(LCG_VIEW) / "setup.sh"
    if not setup.exists():
        raise SystemExit(
            f"[03_render_report] no PyROOT and no LCG view at {setup}. "
            "Set $S1_LCG_VIEW to an LCG view that provides ROOT.")
    print(f"[03_render_report] no PyROOT here -- re-exec under {LCG_VIEW}",
          flush=True)
    os.environ["_S1_LCG_REEXEC"] = "1"
    argv = " ".join(shlex.quote(a) for a in sys.argv)
    os.execvp("bash", ["bash", "-c",
                       f"source {shlex.quote(str(setup))} >/dev/null 2>&1; "
                       f"exec python3 {argv}"])


_bootstrap_lcg()


def montage(args):
    """Use the system ImageMagick with a sanitised environment -- the copy on
    an LCG view has no magic.xml/type.xml and fails on any font."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("LD_LIBRARY_PATH", "MAGICK_HOME", "MAGICK_CONFIGURE_PATH")}
    subprocess.run(["/usr/bin/montage", *args], check=True, env=env)

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

HERE = Path(__file__).resolve().parent      # tt1lWcb/ root -- where panels publish
PKG = HERE / "S1_tagger"                    # config.json / output live here
GALLERY = HERE


def _done_banner(t0, output):
    """One unmissable line at the very end of a run: wall time + where the
    output landed (2026-09-13, user -- the per-step timers were easy to miss
    scrolled past in a long log)."""
    line = f"[DONE  time {(time.time() - t0) / 60.0:.1f} min  output: {output}]"
    print("-" * len(line)); print(line); print("-" * len(line))
SQ = 900                                   # square canvas side
TXT = 0.029                                # one common in-plot text size
MODEL = "S1"                               # S1' dropped
PRETTY = {"S1": "S_{1}(bc)"}
# One distinct hue per ROC curve: grey/black for the two untrained raw sums,
# then green / red / purple for the reference taggers and blue / orange / cyan
# for S1,S2,S3.  No hue is reused, so a curve is identifiable by colour alone.
COL = {"S1": ROOT.kAzure + 1,
       "raw_cb": ROOT.kGray + 1,
       "raw_cb_bb": ROOT.kGray + 2,
       "raw_sum_bc_bb_topbwc": ROOT.kGray + 3,
       "Dbc_old_8node": ROOT.kGreen + 2,
       "Dbc3_bc": ROOT.kRed + 1, "Dbc3_bb": ROOT.kMagenta + 2,
       "Dbc3_bc_plus_bb": ROOT.kViolet + 1,
       "S2": ROOT.kOrange + 7, "S3": ROOT.kCyan + 2, "S4": ROOT.kTeal + 4}
CMPLAB = {"raw_cb": "cb raw",
          "raw_cb_bb": "cb+bb raw",
          "raw_sum_bc_bb_topbwc": "cb+bb+t^{2}(b'c) raw",
          "Dbc_old_8node": "D_{cb} 2cl, 8 inputs",
          "Dbc3_bc": "D_{cb} 3cl, 30 inputs",
          "Dbc3_bb": "D_{bb} 3cl",
          "Dbc3_bc_plus_bb": "D_{cb}+D_{bb} 3cl"}
# legend order, top to bottom; S1/S2/S3 are appended after these
CMP_ORDER = ["raw_cb", "raw_sum_bc_bb_topbwc", "Dbc_old_8node",
             "Dbc3_bc", "Dbc3_bc_plus_bb"]
CMP_STYLE = {"raw_cb": 2, "raw_cb_bb": 7, "raw_sum_bc_bb_topbwc": 7,
             "Dbc_old_8node": 1, "Dbc3_bc": 1, "Dbc3_bb": 1,
             "Dbc3_bc_plus_bb": 1,
             "S2": 1, "S3": 1}

# 2026-09-15, user: "S1, S2, S3 have no other definitions" -- the restricted-
# signal-subset overlay this used to draw (S2 = cb+bb subset, S3 = cb-only
# subset, both re-scorings of the SAME S1 model, no retrain) is RETIRED. It
# predates M3 and collided with the now-established meaning of S1/S2/S3
# (the three separate M3-based taggers, S1(bc)/S2(bb)/S3(bbc)) -- e.g. the
# overtrain panel's 3rd row used to read "S3: cb" where "S3" now means the
# bbc tagger elsewhere in this same report. SUBSET_DEF/SUBSET_LAB/
# SUBSET_ORDER kept only as empty/trivial stubs so any stray reference
# fails loudly rather than silently reintroducing the old curves.
SUBSET_DEF = {}
SUBSET_LAB = {}
SUBSET_ORDER = []
# the merged panel used to overlay 3 signal definitions (S1/S2/S3 subsets)
# as red curves distinguished by line style; only S1(bc) itself remains.
DEF_STYLE = {"S1": 1}
DEF_MARKER = {"S1": 20}
# 2026-09-15, user: relabelled to the actual training target -- S1's own
# training uses y = (topology > 0), i.e. ALL 7 non-bkg topologies count as
# signal (cb, b'c, b'c-prx, b'b, b'bc, Zbb, QCDbb), not a named subset.
DEF_LAB = {"S1": "#splitline{( cb, b'c, b'c-prx,}{b'b, b'bc, Zbb, QCDbb ) jets}"}

# Tagger-input labels, in the 04_Make_plots.py notation.  These are the
# MASS-DECORRELATED ak8_gpt_* scores of a generic massive resonance X, so
# they carry no parent-flavour assumption (W->cb, H+->cb, ... all alike).
# Left column below is the exact ntuple branch suffix.
NODE_LABEL = {
    # 2-prong nodes are defined on a GENERIC resonance X (no parent-flavour
    # assumption -> X(bc) covers W->cb and H+->cb alike).  The top / W / QCD
    # nodes below are defined on the actual SM particles, so X(cs) and W(cs)
    # are genuinely different hypotheses, not a relabelling of one another.
    "bb": "bb", "cc": "cc", "bc": "bc", "bs": "bs", "cs": "cs",
    "qq": "qq", "ss": "ss", "tauhtauh": "#tau_{h}#tau_{h}",
    "qcdb": "QCD(b)", "qcdbb": "QCD(bb)", "qcdc": "QCD(c)",
    "qcdcc": "QCD(cc)", "qcdothers": "QCD(oth)",
    "qcd": "QCD(sum)",   # parent softmax sum of the 5 above (2026-09-13, +feature)
    # t -> b W with the b IN the jet
    "topbwc": "t^{2}(b'c)", "topbwq": "t^{2}(b'q)", "topbws": "t^{2}(b's)",
    "topbwqq": "t^{3}(b'qq)", "topbwcs": "t^{3}(b'cs)",
    "topbwev": "t^{2}(b'e#nu)", "topbwmv": "t^{2}(b'#mu#nu)",
    "topbwtauev": "t^{2}(b'#tau_{e}#nu)",
    "topbwtauhv": "t^{2}(b'#tau_{h}#nu)",
    "topbwtaumv": "t^{2}(b'#tau_{#mu}#nu)",
    # t -> W with the b OUTSIDE the jet: only the W decay products merge
    "topwqq": "W(qq)", "topwcs": "W(cs)",
    "topwev": "W(e#nu)", "topwmv": "W(#mu#nu)",
    "topwtauev": "W(#tau_{e}#nu)", "topwtauhv": "W(#tau_{h}#nu)",
    "topwtaumv": "W(#tau_{#mu}#nu)",
    # the 10 leptonic top/W categories, merged into one input at train time
    "lepq": "t(lq)",
    "ak8_tau21": "#tau_{21}", "ak8_tau32": "#tau_{32}",
    "ak8_nConstituents": "N_{constit}",
}


def node_label(feat):
    return NODE_LABEL.get(feat.replace("ak8_gpt_", ""), feat.replace("ak8_gpt_", ""))


TOPO_COL = {1: ROOT.kAzure + 2, 2: ROOT.kOrange + 7,
            3: ROOT.kGreen + 2, 4: ROOT.kMagenta + 1, 5: ROOT.kCyan + 2,
            6: ROOT.kBlue + 2, 7: ROOT.kRed + 1}
# short forms (no parens) for Wcb/t2b'b/Zbb, 2026-09-13 -- saves legend width
TOPO_LAB = {1: "Wcb", 2: "t^{2}(b'c)", 3: "t^{2}b'b", 4: "t^{3}(b'bc)",
            5: "t^{2}(b'c) proxy", 6: "Zbb adj", 7: "QCD(bb)"}
# M3 (multiclass) per-class colour = the colour of that group's eponymous
# topology above (cb -> Wcb's azure, bb -> t2(b'b)'s green, bbc -> t3(b'bc)'s
# magenta), so M3 plots read consistently with every other plot in this file.
# 2026-09-16, user: "align the color [of overtrain_M3_cb] to be as [roc_M3's
# D_cb curve] color(4)" -- cb overridden to plain ROOT colour 4, matching
# m3_roc_figure's _DARK curve.
M3_COL = {"cb": 4, "bb": TOPO_COL[3], "bbc": TOPO_COL[4]}
# 2026-09-14, user: "S1, Dbc, Dbc(J), Dbb are unclear -- there should be 3
# taggers only: S1(bc), S2(bb), S3(bbc)". Internal keys (cname: cb/bb/bbc,
# used for dict lookups and file names) are UNCHANGED; only the human-
# readable text drawn ON the plots switches to this naming.
# 2026-09-15, user: "S1/S2/S3" collided with the (now-dropped) binary
# tagger's own name "S1", which was a genuine source of confusion across
# this whole session -- renamed to D_cb/D_bb/D_bbc, "S1/S2/S3" retired.
M3_DISP = {"cb": "D_{cb}", "bb": "D_{bb}", "bbc": "D_{bbc}"}
# 2026-09-16, user: the physics components each M3 class actually targets --
# used to spell out "rest" as the OTHER 2 classes + non-targeted jets, not a
# vague "bkg", on the per-class overtraining plots.
M3_COMPONENTS = {"cb": "cb, b'c, b'c-prx", "bb": "b'b, Zbb, QCDbb", "bbc": "b'bc"}
def _m3_rest_components(cname):
    # 2026-09-16, user: "there is no true-BKG, better to say non-targeted
    # jets" -- topology 0 is just jets that match none of the 3 targeted
    # classes, not a separate "background" sample.
    others = ", ".join(M3_COMPONENTS[c] for c in ("cb", "bb", "bbc") if c != cname)
    return "non-targeted jets, " + others

TOP_MARGIN = 0.065     # all four figures (was 0.085)
RIGHT_MARGIN = 0.025   # all four figures (was 0.035)
# Axis tick-NUMBER size.  ROOT scales text by PAD height, so a pad that is
# only a fraction of the canvas needs LSIZE/fraction to come out the same
# physical size -- that is why the score figure's two pads divide by 0.70/0.30.
LSIZE = 0.034
P1_H, P2_H = 0.5, 0.5   # score figure: upper / ratio pad heights
# ratio-pad axis-title sizes quoted as PHYSICAL fractions of the canvas,
# divided by the pad height at use -- so they stay the same size on screen
# if the pad split changes
TSIZE_X_RATIO, TSIZE_Y_RATIO = 0.0394, 0.0348


def cms_style():
    s = ROOT.TStyle("cms", "cms")
    s.SetCanvasBorderMode(0); s.SetCanvasColor(0)
    s.SetPadBorderMode(0); s.SetPadColor(0)
    s.SetFrameBorderMode(0); s.SetFrameFillColor(0)
    s.SetOptStat(0); s.SetOptTitle(0)
    s.SetPadTickX(1); s.SetPadTickY(1)
    s.SetPadLeftMargin(0.125); s.SetPadRightMargin(RIGHT_MARGIN)
    s.SetPadTopMargin(TOP_MARGIN); s.SetPadBottomMargin(0.115)
    s.SetTextFont(42); s.SetLabelFont(42, "XYZ"); s.SetTitleFont(42, "XYZ")
    s.SetLabelSize(0.038, "XYZ"); s.SetTitleSize(0.045, "XYZ")
    s.SetTitleOffset(1.25, "X"); s.SetTitleOffset(1.45, "Y")
    s.SetLegendBorderSize(0); s.SetLegendFillColor(0)
    ROOT.gROOT.SetStyle("cms"); ROOT.gROOT.ForceStyle()


MARGIN = 0.130       # equal on all four sides -> the frame box is square
MARGIN_RANK = 0.160  # ranking plot: the node names need a little more room
CMS_X = 0.165        # canvas NDC (shifted ~5% left)
CMS_GAP = 0.105      # "CMS" -> "Simulation" gap (was 0.082)
CMS_PX_TOP = 46.0    # header baseline, pixels below the canvas top
CMS_SIZE = 1.22      # common scale for the "CMS Simulation ... (13 TeV)" line
CMS_H_REF = 900.0    # canvas height the sizes are quoted for


def cms_header(cfg, canvas=None, dx=0.0, dy=0.0, scale=CMS_SIZE, lumi_dx=0.0):
    """'CMS Simulation' (left) and the lumi (right) on ONE baseline.

    "Simulation" and the lumi share the same font size; only "CMS" is bold
    and larger.  Sizes are converted through pixels so the header is
    physically identical on canvases of different height (the score figure
    is taller because it carries a ratio pad).
    """
    if canvas is not None:
        canvas.cd()
    h = float(canvas.GetWh()) if canvas is not None else CMS_H_REF
    f = CMS_H_REF / h
    y = 1.0 - CMS_PX_TOP / h + dy
    small = 0.032 * scale * f            # "Simulation" and lumi: identical size
    t = ROOT.TLatex(); t.SetNDC()
    t.SetTextFont(61); t.SetTextSize(0.040 * scale * f)
    t.DrawLatex(CMS_X + dx, y, "CMS")
    t.SetTextFont(52); t.SetTextSize(small)
    t.DrawLatex(CMS_X + dx + CMS_GAP * scale, y, cfg.get("cms_label", "Simulation"))
    t.SetTextFont(42); t.SetTextSize(small); t.SetTextAlign(31)
    t.DrawLatex(0.965 + lumi_dx, y, cfg.get("lumi_label", "41.5 fb^{-1} (13 TeV)"))
    ROOT.SetOwnership(t, False)
    return t


def _axes(frame, xt, yt, tsize=0.042, lsize=LSIZE, xoff=1.30, yoff=1.30):
    """DrawFrame() ignores most style settings -- set them explicitly."""
    for ax, title, off in ((frame.GetXaxis(), xt, xoff), (frame.GetYaxis(), yt, yoff)):
        ax.SetTitle(title)
        ax.SetTitleSize(tsize); ax.SetLabelSize(lsize); ax.SetTitleOffset(off)
    return frame


def _legend(x1, y1, x2, y2, size):
    lg = ROOT.TLegend(x1, y1, x2, y2)
    lg.SetBorderSize(0); lg.SetFillStyle(0); lg.SetTextFont(42)
    lg.SetTextSize(size)
    ROOT.SetOwnership(lg, False)
    return lg


def sig_2bin(score, topo, w, sig_classes, nbins=100):
    """Best 2-category selection at high score, maximising the quadrature sum
    of S/sqrt(S+B) -- the same construction as ../Hgg/Root_plot.py
    optimize_n_bins(2), but with both outer edges free so the low-score bulk
    can be discarded.

    Class 5 (t2(b'c) proxy) counts as BACKGROUND: those jets are real W->cs
    background events, promoted to a signal class only to shape the training
    loss.  Counting them as signal would make this metric 99% proxy.

    This is a JET-level figure of merit for ranking taggers -- no event
    selection, no systematics -- not the analysis SR significance.

    Returns (significance, lower score threshold of the selection).
    """
    s_m = np.isin(topo, sig_classes)
    # 2026-09-15 bugfix: topo==5 (proxy) used to be unconditionally counted
    # as background even when the caller explicitly put 5 IN sig_classes
    # (the new "cb,b'c,b'c-prx" curve) -- double-counting those jets into
    # both hs and hb and giving a meaningless significance for that curve.
    # Exclude anything already claimed by s_m.
    b_m = ((topo == 0) | (topo == 5)) & ~s_m
    edges = np.linspace(0.0, 1.0, nbins + 1)
    hs, _ = np.histogram(score[s_m], bins=edges, weights=w[s_m])
    hb, _ = np.histogram(score[b_m], bins=edges, weights=w[b_m])
    # cs[i] / cb[i] = yield summed over bins i .. nbins-1
    cs = np.concatenate([np.cumsum(hs[::-1])[::-1], [0.0]])
    cb = np.concatenate([np.cumsum(hb[::-1])[::-1], [0.0]])
    best_z, best_lo = 0.0, None
    for j in range(1, nbins + 1):                     # upper cat = bins j..end
        s_hi, b_hi = cs[j], cb[j]
        if b_hi < 1.0 or s_hi <= 0:
            continue
        z_hi = s_hi / np.sqrt(s_hi + b_hi)
        for i in range(j):                            # lower cat = bins i..j-1
            s_lo, b_lo = cs[i] - s_hi, cb[i] - b_hi
            if b_lo < 1.0 or s_lo <= 0:
                continue
            z = float(np.hypot(z_hi, s_lo / np.sqrt(s_lo + b_lo)))
            if z > best_z:
                best_z, best_lo = z, float(edges[i])
    return best_z, best_lo


def best_1bin_cut(score, w, sig_mask, bkg_mask, nbins=100):
    """Single-threshold optimum: the score>=thr cut that maximises
    S/sqrt(S+B) (S,B = weighted yields passing the cut). Simpler than
    sig_2bin's 2-category FOM -- one cut, one number, for reading a
    working point straight off a score distribution plot.

    Returns (best threshold, eff_sig at that threshold, eff_bkg at that
    threshold).
    """
    edges = np.linspace(0.0, 1.0, nbins + 1)
    hs, _ = np.histogram(score[sig_mask], bins=edges, weights=w[sig_mask])
    hb, _ = np.histogram(score[bkg_mask], bins=edges, weights=w[bkg_mask])
    cs = np.concatenate([np.cumsum(hs[::-1])[::-1], [0.0]])   # yield >= edges[j]
    cb = np.concatenate([np.cumsum(hb[::-1])[::-1], [0.0]])
    s_tot, b_tot = cs[0], cb[0]
    best_z, best_thr = -1.0, 0.5
    for j in range(1, nbins):
        s, b = cs[j], cb[j]
        if s <= 0:
            continue
        z = s / np.sqrt(s + b) if (s + b) > 0 else 0.0
        if z > best_z:
            best_z, best_thr = z, float(edges[j])
    eff_s = cs[np.searchsorted(edges, best_thr)] / s_tot if s_tot > 0 else 0.0
    eff_b = cb[np.searchsorted(edges, best_thr)] / b_tot if b_tot > 0 else 0.0
    return best_thr, float(eff_s), float(eff_b)


def _roc_points(y, s, w, npts=400):
    """Weighted ROC: signal efficiency vs background efficiency, plus AUC."""
    o = np.argsort(-s)
    y, w = y[o], w[o]
    cs = np.cumsum(w * (y == 1)); cb = np.cumsum(w * (y == 0))
    S, B = cs[-1], cb[-1]
    tpr, fpr = cs / S, cb / B
    auc = float(np.trapz(tpr, fpr))
    k = max(1, len(tpr) // npts)
    return tpr[::k].astype(float), fpr[::k].astype(float), auc


# --------------------------------------------------------------------------- #
def roc_figure(outdir, summary, cfg, out_png, presel=None):
    c = ROOT.TCanvas("roc", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN - 0.010); c.SetLeftMargin(0.105)
    c.SetRightMargin(0.018); c.SetBottomMargin(0.085)
    c.SetLogy(); c.SetGridx(); c.SetGridy()
    # 2026-09-15, user: axis titles corrected -- S1's positive class is NOT
    # cb-only. Checked directly (train_one() in 02_train_tagger.py: y =
    # topology>0, ALL 7 signal-like topologies -- Wcb/t2bc/t2bb/t3bbc/
    # PROXY/Zbb/QCDbb), so "cb-jets" would be actively wrong here (that
    # narrower definition is what M3's S1(bc) class means, a different
    # model). Background is topology==0, any non-signal-matched jet.
    frame = _axes(c.DrawFrame(0.0, 1e-3, 1.0, 0.3),
                  "targeted jets eff.", "Other than targeted jets eff.",
                  xoff=0.96, yoff=1.06)
    frame.GetYaxis().SetLabelOffset(0.005)   # tick numbers 1% closer

    # 2026-09-15 bugfix, user: these 2 curves' LABELS were updated to the
    # exact "S1(cb,b'c,b'c-prx/rest)" / "S1(cb,b'c/rest)" notation, but the
    # underlying y-arrays were never updated to match -- curve 1 was still
    # using e["test_y"] (S1's full 7-topology training target) and curve 2
    # was using "topo!=5" (6 topologies: 1,2,3,4,6,7), NEITHER of which is
    # what their labels claimed ((1,2,5) and (1,2) respectively). Confirmed
    # by the AUC mismatch against roc_M3.png's independently-correct
    # (1,2,5) curve (0.910 here vs 0.886 there for what should be the same
    # definition). Fixed: both now built from topo_e via np.isin, matching
    # their labels exactly.
    # Significance: curve 1 (proxy counted as signal) drops its S/sqrt(B+S)
    # entirely (user: "the 35.2 [now: whatever] significance is misleading
    # due to the proxy") -- same fix already applied in roc_M3.png.
    # entry = (label, colour, style, y, score, w, sig_cls, full score).
    entries = []
    f = outdir / MODEL / "eval.npz"
    if f.exists():
        e = np.load(f)
        topo_e = e["test_topo"]
        y_prx = np.isin(topo_e, (1, 2, 5)).astype(np.int8)
        entries.append(("D_{cb}( cb, b'c, b'c-prx / rest )", COL[MODEL], 1,
                        y_prx, e["test_score"], e["test_w"],
                        None, e["test_score"]))
        y_noprx = np.isin(topo_e, (1, 2)).astype(np.int8)
        entries.append(("D_{cb}( cb, b'c / rest )", COL[MODEL], 7,
                        y_noprx, e["test_score"], e["test_w"],
                        (1, 2), e["test_score"]))

    # 2026-09-15, user: switched to the long "S1(cb,b'c,b'c-prx/rest)"-style
    # notation. bugfix: an earlier attempt widened this box far enough LEFT
    # (to 0.280) that its white background painted over part of the actual
    # curve (which dips down near the bottom of the frame for signal-eff
    # ~0.1-0.25) -- kept the RIGHT edge anchored at the known-good 0.810
    # (same as x_auc below) and only extended left modestly + shrunk the
    # font instead, so the box stays clear of the curve.
    lg = _legend(0.430, 0.335, 0.810, 0.445, 0.0225)
    lg.SetFillStyle(1001); lg.SetFillColor(ROOT.kWhite); lg.SetMargin(0.16)
    keep, rows = [], []
    topo_all = e["test_topo"] if f.exists() else None

    def sig3(x):
        """3 significant digits regardless of magnitude (2026-09-15, user:
        was a fixed 2 or 4 decimals depending on the code path -- 23.69 has
        4 sig figs, 0.910 has 3; %.3g gives 3 always)."""
        return f"{x:.3g}"

    for lab, col, ls, y, s, w, cls, sig_s in entries:
        tpr, fpr, auc = _roc_points(y, s, w)
        g = ROOT.TGraph(len(tpr), tpr, fpr)
        g.SetLineColor(col); g.SetLineWidth(3); g.SetLineStyle(ls)
        g.Draw("L SAME"); keep.append(g)
        text = lab
        # AUC is always <1 -- fixed .3f (always 3 decimals) reads as clean
        # 3 sig figs; %.3g would drop trailing zeros inconsistently between
        # rows (0.910 -> "0.91" but 0.908 stays "0.908", visually uneven).
        rows.append([lab, f"{auc:.3f}", ""])
        if cls is not None and topo_all is not None:
            # significance is evaluated on the FULL test set (proxy in B);
            # the marker is placed on this curve, so its efficiencies use the
            # same arrays the curve was drawn from.
            z, lo_thr = sig_2bin(sig_s, topo_all, e["test_w"], cls)
            if lo_thr is not None:
                rows[-1][2] = "#color[%d]{%s}" % (col, sig3(z))
                sel = s >= lo_thr
                eff_s = w[(y == 1) & sel].sum() / max(w[y == 1].sum(), 1e-12)
                eff_b = w[(y == 0) & sel].sum() / max(w[y == 0].sum(), 1e-12)
                mk = ROOT.TMarker(eff_s, eff_b, 20)
                mk.SetMarkerColor(col); mk.SetMarkerSize(1.7)
                mk.Draw(); keep.append(mk)
        lg.AddEntry(g, text, "l")
    # 2026-09-15, user: "text on the right of the legend closer to the
    # legend" -- x_auc sits flush with the legend box's own right edge
    # (0.810); x_z pulled in from the old 0.895 to 0.885 (kept wide enough
    # that "AUC"/"S/sqrt(B+S)" headers don't collide with each other --
    # 0.875 was tried first and was too tight for that).
    # "shift colored text 1% right" was tried at x_z=0.895 but the
    # "S/sqrt(B+S)" HEADER (wider than any value in the column) then clips
    # against the canvas edge -- reverted to the known-good 0.885.
    x_auc, x_z = 0.810, 0.885
    pvw = ROOT.TPave(0.425, 0.330, 0.955, 0.485, 0, "NDC")
    pvw.SetFillColor(ROOT.kWhite); pvw.SetFillStyle(1001); pvw.SetBorderSize(0)
    pvw.Draw(); ROOT.SetOwnership(pvw, False)
    lg.Draw()
    cms_header(cfg, c, dx=-0.06, dy=0.010, lumi_dx=-0.01)
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.0303)
    # AUC / significance as columns on the legend's own row grid, so the
    # numbers line up regardless of how long each label is
    y1, y2, n = 0.335, 0.445, len(rows)
    t.DrawLatex(x_auc, y2 + 0.012, "AUC")               # row shifted 1% lower
    # 2026-09-15 bugfix: this header used to be drawn at x_z-0.020 while the
    # VALUE rows below use x_z directly -- at the new tighter x_auc/x_z
    # spacing that made "AUC" and "S/sqrt(B+S)" run into each other. Use the
    # same x_z the values use; they don't collide, so the header won't either.
    # 2026-09-16 bugfix, user: "Signif.@WP" is wider than the old
    # "S/sqrt(B+S)" was at this font size -- it clipped past the canvas
    # right edge. Header only (not the value rows) drawn smaller.
    t.SetTextSize(0.0303 * 0.80)
    t.DrawLatex(x_z, y2 + 0.012, "Signif.@WP")
    t.SetTextSize(0.0303)
    for i, (_, a, z) in enumerate(rows):
        yy = y2 - (i + 0.72) * (y2 - y1) / n
        t.DrawLatex(x_auc, yy, a)
        if z:
            t.DrawLatex(x_z, yy, z)
    # preselection, in the empty upper-left corner
    if presel:
        # 2026-09-15, user: was light grey + Courier-Italic to "set this
        # block visually apart" -- now plain black, same font (42) as the
        # rest of the plot's text, per instruction. Still +5% size.
        t.SetTextSize(TXT * 1.10 * 1.05)
        t.SetTextColor(ROOT.kBlack)
        t.SetTextFont(42)
        lines = ["#bf{Preselection}",
                 "p_{T} > %.0f GeV" % presel["jet_pt_min"],
                 "m_{SD} > %.0f GeV" % presel["jet_sdmass_min"],
                 "#tau_{21} < %.2f" % presel["jet_tau21_max"],
                 "#DeltaR(l, J) > %.1f" % presel["dr_lep_jet_min"]]
        for i, ln in enumerate(lines):
            t.DrawLatex(0.165, 0.880 - 0.044 * i, ln)
        t.SetTextColor(ROOT.kBlack)
        t.SetTextFont(42)
    ROOT.SetOwnership(t, False)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return keep


def subset_overtrain(tr_s, tr_w, s_tr, b_tr, te_s, te_w, s_te, b_te, cut):
    """bias / KS for one signal-class subset, same definitions as 02_train_tagger.py."""
    out = {"score_cut": float(cut)}
    worst = 0.0
    for m_tr, m_te, tg in ((s_tr, s_te, "sig"), (b_tr, b_te, "bkg")):
        f_tr = tr_w[m_tr & (tr_s > cut)].sum() / max(tr_w[m_tr].sum(), 1e-12)
        f_te = te_w[m_te & (te_s > cut)].sum() / max(te_w[m_te].sum(), 1e-12)
        rel = 100.0 * abs(f_te - f_tr) / max(f_te, 1e-12)
        out[f"rel_diff_{tg}_pct"] = float(rel)
        worst = max(worst, rel)
        from scipy.stats import ks_2samp
        out[f"ks_{tg}_p"] = float(ks_2samp(tr_s[m_tr], te_s[m_te]).pvalue)
    out["verdict"] = "OK" if worst < 5.0 else ("WARN" if worst < 15.0 else "FAIL")
    return out


def overtrain_figure(outdir, name, summary, cfg, out_png, sig_codes,
                     subset=None, tag=""):
    """subset = signal topology codes (S2/S3 panels).  When given, signal is
    restricted to those classes and the background is the true background
    only -- matching how that panel's ROC curve is built.  The bias/KS numbers
    are then recomputed for the subset, since summary.json describes the full
    S1 signal definition."""
    e = np.load(outdir / name / "eval.npz")
    nb, lo, hi = 30, 0.0, 1.0
    w_te = e["test_w_ot"] if "test_w_ot" in e.files else e["test_w"]
    # 2026-09-15, user: bias evaluated at score > 0.5 (was cfg's 0.6) for this
    # plot specifically -- the M3 overtrain panels keep the cfg default.
    OT_SCORE_CUT = 0.5

    if subset is None:
        ot = summary["models"][name]["overtraining"]
        s_tr = e["train_y"] == 1; b_tr = e["train_y"] == 0
        s_te = e["test_y"] == 1;  b_te = e["test_y"] == 0
    else:
        if "train_topo" not in e.files:
            raise SystemExit("[03_render_report] eval.npz has no train_topo -- "
                             "retrain so the S2/S3 panels can be built")
        s_tr = np.isin(e["train_topo"], subset); b_tr = e["train_topo"] == 0
        s_te = np.isin(e["test_topo"], subset);  b_te = e["test_topo"] == 0
        ot = subset_overtrain(e["train_score"], e["train_w"], s_tr, b_tr,
                              e["test_score"], w_te, s_te, b_te,
                              OT_SCORE_CUT)
        # The background is the SAME jets in every panel, but eval.npz keeps
        # only a 400k plotting subsample of the 4.87M training background, so
        # recomputing its bias here is far noisier than 02_train_tagger.py's full-
        # statistics number (and read WARN spuriously).  Take the background
        # from summary.json and keep only the subset-specific signal number.
        full = summary["models"][name]["overtraining"]
        ot["rel_diff_bkg_pct"] = full["rel_diff_bkg_pct"]
        ot["ks_bkg_p"] = full["ks_bkg_p"]
        worst = max(ot["rel_diff_sig_pct"], ot["rel_diff_bkg_pct"])
        ot["verdict"] = "OK" if worst < 5.0 else ("WARN" if worst < 15.0 else "FAIL")
    n_sig = int(s_te.sum()); n_bkg = int(b_te.sum())

    def mk(tg, score, w, m):
        h = ROOT.TH1F(f"h_{name}{tag}_{tg}", "", nb, lo, hi)
        h.Sumw2()
        for v, ww in zip(score[m], w[m]):
            h.Fill(float(v), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        return h

    # compare like with like: if 02_train_tagger.py stored a class-reweighted test
    # weight, the test shape must use it or the signal train/test mixtures
    # differ and the curves separate for a reason unrelated to overtraining.
    # one (train, test, numbers) set per signal definition; the background is
    # common to all of them, so it is built once
    # 2026-09-15, user: S2/S3 restricted-subset rows retired, see the
    # SUBSET_* note above -- only S1(bc) itself is drawn now.
    # 2026-09-15 bugfix, user: this used to exclude topology 5 (proxy), but
    # S1's own training target is y=(topology>0) -- ALL 7 non-bkg topologies
    # count as signal (see 02_train_tagger.py:502). Excluding the proxy here
    # made this plot assess a different (narrower) definition than what was
    # actually trained -- now uses every code in sig_codes, unmodified.
    defs = [("S1", tuple(sig_codes))]
    sig_sets, ot_by = [], {}
    for dn, dcls in defs:
        m_tr = np.isin(e["train_topo"], dcls); m_te = np.isin(e["test_topo"], dcls)
        htr = mk(f"str{dn}", e["train_score"], e["train_w"], m_tr)
        hte = mk(f"ste{dn}", e["test_score"], w_te, m_te)
        htr.SetLineColor(4); htr.SetMarkerColor(4)
        hte.SetLineColor(4); hte.SetMarkerColor(4)
        htr.SetLineWidth(3); htr.SetLineStyle(DEF_STYLE[dn])
        hte.SetMarkerStyle(DEF_MARKER[dn]); hte.SetMarkerSize(1.0)
        o = subset_overtrain(e["train_score"], e["train_w"], m_tr, b_tr,
                             e["test_score"], w_te, m_te, b_te,
                             OT_SCORE_CUT)
        # 2026-09-15 bugfix, user: "the bias numerics did not change -- do
        # you evaluate bias for >0.5 indeed?" -- they were right to be
        # suspicious. The BKG row used to be silently overwritten with
        # summary.json's TRAINING-TIME value, which 02_train_tagger.py always
        # computes at cfg's overtrain_score_cut (0.6) -- so BKG stayed frozen
        # at the >0.6 number no matter what OT_SCORE_CUT was set to here,
        # while only the SIG row (computed fresh above, from eval.npz) moved
        # to 0.5 -- and that move happened to round to the same displayed
        # ".1f%%" digit, making the whole row look unchanged.
        # eval.npz's background is only a 400k plotting subsample of the
        # 4.87M full training background, so this BKG number is noisier than
        # the frozen full-stats one was -- but it is now actually evaluated
        # at the cut the header claims, which correctness requires.
        worst = max(o["rel_diff_sig_pct"], o["rel_diff_bkg_pct"])
        o["verdict"] = "OK" if worst < 5.0 else ("WARN" if worst < 15.0 else "FAIL")
        ot_by[dn] = o
        sig_sets.append((dn, htr, hte))
    h_btr = mk("btr", e["train_score"], e["train_w"], b_tr)
    h_bte = mk("bte", e["test_score"], w_te, b_te)
    for h in (h_btr, h_bte):
        h.SetLineColor(ROOT.kGray + 1); h.SetMarkerColor(ROOT.kGray + 1)
    h_btr.SetFillColorAlpha(ROOT.kGray, 0.60); h_btr.SetLineWidth(2)
    h_bte.SetMarkerStyle(21); h_bte.SetMarkerSize(1.0)

    # 2026-09-15, user: y-range set from the actual drawn content -- lowest
    # nonzero bin x0.5 to highest bin x2, instead of a fixed 3e-3/0.5.
    all_hists = [h_btr, h_bte] + [h for _, htr, hte in sig_sets for h in (htr, hte)]
    lo_y_vals, hi_y_vals = [], []
    for h in all_hists:
        for ib in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(ib)
            if v > 0:
                lo_y_vals.append(v); hi_y_vals.append(v)
    y_lo = min(lo_y_vals) * 0.5 if lo_y_vals else 3e-3
    y_hi = max(hi_y_vals) * 2.0 if hi_y_vals else 0.5

    c = ROOT.TCanvas(f"ot_{name}", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN); c.SetRightMargin(0.025)
    c.SetLeftMargin(0.095); c.SetBottomMargin(0.075)
    c.SetTicks(1, 1)            # tickmarks on all four sides, as in the score pads
    c.SetLogy()
    frame = _axes(c.DrawFrame(lo, y_lo, hi, y_hi),
                  "D_{cb}( cb, b'c, b'c-prx, b'b, b'bc, Zbb, QCDbb / rest )",
                  "N_{jets} norm. to 1", tsize=0.036, xoff=1.05, yoff=1.315)
    frame.GetYaxis().SetLabelOffset(0.005)   # tick numbers 1% closer
    frame.GetXaxis().SetLabelOffset(-0.005)  # x-axis tick numbers 1% closer
    h_btr.Draw("HIST SAME"); h_bte.Draw("E1 SAME")
    for _, htr, _ in sig_sets:
        htr.Draw("HIST SAME")
    for _, _, hte in sig_sets:
        hte.Draw("E1 SAME")

    def row_verdict(bias):
        return "OK" if bias < 5.0 else ("WARN" if bias < 15.0 else "FAIL")

    # 2026-09-15, user: replaced the old (symbol-legend + separate TLatex
    # table) split with ONE TLegend of 4 real text entries -- one per
    # histogram actually drawn (test bkg, train bkg, test signal, train
    # signal) -- Test entries carry their own bias/KS/verdict inline; Train
    # entries spell out the topology components they're built from, so the
    # legend alone states exactly what each curve is without a side table.
    bkg_bias = ot_by["S1"]["rel_diff_bkg_pct"]; bkg_ks = ot_by["S1"]["ks_bkg_p"]
    sig_o = ot_by["S1"]; sig_bias = sig_o["rel_diff_sig_pct"]; sig_ks = sig_o["ks_sig_p"]
    sig_comp = "( cb, b'c, b'c-prx, b'b, b'bc, Zbb, QCDbb )"

    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.DrawLatex(0.400, 0.885, "#bf{Overtraining test}   70% train / 15% test")
    ROOT.SetOwnership(t, False)

    _, _htr, _hte = sig_sets[0]
    lg = _legend(0.150, 0.640, 0.960, 0.845, TXT * 0.78)
    lg.SetMargin(0.10)
    lg.AddEntry(h_bte, "Test: Rest jets   (bias %.1f%%, KS %.2f, %s)"
                % (bkg_bias, bkg_ks, row_verdict(bkg_bias)), "p")
    lg.AddEntry(h_btr, "Train: Rest jets", "f")
    lg.AddEntry(_hte, "Test: %s jets   (bias %.1f%%, KS %.2f, %s)"
                % (sig_comp, sig_bias, sig_ks, row_verdict(sig_bias)), "p")
    lg.AddEntry(_htr, "Train: %s jets" % sig_comp, "l")
    lg.Draw()

    cms_header(cfg, c, dx=0.095 - CMS_X, lumi_dx=-0.01)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_btr, h_bte] + [h for _, a, b in sig_sets for h in (a, b)]


def permimp_figure(outdir, name, summary, cfg, out_png, ntop=15,
                   subset=None, tag=""):
    """Two bars per input, both on a common '% of total' scale.

      %Imp   = normalised total gain -- the TMVA convention that
               ../Hgg/Plot.py reads out of the weights file
               (variable Importance x 100).  Sums to 100% by construction
               and is what the ranking uses.
      dAUC%  = permutation importance (AUC drop when the input is shuffled),
               also normalised to 100% so the two are directly comparable.
    """
    m = summary["models"][name]
    gain = {r["feature"]: r["gain"] for r in m["feature_importance"]}
    perm = {r["feature"]: max(r["auc_drop"], 0.0) for r in m["permutation_importance"]}
    g_tot = sum(gain.values()) or 1.0
    p_tot = sum(perm.values()) or 1.0
    imp = {f: 100.0 * g / g_tot for f, g in gain.items()}
    pim = {f: 100.0 * perm.get(f, 0.0) / p_tot for f in gain}

    ranked = sorted(pim, key=lambda f: -pim[f])          # RANK BY dAUC
    top = ranked[:ntop][::-1]
    n = len(top)
    # +1 empty bin at the top (no bar drawn there, bins 1..n unchanged) so the
    # highest-ranked bar isn't flush against the frame -- also opens up room
    # to move the top two info lines up there, see below (2026-09-13, user).
    nb_bins = n + 1

    h_imp = ROOT.TH1F(f"imp_{name}{tag}", "", nb_bins, 0, nb_bins)
    h_prm = ROOT.TH1F(f"prm_{name}{tag}", "", nb_bins, 0, nb_bins)
    # without Sumw2 the bins inherit sqrt(content) errors, which ROOT then
    # paints as a stripe under each bar -- these are plain numbers, not counts
    h_imp.Sumw2(); h_prm.Sumw2()
    for i, f in enumerate(top, 1):
        h_imp.SetBinContent(i, imp[f])
        h_prm.SetBinContent(i, pim[f])
        h_prm.GetXaxis().SetBinLabel(i, node_label(f))
    # thin bars, and the two metrics for the SAME node sit right next to each
    # other so a pair reads as one entry (wide gap only between nodes)
    for h, col, off in ((h_prm, ROOT.kAzure + 1, 0.50),
                        (h_imp, ROOT.kOrange + 1, 0.21)):
        h.SetFillColor(col); h.SetLineColor(col); h.SetFillStyle(1001)
        h.SetBarWidth(0.28); h.SetBarOffset(off); h.SetLineWidth(1)
        h.SetMarkerSize(0)

    c = ROOT.TCanvas(f"pi_{name}{tag}", "", SQ, SQ)
    c.SetLeftMargin(MARGIN_RANK - 0.010); c.SetRightMargin(RIGHT_MARGIN)  # -1% (2026-09-13)
    c.SetTopMargin(TOP_MARGIN); c.SetBottomMargin(0.105)
    h_prm.GetYaxis().SetTitle("Share of total importance   [%]")
    h_prm.GetYaxis().SetTitleSize(0.046); h_prm.GetYaxis().SetLabelSize(LSIZE)
    h_prm.GetYaxis().SetTitleOffset(0.95)
    h_prm.GetXaxis().SetLabelSize(0.050)
    h_prm.GetXaxis().SetLabelOffset(0.005)    # node names, offset from the frame
    h_prm.SetMaximum(1.1 * max(max(imp[f] for f in top), max(pim[f] for f in top)))
    h_prm.SetMinimum(0.0)
    h_prm.Draw("HBAR")
    h_imp.Draw("HBAR SAME")

    lg = _legend(0.618, 0.555, 0.965, 0.670, TXT)
    lg.AddEntry(h_prm, "#DeltaAUC  (ranking)", "f")
    lg.AddEntry(h_imp, "%Imp  gain (XGBoost)", "f")
    lg.Draw()

    ot = m["overtraining"]
    xgb = summary["config"]["xgboost"]
    # signal class weights actually used for this run: prefer the config
    # embedded in summary.json so re-rendering an old run cannot pick up
    # shares from a config that has moved on since.
    share = summary["config"].get("class_weight_share") or \
        cfg.get("class_weight_share", {})
    if subset is not None and share:
        # all five classes stay in the table so the panels line up; the ones
        # this tagger does not target read 0%, and the rest are renormalised
        # over the subset (proportions preserved, sum 100%)
        tot = sum(float(v) for k, v in share.items() if int(k) in subset) or 1.0
        share = {k: (float(v) / tot if int(k) in subset else 0.0)
                 for k, v in share.items()}
    # top 2 lines moved up into the empty top bin's headroom, bottom 2 stay
    # with the class-weight table (2026-09-13, user)
    lines_top = [
        "#bf{%s}  ranked by #DeltaAUC   (%d out of %d input param/s)"
        % (PRETTY[name], n, m["n_features"]),
        "normalised = 100 #times x_{i}/#Sigma_{j}x_{j};  #Sigma#DeltaAUC = %.4f #rightarrow 1%% #equiv %.5f"
        % (p_tot, p_tot / 100.0),
    ]
    lines = [
        "trees %d (stop %d)  depth %d  lr %.2f  colsample %.1f"
        % (xgb["n_estimators"], m["best_iteration"], xgb["max_depth"],
           xgb["learning_rate"], xgb["colsample_bytree"]),
        "AUC  train %.3f  test %.3f    bias B %.1f%%  [%s]"
        % (m["train"]["auc"], m["test"]["auc"], ot["rel_diff_bkg_pct"], ot["verdict"]),
    ]
    t_top = ROOT.TLatex(); t_top.SetNDC(); t_top.SetTextFont(42); t_top.SetTextSize(TXT)
    t_top.SetTextAlign(31)
    for i, ln in enumerate(lines_top):
        t_top.DrawLatex(0.955, 0.905 - 0.036 * i, ln)
    ROOT.SetOwnership(t_top, False)
    # signal class weights FIRST, as an aligned 2-row table (names over
    # values), colour-matched to the topology curves of the score figure so
    # the two panels read together; the run info then follows underneath.
    y_top = 0.325
    if share:
        codes = sorted(share, key=int)
        # widened span + smaller text (2026-09-13, user) -- the last 3 columns
        # (t^3(b'bc), t^2(b'c) prx, Zbb adj) used to overlap at the old
        # 0.420-0.878 span / full TXT size
        x0, x1 = 0.360, 0.955                      # first / last column centre
        step = (x1 - x0) / max(len(codes) - 1, 1)
        # 2026-09-15, user: this table was the raw class_weight_share CONFIG
        # echoed verbatim -- the TARGET (sums to 102% by design: it is the
        # OVERALL positive-class share across all 7 topologies at once, a
        # different quantity from M3's per-class subshare tables, which sum
        # to 100% of ONE class's own budget -- not comparable/interchangeable,
        # and previously nothing on the plot said which one this was). Now
        # also shows the ACTUAL achieved train-weighted fraction (grey, 3rd
        # row) computed from eval.npz -- the same real-outcome check M3's
        # per-class table already does -- so a mismatch between "target" and
        # "meas" (e.g. a multiplier not converging, or a bug) is visible
        # directly on the plot instead of assumed.
        _meas = {}
        try:
            _e1 = np.load(outdir / name / "eval.npz")
            _topo_tr, _w_tr = _e1["train_topo"], _e1["train_w"]
            _tot1 = sum(_w_tr[_topo_tr == int(c)].sum() for c in codes) or 1.0
            _meas = {c: 100.0 * _w_tr[_topo_tr == int(c)].sum() / _tot1 for c in codes}
        except Exception:
            pass
        if _meas:
            _hdr = ROOT.TLatex(); _hdr.SetNDC(); _hdr.SetTextFont(42)
            _hdr.SetTextSize(TXT * 0.78); _hdr.SetTextColor(ROOT.kGray + 2)
            _hdr.SetTextAlign(31)
            _hdr.DrawLatex(x1, y_top + 0.040,
                          "#it{target train-weight share  (measured)}")
            ROOT.SetOwnership(_hdr, False)
        tt = ROOT.TLatex(); tt.SetNDC(); tt.SetTextFont(42); tt.SetTextSize(TXT * 0.88)
        tt.SetTextAlign(21)                        # centred on the column
        for i, code in enumerate(codes):          # NB: not 'c' -- that is the canvas
            x = x0 + i * step
            tt.SetTextColor(TOPO_COL[int(code)])
            tt.DrawLatex(x, y_top, TOPO_LAB[int(code)].replace(" proxy", " prx"))
            tt.DrawLatex(x, y_top - 0.036, "%.0f%%" % (100 * float(share[code])))
            if _meas:
                tt.SetTextColor(ROOT.kGray + 2)
                tt.SetTextSize(TXT * 0.72)
                tt.DrawLatex(x, y_top - 0.066, "(%.0f%%)" % _meas[code])
                tt.SetTextSize(TXT * 0.88)
                tt.SetTextColor(TOPO_COL[int(code)])
        tt.SetTextColor(ROOT.kBlack)
        ROOT.SetOwnership(tt, False)
        y_top -= 0.036 * 2 + 0.030 + 0.014          # run info sits below the table

    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.SetTextAlign(31)
    for i, ln in enumerate(lines):
        t.DrawLatex(0.955, y_top - 0.036 * i, ln)
    ROOT.SetOwnership(t, False)

    cms_header(cfg, c)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_imp, h_prm]


def score_figure(outdir, name, summary, cfg, sig_codes, out_png,
                 subset=None, tag=""):
    """subset = signal topology codes; only those components are drawn and the
    'signal all' curve is their sum (S2/S3 panels)."""
    e = np.load(outdir / name / "eval.npz")
    s, w, topo = e["test_score"], e["test_w"], e["test_topo"]
    nb = 30
    # ALL four signal topologies are drawn in every panel -- the analysis
    # reconstructs and measures all of them whichever subset a given tagger
    # was tuned for.  Their shapes are common to S1/S2/S3 (one score); what
    # distinguishes the panels is the TOTAL, which stays that panel's own
    # signal definition, so the ratio pad reads "component vs what this
    # tagger targets".
    all_true = tuple(c for c in sig_codes if int(c) != 5)
    sub = all_true if subset is None else tuple(subset)
    # "signal" total = the mixture the tagger was TRAINED for, i.e. this run's
    # class shares with the proxy dropped -- proxy is W->cs background, and on
    # raw physics weights it is 99% of topo>0, so an unweighted sum would show
    # the proxy shape rather than the Vcb signal.
    # 2026-09-15, user: S2/S3 restricted-subset totals retired, see the
    # SUBSET_* note above -- only S1(bc)'s own total is drawn now.
    defs = [("S1", all_true)]
    sig_m = np.isin(topo, all_true)
    n_bkg = int((topo == 0).sum()); n_sig = int(sig_m.sum())
    cwm = summary["config"].get("class_weight_multiplier") or \
        cfg.get("class_weight_multiplier", {})
    # floor 8 (2026-09-13): topology 7 (QCD(bb)) now exists in test_topo but
    # isn't in class_weight_share -- must still be safely indexable (defaults
    # to multiplier 1.0, i.e. excluded from the "signal all" sum like proxy).
    lut = np.ones(max(8, max((int(k) for k in cwm), default=6) + 1))
    for k, v in cwm.items():
        lut[int(k)] = float(v)
    w_sig = w * lut[topo]                 # per-class multipliers; proxy excluded

    def mk(tg, mask, col, width, style=1, w_over=None):
        h = ROOT.TH1F(f"s_{name}{tag}_{tg}", "", nb, 0.0, 1.0)
        h.Sumw2()
        ws = w if w_over is None else w_over
        for v, ww in zip(s[mask], ws[mask]):
            h.Fill(float(v), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        h.SetLineColor(col); h.SetLineWidth(width); h.SetLineStyle(style)
        h.SetMarkerColor(col)
        return h

    h_bkg = mk("bkg", topo == 0, ROOT.kGray + 1, 4)
    # one red total per signal definition, distinguished by line style
    h_tot = {}
    for dn, dcls in defs:
        m = np.isin(topo, tuple(c for c in dcls if int(c) != 5))
        h_tot[dn] = mk(f"sig{dn}", m, ROOT.kRed + 1, 4, DEF_STYLE[dn], w_over=w_sig)
    h_sig = h_tot["S1"]
    h_cls = {c: mk(f"c{c}", topo == int(c), TOPO_COL[int(c)], 3, 2)
             for c in all_true if (topo == int(c)).sum() > 20}
    # the proxy is the calibration reference: the ratio pad divides every
    # signal component (and the total) by it, so it is drawn here too
    h_prx = mk("prx", topo == 5, TOPO_COL[5], 4, 1)   # solid, same width as the signal total
    n_prx = int((topo == 5).sum())

    # frame width = SQ*(1-2*MARGIN); make the upper pad tall enough that the
    # MAIN frame comes out square even with the ratio pad underneath
    c = ROOT.TCanvas(f"sc_{name}{tag}", "", SQ, SQ)
    p1 = ROOT.TPad(f"p1{name}{tag}", "", 0, P2_H, 1, 1)
    p1.SetBottomMargin(0.02); p1.SetTopMargin(0.128)   # frame shifted lower
    p2 = ROOT.TPad(f"p2{name}{tag}", "", 0, 0.0, 1, P2_H)
    p2.SetTopMargin(0.03); p2.SetBottomMargin(0.25)
    for pp in (p1, p2):
        pp.SetTicks(1, 1); pp.SetLeftMargin(0.145); pp.SetRightMargin(RIGHT_MARGIN)
    p1.Draw(); p2.Draw()

    p1.cd(); p1.SetLogy()
    f1 = p1.DrawFrame(0.0, 3e-3, 1.0, 0.5)
    f1.GetYaxis().SetTitle("N_{jets} norm. to 1")
    f1.GetYaxis().SetTitleSize(0.0385 / P1_H); f1.GetYaxis().SetTitleOffset(0.72)
    f1.GetYaxis().SetLabelSize(LSIZE / P1_H); f1.GetXaxis().SetLabelSize(0)
    f1.GetYaxis().SetLabelOffset(0.005)
    # solid references first, dashed per-topology components on top so the
    # latter stay visible where they overlap
    for h in [h_bkg] + list(h_tot.values()) + [h_prx] + list(h_cls.values()):
        h.Draw("HIST SAME")
    # caption shifted ~20% to the right, with the totals
    # two columns at full readable size: 9 entries in a half-height pad need
    # the width, and this keeps the box in the clear band above the curves
    lg = _legend(0.335, 0.525, 0.790, 0.840, 0.0294 * 1.05 / P1_H)  # text +5%
    lg.SetNColumns(2); lg.SetMargin(0.16); lg.SetColumnSeparation(-0.27)  # cols +1%
    # TLegend fills row-wise, so interleave to get BKG + the three signal
    # definitions + proxy down the LEFT column and the four topology
    # components down the right
    left = ([(h_bkg, "BKG %.2fM jets" % (n_bkg / 1e6))]
            + [(h_tot[dn], DEF_LAB[dn]) for dn, _ in defs]
            + [(h_prx, "t^{2}(b'c) prx")])
    right = [(h, TOPO_LAB[int(cc)]) for cc, h in h_cls.items()]
    for i in range(max(len(left), len(right))):
        if i < len(left):
            lg.AddEntry(left[i][0], left[i][1], "l")
        if i < len(right):
            lg.AddEntry(right[i][0], right[i][1], "l")
    lg.Draw()
    p1.RedrawAxis()
    cms_header(cfg, c)          # canvas NDC -> same as the other three
    p1.cd()

    # ---- ratio pad: component / total signal, with propagated errors -------
    p2.cd()
    f2 = p2.DrawFrame(0.0, 0.0, 1.0, 2.0)
    f2.GetXaxis().SetTitle("BDT fine-tuned score")
    f2.GetYaxis().SetTitle("Signal / proxy")
    f2.GetXaxis().SetTitleSize(TSIZE_X_RATIO / P2_H); f2.GetXaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetXaxis().SetTitleOffset(1.12)
    f2.GetYaxis().SetTitleSize(TSIZE_Y_RATIO / P2_H); f2.GetYaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetYaxis().SetTitleOffset(0.97); f2.GetYaxis().SetNdivisions(505)
    f2.GetYaxis().SetLabelOffset(0.005)
    ratios = []
    # total and each component divided by the PROXY -- the calibration check:
    # a flat line at 1 means the proxy reproduces that component's tagger
    # shape and can stand in for it.  Numerator and denominator are disjoint
    # samples here (different topologies), so the errors are uncorrelated:
    # var(r) = r^2 [ (sx/x)^2 + (sp/p)^2 ]
    for cc, h in [(f"tot{d}", h_tot[d]) for d, _ in defs] + list(h_cls.items()):
        r = h.Clone(f"r_{name}{tag}_{cc}")
        for b in range(1, nb + 1):
            x, sx = h.GetBinContent(b), h.GetBinError(b)
            pp, sp = h_prx.GetBinContent(b), h_prx.GetBinError(b)
            if pp <= 0 or x <= 0:
                r.SetBinContent(b, 0.0); r.SetBinError(b, 0.0); continue
            v = x / pp
            var = v * v * ((sx / x) ** 2 + (sp / pp) ** 2)
            r.SetBinContent(b, v); r.SetBinError(b, float(np.sqrt(max(var, 0.0))))
        # keep the clone's inherited style so the ratio matches the top
        # pad: dashed per-topology components, solid total
        r.SetLineWidth(4 if str(cc).startswith("tot") else 3)
        r.SetMarkerSize(0); r.SetMarkerColor(h.GetLineColor())
        r.Draw("HIST SAME")
        r.Draw("E1 SAME")
        ratios.append(r)
    one = ROOT.TLine(0.0, 1.0, 1.0, 1.0)
    one.SetLineStyle(2); one.SetLineColor(ROOT.kBlack); one.Draw()
    ROOT.SetOwnership(one, False)
    p2.RedrawAxis()
    c.SaveAs(str(out_png))
    return [h_bkg, h_prx] + list(h_tot.values()) + list(h_cls.values()) + ratios


# --------------------------------------------------------------------------- #
# 2026-09-16, user: REPLACED ENTIRELY. This used to plot S1's OLD BINARY
# score (from S1/eval.npz), just re-labelled as "D_cb" -- but that score was
# never trained to separate cb-jets from everything else specifically (S1's
# target was ALL 7 topologies as one undifferentiated bucket). Labelling it
# "D_cb" implied a discriminant that doesn't exist.
# Fixed: N nested-topology-inclusion curves like the original design, but
# ALL of them genuinely use that ONE class's REAL M3 score (its own softmax
# probability, test_proba[:, class_idx] from M3/eval.npz) -- only the
# TOPOLOGY SET called "signal" changes between curves, the score never does.
# One call per class, each producing its own separate plot -- a D_cb plot
# never mixes in D_bb/D_bbc curves or vice versa (user: "make this post only
# with D_cb info").
#   D_cb curves:  (cb,b'c,b'c-prx/rest), (cb,b'c/rest), (cb,b'c,b'b,b'bc/rest)
#   D_bb curves:  (b'b,Zbb,QCDbb/rest), (b'b/rest), (cb,b'c,b'b,b'bc/rest)
# --------------------------------------------------------------------------- #
def m3_roc_figure(outdir, cfg, out_png, sig_classes, cname, curves, colors,
                  presel=None):
    """N ROC curves of ONE class's REAL M3 score (its own softmax
    probability), one per nested topology-inclusion definition of "signal"
    for that class -- see module comment above. `curves` = list of
    (label, topology codes); `colors` = list of ROOT colour indices, same
    length, dark-to-light order."""
    c_idx = next(i for i, cn in sig_classes if cn == cname)
    e = np.load(outdir / "M3" / "eval.npz")
    w, topo = e["test_w"], e["test_topo"]
    score = e["test_proba"][:, c_idx]

    c = ROOT.TCanvas(f"roc_m3_{cname}", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN - 0.010); c.SetLeftMargin(0.105)
    c.SetRightMargin(0.018); c.SetBottomMargin(0.085)
    c.SetLogy(); c.SetGridx(); c.SetGridy()
    frame = _axes(c.DrawFrame(0.0, 1e-3, 1.0, 0.3),
                  f"{M3_DISP[cname]} score", "Other than targeted jets eff.", xoff=0.96, yoff=1.06)
    frame.GetYaxis().SetLabelOffset(0.005)

    entries = [(lab, col, 1, np.isin(topo, codes).astype(np.int8))
               for (lab, codes), col in zip(curves, colors)]

    n = len(entries)
    lg = _legend(0.420, 0.130, 0.780, 0.130 + 0.06 * n, 0.0235)
    lg.SetFillStyle(1001); lg.SetFillColor(ROOT.kWhite); lg.SetMargin(0.16)
    pvw = ROOT.TPave(0.415, 0.125, 0.940, 0.135 + 0.06 * n, 0, "NDC")
    pvw.SetFillColor(ROOT.kWhite); pvw.SetFillStyle(1001); pvw.SetBorderSize(0)
    pvw.Draw(); ROOT.SetOwnership(pvw, False)

    def sig3(x):
        return f"{x:.3g}"

    keep, rows = [pvw], []
    for lab, col, ls, y in entries:
        tpr, fpr, auc = _roc_points(y, score, w)
        g = ROOT.TGraph(len(tpr), tpr, fpr)
        g.SetLineColor(col); g.SetLineWidth(3); g.SetLineStyle(ls)
        g.Draw("L SAME"); keep.append(g)
        lg.AddEntry(g, lab, "l")
        # marker + significance both from the same best_1bin_cut optimum
        # (same method as overtrain_M3_cb.png's WP line).
        cut_thr, eff_s, eff_b = best_1bin_cut(score, w, y == 1, y == 0)
        s_tot, b_tot = w[y == 1].sum(), w[y == 0].sum()
        S, B = eff_s * s_tot, eff_b * b_tot
        z = S / np.sqrt(S + B) if (S + B) > 0 else 0.0
        rows.append([f"{auc:.3f}", "#color[%d]{%s}" % (col, sig3(z))])
        mk = ROOT.TMarker(eff_s, eff_b, 20)
        mk.SetMarkerColor(col); mk.SetMarkerSize(1.7)
        mk.Draw(); keep.append(mk)
    lg.Draw()

    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.0303)
    x_auc, x_z = 0.780, 0.865
    y1, y2 = 0.130, 0.130 + 0.06 * n
    t.DrawLatex(x_auc, y2 + 0.012, "AUC")
    # 2026-09-16 bugfix, user: "Signif.@WP" clipped past the canvas right
    # edge at the value-row font size -- header only drawn smaller.
    t.SetTextSize(0.0303 * 0.80)
    t.DrawLatex(x_z, y2 + 0.012, "Signif.@WP")
    t.SetTextSize(0.0303)
    for i, (a, z) in enumerate(rows):
        yy = y2 - (i + 0.72) * (y2 - y1) / n
        t.DrawLatex(x_auc, yy, a)
        if z:
            t.DrawLatex(x_z, yy, z)
    ROOT.SetOwnership(t, False)

    cms_header(cfg, c, dx=-0.06, dy=0.010, lumi_dx=-0.01)
    if presel:
        # 2026-09-15, user: same styling as roc.png -- black, font 42 (was
        # light grey + Courier-Italic).
        t2 = ROOT.TLatex(); t2.SetNDC(); t2.SetTextFont(42)
        t2.SetTextColor(ROOT.kBlack); t2.SetTextSize(TXT * 1.10 * 1.05)
        lines = ["#bf{Preselection}",
                 "p_{T} > %.0f GeV" % presel["jet_pt_min"],
                 "m_{SD} > %.0f GeV" % presel["jet_sdmass_min"],
                 "#tau_{21} < %.2f" % presel["jet_tau21_max"],
                 "#DeltaR(l, J) > %.1f" % presel["dr_lep_jet_min"]]
        for i, ln in enumerate(lines):
            t2.DrawLatex(0.165, 0.880 - 0.044 * i, ln)
        ROOT.SetOwnership(t2, False)
        keep.append(t2)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return keep


def m3_overtrain_figure(outdir, name, cname, class_idx, summary, cfg, out_png):
    """One-vs-rest overtraining check for ONE M3 class -- simpler than the
    binary overtrain_figure since there is only ONE tagger per class here
    (no S1/S2/S3-style multi-definition overlay). The overtraining numbers
    themselves are read from summary.json (already computed by
    train_one_multiclass in 02_train_tagger.py, same pattern the binary
    overtrain_figure uses) -- NOT recomputed here, since overtrain_metrics()
    lives only in 02_train_tagger.py's venv-side code, not this PyROOT side."""
    e = np.load(outdir / name / "eval.npz")
    nb, lo, hi = 30, 0.0, 1.0
    y_tr, y_te = e["train_y"], e["test_y"]
    w_tr, w_te = e["train_w"], e["test_w"]
    s_tr, s_te = e["train_proba"][:, class_idx], e["test_proba"][:, class_idx]
    yb_tr = (y_tr == class_idx).astype(np.int8)
    yb_te = (y_te == class_idx).astype(np.int8)

    def mk(tg, score, w, m):
        h = ROOT.TH1F(f"h_m3_{cname}_{tg}", "", nb, lo, hi)
        h.Sumw2()
        for v, ww in zip(score[m], w[m]):
            h.Fill(float(v), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        return h

    h_str = mk("str", s_tr, w_tr, yb_tr == 1); h_ste = mk("ste", s_te, w_te, yb_te == 1)
    h_btr = mk("btr", s_tr, w_tr, yb_tr == 0); h_bte = mk("bte", s_te, w_te, yb_te == 0)
    col = M3_COL[cname]
    h_str.SetLineColor(col); h_str.SetLineWidth(3); h_str.SetMarkerColor(col)
    h_str.SetMarkerSize(0)   # error bars only (2026-09-13, user) -- no marker dot
    h_ste.SetMarkerColor(col); h_ste.SetMarkerStyle(20); h_ste.SetMarkerSize(1.0)
    # 2026-09-16, user: BKG/rest -> light grey, matching the binary
    # overtrain_figure's fix (was kAzure+2).
    h_btr.SetLineColor(ROOT.kGray + 1); h_btr.SetMarkerColor(ROOT.kGray + 1)
    h_btr.SetFillColorAlpha(ROOT.kGray, 0.60); h_btr.SetLineWidth(2)
    h_btr.SetMarkerSize(0)   # error bars only (2026-09-13, user) -- no marker dot
    h_bte.SetMarkerColor(ROOT.kGray + 1); h_bte.SetMarkerStyle(21); h_bte.SetMarkerSize(1.0)

    # 2026-09-16, user: y-range from the actual drawn content (lowest nonzero
    # bin x0.5 to highest x2), same fix as the binary overtrain_figure.
    all_hists = [h_btr, h_bte, h_str, h_ste]
    lo_y_vals, hi_y_vals = [], []
    for h in all_hists:
        for ib in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(ib)
            if v > 0:
                lo_y_vals.append(v); hi_y_vals.append(v)
    y_lo = min(lo_y_vals) * 0.5 if lo_y_vals else 3e-3
    y_hi = max(hi_y_vals) * 5.0 if hi_y_vals else 0.5

    c = ROOT.TCanvas(f"ot_m3_{cname}", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN); c.SetRightMargin(0.025)
    c.SetLeftMargin(0.095); c.SetBottomMargin(0.075)
    c.SetTicks(1, 1); c.SetLogy()
    comp = M3_COMPONENTS[cname]
    frame = _axes(c.DrawFrame(lo, y_lo, hi, y_hi),
                  f"{M3_DISP[cname]}( {comp} / rest )",
                  "N_{jets} norm. to 1", tsize=0.036, xoff=1.05, yoff=1.315)
    frame.GetYaxis().SetLabelOffset(0.005)   # tick numbers 1% closer
    frame.GetXaxis().SetLabelOffset(-0.005)  # x-axis tick numbers 1% closer
    # train histograms now ALSO carry error bars (2026-09-13, user), not just
    # the outline/fill -- "HIST E1" draws both together.
    h_btr.Draw("HIST E1 SAME"); h_bte.Draw("E1 SAME")
    h_str.Draw("HIST E1 SAME"); h_ste.Draw("E1 SAME")

    # 2026-09-16, user: "evaluate the 1-bin optimal cut of this variable" --
    # single score>=thr working point maximising S/sqrt(S+B) on the TEST
    # sample, marked with a dashed grey vertical line + efficiencies quoted
    # next to it.
    cut_thr, cut_eff_s, cut_eff_b = best_1bin_cut(s_te, w_te, yb_te == 1, yb_te == 0)
    ln = ROOT.TLine(cut_thr, y_lo, cut_thr, y_hi)
    ln.SetLineColor(ROOT.kGray + 2); ln.SetLineStyle(2); ln.SetLineWidth(2)
    ln.Draw(); ROOT.SetOwnership(ln, False)
    # 2026-09-16 bugfix, user: the annotation used to sit in DATA coordinates
    # near the top of the (now much taller, x5/x10 headroom) frame, which
    # collided with the "70% train / 15% test" header -- switched to NDC
    # (left/right margins 0.095/0.025 from this function's own
    # c.SetLeftMargin/SetRightMargin).
    # 2026-09-16, user: "the line at lower 1/3 of y-axis range, eff values on
    # its right" -- text placed at the NDC height corresponding to 1/3 up the
    # log-scale y-axis (bottom margin 0.075, top margin TOP_MARGIN=0.065), to
    # the right of the line; the line itself still spans the full frame.
    x_ndc = min(0.095 + cut_thr * (1.0 - 0.095 - 0.025) + 0.015, 0.860)
    y_ndc = 0.075 + (1.0 / 3.0) * (1.0 - TOP_MARGIN - 0.075) - 0.10
    t_cut = ROOT.TLatex(); t_cut.SetNDC(True); t_cut.SetTextFont(42)
    t_cut.SetTextSize(0.024); t_cut.SetTextColor(ROOT.kGray + 3)
    t_cut.SetTextAlign(11)
    t_cut.DrawLatex(x_ndc, y_ndc,
                    "#splitline{cut = %.2f}{#splitline{eff_{%s} = %.1f%%}{eff_{rest} = %.1f%%}}"
                    % (cut_thr, cname, 100 * cut_eff_s, 100 * cut_eff_b))
    ROOT.SetOwnership(t_cut, False)

    # 2026-09-16, user: bias re-evaluated at score>0.5 (was cfg's 0.6),
    # recomputed here directly rather than trusting the training-time
    # summary.json number, same bugfix as the binary overtrain_figure (that
    # value would otherwise stay silently frozen at the old cut).
    OT_SCORE_CUT = 0.5
    ot = subset_overtrain(s_tr, w_tr, yb_tr == 1, yb_tr == 0,
                          s_te, w_te, yb_te == 1, yb_te == 0, OT_SCORE_CUT)

    def row_verdict(bias):
        return "OK" if bias < 5.0 else ("WARN" if bias < 15.0 else "FAIL")

    def row_verdict_colored(bias):
        word = row_verdict(bias)
        col = {"OK": ROOT.kGreen + 2, "WARN": ROOT.kOrange + 1,
               "FAIL": ROOT.kRed + 1}[word]
        return "#color[%d]{%s}" % (int(col), word)

    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.DrawLatex(0.220, 0.885, f"#bf{{Overtraining test -- {M3_DISP[cname]}}}   70% train / 15% test")
    ROOT.SetOwnership(t, False)

    # 2026-09-16, user: "legend is overwhelming, repetitive info" -- dropped
    # the spelled-out topology components from every row (the x-axis title
    # already states them, in full, once) -- each row now only says WHICH
    # sample (Rest/D_cb, Test/Train) and, for Test only, its bias/KS/verdict.
    # 2026-09-16, user: legend shifted 20% right; verdict word now coloured
    # (green/orange/red for OK/WARN/FAIL) instead of plain text.
    pvw = ROOT.TPave(0.245, 0.690, 0.730, 0.860, 0, "NDC")
    pvw.SetFillColor(ROOT.kWhite); pvw.SetFillStyle(1001); pvw.SetBorderSize(0)
    pvw.Draw(); ROOT.SetOwnership(pvw, False)
    lg = _legend(0.250, 0.695, 0.725, 0.855, TXT * 0.85)
    lg.SetMargin(0.16)
    # 2026-09-16, user: "at the train entries note the components" -- Test
    # rows stay compact (bias/KS/verdict); Train rows spell out their own
    # topology composition instead, since that info isn't duplicated there.
    # 2026-09-16, user: "place first the train and then the test".
    lg.AddEntry(h_btr, "Rest, train ( %s )" % _m3_rest_components(cname), "f")
    lg.AddEntry(h_bte, "Rest, test (bias %.1f%%, KS %.2f, %s)"
                % (ot["rel_diff_bkg_pct"], ot["ks_bkg_p"],
                   row_verdict_colored(ot["rel_diff_bkg_pct"])), "p")
    lg.AddEntry(h_str, "%s, train ( %s )" % (M3_DISP[cname], comp), "l")
    lg.AddEntry(h_ste, "%s, test (bias %.1f%%, KS %.2f, %s)"
                % (M3_DISP[cname], ot["rel_diff_sig_pct"], ot["ks_sig_p"],
                   row_verdict_colored(ot["rel_diff_sig_pct"])), "p")
    lg.Draw()

    cms_header(cfg, c, dx=0.095 - CMS_X, lumi_dx=-0.01)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_btr, h_bte, h_str, h_ste]


def m3_permimp_figure(outdir, name, cname, class_idx, class_groups, summary, cfg, out_png, ntop=22):
    """Ranking bars for ONE M3 class's one-vs-rest permutation importance,
    PLUS (2026-09-15, user: "prepare the code so 2 bars/feature next
    iteration") a 2nd bar = this feature's GLOBAL XGBoost gain importance
    (m["feature_importance"], the SAME one binary permimp_figure plots) --
    unlike the per-class dAUC series, gain is NOT separable per one-vs-rest
    class in a shared multiclass booster (one tree can split on a feature
    for any/all classes at once), so this bar is identical across cb/bb/bbc
    panels for a given feature -- labelled explicitly as "(global)" so it
    is not mistaken for a class-specific number. Data already exists in
    summary.json from the last training run -- this is a pure rendering
    change, no retrain needed for it to appear."""
    m = summary["models"][name]
    pc = m["per_class"][cname]
    perm = {r["feature"]: max(r["auc_drop"], 0.0) for r in pc["permutation_importance"]}
    gain = {r["feature"]: r["gain"] for r in m["feature_importance"]}
    p_tot = sum(perm.values()) or 1.0
    g_tot = sum(gain.values()) or 1.0
    pim = {f: 100.0 * v / p_tot for f, v in perm.items()}
    gim = {f: 100.0 * gain.get(f, 0.0) / g_tot for f in perm}
    ranked = sorted(pim, key=lambda f: -pim[f])          # RANK BY dAUC (per-class)
    top = ranked[:ntop][::-1]
    n = len(top)
    # 2026-09-15, user: 2 empty bins at the top now (was 1) -- more headroom
    # for the text block, and room to show all 22 features (was capped at
    # 15) to see whether any have genuinely negligible impact.
    nb_bins = n + 2

    h = ROOT.TH1F(f"prm_m3_{cname}", "", nb_bins, 0, nb_bins)
    h_g = ROOT.TH1F(f"gim_m3_{cname}", "", nb_bins, 0, nb_bins)
    h.Sumw2(); h_g.Sumw2()
    for i, f in enumerate(top, 1):
        h.SetBinContent(i, pim[f])
        h_g.SetBinContent(i, gim[f])
        h.GetXaxis().SetBinLabel(i, node_label(f))
    col = M3_COL[cname]
    # thin side-by-side bars per feature, sharing an EXACT common edge at
    # the bin centre (0.50) so there is never a gap (2026-09-15, user:
    # "sometimes touching, sometimes a thin gap" -- the old offsets left a
    # 0.01-wide sliver that rounded differently depending on the pad size).
    h.SetFillColor(col); h.SetLineColor(col); h.SetFillStyle(1001)
    h.SetBarWidth(0.28); h.SetBarOffset(0.50); h.SetLineWidth(1); h.SetMarkerSize(0)
    h_g.SetFillColor(ROOT.kOrange + 1); h_g.SetLineColor(ROOT.kOrange + 1)
    h_g.SetFillStyle(1001)
    h_g.SetBarWidth(0.28); h_g.SetBarOffset(0.22); h_g.SetLineWidth(1); h_g.SetMarkerSize(0)

    c = ROOT.TCanvas(f"pi_m3_{cname}", "", SQ, SQ)
    c.SetLeftMargin(MARGIN_RANK - 0.010); c.SetRightMargin(RIGHT_MARGIN)
    c.SetTopMargin(TOP_MARGIN); c.SetBottomMargin(0.105)
    # 2026-09-15, user: "Share of #DeltaAUC..." read as unclear -- spell out
    # what the bars actually mean, per-class tagger name for consistency
    # with the D_cb/D_bb/D_bbc naming used everywhere else.
    h.GetYaxis().SetTitle("%s: AUC drop from gpt-score removal   [%%]"
                          % M3_DISP[cname])
    h.GetYaxis().SetTitleSize(0.040); h.GetYaxis().SetLabelSize(LSIZE)
    h.GetYaxis().SetTitleOffset(0.95)
    h.GetXaxis().SetLabelSize(0.050); h.GetXaxis().SetLabelOffset(0.005)
    h.SetMaximum(1.1 * max(max(pim[f] for f in top), max(gim[f] for f in top)))
    h.SetMinimum(0.0)
    h.Draw("HBAR")
    h_g.Draw("HBAR SAME")

    # legend + text block moved down (2026-09-15, user) into the extra
    # headroom from the 2nd empty bin, and pulled in from 0.965 to 0.950 on
    # the right so text never touches the frame (RIGHT_MARGIN puts the
    # frame's own right edge at 1-0.025=0.975 -- 0.965 left only a 1%
    # gap, easy to touch at some font/DPI combinations).
    Y0 = 0.860
    lg = _legend(0.600, Y0 - 0.350, 0.950, Y0 - 0.235, TXT)
    lg.AddEntry(h, "#DeltaAUC vs rest  (this class)", "f")
    lg.AddEntry(h_g, "%Imp gain  (global)", "f")
    lg.Draw()

    ot = pc["overtraining"]
    lines = [
        "#bf{M3 -- %s}  ranked by #DeltaAUC vs rest   (%d of %d params)"
        % (M3_DISP[cname], n, m["n_features"]),
        "AUC(vs rest) test %.3f    bias B %.1f%%  [%s]"
        % (pc["auc_ovr"], ot["rel_diff_bkg_pct"], ot["verdict"]),
    ]
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.SetTextAlign(31)
    for i, ln in enumerate(lines):
        t.DrawLatex(0.950, Y0 - 0.036 * i, ln)
    ROOT.SetOwnership(t, False)

    # topology composition of THIS class's own train-split signal, shown as
    # the RAW JET-YIELD fraction entering training (2026-09-15, user: "I
    # want the % to be the relative fraction of event/jet yields entering
    # training as signal, not the weight" -- checked, the previous version
    # used train_w, the POST-MULTIPLIER weight, which is circular: it just
    # echoes M3_topo_subshare's own config target back (e.g. cb's 60/20/20
    # verbatim) rather than showing anything about the actual data. Plain
    # counts show the real, physical imbalance the multiplier corrects for.
    my_codes = sorted(int(k) for k, v in class_groups.items() if int(v) == class_idx)
    if my_codes:
        e = np.load(outdir / name / "eval.npz")
        topo = e["train_topo"]
        tot_n = sum(int((topo == tc).sum()) for tc in my_codes) or 1
        frac = {tc: 100.0 * int((topo == tc).sum()) / tot_n for tc in my_codes}

        cap = ROOT.TLatex(); cap.SetNDC(); cap.SetTextFont(42)
        cap.SetTextSize(TXT * 0.92); cap.SetTextAlign(31)
        cap.DrawLatex(0.950, Y0 - 0.036 * len(lines) - 0.018,
                     "#it{%s signal composition (train, jet yield):}" % M3_DISP[cname])

        y_top = Y0 - 0.036 * len(lines) - 0.018 - 0.044
        x0, x1 = 0.625, 0.950
        step = (x1 - x0) / len(my_codes)
        tt = ROOT.TLatex(); tt.SetNDC(); tt.SetTextFont(42); tt.SetTextSize(TXT)
        tt.SetTextAlign(21)
        for i, tc in enumerate(my_codes):
            x = x0 + (i + 0.5) * step
            tt.SetTextColor(TOPO_COL[tc])
            tt.DrawLatex(x, y_top, TOPO_LAB[tc].replace(" proxy", " prx"))
            tt.DrawLatex(x, y_top - 0.036, "%.0f%%" % frac[tc])
        tt.SetTextColor(ROOT.kBlack)
        ROOT.SetOwnership(cap, False); ROOT.SetOwnership(tt, False)

    cms_header(cfg, c)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h, h_g]


def m3_score_figure(outdir, name, cname, class_idx, class_groups, cfg, out_png,
                    proxy_mask, proxy_label, show_own_components=True,
                    exclude_codes=()):
    """One-vs-rest score composition for ONE M3 class, ratio-to-proxy pad
    style like the binary score_figure ("S/proxy-inspection", user's name
    for this plot type).

    The reference ("proxy") population is now PASSED IN per class rather
    than hardcoded to topology 5 for all three (2026-09-13, user):
      cb  -> topology 5 (t2(b'c) proxy) -- unchanged, genuinely its own
             constituent, so that curve's ratio is a flat-line-at-1 check.
      bb  -> topology {6,7} (Z(bb)-adj + QCD(bb)) -- these two DOMINATE the
             physical (raw-weight) "bb" population (~99%, see investigation
             2026-09-13: QCD(bb) alone is 94% of it), so the old borrowed
             t2(b'c) proxy was not just imperfect but irrelevant here; this
             makes the ratio pad read "how does t2(b'b) compare to the
             QCD+Z-dominated bulk" instead.
      bbc -> the t3(b'cq) proxy flag (build_trainset.py, mirrors the t2(b'c)
             proxy idea for the fully-merged topology) -- bbc has no
             adequate real proxy at all; used anyway per explicit instruction
             ("despite knowing it is not the adequate choice").
    show_own_components=False (bb only) drops the my_codes per-topology
    overlay entirely -- with proxy now BEING two of bb's own three
    constituents, drawing them twice (once as "component", once folded into
    "proxy") would be redundant/confusing; only the total and t2(b'b) itself
    are still meaningful to show.

    exclude_codes (2026-09-15, user: "confusing -- t2(b'c) prx AND t2(b'c)
    proxy?"): drop these topology codes from the per-topology h_cls overlay
    even when show_own_components=True. cb's proxy_mask IS topology 5, one
    of cb's own my_codes -- without this it was drawn TWICE, once as the
    black dashed proxy-reference line (intentional self-consistency check,
    ratio == 1 by construction) and AGAIN as its own coloured component line
    with a near-identical but not-identical label ("prx" vs "proxy"). Now
    passed as (5,) for cb so it appears only once (the black reference).
    """
    e = np.load(outdir / name / "eval.npz")
    s, w, topo = e["test_proba"][:, class_idx], e["test_w"], e["test_topo"]
    nb = 30
    my_codes = sorted(int(k) for k, v in class_groups.items() if int(v) == class_idx)

    def mk(tg, mask, col, width, style=1):
        h = ROOT.TH1F(f"s_m3_{cname}_{tg}", "", nb, 0.0, 1.0)
        h.Sumw2()
        for v, ww in zip(s[mask], w[mask]):
            h.Fill(float(v), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        h.SetLineColor(col); h.SetLineWidth(width); h.SetLineStyle(style)
        h.SetMarkerColor(col)
        return h

    h_bkg = mk("bkg", topo == 0, ROOT.kGray + 1, 4)
    h_tot = mk("tot", np.isin(topo, my_codes), M3_COL[cname], 4)
    show_codes = my_codes if show_own_components else [c for c in my_codes if c == 3]
    show_codes = [c for c in show_codes if c not in exclude_codes]
    h_cls = {c: mk(f"c{c}", topo == c, TOPO_COL[c], 3, 2) for c in show_codes
             if (topo == c).sum() > 20}
    h_prx = mk("prx", proxy_mask, ROOT.kBlack, 4, 1)
    n_bkg = int((topo == 0).sum())

    c = ROOT.TCanvas(f"sc_m3_{cname}", "", SQ, SQ)
    p1 = ROOT.TPad(f"p1_m3_{cname}", "", 0, P2_H, 1, 1)
    p1.SetBottomMargin(0.02); p1.SetTopMargin(0.128)
    p2 = ROOT.TPad(f"p2_m3_{cname}", "", 0, 0.0, 1, P2_H)
    p2.SetTopMargin(0.03); p2.SetBottomMargin(0.25)
    for pp in (p1, p2):
        pp.SetTicks(1, 1); pp.SetLeftMargin(0.145); pp.SetRightMargin(RIGHT_MARGIN)
    p1.Draw(); p2.Draw()

    p1.cd(); p1.SetLogy()
    all_h = [h_bkg, h_tot, h_prx] + list(h_cls.values())
    # y-axis span fit to what's ACTUALLY drawn (2026-09-13, user: cb/bbc
    # panels were clipping real yield at both the top and bottom of the old
    # fixed 3e-3..0.5 window) instead of a one-size-fits-all fixed range.
    peak = max((h.GetMaximum() for h in all_h), default=0.5)
    floor = min((h.GetBinContent(b) for h in all_h for b in range(1, nb + 1)
                if h.GetBinContent(b) > 0), default=3e-3)
    f1 = p1.DrawFrame(0.0, floor * 0.5, 1.0, peak * 1.8)
    f1.GetYaxis().SetTitle("N_{jets} norm. to 1")
    f1.GetYaxis().SetTitleSize(0.0385 / P1_H); f1.GetYaxis().SetTitleOffset(0.72)
    f1.GetYaxis().SetLabelSize(LSIZE / P1_H); f1.GetXaxis().SetLabelSize(0)
    f1.GetYaxis().SetLabelOffset(0.005)
    for h in all_h:
        h.Draw("HIST SAME")

    lg = _legend(0.335, 0.610, 0.790, 0.840, 0.0294 * 1.05 / P1_H)
    # column separation was -0.27 (compressed, 2026-09-15 user: "give more
    # space between the 2 columns") -- now a small POSITIVE gap instead.
    lg.SetNColumns(2); lg.SetMargin(0.16); lg.SetColumnSeparation(0.06)
    left = [(h_bkg, "BKG %.2fM jets" % (n_bkg / 1e6)), (h_tot, f"{M3_DISP[cname]} (total)"),
            (h_prx, proxy_label)]
    right = [(h, TOPO_LAB[c]) for c, h in h_cls.items()]
    for i in range(max(len(left), len(right))):
        if i < len(left):
            lg.AddEntry(left[i][0], left[i][1], "l")
        if i < len(right):
            lg.AddEntry(right[i][0], right[i][1], "l")
    lg.Draw()
    p1.RedrawAxis()
    cms_header(cfg, c)
    p1.cd()

    p2.cd()
    f2 = p2.DrawFrame(0.0, 0.0, 1.0, 2.0)
    f2.GetXaxis().SetTitle(f"{M3_DISP[cname]} score")
    f2.GetYaxis().SetTitle("Signal / proxy")
    f2.GetXaxis().SetTitleSize(TSIZE_X_RATIO / P2_H); f2.GetXaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetXaxis().SetTitleOffset(1.12)
    f2.GetYaxis().SetTitleSize(TSIZE_Y_RATIO / P2_H); f2.GetYaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetYaxis().SetTitleOffset(0.97); f2.GetYaxis().SetNdivisions(505)
    f2.GetYaxis().SetLabelOffset(0.005)
    # 2026-09-15, user: "unclear what the lower ratio pads are, what is the
    # numerator and denominator" -- spell it out. Denominator is ALWAYS the
    # black dashed proxy_label curve (h_prx); numerator is whichever
    # same-coloured curve from the top pad the ratio line matches.
    _rcap = ROOT.TLatex(); _rcap.SetNDC(); _rcap.SetTextFont(42)
    _rcap.SetTextSize(TXT * 0.80); _rcap.SetTextColor(ROOT.kGray + 2)
    _rcap.SetTextAlign(13)
    _rcap.DrawLatex(0.150, 0.965,
                    "#it{curve = (top-pad line of the same colour) / (black "
                    + proxy_label + " line)}")
    ROOT.SetOwnership(_rcap, False)
    ratios = []
    for cc, h in [("tot", h_tot)] + list(h_cls.items()):
        r = h.Clone(f"r_m3_{cname}_{cc}")
        for b in range(1, nb + 1):
            x, sx = h.GetBinContent(b), h.GetBinError(b)
            pp, sp = h_prx.GetBinContent(b), h_prx.GetBinError(b)
            if pp <= 0 or x <= 0:
                r.SetBinContent(b, 0.0); r.SetBinError(b, 0.0); continue
            v = x / pp
            var = v * v * ((sx / x) ** 2 + (sp / pp) ** 2)
            r.SetBinContent(b, v); r.SetBinError(b, float(np.sqrt(max(var, 0.0))))
        r.SetLineWidth(4 if cc == "tot" else 3)
        r.SetMarkerSize(0); r.SetMarkerColor(h.GetLineColor())
        r.Draw("HIST SAME"); r.Draw("E1 SAME")
        ratios.append(r)
    one = ROOT.TLine(0.0, 1.0, 1.0, 1.0)
    one.SetLineStyle(2); one.SetLineColor(ROOT.kBlack); one.Draw()
    ROOT.SetOwnership(one, False)
    p2.RedrawAxis()
    c.SaveAs(str(out_png))
    return [h_bkg, h_tot, h_prx] + list(h_cls.values()) + ratios


# --------------------------------------------------------------------------- #
def build_report(cfg, outdir):
    outdir = Path(outdir)
    summary = json.loads((outdir / "summary.json").read_text())
    tag = cfg.get("run_tag", cfg["dataset_tag"])
    ds_meta = json.loads((Path(summary["dataset_dir"]) / "dataset_complete.json").read_text())
    sig_codes = {int(k): v for k, v in ds_meta["signal_codes"].items()}
    if MODEL not in summary["models"] or not (outdir / MODEL / "eval.npz").exists():
        raise SystemExit(f"[{Path(__file__).name}] no {MODEL} in {outdir} -- run 02_train_tagger.py first")

    cms_style()
    keep = []
    roc = outdir / "roc.png"
    keep += roc_figure(outdir, summary, cfg, roc, ds_meta.get('preselection_used'))

    # One panel per signal definition.  All three share the SAME model, so the
    # ROC (which shows all of them together) and the input ranking are common;
    # only the overtraining and score-composition quadrants are restricted to
    # that panel's signal classes.
    # ONE merged panel: the ROC and the input ranking are common to the three
    # signal definitions, and the two right-hand figures overlay S1/S2/S3 as
    # red curves distinguished by line style.
    pi = outdir / f"permimp_{MODEL}.png"
    sc = outdir / f"score_{MODEL}.png"
    keep += permimp_figure(outdir, MODEL, summary, cfg, pi)
    keep += score_figure(outdir, MODEL, summary, cfg, sig_codes, sc)

    # 2026-09-16, user: "delete the [old S1 overtrain plot]" -- it could never
    # be labelled correctly (S1's binary training target is ALL 7 topologies
    # as one signal bucket, not any single named subset), and the genuinely
    # correct per-class version already exists as overtrain_M3_{cb,bb,bbc}.png
    # (m3_overtrain_figure). Dropped from generation + panel (now 1x3, not
    # 2x2) + publishing; any stale published copy is removed below.
    panel = outdir / "panel.png"
    singles = [roc, pi, sc]
    montage([str(roc), str(pi), str(sc),
             "-tile", "3x1", "-geometry", f"{SQ}x{SQ}+0+0",
             "-background", "white", str(panel)])
    published = {f"S1_tagger_{tag}_panel.png": panel,
                 f"S1_tagger_{tag}_roc.png": roc,
                 f"S1_tagger_{tag}_permimp.png": pi,
                 f"S1_tagger_{tag}_score.png": sc}
    for gname, src in published.items():
        shutil.copyfile(src, GALLERY / gname)
    stale_ov = GALLERY / f"S1_tagger_{tag}_overtrain.png"
    if stale_ov.exists():
        stale_ov.unlink()
        print(f"  removed stale {stale_ov.name} (old S1 overtrain plot retired)")

    print(f"  wrote {panel.name}  ({panel.stat().st_size // 1024} KB)")
    for f_ in singles:
        print(f"        {f_.name}  ({f_.stat().st_size // 1024} KB)")
    print(f"  published {len(published)} -> {GALLERY}/S1_tagger_{tag}_*.png")

    # M3 (multiclass): ROC kept SEPARATE (user's instruction), overtrain /
    # ranking / score drawn once per class (cb/bb/bbc) and tiled into a 3x3
    # panel -- see the "M3 (multiclass) figures" block above for why this is
    # shaped differently from S1's 2x2.
    if "M3" in summary["models"] and (outdir / "M3" / "eval.npz").exists():
        class_names = cfg.get("M3_class_names", ["bkg", "cb", "bb", "bbc"])
        class_groups = cfg.get("M3_class_groups", {})
        sig_classes = [(i, cn) for i, cn in enumerate(class_names) if i != 0]

        # 2026-09-16, user: "roc_M3/overtrain_M3_cb have nothing to do with
        # bb -- update the name" + "reproduce the similar pair for D_bb" --
        # roc_M3.png renamed roc_M3_cb.png (it was always cb-only, just not
        # named as such), and a genuinely parallel roc_M3_bb.png added,
        # using D_bb's OWN score (not D_cb's) throughout.
        _presel = ds_meta.get('preselection_used')
        roc_m3_cb = outdir / "roc_M3_cb.png"
        keep += m3_roc_figure(
            outdir, cfg, roc_m3_cb, sig_classes, "cb",
            curves=[("D_{cb}( cb, b'c, b'c-prx / rest )", (1, 2, 5)),
                    ("D_{cb}( cb, b'c / rest )", (1, 2)),
                    ("D_{cb}( cb, b'c, b'b, b'bc / rest )", (1, 2, 3, 4))],
            colors=[4, 62, 65], presel=_presel)
        # 2026-09-16, user's own 3 curves for D_bb (typed explicitly, not
        # guessed -- bb's components are physically distinct processes with
        # no natural cb-style narrowing order).
        roc_m3_bb = outdir / "roc_M3_bb.png"
        keep += m3_roc_figure(
            outdir, cfg, roc_m3_bb, sig_classes, "bb",
            curves=[("D_{bb}( b'b, Zbb, QCDbb / rest )", (3, 6, 7)),
                    ("D_{bb}( b'b / rest )", (3,)),
                    ("D_{bb}( cb, b'c, b'b, b'bc / rest )", (1, 2, 3, 4))],
            colors=[ROOT.kGreen + 4, ROOT.kGreen + 2, ROOT.kGreen + 0],
            presel=_presel)

        # per-class proxy reference for the score/S-proxy-inspection plot
        # (2026-09-13, user) -- see m3_score_figure's docstring for why each
        # of these three is different.
        e_m3 = np.load(outdir / "M3" / "eval.npz")
        _topo_m3 = e_m3["test_topo"]
        # 2026-09-15, user: "t2(b'c) prx" vs "t2(b'c) proxy" as TWO separate
        # legend entries was confusing -- they were the SAME topology-5
        # population drawn twice (once as this black proxy-reference line,
        # once again as its own coloured component inside h_cls). Full word
        # "proxy" now used consistently everywhere (not abbreviated "prx"),
        # and cb's own topology-5 component is excluded from h_cls (4th
        # tuple element) so it only appears once, as this reference line.
        m3_proxy = {
            "cb":  (_topo_m3 == 5, "t^{2}(b'c) proxy", True, (5,)),
            "bb":  (np.isin(_topo_m3, (6, 7)), "Zbb+QCD(bb)", False, ()),
            "bbc": (e_m3["test_t3bcq_proxy"], "t^{3}(b'cq) proxy", True, ()),
        }

        m3_singles = {}   # cname -> (overtrain, permimp, score) paths
        for c_idx, cname in sig_classes:
            ov3 = outdir / f"overtrain_M3_{cname}.png"
            pi3 = outdir / f"permimp_M3_{cname}.png"
            sc3 = outdir / f"score_M3_{cname}.png"
            keep += m3_overtrain_figure(outdir, "M3", cname, c_idx, summary, cfg, ov3)
            keep += m3_permimp_figure(outdir, "M3", cname, c_idx, class_groups, summary, cfg, pi3)
            pmask, plabel, pshow, pexcl = m3_proxy.get(
                cname, (_topo_m3 == 5, "t^{2}(b'c) proxy", True, (5,)))
            keep += m3_score_figure(outdir, "M3", cname, c_idx, class_groups, cfg, sc3,
                                    pmask, plabel, pshow, exclude_codes=pexcl)
            m3_singles[cname] = (ov3, pi3, sc3)

        panel_m3 = outdir / "panel_M3.png"
        # rows = plot type (overtrain / ranking / score), columns = class
        # (cb, bb, bbc), in that fixed order regardless of dict iteration --
        # a stated design choice, not derived from anything upstream.
        order = [cn for cn in ("cb", "bb", "bbc") if cn in m3_singles]
        tile_row_ov = [str(m3_singles[cn][0]) for cn in order]
        tile_row_pi = [str(m3_singles[cn][1]) for cn in order]
        tile_row_sc = [str(m3_singles[cn][2]) for cn in order]
        montage(tile_row_ov + tile_row_pi + tile_row_sc +
                ["-tile", "3x3", "-geometry", f"{SQ}x{SQ}+0+0",
                 "-background", "white", str(panel_m3)])

        published_m3 = {f"S1_tagger_{tag}_panel_M3.png": panel_m3,
                        f"S1_tagger_{tag}_roc_M3_cb.png": roc_m3_cb,
                        f"S1_tagger_{tag}_roc_M3_bb.png": roc_m3_bb}
        for cn in order:
            ov3, pi3, sc3 = m3_singles[cn]
            published_m3[f"S1_tagger_{tag}_overtrain_M3_{cn}.png"] = ov3
            published_m3[f"S1_tagger_{tag}_permimp_M3_{cn}.png"] = pi3
            published_m3[f"S1_tagger_{tag}_score_M3_{cn}.png"] = sc3
        for gname, src in published_m3.items():
            shutil.copyfile(src, GALLERY / gname)

        print(f"  wrote {panel_m3.name}  ({panel_m3.stat().st_size // 1024} KB)  "
              f"(3x3: rows=overtrain/ranking/score, cols={order})")
        print(f"        {roc_m3_cb.name}  ({roc_m3_cb.stat().st_size // 1024} KB)  "
              f"(kept OUT of the panel, per instruction)")
        print(f"        {roc_m3_bb.name}  ({roc_m3_bb.stat().st_size // 1024} KB)  "
              f"(kept OUT of the panel, per instruction)")
        print(f"  published {len(published_m3)} -> {GALLERY}/S1_tagger_{tag}_*_M3*.png")

        stale_roc_m3 = GALLERY / f"S1_tagger_{tag}_roc_M3.png"
        if stale_roc_m3.exists():
            stale_roc_m3.unlink()
            print(f"  removed stale {stale_roc_m3.name} (renamed to roc_M3_cb.png)")


def main():
    t_wall0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "config.json")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    outdir = PKG / cfg["output_dir"] / cfg.get("run_tag", cfg["dataset_tag"])
    build_report(cfg, outdir)
    print()
    _done_banner(t_wall0, f"{outdir}  (published: {GALLERY}/S1_tagger_"
                          f"{cfg.get('run_tag', cfg['dataset_tag'])}_*.png)")


if __name__ == "__main__":
    main()
