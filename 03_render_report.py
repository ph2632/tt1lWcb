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
PRETTY = {"S1": "S_{1}"}
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

# S2/S3/S4 are the SAME trained S1 model evaluated against a restricted signal
# definition -- the background is untouched and nothing is retrained.  They
# answer "how cleanly does the one tagger isolate THIS topology", which is the
# same construction the old "S1 no t2(b'c)" curve used.
#   topology codes: 1 W(cb)  2 t2(b'c)  3 t2(b'b)  4 t3(b'bc)  5 t2(b'c) proxy
SUBSET_DEF = {"S2": (1, 3, 6, 7), "S3": (1,)}   # S2 bb-family: +Z(bb)-adj, +QCD(bb) (2026-09-13)
SUBSET_LAB = {"S2": "S_{2}: cb+bb",
              "S3": "S_{3}: cb"}
SUBSET_ORDER = ["S2", "S3"]
# the merged panel overlays the three signal definitions as red curves,
# distinguished by line style only
DEF_STYLE = {"S1": 1, "S2": 2, "S3": 3}
DEF_MARKER = {"S1": 20, "S2": 21, "S3": 22}
DEF_LAB = {"S1": "S_{1}: cb+bb+t^{2}(bc)+Zbb",
           "S2": "S_{2}: cb+bb", "S3": "S_{3}: cb"}

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
M3_COL = {"cb": TOPO_COL[1], "bb": TOPO_COL[3], "bbc": TOPO_COL[4]}

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
    b_m = (topo == 0) | (topo == 5)
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
    frame = _axes(c.DrawFrame(0.0, 1e-3, 1.0, 0.3),
                  "Signal eff.", "BKG eff.",
                  xoff=0.96, yoff=1.06)
    frame.GetYaxis().SetLabelOffset(0.005)   # tick numbers 1% closer

    # legend order: the untrained raw sums first (cumulative), then the old and
    # 3-class Dbc models, then S1 itself last.
    # entry = (label, colour, style, y, score, w, signal classes, full score).
    # The cmp_*.npz arrays are built from the same test dataframe, in the same
    # row order, as eval.npz -- so eval's test_topo applies to them too and
    # every curve can get the 2-bin significance.  The last field is the
    # UNMASKED score of that curve, which the significance must use (the
    # plotted score may be subset-masked, and is a different discriminant
    # entirely for the baselines).
    entries = []
    for key in CMP_ORDER:
        f = outdir / f"cmp_{key}.npz"
        if f.exists():
            d = np.load(f)
            entries.append((CMPLAB[key], COL[key], CMP_STYLE[key],
                            d["y"], d["score"], d["w"], (1, 2, 3, 4, 6, 7), d["score"]))
    f = outdir / MODEL / "eval.npz"
    if f.exists():
        e = np.load(f)
        entries.append(("S_{1}: cb+bb+t^{2}(bc)", COL[MODEL], 1,
                        e["test_y"], e["test_score"], e["test_w"],
                        (1, 2, 3, 4, 6, 7), e["test_score"]))
        # S2/S3/S4: the SAME S1 model, re-scored against a restricted signal
        # definition (background untouched).  These are not separate trainings
        # -- they show how well the one tagger isolates each topology.
        topo = e["test_topo"]
        for key in SUBSET_ORDER:
            cls = SUBSET_DEF[key]
            m = (topo == 0) | np.isin(topo, cls)
            if int(np.isin(topo, cls).sum()) < 20:
                continue
            entries.append((SUBSET_LAB[key], COL[key], CMP_STYLE[key],
                            e["test_y"][m], e["test_score"][m], e["test_w"][m],
                            cls, e["test_score"]))

    lg = _legend(0.510, 0.130, 0.810, 0.445, 0.0303)   # +1% right (2026-09-13)
    lg.SetFillStyle(1001); lg.SetFillColor(ROOT.kWhite); lg.SetMargin(0.16)
    keep, rows = [], []
    topo_all = e["test_topo"] if f.exists() else None
    for lab, col, ls, y, s, w, cls, sig_s in entries:
        tpr, fpr, auc = _roc_points(y, s, w)
        g = ROOT.TGraph(len(tpr), tpr, fpr)
        g.SetLineColor(col); g.SetLineWidth(3); g.SetLineStyle(ls)
        g.Draw("L SAME"); keep.append(g)
        text = lab
        rows.append([lab, f"{auc:.3f}", ""])
        if cls is not None and topo_all is not None:
            # significance is evaluated on the FULL test set (proxy in B);
            # the marker is placed on this curve, so its efficiencies use the
            # same arrays the curve was drawn from.
            z, lo_thr = sig_2bin(sig_s, topo_all, e["test_w"], cls)
            if lo_thr is not None:
                rows[-1][2] = "#color[%d]{%.2f}" % (col, z)
                sel = s >= lo_thr
                eff_s = w[(y == 1) & sel].sum() / max(w[y == 1].sum(), 1e-12)
                eff_b = w[(y == 0) & sel].sum() / max(w[y == 0].sum(), 1e-12)
                mk = ROOT.TMarker(eff_s, eff_b, 20)
                mk.SetMarkerColor(col); mk.SetMarkerSize(1.7)
                mk.Draw(); keep.append(mk)
        lg.AddEntry(g, text, "l")
    pvw = ROOT.TPave(0.505, 0.125, 0.970, 0.485, 0, "NDC")   # +1% right (2026-09-13)
    pvw.SetFillColor(ROOT.kWhite); pvw.SetFillStyle(1001); pvw.SetBorderSize(0)
    pvw.Draw(); ROOT.SetOwnership(pvw, False)
    lg.Draw()
    cms_header(cfg, c, dx=-0.06, dy=0.010, lumi_dx=-0.01)
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.0303)
    # AUC / significance as columns on the legend's own row grid, so the
    # numbers line up regardless of how long each label is
    y1, y2, n = 0.130, 0.445, len(rows)
    # +1% right (tied to lg's new x2=0.810) and 1% less gap between the two
    # columns than before (was 0.800/0.895 -> net x_z unchanged) (2026-09-13)
    x_auc, x_z = 0.810, 0.895
    t.DrawLatex(x_auc, y2 + 0.012, "AUC")               # row shifted 1% lower
    t.DrawLatex(x_z - 0.020, y2 + 0.012, "S/#sqrt{B+S}")
    for i, (_, a, z) in enumerate(rows):
        yy = y2 - (i + 0.72) * (y2 - y1) / n
        t.DrawLatex(x_auc, yy, a)
        if z:
            t.DrawLatex(x_z, yy, z)
    # preselection, in the empty upper-left corner
    if presel:
        # +5% size, light grey (was effectively plain black), font family 9
        # = Courier-Italic (precision 2 -> 92) to set this block visually
        # apart from the rest of the plot (2026-09-13, user)
        t.SetTextSize(TXT * 1.10 * 1.05)
        t.SetTextColor(ROOT.kGray + 1)
        t.SetTextFont(92)
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
                              cfg.get("overtrain_score_cut", 0.6))
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
    defs = [("S1", tuple(k for k in sig_codes if int(k) != 5))] + \
           [(k, SUBSET_DEF[k]) for k in cfg.get("panel_subsets", SUBSET_ORDER)]
    sig_sets, ot_by = [], {}
    for dn, dcls in defs:
        m_tr = np.isin(e["train_topo"], dcls); m_te = np.isin(e["test_topo"], dcls)
        htr = mk(f"str{dn}", e["train_score"], e["train_w"], m_tr)
        hte = mk(f"ste{dn}", e["test_score"], w_te, m_te)
        htr.SetLineColor(ROOT.kRed + 1); htr.SetMarkerColor(ROOT.kRed + 1)
        hte.SetLineColor(ROOT.kRed + 1); hte.SetMarkerColor(ROOT.kRed + 1)
        htr.SetLineWidth(3); htr.SetLineStyle(DEF_STYLE[dn])
        hte.SetMarkerStyle(DEF_MARKER[dn]); hte.SetMarkerSize(1.0)
        o = subset_overtrain(e["train_score"], e["train_w"], m_tr, b_tr,
                             e["test_score"], w_te, m_te, b_te,
                             cfg.get("overtrain_score_cut", 0.6))
        # eval.npz holds only a 400k plotting subsample of the 4.87M training
        # background, so its bias is far noisier than 02_train_tagger.py's full-stats
        # value -- take the background from summary.json, keep the signal here
        full = summary["models"][name]["overtraining"]
        o["rel_diff_bkg_pct"] = full["rel_diff_bkg_pct"]
        o["ks_bkg_p"] = full["ks_bkg_p"]
        # verdict must be re-derived AFTER the background swap, or it still
        # reflects the noisy subsample value
        worst = max(o["rel_diff_sig_pct"], o["rel_diff_bkg_pct"])
        o["verdict"] = "OK" if worst < 5.0 else ("WARN" if worst < 15.0 else "FAIL")
        ot_by[dn] = o
        sig_sets.append((dn, htr, hte))
    h_btr = mk("btr", e["train_score"], e["train_w"], b_tr)
    h_bte = mk("bte", e["test_score"], w_te, b_te)
    for h in (h_btr, h_bte):
        h.SetLineColor(ROOT.kAzure + 2); h.SetMarkerColor(ROOT.kAzure + 2)
    h_btr.SetFillColorAlpha(ROOT.kAzure + 2, 0.30); h_btr.SetLineWidth(2)
    h_bte.SetMarkerStyle(21); h_bte.SetMarkerSize(1.0)

    c = ROOT.TCanvas(f"ot_{name}", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN); c.SetRightMargin(0.025)
    c.SetLeftMargin(0.095); c.SetBottomMargin(0.075)
    c.SetTicks(1, 1)            # tickmarks on all four sides, as in the score pads
    c.SetLogy()
    frame = _axes(c.DrawFrame(lo, 3e-3, hi, 0.5),
                  "BDT fine-tuned score",
                  "N_{jets} norm. to 1", tsize=0.036, xoff=1.05, yoff=1.32)
    frame.GetYaxis().SetLabelOffset(0.005)   # tick numbers 1% closer
    h_btr.Draw("HIST SAME"); h_bte.Draw("E1 SAME")
    for _, htr, _ in sig_sets:
        htr.Draw("HIST SAME")
    for _, _, hte in sig_sets:
        hte.Draw("E1 SAME")

    # One table: Test marker | Train style | name | bias | KS | verdict.
    # The two symbol columns are a TLegend with blank labels; everything else
    # is TLatex on the same row grid, so all six columns line up.
    y0, dy = 0.835, 0.040
    x0 = 0.340                     # name column
    # bias/KS/verdict columns pushed 0.04 further right of the name column
    # (2026-09-13, user) -- room for the now-longest label, "S_1:
    # cb+bb+t^{2}(bc)+Zbb", which used to run into the bias % (see below);
    # the three columns keep their old relative spacing to each other.
    xb, xk, xv = x0 + 0.260, x0 + 0.380, x0 + 0.480
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.DrawLatex(0.350, y0 + 0.048, "#bf{Overtraining test}   70% train / 15% test")
    # "Test"/"Train" centred on the swatch columns of the symbol-only legend
    # below (x1=0.155, x2=0.335, 2 columns, margin 0.50 -> swatch centres at
    # 0.1775 / 0.2675) instead of left-aligned at an eyeballed x -- guarantees
    # the header sits over its symbol regardless of label width (2026-09-13).
    t.SetTextAlign(21)
    t.DrawLatex(0.1775, y0, "Test")
    t.DrawLatex(0.2675, y0, "Train")
    t.SetTextAlign(11)
    t.DrawLatex(xb, y0, "Bias >%.1f" % cfg.get("overtrain_score_cut", 0.6))
    t.DrawLatex(xk, y0, "KS p-val.")
    t.DrawLatex(x0, y0 - dy, "BKG")
    t.DrawLatex(xb, y0 - dy, "%.1f%%" % ot_by["S1"]["rel_diff_bkg_pct"])
    t.DrawLatex(xk, y0 - dy, "%.2f" % ot_by["S1"]["ks_bkg_p"])
    for i, (dn, _, _) in enumerate(sig_sets):
        o = ot_by[dn]
        yy = y0 - (2 + i) * dy
        t.SetTextColor(ROOT.kBlack)
        t.DrawLatex(x0, yy, DEF_LAB[dn])
        t.DrawLatex(xb, yy, "%.1f%%" % o["rel_diff_sig_pct"])
        t.DrawLatex(xk, yy, "%.2f" % o["ks_sig_p"])
        okd = max(o["rel_diff_sig_pct"], o["rel_diff_bkg_pct"]) < 5.0
        t.SetTextColor(ROOT.kGreen + 2 if okd else ROOT.kRed + 1)
        t.DrawLatex(xv, yy, "#bf{%s}" % o["verdict"])
    t.SetTextColor(ROOT.kBlack)
    ROOT.SetOwnership(t, False)

    # symbols only -- blank labels, four rows aligned with the table rows
    lg = _legend(0.155, y0 - 4.5 * dy, 0.335, y0 - 0.5 * dy, TXT * 1.05)
    lg.SetNColumns(2); lg.SetColumnSeparation(0.0); lg.SetMargin(0.50)
    lg.AddEntry(h_bte, " ", "p")
    lg.AddEntry(h_btr, " ", "f")
    for dn, htr, hte in sig_sets:
        lg.AddEntry(hte, " ", "p")
        lg.AddEntry(htr, " ", "l")
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
    lg.AddEntry(h_imp, "%Imp  gain (TMVA)", "f")
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
        tt = ROOT.TLatex(); tt.SetNDC(); tt.SetTextFont(42); tt.SetTextSize(TXT * 0.88)
        tt.SetTextAlign(21)                        # centred on the column
        for i, code in enumerate(codes):          # NB: not 'c' -- that is the canvas
            x = x0 + i * step
            tt.SetTextColor(TOPO_COL[int(code)])
            tt.DrawLatex(x, y_top, TOPO_LAB[int(code)].replace(" proxy", " prx"))
            tt.DrawLatex(x, y_top - 0.036, "%.0f%%" % (100 * float(share[code])))
        tt.SetTextColor(ROOT.kBlack)
        ROOT.SetOwnership(tt, False)
        y_top -= 0.036 * 2 + 0.014                 # run info sits below the table

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
    defs = [("S1", all_true)] + [(k, SUBSET_DEF[k])
                                 for k in cfg.get("panel_subsets", SUBSET_ORDER)]
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
# M3 (multiclass, bkg/cb/bb/bbc) figures, 2026-09-13.
#
# User's framing: "M1 and M3 in parallel... M3 will be just S, as we will
# have one tagger per class (cb, bb, bbc)" -- so unlike the binary S1/S2/S3
# overlay (one tagger, three restricted-signal re-scorings drawn as line
# styles on the SAME plot), M3's overtrain/ranking/score figures are drawn
# ONCE PER CLASS (3 separate figures each), tiled into a 3x3 panel. The ROC
# is kept OUT of that panel (user's instruction) as its own single figure
# with all 3 one-vs-rest curves overlaid, alongside S1's binary curve for
# direct architecture comparison.
# --------------------------------------------------------------------------- #
def m3_roc_figure(outdir, cfg, out_png, presel=None):
    """3 one-vs-rest ROC curves (cb/bb/bbc vs rest) from M3, plus the S1
    binary curve on the SAME test split for direct comparison. This is the
    "+1" ROC figure the user asked to keep separate from the 3x3 panel."""
    e = np.load(outdir / "M3" / "eval.npz")
    class_names = [str(x) for x in e["class_names"]]
    y_te, w_te, proba_te = e["test_y"], e["test_w"], e["test_proba"]
    topo_te = e["test_topo"]

    c = ROOT.TCanvas("roc_m3", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN - 0.010); c.SetLeftMargin(0.105)
    c.SetRightMargin(0.018); c.SetBottomMargin(0.085)
    c.SetLogy(); c.SetGridx(); c.SetGridy()
    frame = _axes(c.DrawFrame(0.0, 1e-3, 1.0, 0.3),
                  "Signal eff.", "BKG eff.", xoff=0.96, yoff=1.06)
    frame.GetYaxis().SetLabelOffset(0.005)

    entries = []
    for c_idx, cname in enumerate(class_names):
        if c_idx == 0:
            continue
        yb = (y_te == c_idx).astype(np.int8)
        entries.append((f"M3 {cname} (vs rest)", M3_COL[cname], 1, yb, proba_te[:, c_idx], w_te))

    # restricted-definition subset curves (2026-09-13, user), same pattern as
    # S1's S2/S3 subset ROCs: keep bkg + ONLY the named topologies (drop
    # every other topology from consideration entirely, not just from the
    # positive class), scored on that class's own probability column.
    for lab, col, cls_idx, codes in [
        ("Wcb+t^{2}(b'c)", ROOT.kCyan + 2, 1, (1, 2)),          # excludes the proxy (5)
        ("QCD(bb)+Zbb", ROOT.kOrange + 7, 2, (6, 7)),            # excludes true t2(b'b) (3)
    ]:
        m = (topo_te == 0) | np.isin(topo_te, codes)
        yb = np.isin(topo_te[m], codes).astype(np.int8)
        entries.append((lab, col, 3, yb, proba_te[m, cls_idx], w_te[m]))

    f_s1 = outdir / MODEL / "eval.npz"
    if f_s1.exists():
        e1 = np.load(f_s1)
        entries.append(("S1 (binary)", COL[MODEL], 2, e1["test_y"], e1["test_score"], e1["test_w"]))

    n = len(entries)
    lg = _legend(0.480, 0.130, 0.870, 0.130 + 0.055 * n, 0.0303)
    lg.SetFillStyle(1001); lg.SetFillColor(ROOT.kWhite); lg.SetMargin(0.16)
    pvw = ROOT.TPave(0.475, 0.125, 0.960, 0.135 + 0.055 * n, 0, "NDC")
    pvw.SetFillColor(ROOT.kWhite); pvw.SetFillStyle(1001); pvw.SetBorderSize(0)
    pvw.Draw(); ROOT.SetOwnership(pvw, False)

    keep = [pvw]
    for lab, col, ls, y, s, w in entries:
        tpr, fpr, auc = _roc_points(y, s, w)
        g = ROOT.TGraph(len(tpr), tpr, fpr)
        g.SetLineColor(col); g.SetLineWidth(3); g.SetLineStyle(ls)
        g.Draw("L SAME"); keep.append(g)
        lg.AddEntry(g, f"{lab}   AUC {auc:.3f}", "l")
    lg.Draw()
    cms_header(cfg, c, dx=-0.06, dy=0.010, lumi_dx=-0.01)
    if presel:
        t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(92)
        t.SetTextColor(ROOT.kGray + 1); t.SetTextSize(TXT * 1.10 * 1.05)
        lines = ["#bf{Preselection}",
                 "p_{T} > %.0f GeV" % presel["jet_pt_min"],
                 "m_{SD} > %.0f GeV" % presel["jet_sdmass_min"],
                 "#tau_{21} < %.2f" % presel["jet_tau21_max"],
                 "#DeltaR(l, J) > %.1f" % presel["dr_lep_jet_min"]]
        for i, ln in enumerate(lines):
            t.DrawLatex(0.165, 0.880 - 0.044 * i, ln)
        ROOT.SetOwnership(t, False)
        keep.append(t)
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
    h_btr.SetLineColor(ROOT.kAzure + 2); h_btr.SetMarkerColor(ROOT.kAzure + 2)
    h_btr.SetFillColorAlpha(ROOT.kAzure + 2, 0.30); h_btr.SetLineWidth(2)
    h_btr.SetMarkerSize(0)   # error bars only (2026-09-13, user) -- no marker dot
    h_bte.SetMarkerColor(ROOT.kAzure + 2); h_bte.SetMarkerStyle(21); h_bte.SetMarkerSize(1.0)

    c = ROOT.TCanvas(f"ot_m3_{cname}", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN); c.SetRightMargin(0.025)
    c.SetLeftMargin(0.095); c.SetBottomMargin(0.075)
    c.SetTicks(1, 1); c.SetLogy()
    frame = _axes(c.DrawFrame(lo, 3e-3, hi, 0.5), f"P({cname}) score",
                  "N_{jets} norm. to 1", tsize=0.036, xoff=1.05, yoff=1.32)
    frame.GetYaxis().SetLabelOffset(0.005)
    # train histograms now ALSO carry error bars (2026-09-13, user), not just
    # the outline/fill -- "HIST E1" draws both together.
    h_btr.Draw("HIST E1 SAME"); h_bte.Draw("E1 SAME")
    h_str.Draw("HIST E1 SAME"); h_ste.Draw("E1 SAME")

    ot = summary["models"][name]["per_class"][cname]["overtraining"]
    y0, dy = 0.835, 0.040
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.DrawLatex(0.320, y0 + 0.048, f"#bf{{Overtraining test -- M3 {cname}}}   70% train / 15% test")
    x0 = 0.340; xb, xk, xv = x0 + 0.260, x0 + 0.380, x0 + 0.480
    # "Test"/"Train" centred over the legend's swatch columns below, same
    # convention as the binary overtrain_figure (2026-09-13)
    t.SetTextAlign(21)
    t.DrawLatex(0.1775, y0, "Test"); t.DrawLatex(0.2675, y0, "Train")
    t.SetTextAlign(11)
    t.DrawLatex(xb, y0, "Bias >%.1f" % cfg.get("overtrain_score_cut", 0.6))
    t.DrawLatex(xk, y0, "KS p-val.")
    t.DrawLatex(x0, y0 - dy, "bkg (rest)")
    t.DrawLatex(xb, y0 - dy, "%.1f%%" % ot["rel_diff_bkg_pct"])
    t.DrawLatex(xk, y0 - dy, "%.2f" % ot["ks_bkg_p"])
    t.DrawLatex(x0, y0 - 2 * dy, cname)
    t.DrawLatex(xb, y0 - 2 * dy, "%.1f%%" % ot["rel_diff_sig_pct"])
    t.DrawLatex(xk, y0 - 2 * dy, "%.2f" % ot["ks_sig_p"])
    okd = ot["verdict"] == "OK"
    t.SetTextColor(ROOT.kGreen + 2 if okd else ROOT.kRed + 1)
    t.DrawLatex(xv, y0 - 2 * dy, "#bf{%s}" % ot["verdict"])
    t.SetTextColor(ROOT.kBlack)
    ROOT.SetOwnership(t, False)

    lg = _legend(0.155, y0 - 2.5 * dy, 0.335, y0 - 0.5 * dy, TXT * 1.05)
    lg.SetNColumns(2); lg.SetColumnSeparation(-0.27); lg.SetMargin(0.50)
    lg.AddEntry(h_bte, " ", "p"); lg.AddEntry(h_btr, " ", "f")
    lg.AddEntry(h_ste, " ", "p"); lg.AddEntry(h_str, " ", "l")
    lg.Draw()

    cms_header(cfg, c, dx=0.095 - CMS_X, lumi_dx=-0.01)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_btr, h_bte, h_str, h_ste]


def m3_permimp_figure(outdir, name, cname, class_idx, class_groups, summary, cfg, out_png, ntop=15):
    """Ranking bars for ONE M3 class's one-vs-rest permutation importance.
    Single series only (this class's dAUC) -- unlike the binary
    permimp_figure's dAUC+gain pair, XGBoost's gain accounting does not
    cleanly separate by class in a shared multiclass booster, so no gain
    overlay is drawn here."""
    m = summary["models"][name]
    pc = m["per_class"][cname]
    perm = {r["feature"]: max(r["auc_drop"], 0.0) for r in pc["permutation_importance"]}
    p_tot = sum(perm.values()) or 1.0
    pim = {f: 100.0 * v / p_tot for f, v in perm.items()}
    ranked = sorted(pim, key=lambda f: -pim[f])
    top = ranked[:ntop][::-1]
    n = len(top)
    nb_bins = n + 1   # +1 empty bin at the top, same convention as permimp_figure

    h = ROOT.TH1F(f"prm_m3_{cname}", "", nb_bins, 0, nb_bins)
    h.Sumw2()
    for i, f in enumerate(top, 1):
        h.SetBinContent(i, pim[f])
        h.GetXaxis().SetBinLabel(i, node_label(f))
    col = M3_COL[cname]
    h.SetFillColor(col); h.SetLineColor(col); h.SetFillStyle(1001)
    h.SetBarWidth(0.55); h.SetBarOffset(0.22); h.SetLineWidth(1); h.SetMarkerSize(0)

    c = ROOT.TCanvas(f"pi_m3_{cname}", "", SQ, SQ)
    c.SetLeftMargin(MARGIN_RANK - 0.010); c.SetRightMargin(RIGHT_MARGIN)
    c.SetTopMargin(TOP_MARGIN); c.SetBottomMargin(0.105)
    h.GetYaxis().SetTitle("Share of #DeltaAUC (vs rest)   [%]")
    h.GetYaxis().SetTitleSize(0.046); h.GetYaxis().SetLabelSize(LSIZE)
    h.GetYaxis().SetTitleOffset(0.95)
    h.GetXaxis().SetLabelSize(0.050); h.GetXaxis().SetLabelOffset(0.005)
    h.SetMaximum(1.1 * max(pim[f] for f in top)); h.SetMinimum(0.0)
    h.Draw("HBAR")

    ot = pc["overtraining"]
    lines = [
        "#bf{M3 -- %s}  ranked by #DeltaAUC vs rest   (%d of %d params)"
        % (cname, n, m["n_features"]),
        "AUC(vs rest) test %.3f    bias B %.1f%%  [%s]"
        % (pc["auc_ovr"], ot["rel_diff_bkg_pct"], ot["verdict"]),
    ]
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.SetTextAlign(31)
    for i, ln in enumerate(lines):
        t.DrawLatex(0.955, 0.905 - 0.036 * i, ln)
    ROOT.SetOwnership(t, False)

    # topology composition of THIS class's own train-split signal weight
    # (2026-09-13, user) -- e.g. makes cb's proxy-dominance directly visible,
    # which is exactly what drives cb's pull on the softmax discussed
    # alongside the overtraining numbers above. Ratios are invariant to the
    # (uniform, per-class) M3 weight multiplier, so the already-saved
    # train_w/train_topo in eval.npz give this directly, no retrain needed.
    my_codes = sorted(int(k) for k, v in class_groups.items() if int(v) == class_idx)
    if my_codes:
        e = np.load(outdir / name / "eval.npz")
        topo, wtr = e["train_topo"], e["train_w"]
        tot_w = sum(wtr[topo == tc].sum() for tc in my_codes) or 1.0
        frac = {tc: 100.0 * wtr[topo == tc].sum() / tot_w for tc in my_codes}

        cap = ROOT.TLatex(); cap.SetNDC(); cap.SetTextFont(42)
        cap.SetTextSize(TXT * 0.92); cap.SetTextAlign(31)
        cap.DrawLatex(0.955, 0.905 - 0.036 * len(lines) - 0.018,
                     "#it{%s signal composition (train, weighted):}" % cname)

        y_top = 0.905 - 0.036 * len(lines) - 0.018 - 0.044
        x0, x1 = 0.630, 0.955
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
    return [h]


def m3_score_figure(outdir, name, cname, class_idx, class_groups, cfg, out_png,
                    proxy_mask, proxy_label, show_own_components=True):
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
    lg.SetNColumns(2); lg.SetMargin(0.16); lg.SetColumnSeparation(-0.27)
    left = [(h_bkg, "BKG %.2fM jets" % (n_bkg / 1e6)), (h_tot, f"M3 {cname} (total)"),
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
    f2.GetXaxis().SetTitle(f"P({cname}) score")
    f2.GetYaxis().SetTitle("Signal / proxy")
    f2.GetXaxis().SetTitleSize(TSIZE_X_RATIO / P2_H); f2.GetXaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetXaxis().SetTitleOffset(1.12)
    f2.GetYaxis().SetTitleSize(TSIZE_Y_RATIO / P2_H); f2.GetYaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetYaxis().SetTitleOffset(0.97); f2.GetYaxis().SetNdivisions(505)
    f2.GetYaxis().SetLabelOffset(0.005)
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
    ov = outdir / f"overtrain_{MODEL}.png"
    sc = outdir / f"score_{MODEL}.png"
    keep += permimp_figure(outdir, MODEL, summary, cfg, pi)
    keep += overtrain_figure(outdir, MODEL, summary, cfg, ov, sig_codes)
    keep += score_figure(outdir, MODEL, summary, cfg, sig_codes, sc)

    panel = outdir / "panel.png"
    singles = [roc, ov, pi, sc]
    montage([str(roc), str(ov), str(pi), str(sc),
             "-tile", "2x2", "-geometry", f"{SQ}x{SQ}+0+0",
             "-background", "white", str(panel)])
    # keep the four singles alongside the panel and publish them too, so any
    # quadrant can be used on its own
    published = {f"S1_tagger_{tag}_panel.png": panel,
                 f"S1_tagger_{tag}_roc.png": roc,
                 f"S1_tagger_{tag}_overtrain.png": ov,
                 f"S1_tagger_{tag}_permimp.png": pi,
                 f"S1_tagger_{tag}_score.png": sc}
    for gname, src in published.items():
        shutil.copyfile(src, GALLERY / gname)

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

        roc_m3 = outdir / "roc_M3.png"
        keep += m3_roc_figure(outdir, cfg, roc_m3, ds_meta.get('preselection_used'))

        # per-class proxy reference for the score/S-proxy-inspection plot
        # (2026-09-13, user) -- see m3_score_figure's docstring for why each
        # of these three is different.
        e_m3 = np.load(outdir / "M3" / "eval.npz")
        _topo_m3 = e_m3["test_topo"]
        m3_proxy = {
            "cb":  (_topo_m3 == 5, "t^{2}(b'c) prx", True),
            "bb":  (np.isin(_topo_m3, (6, 7)), "Zbb+QCD(bb)", False),
            "bbc": (e_m3["test_t3bcq_proxy"], "t^{3}(b'cq) prx", True),
        }

        m3_singles = {}   # cname -> (overtrain, permimp, score) paths
        for c_idx, cname in sig_classes:
            ov3 = outdir / f"overtrain_M3_{cname}.png"
            pi3 = outdir / f"permimp_M3_{cname}.png"
            sc3 = outdir / f"score_M3_{cname}.png"
            keep += m3_overtrain_figure(outdir, "M3", cname, c_idx, summary, cfg, ov3)
            keep += m3_permimp_figure(outdir, "M3", cname, c_idx, class_groups, summary, cfg, pi3)
            pmask, plabel, pshow = m3_proxy.get(cname, (_topo_m3 == 5, "t^{2}(b'c) prx", True))
            keep += m3_score_figure(outdir, "M3", cname, c_idx, class_groups, cfg, sc3,
                                    pmask, plabel, pshow)
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
                        f"S1_tagger_{tag}_roc_M3.png": roc_m3}
        for cn in order:
            ov3, pi3, sc3 = m3_singles[cn]
            published_m3[f"S1_tagger_{tag}_overtrain_M3_{cn}.png"] = ov3
            published_m3[f"S1_tagger_{tag}_permimp_M3_{cn}.png"] = pi3
            published_m3[f"S1_tagger_{tag}_score_M3_{cn}.png"] = sc3
        for gname, src in published_m3.items():
            shutil.copyfile(src, GALLERY / gname)

        print(f"  wrote {panel_m3.name}  ({panel_m3.stat().st_size // 1024} KB)  "
              f"(3x3: rows=overtrain/ranking/score, cols={order})")
        print(f"        {roc_m3.name}  ({roc_m3.stat().st_size // 1024} KB)  "
              f"(kept OUT of the panel, per instruction)")
        print(f"  published {len(published_m3)} -> {GALLERY}/S1_tagger_{tag}_*_M3*.png")


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
