#!/usr/bin/env python3
"""
PyROOT / CMS-style figures for the S1 boosted-cb tagger.

Reads only summary.json + the per-model eval.npz / cmp_*.npz written by
train.py, so it runs under an LCG view (ROOT) while training runs in the venv.

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

  python3 XGBoost_training.py [--config S1_tagger/config.json]
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# LCG view providing PyROOT.  Override with $S1_LCG_VIEW.
LCG_VIEW = os.environ.get(
    "S1_LCG_VIEW", "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt")


def _bootstrap_lcg():
    """Re-exec under the LCG view when PyROOT is missing.

    The login shell here is tcsh but LCG ships a bash setup.sh, so
    'source setup.sh && python3 ...' aborts before running anything.  Rather
    than depend on the caller's shell, sourcing is done in an explicit bash
    subshell -- so plain 'python XGBoost_training.py' works from tcsh, bash,
    or the project venv.
    """
    try:
        import ROOT  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_S1_LCG_REEXEC"):        # already tried -- don't loop
        raise SystemExit(
            "[XGBoost_training] PyROOT still missing after sourcing "
            f"{LCG_VIEW}. Set $S1_LCG_VIEW to a view that provides ROOT.")
    setup = Path(LCG_VIEW) / "setup.sh"
    if not setup.exists():
        raise SystemExit(
            f"[XGBoost_training] no PyROOT and no LCG view at {setup}. "
            "Set $S1_LCG_VIEW to an LCG view that provides ROOT.")
    print(f"[XGBoost_training] no PyROOT here -- re-exec under {LCG_VIEW}",
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

HERE = Path(__file__).resolve().parent
GALLERY = HERE.parent
SQ = 900                                   # square canvas side
TXT = 0.029                                # one common in-plot text size
MODEL = "S1"                               # S1' dropped
PRETTY = {"S1": "S_{1}"}
# One distinct hue per ROC curve: grey/black for the two untrained raw sums,
# then green / red / purple for the reference taggers and blue / orange / cyan
# for S1,S2,S3.  No hue is reused, so a curve is identifiable by colour alone.
COL = {"S1": ROOT.kAzure + 1,
       "raw_cb": ROOT.kGray + 2,
       "raw_cb_bb": ROOT.kGray + 3,
       "raw_sum_bc_bb_topbwc": ROOT.kBlack,
       "Dbc_old_8node": ROOT.kGreen + 2,
       "Dbc3_bc": ROOT.kRed + 1, "Dbc3_bb": ROOT.kMagenta + 2,
       "Dbc3_bc_plus_bb": ROOT.kViolet + 1,
       "S2": ROOT.kOrange + 7, "S3": ROOT.kCyan + 2, "S4": ROOT.kTeal + 4}
CMPLAB = {"raw_cb": "cb raw",
          "raw_cb_bb": "cb+bb raw",
          "raw_sum_bc_bb_topbwc": "cb+bb+t^{2}(b'c) raw",
          "Dbc_old_8node": "D_{cb} old",
          "Dbc3_bc": "D_{cb} 3cl",
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
SUBSET_DEF = {"S2": (1, 3), "S3": (1,)}
SUBSET_LAB = {"S2": "S_{2}: cb+bb",
              "S3": "S_{3}: cb"}
SUBSET_ORDER = ["S2", "S3"]

# Tagger-input labels, in the Make_plots.py notation.  These are the
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
            3: ROOT.kGreen + 2, 4: ROOT.kMagenta + 1, 5: ROOT.kCyan + 2}
TOPO_LAB = {1: "W(cb)", 2: "t^{2}(b'c)", 3: "t^{2}(b'b)", 4: "t^{3}(b'bc)",
            5: "t^{2}(b'c) proxy"}

TOP_MARGIN = 0.065     # all four figures (was 0.085)
RIGHT_MARGIN = 0.025   # all four figures (was 0.035)
# Axis tick-NUMBER size.  ROOT scales text by PAD height, so a pad that is
# only a fraction of the canvas needs LSIZE/fraction to come out the same
# physical size -- that is why the score figure's two pads divide by 0.70/0.30.
LSIZE = 0.034
P1_H, P2_H = 0.70, 0.30


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


def _axes(frame, xt, yt, tsize=0.040, lsize=LSIZE, xoff=1.30, yoff=1.30):
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
def roc_figure(outdir, summary, cfg, out_png):
    c = ROOT.TCanvas("roc", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN); c.SetLeftMargin(0.115)
    c.SetRightMargin(0.018); c.SetBottomMargin(0.095)
    c.SetLogy(); c.SetGridx(); c.SetGridy()
    frame = _axes(c.DrawFrame(0.0, 1e-3, 1.0, 0.3),
                  "signal efficiency", "background efficiency",
                  xoff=0.95, yoff=1.05)

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
                            d["y"], d["score"], d["w"], (1, 2, 3, 4), d["score"]))
    f = outdir / MODEL / "eval.npz"
    if f.exists():
        e = np.load(f)
        entries.append(("S_{1}: cb+bb+t^{2}(bc)", COL[MODEL], 1,
                        e["test_y"], e["test_score"], e["test_w"],
                        (1, 2, 3, 4), e["test_score"]))
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

    lg = _legend(0.558, 0.130, 0.922, 0.445, 0.0289)
    lg.SetFillStyle(1001); lg.SetFillColor(ROOT.kWhite); lg.SetMargin(0.16)
    keep = []
    topo_all = e["test_topo"] if f.exists() else None
    for lab, col, ls, y, s, w, cls, sig_s in entries:
        tpr, fpr, auc = _roc_points(y, s, w)
        g = ROOT.TGraph(len(tpr), tpr, fpr)
        g.SetLineColor(col); g.SetLineWidth(3); g.SetLineStyle(ls)
        g.Draw("L SAME"); keep.append(g)
        text = f"{lab}  {auc:.3f}"
        if cls is not None and topo_all is not None:
            # significance is evaluated on the FULL test set (proxy in B);
            # the marker is placed on this curve, so its efficiencies use the
            # same arrays the curve was drawn from.
            z, lo_thr = sig_2bin(sig_s, topo_all, e["test_w"], cls)
            if lo_thr is not None:
                text += "  #color[%d]{[%.2f]}" % (col, z)
                sel = s >= lo_thr
                eff_s = w[(y == 1) & sel].sum() / max(w[y == 1].sum(), 1e-12)
                eff_b = w[(y == 0) & sel].sum() / max(w[y == 0].sum(), 1e-12)
                mk = ROOT.TMarker(eff_s, eff_b, 20)
                mk.SetMarkerColor(col); mk.SetMarkerSize(1.7)
                mk.Draw(); keep.append(mk)
        lg.AddEntry(g, text, "l")
    lg.Draw()
    cms_header(cfg, c, dx=-0.06, lumi_dx=-0.01)
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.DrawLatex(0.665, 0.460, "AUC,  [S/#sqrt{B+S}]")
    ROOT.SetOwnership(t, False)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return keep


def subset_overtrain(tr_s, tr_w, s_tr, b_tr, te_s, te_w, s_te, b_te, cut):
    """bias / KS for one signal-class subset, same definitions as train.py."""
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


def overtrain_figure(outdir, name, summary, cfg, out_png, subset=None, tag=""):
    """subset = signal topology codes (S2/S3 panels).  When given, signal is
    restricted to those classes and the background is the true background
    only -- matching how that panel's ROC curve is built.  The bias/KS numbers
    are then recomputed for the subset, since summary.json describes the full
    S1 signal definition."""
    e = np.load(outdir / name / "eval.npz")
    nb, lo, hi = 40, 0.0, 1.0
    w_te = e["test_w_ot"] if "test_w_ot" in e.files else e["test_w"]

    if subset is None:
        ot = summary["models"][name]["overtraining"]
        s_tr = e["train_y"] == 1; b_tr = e["train_y"] == 0
        s_te = e["test_y"] == 1;  b_te = e["test_y"] == 0
    else:
        if "train_topo" not in e.files:
            raise SystemExit("[XGBoost_training] eval.npz has no train_topo -- "
                             "retrain so the S2/S3 panels can be built")
        s_tr = np.isin(e["train_topo"], subset); b_tr = e["train_topo"] == 0
        s_te = np.isin(e["test_topo"], subset);  b_te = e["test_topo"] == 0
        ot = subset_overtrain(e["train_score"], e["train_w"], s_tr, b_tr,
                              e["test_score"], w_te, s_te, b_te,
                              cfg.get("overtrain_score_cut", 0.6))
        # The background is the SAME jets in every panel, but eval.npz keeps
        # only a 400k plotting subsample of the 4.87M training background, so
        # recomputing its bias here is far noisier than train.py's full-
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

    # compare like with like: if train.py stored a class-reweighted test
    # weight, the test shape must use it or the signal train/test mixtures
    # differ and the curves separate for a reason unrelated to overtraining.
    h_str = mk("str", e["train_score"], e["train_w"], s_tr)
    h_ste = mk("ste", e["test_score"], w_te, s_te)
    h_btr = mk("btr", e["train_score"], e["train_w"], b_tr)
    h_bte = mk("bte", e["test_score"], w_te, b_te)

    for h, col in ((h_str, ROOT.kRed + 1), (h_ste, ROOT.kRed + 1),
                   (h_btr, ROOT.kAzure + 2), (h_bte, ROOT.kAzure + 2)):
        h.SetLineColor(col); h.SetMarkerColor(col)
    h_str.SetFillColorAlpha(ROOT.kRed + 1, 0.30); h_str.SetLineWidth(2)
    h_btr.SetFillColorAlpha(ROOT.kAzure + 2, 0.30); h_btr.SetLineWidth(2)
    h_ste.SetMarkerStyle(20); h_ste.SetMarkerSize(1.1)
    h_bte.SetMarkerStyle(21); h_bte.SetMarkerSize(1.1)

    c = ROOT.TCanvas(f"ot_{name}", "", SQ, SQ)
    c.SetTopMargin(TOP_MARGIN); c.SetRightMargin(0.015)
    c.SetLeftMargin(0.115); c.SetBottomMargin(0.095)
    c.SetLogy()
    frame = _axes(c.DrawFrame(lo, 1e-3, hi, 1.0),
                  f"{PRETTY[name]}  (BDT fine-tuned score)",
                  "N_{jets} normalized to 1", xoff=1.15, yoff=1.55)
    for h in (h_btr, h_str):
        h.Draw("HIST SAME")
    for h in (h_bte, h_ste):
        h.Draw("E1 SAME")

    # top block, in order:  verdict / text / legend
    ok = ot["verdict"] == "OK"
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    x0, y0, dy = 0.355, 0.880, 0.040
    t.DrawLatex(x0, y0, "#bf{Overtraining test}  (test = held-out 15%)")
    t.DrawLatex(x0, y0 - dy, "S: %s   B: %s  (test jets)"
                % (f"{n_sig:,}", f"{n_bkg:,}"))
    t.DrawLatex(x0, y0 - 2 * dy, "Training bias > %.1f:  S: %.1f%%,  B: %.1f%%"
                % (ot["score_cut"], ot["rel_diff_sig_pct"], ot["rel_diff_bkg_pct"]))
    y_ks = y0 - 3 * dy
    t.DrawLatex(x0, y_ks, "KS p-value:  S: %.3f,  B: %.3f"
                % (ot["ks_sig_p"], ot["ks_bkg_p"]))
    # verdict sits on the KS line, just to its right
    t.SetTextColor(ROOT.kGreen + 2 if ok else ROOT.kRed + 1)
    t.DrawLatex(x0 + 0.400, y_ks, "#bf{verdict: %s}" % ot["verdict"])
    t.SetTextColor(ROOT.kBlack)
    ROOT.SetOwnership(t, False)

    # legend: top middle, directly below the text block
    lg = _legend(0.340, y0 - 6.6 * dy, 0.960, y0 - 4.6 * dy, TXT)
    lg.SetNColumns(2); lg.SetColumnSeparation(0.0); lg.SetMargin(0.14)
    lg.AddEntry(h_str, "Signal train (70%)", "f")
    lg.AddEntry(h_ste, "Signal test (15%)", "pe")
    lg.AddEntry(h_btr, "BKG train", "f")
    lg.AddEntry(h_bte, "BKG test", "pe")
    lg.Draw()

    cms_header(cfg, c, lumi_dx=-0.01)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_str, h_ste, h_btr, h_bte]


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

    h_imp = ROOT.TH1F(f"imp_{name}{tag}", "", n, 0, n)
    h_prm = ROOT.TH1F(f"prm_{name}{tag}", "", n, 0, n)
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
    c.SetLeftMargin(MARGIN_RANK); c.SetRightMargin(RIGHT_MARGIN)
    c.SetTopMargin(TOP_MARGIN); c.SetBottomMargin(0.105)
    h_prm.GetYaxis().SetTitle("share of total importance   [%]")
    h_prm.GetYaxis().SetTitleSize(0.044); h_prm.GetYaxis().SetLabelSize(LSIZE)
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
        # this panel's signal is only `subset`, so quote the shares
        # renormalised over those classes (proportions preserved, sum 100%)
        share = {k: v for k, v in share.items() if int(k) in subset}
        tot = sum(float(v) for v in share.values()) or 1.0
        share = {k: float(v) / tot for k, v in share.items()}
    lines = [
        "#bf{%s}  ranked by #DeltaAUC   (%d out of %d input param/s)"
        % (PRETTY[name], n, m["n_features"]),
        "normalised = 100 #times x_{i}/#Sigma_{j}x_{j};  #Sigma#DeltaAUC = %.4f #rightarrow 1%% #equiv %.5f"
        % (p_tot, p_tot / 100.0),
        "trees %d (stop %d)  depth %d  lr %.2f  colsample %.1f"
        % (xgb["n_estimators"], m["best_iteration"], xgb["max_depth"],
           xgb["learning_rate"], xgb["colsample_bytree"]),
        "AUC  train %.3f  test %.3f    bias B %.1f%%  [%s]"
        % (m["train"]["auc"], m["test"]["auc"], ot["rel_diff_bkg_pct"], ot["verdict"]),
    ]
    # signal class weights FIRST, as an aligned 2-row table (names over
    # values), colour-matched to the topology curves of the score figure so
    # the two panels read together; the run info then follows underneath.
    y_top = 0.325
    if share:
        codes = sorted(share, key=int)
        x0, x1 = 0.420, 0.878                      # first / last column centre
        step = (x1 - x0) / max(len(codes) - 1, 1)
        tt = ROOT.TLatex(); tt.SetNDC(); tt.SetTextFont(42); tt.SetTextSize(TXT)
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
    nb = 40
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
    true_codes = tuple(c for c in sub if int(c) != 5)
    sig_m = np.isin(topo, true_codes)
    n_bkg = int((topo == 0).sum()); n_sig = int(sig_m.sum())
    cwm = summary["config"].get("class_weight_multiplier") or \
        cfg.get("class_weight_multiplier", {})
    lut = np.ones(6)
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

    h_bkg = mk("bkg", topo == 0, ROOT.kGray + 2, 3)
    h_sig = mk("sig", sig_m, ROOT.kRed + 1, 3, w_over=w_sig)
    h_cls = {c: mk(f"c{c}", topo == int(c), TOPO_COL[int(c)], 2, 2)
             for c in all_true if (topo == int(c)).sum() > 20}
    # the proxy is the calibration reference: the ratio pad divides every
    # signal component (and the total) by it, so it is drawn here too
    h_prx = mk("prx", topo == 5, TOPO_COL[5], 2, 2)
    n_prx = int((topo == 5).sum())

    # frame width = SQ*(1-2*MARGIN); make the upper pad tall enough that the
    # MAIN frame comes out square even with the ratio pad underneath
    c = ROOT.TCanvas(f"sc_{name}{tag}", "", SQ, SQ)
    p1 = ROOT.TPad(f"p1{name}{tag}", "", 0, P2_H, 1, 1)
    p1.SetBottomMargin(0.02); p1.SetTopMargin(0.098)
    p2 = ROOT.TPad(f"p2{name}{tag}", "", 0, 0.0, 1, P2_H)
    p2.SetTopMargin(0.03); p2.SetBottomMargin(0.33)
    for pp in (p1, p2):
        pp.SetTicks(1, 1); pp.SetLeftMargin(0.125); pp.SetRightMargin(RIGHT_MARGIN)
    p1.Draw(); p2.Draw()

    p1.cd(); p1.SetLogy()
    f1 = p1.DrawFrame(0.0, 1e-3, 1.0, 1.0)
    f1.GetYaxis().SetTitle("normalised to unit area")
    f1.GetYaxis().SetTitleSize(0.052); f1.GetYaxis().SetTitleOffset(0.91)
    f1.GetYaxis().SetLabelSize(LSIZE / P1_H); f1.GetXaxis().SetLabelSize(0)
    for h in list(h_cls.values()) + [h_prx, h_bkg, h_sig]:
        h.Draw("HIST SAME")
    # caption shifted ~20% to the right, with the totals
    lg = _legend(0.345, 0.525, 0.78, 0.855, 0.0400)
    lg.AddEntry(h_bkg, "BKG  (%s jets)" % f"{n_bkg:,}", "l")
    lg.AddEntry(h_sig, "Signal W#rightarrowcb, class-wgt  (%s)" % f"{n_sig:,}", "l")
    for cc, h in h_cls.items():
        lg.AddEntry(h, TOPO_LAB[int(cc)], "l")
    lg.AddEntry(h_prx, "t^{2}(b'c) prx  (%s)  [ratio denom.]" % f"{n_prx:,}", "l")
    lg.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.042)
    t.DrawLatex(0.80, 0.862, PRETTY[name])
    ROOT.SetOwnership(t, False)
    p1.RedrawAxis()
    cms_header(cfg, c)          # canvas NDC -> same as the other three
    p1.cd()

    # ---- ratio pad: component / total signal, with propagated errors -------
    p2.cd()
    f2 = p2.DrawFrame(0.0, 0.0, 1.0, 2.0)
    f2.GetXaxis().SetTitle(f"{PRETTY[name]}  (BDT fine-tuned score)")
    f2.GetYaxis().SetTitle("/ proxy")
    f2.GetXaxis().SetTitleSize(0.125); f2.GetXaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetXaxis().SetTitleOffset(1.12)
    f2.GetYaxis().SetTitleSize(0.105); f2.GetYaxis().SetLabelSize(LSIZE / P2_H)
    f2.GetYaxis().SetTitleOffset(0.39); f2.GetYaxis().SetNdivisions(505)
    ratios = []
    # total and each component divided by the PROXY -- the calibration check:
    # a flat line at 1 means the proxy reproduces that component's tagger
    # shape and can stand in for it.  Numerator and denominator are disjoint
    # samples here (different topologies), so the errors are uncorrelated:
    # var(r) = r^2 [ (sx/x)^2 + (sp/p)^2 ]
    for cc, h in [("tot", h_sig)] + list(h_cls.items()):
        r = h.Clone(f"r_{name}{tag}_{cc}")
        for b in range(1, nb + 1):
            x, sx = h.GetBinContent(b), h.GetBinError(b)
            pp, sp = h_prx.GetBinContent(b), h_prx.GetBinError(b)
            if pp <= 0 or x <= 0:
                r.SetBinContent(b, 0.0); r.SetBinError(b, 0.0); continue
            v = x / pp
            var = v * v * ((sx / x) ** 2 + (sp / pp) ** 2)
            r.SetBinContent(b, v); r.SetBinError(b, float(np.sqrt(max(var, 0.0))))
        r.SetLineWidth(3 if cc == "tot" else 2); r.SetLineStyle(1)
        r.SetMarkerSize(0); r.SetMarkerColor(h.GetLineColor())
        r.Draw("HIST SAME")
        r.Draw("E1 SAME")
        ratios.append(r)
    one = ROOT.TLine(0.0, 1.0, 1.0, 1.0)
    one.SetLineStyle(2); one.SetLineColor(ROOT.kBlack); one.Draw()
    ROOT.SetOwnership(one, False)
    p2.RedrawAxis()
    c.SaveAs(str(out_png))
    return [h_bkg, h_sig, h_prx] + list(h_cls.values()) + ratios


# --------------------------------------------------------------------------- #
def build_report(cfg, outdir):
    outdir = Path(outdir)
    summary = json.loads((outdir / "summary.json").read_text())
    tag = cfg.get("run_tag", cfg["dataset_tag"])
    ds_meta = json.loads((Path(summary["dataset_dir"]) / "dataset_complete.json").read_text())
    sig_codes = {int(k): v for k, v in ds_meta["signal_codes"].items()}
    if MODEL not in summary["models"] or not (outdir / MODEL / "eval.npz").exists():
        raise SystemExit(f"[{Path(__file__).name}] no {MODEL} in {outdir} -- run train.py first")

    cms_style()
    keep = []
    roc = outdir / "roc.png"
    keep += roc_figure(outdir, summary, cfg, roc)

    # One panel per signal definition.  All three share the SAME model, so the
    # ROC (which shows all of them together) and the input ranking are common;
    # only the overtraining and score-composition quadrants are restricted to
    # that panel's signal classes.
    panels = [("S1", None)] + [(k, SUBSET_DEF[k]) for k in SUBSET_ORDER]
    published = {f"S1_tagger_{tag}_roc.png": roc}
    made = []
    for pname, subset in panels:
        sfx = "" if subset is None else f"_{pname}"
        ov = outdir / f"overtrain_{MODEL}{sfx}.png"
        sc = outdir / f"score_{MODEL}{sfx}.png"
        pi = outdir / f"permimp_{MODEL}{sfx}.png"
        keep += permimp_figure(outdir, MODEL, summary, cfg, pi,
                               subset=subset, tag=sfx)
        keep += overtrain_figure(outdir, MODEL, summary, cfg, ov,
                                 subset=subset, tag=sfx)
        keep += score_figure(outdir, MODEL, summary, cfg, sig_codes, sc,
                             subset=subset, tag=sfx)
        panel = outdir / f"panel_{pname}.png"
        montage([str(roc), str(ov), str(pi), str(sc),
                 "-tile", "2x2", "-geometry", f"{SQ}x{SQ}+2+2",
                 "-background", "white", str(panel)])
        published[f"S1_tagger_{tag}_panel_{pname}.png"] = panel
        published[f"S1_tagger_{tag}_overtrain_{pname}.png"] = ov
        published[f"S1_tagger_{tag}_permimp_{pname}.png"] = pi
        published[f"S1_tagger_{tag}_score_{pname}.png"] = sc
        made.append((pname, panel, ov, sc))

    for gname, src in published.items():
        shutil.copyfile(src, GALLERY / gname)

    for pname, panel, ov, sc in made:
        lab = "all 4 classes" if pname == "S1" else \
            SUBSET_LAB[pname].split(":", 1)[1].strip()
        print(f"  {panel.name:16s} signal = {lab}")
    print(f"  published {len(published)} -> {GALLERY}/S1_tagger_{tag}_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    build_report(cfg, HERE / cfg["output_dir"] / cfg.get("run_tag", cfg["dataset_tag"]))


if __name__ == "__main__":
    main()
