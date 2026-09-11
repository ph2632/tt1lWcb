#!/usr/bin/env python3
"""
PyROOT / CMS-style figures for the S1 and S1' boosted-cb taggers.

Reads only summary.json + the per-model eval.npz / cmp_*.npz written by
train.py, so it runs under an LCG view (ROOT) while training runs in the venv.

Figures (all square canvases, frameless legends, CMS Simulation + lumi):
  roc.png                 S1 / S1' / raw-sum baseline / old Dbc / 3-class Dbc
  overtrain_<M>.png       train (filled) vs test (points) + KS and |t-tr|/t
  permimp_<M>.png         permutation importance, one bar per input
  score_<M>.png           score by W->cb topology + ratio pad (component/total)
  panel_<M>.png           2x2 montage of the four

Run:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
  python3 S1_tagger/make_plots_root.py [--config S1_tagger/config.json]
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


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
PRETTY = {"S1": "S_{1}", "S1p": "S_{1}'"}
COL = {"S1": ROOT.kAzure + 1, "S1p": ROOT.kOrange + 7,
       "raw_sum_bc_bb_topbwc": ROOT.kBlack,
       "Dbc_old_8node": ROOT.kGreen + 2,
       "Dbc3_bc": ROOT.kRed + 1, "Dbc3_bb": ROOT.kMagenta + 2,
       "Dbc3_bc_plus_bb": ROOT.kViolet - 1}
CMPLAB = {"raw_sum_bc_bb_topbwc": "raw bc+bb+t^{2}(b'c)",
          "Dbc_old_8node": "D_{bc} old, 8 nodes",
          "Dbc3_bc": "3-class  D_{bc} only",
          "Dbc3_bb": "3-class  D_{bb} only",
          "Dbc3_bc_plus_bb": "3-class  D_{bc}+D_{bb}"}
CMP_ORDER = ["raw_sum_bc_bb_topbwc", "Dbc_old_8node", "Dbc3_bc_plus_bb"]
CMP_STYLE = {"raw_sum_bc_bb_topbwc": 2, "Dbc_old_8node": 1,
             "Dbc3_bc_plus_bb": 1, "Dbc3_bc": 7, "Dbc3_bb": 3}

# Tagger-input labels, in the Make_plots.py notation.  These are the
# MASS-DECORRELATED ak8_gpt_* scores of a generic massive resonance X, so
# they carry no parent-flavour assumption (W->cb, H+->cb, ... all alike).
# Left column below is the exact ntuple branch suffix.
NODE_LABEL = {
    # 2-prong nodes are defined on a GENERIC resonance X (no parent-flavour
    # assumption -> X(bc) covers W->cb and H+->cb alike).  The top / W / QCD
    # nodes below are defined on the actual SM particles, so X(cs) and W(cs)
    # are genuinely different hypotheses, not a relabelling of one another.
    "bb": "X(bb)", "cc": "X(cc)", "bc": "X(bc)", "bs": "X(bs)", "cs": "X(cs)",
    "qq": "X(qq)", "ss": "X(ss)", "tauhtauh": "X(#tau_{h}#tau_{h})",
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
    "ak8_tau21": "#tau_{21}", "ak8_tau32": "#tau_{32}",
    "ak8_nConstituents": "N_{constit}",
}


def node_label(feat):
    return NODE_LABEL.get(feat.replace("ak8_gpt_", ""), feat.replace("ak8_gpt_", ""))
TOPO_COL = {1: ROOT.kAzure + 2, 2: ROOT.kOrange + 7,
            3: ROOT.kGreen + 2, 4: ROOT.kMagenta + 1, 5: ROOT.kCyan + 2}
TOPO_LAB = {1: "W(cb)", 2: "t^{2}(b'c)", 3: "t^{2}(b'b)", 4: "t^{3}(b'bc)",
            5: "t^{2}(b'c) proxy"}


def cms_style():
    s = ROOT.TStyle("cms", "cms")
    s.SetCanvasBorderMode(0); s.SetCanvasColor(0)
    s.SetPadBorderMode(0); s.SetPadColor(0)
    s.SetFrameBorderMode(0); s.SetFrameFillColor(0)
    s.SetOptStat(0); s.SetOptTitle(0)
    s.SetPadTickX(1); s.SetPadTickY(1)
    s.SetPadLeftMargin(0.125); s.SetPadRightMargin(0.035)
    s.SetPadTopMargin(0.085); s.SetPadBottomMargin(0.115)
    s.SetTextFont(42); s.SetLabelFont(42, "XYZ"); s.SetTitleFont(42, "XYZ")
    s.SetLabelSize(0.038, "XYZ"); s.SetTitleSize(0.045, "XYZ")
    s.SetTitleOffset(1.25, "X"); s.SetTitleOffset(1.45, "Y")
    s.SetLegendBorderSize(0); s.SetLegendFillColor(0)
    ROOT.gROOT.SetStyle("cms"); ROOT.gROOT.ForceStyle()


MARGIN = 0.130       # equal on all four sides -> the frame box is square
MARGIN_RANK = 0.170  # ranking plot: the node names need a little more room
CMS_X = 0.215        # canvas NDC
CMS_PX_TOP = 58.0    # header baseline, pixels below the canvas top
CMS_SIZE = 1.22      # common scale for the "CMS Simulation ... (13 TeV)" line
CMS_H_REF = 900.0    # canvas height the sizes are quoted for


def cms_header(cfg, canvas=None, dx=0.0, dy=0.0, scale=CMS_SIZE):
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
    t.DrawLatex(CMS_X + dx + 0.082 * scale, y, cfg.get("cms_label", "Simulation"))
    t.SetTextFont(42); t.SetTextSize(small); t.SetTextAlign(31)
    t.DrawLatex(0.965, y, cfg.get("lumi_label", "41.5 fb^{-1} (13 TeV)"))
    ROOT.SetOwnership(t, False)
    return t


def _axes(frame, xt, yt, tsize=0.040, lsize=0.034, xoff=1.30, yoff=1.55):
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
    c.SetLogy(); c.SetGridx(); c.SetGridy()
    frame = _axes(c.DrawFrame(0.0, 1e-3, 1.0, 1.0),
                  "signal efficiency", "background efficiency")

    entries = []
    for name in ("S1", "S1p"):
        f = outdir / name / "eval.npz"
        if not f.exists():
            continue
        e = np.load(f)
        entries.append((PRETTY[name], COL[name], 1, e["test_y"], e["test_score"], e["test_w"]))
    # S1 with the t2(b'c) class removed from the signal: a b'+c jet from W->cb
    # and one from W->cs contain the same two partons, so that class is very
    # nearly irreducible for ANY jet tagger -- this curve shows the headroom.
    e = np.load(outdir / "S1" / "eval.npz")
    keep = (e["test_topo"] != 2)
    entries.append((PRETTY["S1"] + " no t^{2}(b'c)", ROOT.kAzure + 1, 9,
                    e["test_y"][keep], e["test_score"][keep], e["test_w"][keep]))
    for key in CMP_ORDER:
        f = outdir / f"cmp_{key}.npz"
        if f.exists():
            d = np.load(f)
            entries.append((CMPLAB[key], COL[key], CMP_STYLE[key],
                            d["y"], d["score"], d["w"]))

    lg = _legend(0.535, 0.170, 0.950, 0.455, 0.0250)
    lg.SetFillStyle(1001); lg.SetFillColor(ROOT.kWhite); lg.SetMargin(0.20)
    keep = []
    for lab, col, ls, y, s, w in entries:
        tpr, fpr, auc = _roc_points(y, s, w)
        g = ROOT.TGraph(len(tpr), tpr, fpr)
        g.SetLineColor(col); g.SetLineWidth(3); g.SetLineStyle(ls)
        g.Draw("L SAME"); keep.append(g)
        lg.AddEntry(g, f"{lab}  AUC {auc:.3f}", "l")
    lg.Draw()
    cms_header(cfg, c)
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.DrawLatex(0.155, 0.862, "test jets, preselection")
    ROOT.SetOwnership(t, False)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return keep


def overtrain_figure(outdir, name, summary, cfg, out_png):
    e = np.load(outdir / name / "eval.npz")
    ot = summary["models"][name]["overtraining"]
    nb, lo, hi = 40, 0.0, 1.0

    def mk(tag, score, y, w, cls):
        h = ROOT.TH1F(f"h_{name}_{tag}", "", nb, lo, hi)
        h.Sumw2()
        m = y == cls
        for v, ww in zip(score[m], w[m]):
            h.Fill(float(v), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        return h

    h_str = mk("str", e["train_score"], e["train_y"], e["train_w"], 1)
    h_ste = mk("ste", e["test_score"], e["test_y"], e["test_w"], 1)
    h_btr = mk("btr", e["train_score"], e["train_y"], e["train_w"], 0)
    h_bte = mk("bte", e["test_score"], e["test_y"], e["test_w"], 0)

    for h, col in ((h_str, ROOT.kRed + 1), (h_ste, ROOT.kRed + 1),
                   (h_btr, ROOT.kAzure + 2), (h_bte, ROOT.kAzure + 2)):
        h.SetLineColor(col); h.SetMarkerColor(col)
    h_str.SetFillColorAlpha(ROOT.kRed + 1, 0.30); h_str.SetLineWidth(2)
    h_btr.SetFillColorAlpha(ROOT.kAzure + 2, 0.30); h_btr.SetLineWidth(2)
    h_ste.SetMarkerStyle(20); h_ste.SetMarkerSize(1.1)
    h_bte.SetMarkerStyle(21); h_bte.SetMarkerSize(1.1)

    c = ROOT.TCanvas(f"ot_{name}", "", SQ, SQ)
    c.SetLogy()
    ymax = max(h.GetMaximum() for h in (h_str, h_ste, h_btr, h_bte))
    frame = _axes(c.DrawFrame(lo, 3e-4, hi, ymax * 120),
                  f"{PRETTY[name]}  (BDT fine-tuned score)", "normalised to unit area")
    for h in (h_btr, h_str):
        h.Draw("HIST SAME")
    for h in (h_bte, h_ste):
        h.Draw("E1 SAME")

    # lower legend: two columns kept close together, frameless
    lg = _legend(0.16, 0.165, 0.80, 0.295, TXT)
    lg.SetNColumns(2); lg.SetColumnSeparation(-0.28); lg.SetMargin(0.16)
    lg.AddEntry(h_str, "signal train (70%)", "f")
    lg.AddEntry(h_ste, "signal test (15%)", "pe")
    lg.AddEntry(h_btr, "background train", "f")
    lg.AddEntry(h_bte, "background test", "pe")
    lg.Draw()

    ok = ot["verdict"] == "OK"
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    x0, y0, dy = 0.355, 0.865, 0.040
    t.DrawLatex(x0, y0, "#bf{overtraining test}  (test = held-out 15%)")
    t.DrawLatex(x0, y0 - dy, "Training bias > %.1f:  S %.1f%%,  B %.1f%%"
                % (ot["score_cut"], ot["rel_diff_sig_pct"], ot["rel_diff_bkg_pct"]))
    t.DrawLatex(x0, y0 - 2 * dy, "KS p-value:  S %.3f,  B %.3f"
                % (ot["ks_sig_p"], ot["ks_bkg_p"]))
    t.SetTextColor(ROOT.kGreen + 2 if ok else ROOT.kRed + 1)
    t.DrawLatex(x0, y0 - 3 * dy, "#bf{verdict: %s}" % ot["verdict"])
    ROOT.SetOwnership(t, False)

    cms_header(cfg, c)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_str, h_ste, h_btr, h_bte]


def permimp_figure(outdir, name, summary, cfg, out_png, ntop=15):
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

    h_imp = ROOT.TH1F(f"imp_{name}", "", n, 0, n)
    h_prm = ROOT.TH1F(f"prm_{name}", "", n, 0, n)
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

    c = ROOT.TCanvas(f"pi_{name}", "", SQ, SQ)
    c.SetLeftMargin(0.178); c.SetRightMargin(0.035)
    c.SetTopMargin(0.085); c.SetBottomMargin(0.105)
    h_prm.GetYaxis().SetTitle("share of total importance   [%]")
    h_prm.GetYaxis().SetTitleSize(0.044); h_prm.GetYaxis().SetLabelSize(0.036)
    h_prm.GetYaxis().SetTitleOffset(0.92)
    h_prm.GetXaxis().SetLabelSize(0.050)
    h_prm.SetMaximum(1.1 * max(max(imp[f] for f in top), max(pim[f] for f in top)))
    h_prm.SetMinimum(0.0)
    h_prm.Draw("HBAR")
    h_imp.Draw("HBAR SAME")

    lg = _legend(0.455, 0.585, 0.945, 0.700, TXT)
    lg.AddEntry(h_prm, "#DeltaAUC  (ranking)", "f")
    lg.AddEntry(h_imp, "%Imp  normalised gain (TMVA)", "f")
    lg.Draw()

    ot = m["overtraining"]
    xgb = summary["config"]["xgboost"]
    lines = [
        "#bf{%s}  ranked by #DeltaAUC   (%d out of %d input param/s)" % (PRETTY[name], n, m["n_features"]),
        "normalised = 100 #times x_{i}/#Sigma_{j}x_{j};  #Sigma#DeltaAUC = %.4f #rightarrow 1%% #equiv %.5f"
        % (p_tot, p_tot / 100.0),
        "trees %d (stop %d)  depth %d  lr %.2f  colsample %.1f"
        % (xgb["n_estimators"], m["best_iteration"], xgb["max_depth"],
           xgb["learning_rate"], xgb["colsample_bytree"]),
        "AUC  train %.3f  test %.3f    bias B %.1f%%  [%s]"
        % (m["train"]["auc"], m["test"]["auc"], ot["rel_diff_bkg_pct"], ot["verdict"]),
    ]
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(TXT)
    t.SetTextAlign(31)
    for i, ln in enumerate(lines):
        t.DrawLatex(0.955, 0.325 - 0.036 * i, ln)
    ROOT.SetOwnership(t, False)

    cms_header(cfg, c)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return [h_imp, h_prm]


def score_figure(outdir, name, summary, cfg, sig_codes, out_png):
    e = np.load(outdir / name / "eval.npz")
    s, w, topo = e["test_score"], e["test_w"], e["test_topo"]
    nb = 40
    n_bkg = int((topo == 0).sum()); n_sig = int((topo > 0).sum())

    def mk(tag, mask, col, width, style=1):
        h = ROOT.TH1F(f"s_{name}_{tag}", "", nb, 0.0, 1.0)
        h.Sumw2()
        for v, ww in zip(s[mask], w[mask]):
            h.Fill(float(v), float(ww))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        h.SetLineColor(col); h.SetLineWidth(width); h.SetLineStyle(style)
        h.SetMarkerColor(col)
        return h

    h_bkg = mk("bkg", topo == 0, ROOT.kAzure + 2, 3)
    h_sig = mk("sig", topo > 0, ROOT.kRed + 1, 3)
    h_cls = {c: mk(f"c{c}", topo == int(c), TOPO_COL[int(c)], 2, 2)
             for c in sig_codes if (topo == int(c)).sum() > 20}

    # frame width = SQ*(1-2*MARGIN); make the upper pad tall enough that the
    # MAIN frame comes out square even with the ratio pad underneath
    c = ROOT.TCanvas(f"sc_{name}", "", SQ, SQ)
    p1 = ROOT.TPad(f"p1{name}", "", 0, 0.30, 1, 1)
    p1.SetBottomMargin(0.02); p1.SetTopMargin(0.118)
    p2 = ROOT.TPad(f"p2{name}", "", 0, 0.0, 1, 0.30)
    p2.SetTopMargin(0.03); p2.SetBottomMargin(0.33)
    for pp in (p1, p2):
        pp.SetTicks(1, 1); pp.SetLeftMargin(0.125); pp.SetRightMargin(0.035)
    p1.Draw(); p2.Draw()

    p1.cd(); p1.SetLogy()
    ymax = max([h_bkg.GetMaximum(), h_sig.GetMaximum()]
               + [h.GetMaximum() for h in h_cls.values()])
    f1 = p1.DrawFrame(0.0, 2e-4, 1.0, ymax * 45)
    f1.GetYaxis().SetTitle("normalised to unit area")
    f1.GetYaxis().SetTitleSize(0.052); f1.GetYaxis().SetTitleOffset(1.10)
    f1.GetYaxis().SetLabelSize(0.044); f1.GetXaxis().SetLabelSize(0)
    for h in list(h_cls.values()) + [h_bkg, h_sig]:
        h.Draw("HIST SAME")
    # caption shifted ~20% to the right, with the totals
    lg = _legend(0.345, 0.525, 0.78, 0.855, 0.0400)
    lg.AddEntry(h_bkg, "background  (%s jets)" % f"{n_bkg:,}", "l")
    lg.AddEntry(h_sig, "signal all W#rightarrowcb  (%s)" % f"{n_sig:,}", "l")
    for cc, h in h_cls.items():
        lg.AddEntry(h, TOPO_LAB[int(cc)], "l")
    lg.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.042)
    t.DrawLatex(0.80, 0.845, PRETTY[name])
    ROOT.SetOwnership(t, False)
    p1.RedrawAxis()
    cms_header(cfg, c)          # canvas NDC -> same as the other three
    p1.cd()

    # ---- ratio pad: component / total signal, with propagated errors -------
    p2.cd()
    f2 = p2.DrawFrame(0.0, 0.0, 1.0, 2.0)
    f2.GetXaxis().SetTitle(f"{PRETTY[name]}  (BDT fine-tuned score)")
    f2.GetYaxis().SetTitle("comp. / total")
    f2.GetXaxis().SetTitleSize(0.125); f2.GetXaxis().SetLabelSize(0.100)
    f2.GetXaxis().SetTitleOffset(1.12)
    f2.GetYaxis().SetTitleSize(0.105); f2.GetYaxis().SetLabelSize(0.092)
    f2.GetYaxis().SetTitleOffset(0.48); f2.GetYaxis().SetNdivisions(505)
    ratios = []
    for cc, h in h_cls.items():
        r = h.Clone(f"r_{name}_{cc}")
        # the component is a SUBSET of the total, so the two are correlated:
        # var(r) = r^2 [ (1-2r) sx^2/x^2 + st^2/t^2 ]   (binomial-like)
        for b in range(1, nb + 1):
            x, sx = h.GetBinContent(b), h.GetBinError(b)
            tt, st = h_sig.GetBinContent(b), h_sig.GetBinError(b)
            if tt <= 0 or x < 0:
                r.SetBinContent(b, 0.0); r.SetBinError(b, 0.0); continue
            v = x / tt
            var = v * v * ((1.0 - 2.0 * v) * (sx / x) ** 2 + (st / tt) ** 2) \
                if x > 0 else 0.0
            r.SetBinContent(b, v); r.SetBinError(b, float(np.sqrt(max(var, 0.0))))
        r.SetLineWidth(2); r.SetLineStyle(1)
        r.SetMarkerSize(0); r.SetMarkerColor(h.GetLineColor())
        r.Draw("HIST SAME")          # same histogram style for all 4
        r.Draw("E1 SAME")            # ... plus the propagated errors
        ratios.append(r)
    one = ROOT.TLine(0.0, 1.0, 1.0, 1.0)
    one.SetLineStyle(2); one.SetLineColor(ROOT.kBlack); one.Draw()
    ROOT.SetOwnership(one, False)
    p2.RedrawAxis()
    c.SaveAs(str(out_png))
    return [h_bkg, h_sig] + list(h_cls.values()) + ratios


# --------------------------------------------------------------------------- #
def build_report(cfg, outdir):
    outdir = Path(outdir)
    summary = json.loads((outdir / "summary.json").read_text())
    tag = cfg.get("run_tag", cfg["dataset_tag"])
    ds_meta = json.loads((Path(summary["dataset_dir"]) / "dataset_complete.json").read_text())
    sig_codes = {int(k): v for k, v in ds_meta["signal_codes"].items()}

    cms_style()
    keep = []
    roc = outdir / "roc.png"
    keep += roc_figure(outdir, summary, cfg, roc)

    published = {f"S1_tagger_{tag}_roc.png": roc}
    for name in ("S1", "S1p"):
        if name not in summary["models"] or not (outdir / name / "eval.npz").exists():
            print(f"  [{name}] no eval.npz / not in summary -- skipping its plots")
            continue
        ov, pi, sc = (outdir / f"{k}_{name}.png"
                      for k in ("overtrain", "permimp", "score"))
        keep += overtrain_figure(outdir, name, summary, cfg, ov)
        keep += permimp_figure(outdir, name, summary, cfg, pi)
        keep += score_figure(outdir, name, summary, cfg, sig_codes, sc)
        panel = outdir / f"panel_{name}.png"
        montage([str(roc), str(ov), str(pi), str(sc),
                 "-tile", "2x2", "-geometry", f"{SQ}x{SQ}+2+2",
                 "-background", "white", str(panel)])
        published[f"S1_tagger_{tag}_panel_{name}.png"] = panel
        published[f"S1_tagger_{tag}_overtrain_{name}.png"] = ov
        published[f"S1_tagger_{tag}_permimp_{name}.png"] = pi
        published[f"S1_tagger_{tag}_score_{name}.png"] = sc

    for gname, src in published.items():
        shutil.copyfile(src, GALLERY / gname)
    print(f"  wrote {len(published)} figures -> {outdir}")
    print(f"  published -> {GALLERY}/S1_tagger_{tag}_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    build_report(cfg, HERE / cfg["output_dir"] / cfg.get("run_tag", cfg["dataset_tag"]))


if __name__ == "__main__":
    main()
