#!/usr/bin/env python3
"""
Score-spectrum comparison: t3(b'bc) signal vs t3(b'cq) proxy, and the
t2(b'c) reference pair, for S1/S2/S3.  Same CMS style as XGBoost_training.py.

  ./.venv/bin/python S1_tagger/t3_proxy_hist.py
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LCG_VIEW = os.environ.get(
    "S1_LCG_VIEW", "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt")
try:
    import ROOT
except ImportError:
    setup = Path(LCG_VIEW) / "setup.sh"
    argv = " ".join(shlex.quote(a) for a in sys.argv)
    os.execvp("bash", ["bash", "-c",
                       f"source {shlex.quote(str(setup))} >/dev/null 2>&1; "
                       f"exec python3 {argv}"])

import numpy as np
import pandas as pd

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

GALLERY = HERE.parent

# 03_render_report.py (moved to tt1lWcb/ root, 2026-09-13; was S1_tagger/
# XGBoost_training.py) starts with a digit, so it isn't a valid module name
# for a plain "import" -- load it by path instead.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_render_report", GALLERY / "03_render_report.py")
_render_report = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_render_report)
cms_style, cms_header, _axes, _legend, SQ, TXT, montage = (
    _render_report.cms_style, _render_report.cms_header, _render_report._axes,
    _render_report._legend, _render_report.SQ, _render_report.TXT,
    _render_report.montage)
PAIRS = [("t3bbc", "sig_t3bbc", "prx_t3bcq", "t^{3}(b'bc)", "t^{3}(b'cq) proxy"),
         ("t2bc", "sig_t2bc", "prx_t2bc", "t^{2}(b'c)", "t^{2}(b'c) proxy")]


def make(name, score_col, pretty, cfg, out_png):
    df = pd.read_parquet(HERE / "t3_proxy_scored.parquet")
    c = ROOT.TCanvas(f"c_{name}", "", SQ, SQ)
    c.SetTopMargin(0.075); c.SetRightMargin(0.03)
    c.SetLeftMargin(0.13); c.SetBottomMargin(0.11); c.SetLogy()
    frame = _axes(c.DrawFrame(0, 1e-3, 1, 12), f"{pretty}  score", "N_{jets} norm. to 1")

    cols = {"sig_t3bbc": ROOT.kRed + 1, "prx_t3bcq": ROOT.kMagenta + 1,
            "sig_t2bc": ROOT.kAzure + 1, "prx_t2bc": ROOT.kGreen + 2}
    styles = {"sig_t3bbc": 1, "prx_t3bcq": 2, "sig_t2bc": 1, "prx_t2bc": 2}
    labs = {"sig_t3bbc": "t^{3}(b'bc) signal", "prx_t3bcq": "t^{3}(b'cq) proxy",
            "sig_t2bc": "t^{2}(b'c) signal", "prx_t2bc": "t^{2}(b'c) proxy"}
    keep, entries = [], []
    for cat in ("sig_t3bbc", "prx_t3bcq", "sig_t2bc", "prx_t2bc"):
        sub = df[df.cat == cat]
        h = ROOT.TH1F(f"h_{name}_{cat}", "", 25, 0, 1)
        h.Sumw2()
        for v, w in zip(sub[score_col].values, sub["w"].values):
            h.Fill(float(v), float(w))
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        h.SetLineColor(cols[cat]); h.SetLineWidth(3); h.SetLineStyle(styles[cat])
        h.Draw("HIST SAME"); keep.append(h); entries.append((h, labs[cat]))

    lg = _legend(0.40, 0.65, 0.90, 0.90, TXT * 1.1)
    for h, lab in entries:
        lg.AddEntry(h, lab, "l")
    lg.Draw()
    cms_header(cfg, c)
    c.RedrawAxis(); c.SaveAs(str(out_png))
    return keep


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    cms_style()
    singles = []
    for tg in ("S1", "S2", "S3"):
        col = f"score_{tg}"
        df0 = pd.read_parquet(HERE / "t3_proxy_scored.parquet", columns=[col]) \
            if singles else None
        out = HERE / f"t3_proxy_{tg}.png"
        try:
            make(tg, col, tg, cfg, out)
        except KeyError:
            print(f"  [skip] {tg}: not scored"); continue
        singles.append(out)
        print(f"  wrote {out.name}")
    if singles:
        panel = HERE / "t3_proxy_panel.png"
        montage([str(p) for p in singles] + ["-tile", f"{len(singles)}x1",
                "-geometry", f"{SQ}x{SQ}+0+0", "-background", "white", str(panel)])
        for f in [panel] + singles:
            import shutil
            shutil.copyfile(f, GALLERY / f"S1_tagger_t3proxy_{f.stem}.png")
        print(f"  wrote {panel.name}, published {len(singles)+1} files")


if __name__ == "__main__":
    main()
