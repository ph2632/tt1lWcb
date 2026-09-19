#!/usr/bin/env python3
"""Montages of the MAT single plots (built after `04_Make_plots.py MAT-MULTI ...`).

1) MAT_ttreco_mSD_jstar_4block_6SR_{sig,bkg,combo}.png
   2 rows (SR a=looser, b=tighter) x 3 cols (SR1=D_cb, SR2=D_bb, SR3=D_bbc)
2) MAT_score_M3_9panel_PRE.png
   3 rows (Signal-only, Bkg-only, Signal+Bkg) x 3 cols (D_cb, D_bb, D_bbc), PRE

Each montage is built only if ALL of its singles exist; then exactly those
singles are deleted (per-panel cleanup, 2026-09-19 user).  Regenerate singles
with:  WCB_MAT_SPECS=... python 04_Make_plots.py MAT-MULTI SR1A,SR1B,...
usage: .venv/bin/python make_sr_montages.py [--keep]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORKDIR = Path(__file__).resolve().parent
KEEP = "--keep" in sys.argv
GUTTER, TRIM_R, TRIM_T, TRIM_B = 4, 16, 6, 10

try:
    FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 32)
except Exception:
    FONT = ImageFont.load_default()


def _trim(im, trim_b=None):
    w, h = im.size
    tb = TRIM_B if trim_b is None else trim_b
    return im.crop((0, TRIM_T, max(1, w - TRIM_R), max(1, h - tb)))


def build(grid, col_labels, row_labels, out_name, header_h, side_w, trim_b=None):
    """grid[r][c] = Path of a single PNG."""
    missing = [p.name for row in grid for p in row if not p.exists()]
    if missing:
        print(f"[SKIP] {out_name}: missing {len(missing)} single(s): {missing[:4]}...")
        return False
    cells = [[_trim(Image.open(p).convert("RGB"), trim_b) for p in row] for row in grid]
    cw = max(im.width for row in cells for im in row)
    ch = max(im.height for row in cells for im in row)
    nr, nc = len(cells), len(cells[0])
    W = side_w + nc * cw + (nc - 1) * GUTTER
    H = header_h + nr * ch + (nr - 1) * GUTTER
    panel = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(panel)
    for c, lbl in enumerate(col_labels):
        x = side_w + c * (cw + GUTTER) + cw // 2
        tw = d.textbbox((0, 0), lbl, font=FONT)[2]
        d.text((x - tw // 2, 10), lbl, fill="black", font=FONT)
    for r, lbl in enumerate(row_labels):
        t = Image.new("RGBA", (ch, side_w), (255, 255, 255, 0))
        td = ImageDraw.Draw(t)
        tw = td.textbbox((0, 0), lbl, font=FONT)[2]
        td.text((ch // 2 - tw // 2, 5), lbl, fill="black", font=FONT)
        t = t.rotate(90, expand=True)
        panel.paste(t, (0, header_h + r * (ch + GUTTER)), t)
    for r, row in enumerate(cells):
        for c, im in enumerate(row):
            panel.paste(im, (side_w + c * (cw + GUTTER), header_h + r * (ch + GUTTER)))
    panel.save(WORKDIR / out_name)
    print(f"[OK] {out_name}  ({W}x{H})")
    if not KEEP:
        n = 0
        for row in grid:
            for p in row:
                if p.exists():
                    p.unlink()
                    n += 1
        print(f"[CLEANUP] removed {n} single(s)")
    return True


def main():
    base = "MAT_ttreco_mSD_jstar_4block"
    for suffix, tag in (("_sig", "sig"), ("_bkg", "bkg"), ("", "combo")):
        grid = [[WORKDIR / f"{base}_SR{k}{ab}{suffix}.png" for k in (1, 2, 3)]
                for ab in ("A", "B")]
        build(grid, ["SR1 (D_cb)", "SR2 (D_bb)", "SR3 (D_bbc)"],
              ["a (looser)", "b (tighter)"], f"{base}_6SR_{tag}.png", 60, 60)
    # rows = modes, columns = taggers (transposed 2026-09-19, user)
    grid = [[WORKDIR / f"{b}_PRE{s}.png"
             for b in ("MAT_score_M3_cb", "MAT_score_S2", "MAT_score_S3")]
            for s in ("_sig", "_bkg", "")]
    build(grid, ["D_cb", "D_bb", "D_bbc"],
          ["Signal-only", "Bkg-only", "Signal + Bkg"],
          "MAT_score_M3_9panel_PRE.png", 50, 90, trim_b=0)   # x-titles sit at the very bottom


if __name__ == "__main__":
    main()
