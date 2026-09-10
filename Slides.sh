#!/bin/sh
# One-shot: build the 3x2 MAT panels, then bundle them into the slide PDF.
#
#   ./Slides.sh [OUTPUT.pdf]
#
# Equivalent to running, in order:
#     ./montage_mat_3x2.sh          # MAT_*_{PRE,SR}{,_sig,_bkg}.png  -> panel3x2_MAT_*.png
#     ./panels_to_pdf.sh [OUTPUT]   # panel3x2_MAT_*.png              -> OUTPUT (default panel_MAT_all.pdf)
#
# The PDF is byte-for-byte what `./panels_to_pdf.sh` alone would produce.

set -eu

cd "$(dirname "$0")"

echo "==> montage_mat_3x2.sh"
./montage_mat_3x2.sh

echo
echo "==> panels_to_pdf.sh"
./panels_to_pdf.sh "$@"
