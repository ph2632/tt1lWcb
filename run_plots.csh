#!/bin/tcsh -f
# Produce the SD-mass + Dbc-score Data/MC plots (process and matched-jets
# decompositions) at 600x600 into this directory, popping each up with `display`
# as it is written.
#
#   ./run_plots.csh              # ALL mode (default): the normal Data/MC batch plots
#   ./run_plots.csh MAT          # matching-truth overlay, PRESEL (ak8 pt>200)  -> MAT_*_PRE.png
#   ./run_plots.csh MAT SR       # SR cut (score_Dbc > 0.95) -> MAT_*_SR.png
#   ./run_plots.csh MAT JB       # jb-Region: mSD(j_b) 65-105 & tau21(j_b)<0.5 & Dbc(j_a)>0.8 -> MAT_*_JB.png
#   ./run_plots.csh MAT NORM SR  # + shape-normalized y-axis (ABS|NORM and PRESEL|SR|JB, any order)
#
# Everything is self-contained in this directory: the Python interpreter lives
# in ./.venv , the inputs are the parquet caches under ./cache_batch_parquet ,
# and the PNGs land here.
#
# Env knobs:  SHOW=0 (no pop-ups)   PNG_VIEWER=eog   (alt viewer)

set here = `dirname $0`
cd $here
./.venv/bin/python Make_plots.py $argv
