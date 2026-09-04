#!/bin/bash
# Assemble summary panels from the MAT (matching-truth overlay) PNGs that
# `Make_plots.py MAT` writes into the figure directory. Called automatically
# at the end of that run; can also be run by hand:
#
#     ./montage_mat.sh [FIGURE_DIR] [SUFFIX]   # default dir ".", suffix ""
#
# SUFFIX ("_PRE" / "_SR") picks the per-selection MAT_*<SUFFIX>.png set and is
# also appended to the montage_* output names.
#
# Produces (montage_MAT_*<SUFFIX>.png):
#   montage_MAT_mSD     3x2 : the 6 m_SD orderings
#                             (j_a = cb candidate / j^1 = max mSD /
#                              j_1 = max pT / j_2 / j^2 / j_b)
#   montage_MAT_scores  3x3 : the raw GloParT leading-AK8 tagger scores
#   montage_MAT_pT      3x3 : the pT / energy-scale observables
#                             (AK8/AK4/lepton pT, MET, mT^W, V pT,
#                              H_T, H_T(bc), pT^l/pT^J)

DIR="${1:-.}"
SFX="${2:-}"
cd "$DIR" || { echo "[montage_mat] cannot cd to $DIR"; exit 1; }

if ! command -v montage >/dev/null 2>&1; then
    echo "[montage_mat] ImageMagick 'montage' not on PATH -- skipping panels."
    exit 0
fi

GEOM="-geometry +4+4 -background white"

make_panel() {
    # $1 = output png, $2 = tile spec, $3.. = ordered candidate input pngs
    local out="$1" tile="$2"; shift 2
    local files=()
    for f in "$@"; do
        if [ -f "$f" ]; then files+=("$f"); else echo "[montage_mat]   missing: $f"; fi
    done
    if [ "${#files[@]}" -eq 0 ]; then
        echo "[montage_mat] no inputs for $out -- skipped."
        return
    fi
    montage "${files[@]}" -tile "$tile" $GEOM "$out"
    echo "[montage_mat] wrote $out  (${#files[@]} plots, tile $tile)"
}

# --- Panel 1: soft-drop mass under the six jet orderings -------------------
make_panel "montage_MAT_mSD${SFX}.png" 3x2 \
    "MAT_ak8_sdmass_0${SFX}.png" \
    "MAT_ak8_sdmass_maxmass_0${SFX}.png" \
    "MAT_ak8_sdmass_lead_pt_0${SFX}.png" \
    "MAT_ak8_sdmass_sub_pt_0${SFX}.png" \
    "MAT_ak8_sdmass_sub_mass_0${SFX}.png" \
    "MAT_ak8_sdmass_sub_bc_0${SFX}.png"

# --- Panel 2: raw GloParT tagger scores (leading AK8) -- 9 live nodes -----
# (ak8_gpt_bqq is identically 0 in these ntuples -> omitted)
make_panel "montage_MAT_scores${SFX}.png" 3x3 \
    "MAT_gpt_bc${SFX}.png" \
    "MAT_gpt_bb${SFX}.png" \
    "MAT_gpt_cc${SFX}.png" \
    "MAT_gpt_cs${SFX}.png" \
    "MAT_gpt_bs${SFX}.png" \
    "MAT_gpt_qq${SFX}.png" \
    "MAT_gpt_qcd${SFX}.png" \
    "MAT_gpt_topbw${SFX}.png" \
    "MAT_gpt_topw${SFX}.png"

# --- Panel 3: pT / energy-scale observables (incl. MET and H_T) ----------
make_panel "montage_MAT_pT${SFX}.png" 3x3 \
    "MAT_ak8_pt_0${SFX}.png" \
    "MAT_lep1_pt${SFX}.png" \
    "MAT_met${SFX}.png" \
    "MAT_mTW${SFX}.png" \
    "MAT_v_pt${SFX}.png" \
    "MAT_ht${SFX}.png" \
    "MAT_ht_bc${SFX}.png" \
    "MAT_ptrel_lep_ak8${SFX}.png"

echo "[montage_mat] done."
