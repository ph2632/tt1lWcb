#!/bin/bash
# Merge each sextuplet of MAT plots into one 3x2 panel.
#
# For a given base name B (as written by Make_plots.py: "MAT_<var>"), the six
# single plots are the three composition modes x the two selections:
#
#                Signal-only        Bkg-only          Signal + Bkg
#   Preselection  B_PRE_sig.png      B_PRE_bkg.png     B_PRE.png
#   SR            B_SR_sig.png       B_SR_bkg.png      B_SR.png
#
#   ( _sig  <- `Make_plots.py MAT-SIGNAL`,
#     _bkg  <- `Make_plots.py MAT-BKG`,
#     none  <- `Make_plots.py MAT` )
#
# Layout: columns = signal-only / bkg-only / signal+bkg,
#         rows    = Preselection (top) / SR (bottom).
#
# - All 6 present  -> panel is written AND the 6 single plots are deleted.
# - 2..5 present   -> panel is still written (missing cells become a grey
#                     "(not found)" placeholder), but NO single plot is deleted.
# - 0 or 1         -> skipped, nothing touched.
#
# Usage:
#   ./montage_mat_3x2.sh                 # every MAT_* base found here
#   ./montage_mat_3x2.sh -k              # never delete the single plots
#   ./montage_mat_3x2.sh MAT_met ...     # only the listed base name(s)
#
# See ../Hgg/montage*.sh for the montage/convert idiom this follows.

set -u

KEEP=0
if [ "${1:-}" = "-k" ] || [ "${1:-}" = "--keep" ]; then
    KEEP=1
    shift
fi

OUT_PREFIX="panel3x2_"       # output file:  ${OUT_PREFIX}<base>.png
GEOM="+0+0"                   # padding between cells
HEADER_H=30                   # px added on top for the column labels
SIDE_W=34                     # px added on the left for the (rotated) row labels
PH_W=600                      # placeholder cell size (matches the ROOT canvas)
PH_H=600

COL_LABELS=( "Signal-only" "Bkg-only" "Signal + Bkg" )
ROW_LABELS=( "Preselection" "SR" )

# per-base file suffixes, row-major (row = selection, col = composition mode)
#   row 0 = PRE : _PRE_sig  _PRE_bkg  _PRE
#   row 1 = SR  : _SR_sig   _SR_bkg   _SR
CELL_SUFFIX=( "_PRE_sig.png" "_PRE_bkg.png" "_PRE.png" \
              "_SR_sig.png"  "_SR_bkg.png"  "_SR.png" )
NCOL=3
NROW=2

# ── Collect the base names to process ───────────────────────────────────────
BASES=()
if [ $# -gt 0 ]; then
    BASES=( "$@" )
else
    shopt -s nullglob
    for f in MAT_*_PRE_sig.png MAT_*_SR_sig.png \
             MAT_*_PRE_bkg.png MAT_*_SR_bkg.png \
             MAT_*_PRE.png     MAT_*_SR.png; do
        b="$f"
        b="${b%_PRE_sig.png}"; b="${b%_SR_sig.png}"
        b="${b%_PRE_bkg.png}"; b="${b%_SR_bkg.png}"
        b="${b%_PRE.png}";     b="${b%_SR.png}"
        BASES+=( "$b" )
    done
    shopt -u nullglob
    if [ ${#BASES[@]} -gt 0 ]; then
        IFS=$'\n' BASES=( $(printf '%s\n' "${BASES[@]}" | sort -u) ); unset IFS
    fi
fi

if [ ${#BASES[@]} -eq 0 ]; then
    echo "No MAT_* single-plot PNGs found here - nothing to do."
    exit 0
fi

N_DONE=0
N_PARTIAL=0
N_SKIP=0

for BASE in "${BASES[@]}"; do
    FILES=()
    PRESENT=()
    for sfx in "${CELL_SUFFIX[@]}"; do
        f="${BASE}${sfx}"
        FILES+=( "$f" )
        [ -f "$f" ] && PRESENT+=( "$f" )
    done
    NP=${#PRESENT[@]}

    if [ "$NP" -lt 2 ]; then
        echo "SKIP    ${BASE}  (${NP}/6 present)"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    OUTPUT="${OUT_PREFIX}${BASE}.png"
    TEMP="$(mktemp --suffix=.png 2>/dev/null || echo "/tmp/panel3x2_$$_${BASE//\//_}.png")"
    PH=""

    make_ph() {
        [ -n "$PH" ] && return
        PH="$(mktemp --suffix=.png 2>/dev/null || echo "/tmp/ph3x2_$$.png")"
        convert -size ${PH_W}x${PH_H} xc:grey92 \
            -font Helvetica -pointsize 30 -fill grey45 \
            -gravity center -annotate 0 "(not found)" "$PH" 2>/dev/null
    }

    CELLS=()
    for f in "${FILES[@]}"; do
        if [ -f "$f" ]; then
            CELLS+=( "$f" )
        else
            make_ph
            CELLS+=( "$PH" )
        fi
    done

    # ── raw 3x2 montage, row-major ────────────────────────────────────────
    montage "${CELLS[@]}" \
        -tile ${NCOL}x${NROW} -geometry "$GEOM" -background white "$TEMP"
    if [ $? -ne 0 ]; then
        echo "ERROR   ${BASE}  (montage failed)"
        rm -f "$TEMP" "$PH"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    TOTAL_W=$(identify -format "%w" "$TEMP" 2>/dev/null)
    TOTAL_H=$(identify -format "%h" "$TEMP" 2>/dev/null)
    CELL_W=$(( ${TOTAL_W:-1800} / NCOL ))
    CELL_H=$(( ${TOTAL_H:-1200} / NROW ))

    # ── bold column headers (top strip) ──────────────────────────────────
    LTMP="$(mktemp --suffix=.png 2>/dev/null || echo /tmp/lbl3x2_$$.png)"
    HDR_ARGS=()
    for c in $(seq 0 $((NCOL - 1))); do
        lbl="${COL_LABELS[$c]}"
        # rough centring: ~9 px per bold char at pointsize 29
        half=$(( ${#lbl} * 9 / 2 ))
        x=$(( SIDE_W + CELL_W * c + CELL_W / 2 - half ))
        HDR_ARGS+=( -annotate +${x}+3 "$lbl" )
    done
    convert "$TEMP" \
        -gravity North -splice 0x${HEADER_H} \
        -gravity West  -splice ${SIDE_W}x0   \
        -font Helvetica-Bold -fill black -pointsize 29 \
        -gravity NorthWest \
        "${HDR_ARGS[@]}" \
        "$LTMP"

    # ── bold rotated row labels (left strip) ──────────────────────────────
    CONV_ARGS=( "$LTMP" )
    RLS=()
    for r in $(seq 0 $((NROW - 1))); do
        rl="$(mktemp --suffix=.png 2>/dev/null || echo /tmp/rl3x2_${r}_$$.png)"
        RLS+=( "$rl" )
        convert -background none -fill black -font Helvetica-Bold -pointsize 26 \
            -gravity center -size ${CELL_H}x${SIDE_W} caption:"${ROW_LABELS[$r]}" \
            -rotate -90 "$rl"
        CONV_ARGS+=( "$rl" -gravity NorthWest \
                     -geometry +0+$(( HEADER_H + CELL_H * r )) -composite )
    done
    convert "${CONV_ARGS[@]}" "$OUTPUT"
    STATUS=$?
    rm -f "$TEMP" "$PH" "$LTMP" "${RLS[@]}"

    if [ $STATUS -ne 0 ]; then
        echo "ERROR   ${BASE}  (convert labelling failed)"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    if [ "$NP" -eq 6 ]; then
        echo "OK      ${BASE}  ->  ${OUTPUT}"
        N_DONE=$((N_DONE + 1))
        if [ $KEEP -eq 0 ]; then
            rm -f "${FILES[@]}"
        fi
    else
        echo "PARTIAL ${BASE}  ->  ${OUTPUT}   (${NP}/6; singles kept)"
        N_PARTIAL=$((N_PARTIAL + 1))
    fi
done

echo "----"
echo "full panels: ${N_DONE}   partial panels: ${N_PARTIAL}   skipped: ${N_SKIP}"
[ $KEEP -eq 1 ] && echo "(-k: no single plots deleted)"
