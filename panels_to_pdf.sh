#!/bin/sh
# Bundle the MAT panel plots in the current directory into ONE pdf,
# one panel per page.
#
# Only the 3x2 layout (panel3x2_MAT_<var>.png) is used -- the 2x2 panels are
# NOT included (2026-09-07, user).  Variables with no 3x2 file are skipped.
#
#   ./panels_to_pdf.sh [OUTPUT.pdf]
#
#   OUTPUT.pdf   name of the combined file   (default: panel_MAT_all.pdf)
#
# The panel prefix list can be overridden with PANEL_PREFIXES (space
# separated, highest priority first).  Needs ImageMagick `convert`.

set -eu

OUT=${1:-panel_MAT_all.pdf}
PREFIXES=${PANEL_PREFIXES:-'panel3x2_MAT_'}
# Variables hoisted to the front of the PDF, in this order, when a panel for
# them exists (space separated, override with PANEL_FIRST).  Default: the
# gen-content composition plot (MAT_ttreco_ja_truth -- 15 category bins, every
# component shown per bin) leads the deck.
FIRST_VARS=${PANEL_FIRST:-'ttreco_ja_truth'}

# --- collect the set of variable names across every layout ----------------
ALL_VARS=$(
    for p in $PREFIXES; do
        for f in ${p}*.png; do
            [ -e "$f" ] || continue
            v=${f#"$p"}; v=${v%.png}
            printf '%s\n' "$v"
        done
    done | sort -u -V
)

if [ -z "$ALL_VARS" ]; then
    echo "panels_to_pdf: no panel*_MAT_*.png in $(pwd) -- nothing to do." >&2
    exit 1
fi

# --- hoist the FIRST_VARS that actually have a panel, keep the rest sorted --
VARS=""
for fv in $FIRST_VARS; do
    for v in $ALL_VARS; do
        [ "$v" = "$fv" ] && { VARS="$VARS $v"; break; }
    done
done
for v in $ALL_VARS; do
    skip=0
    for fv in $FIRST_VARS; do
        [ "$v" = "$fv" ] && skip=1
    done
    [ "$skip" = 0 ] && VARS="$VARS $v"
done

# --- for each variable pick the first layout that exists -----------------
FILES=""
for v in $VARS; do
    for p in $PREFIXES; do
        cand="${p}${v}.png"
        if [ -e "$cand" ]; then
            FILES="$FILES $cand"
            break
        fi
    done
done

N=$(printf '%s\n' $FILES | wc -l | tr -d ' ')
echo "panels_to_pdf: $N panel(s) -> $OUT"
printf '  %s\n' $FILES

# --- stamp a slide number (lower-left) onto each panel -------------------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
i=0
STAMPED=""
for f in $FILES; do
    i=$((i + 1))
    out=$(printf '%s/%03d.png' "$TMP" "$i")
    # point size scaled to the panel height so the stamp looks the same on
    # every layout; white under-box so the digits stay readable on any
    # background.  SouthWest gravity + a small proportional inset.
    h=$(identify -format '%h' "$f" 2>/dev/null || echo 600)
    ps=$(( h / 30 )); [ "$ps" -lt 14 ] && ps=14
    off=$(( ps / 2 ))
    convert "$f" -auto-orient \
        -gravity SouthWest -pointsize "$ps" -font Helvetica \
        -fill black -undercolor '#ffffffcc' \
        -annotate +${off}+${off} " $i / $N " \
        "$out"
    STAMPED="$STAMPED $out"
done

# default Zip (lossless) compression.
convert $STAMPED "$OUT"

echo "panels_to_pdf: wrote $OUT ($(du -h "$OUT" | cut -f1))"
