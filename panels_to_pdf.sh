#!/bin/sh
# Bundle the MAT panel plots in the current directory into ONE pdf,
# one panel per page.
#
# For every variable we prefer the 3x2 layout (panel3x2_MAT_<var>.png) and
# fall back to the 2x2 layout (panel2x2_MAT_<var>.png) only when the 3x2
# file is absent.  Variables with neither file are simply skipped.
#
#   ./panels_to_pdf.sh [OUTPUT.pdf]
#
#   OUTPUT.pdf   name of the combined file   (default: panel_MAT_all.pdf)
#
# Layout preference order can be overridden with PANEL_PREFIXES (space
# separated, highest priority first).  Needs ImageMagick `convert`.

set -eu

OUT=${1:-panel_MAT_all.pdf}
PREFIXES=${PANEL_PREFIXES:-'panel3x2_MAT_ panel2x2_MAT_'}

# --- collect the set of variable names across every layout ----------------
VARS=$(
    for p in $PREFIXES; do
        for f in ${p}*.png; do
            [ -e "$f" ] || continue
            v=${f#"$p"}; v=${v%.png}
            printf '%s\n' "$v"
        done
    done | sort -u -V
)

if [ -z "$VARS" ]; then
    echo "panels_to_pdf: no panel*_MAT_*.png in $(pwd) -- nothing to do." >&2
    exit 1
fi

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

# -auto-orient: honour any EXIF rotation; default Zip (lossless) compression.
convert $FILES -auto-orient "$OUT"

echo "panels_to_pdf: wrote $OUT ($(du -h "$OUT" | cut -f1))"
