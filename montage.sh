#!/bin/bash
# Create montage panel for a given region and channel
# Usage: ./montage.sh <region> <channel>
# Example: ./montage.sh Preselection 0l
#          ./montage.sh Preselection 1l
#          ./montage.sh Preselection 2l
#          ./montage.sh BDT              # 3x2 panel of TMVA_BDT summaries
#          ./montage.sh 0l               # creates 2 panels: ParT (~23) + KIN (38-64)

# Special case: BDT summary montage (3x2 panel)
if [ "$1" == "BDT" ]; then
    echo "Creating BDT summary montage (3x2: ParT|KIN x 0l|1l|2l)"

    # Define files in order: rows=channels, cols=ParT|KIN
    FILES=""
    for ch in 0l 1l 2l; do
        for type in ParT KIN; do
            f="TMVA_BDT_${type}_${ch}_summary.png"
            if [ -f "$f" ]; then
                FILES="$FILES $f"
            else
                echo "Warning: $f not found"
            fi
        done
    done

    if [ -z "$FILES" ]; then
        echo "No BDT summary files found"
        exit 1
    fi

    # Create basic montage first
    TEMP_MONTAGE="/tmp/montage_bdt_temp.png"
    OUTPUT="montage_BDT_summary.png"
    montage $FILES -tile 2x3 -geometry +4+4 -background white $TEMP_MONTAGE

    # Get dimensions for label positioning
    W=$(identify -format "%w" $TEMP_MONTAGE)
    H=$(identify -format "%h" $TEMP_MONTAGE)
    CELL_W=$((W / 2))
    CELL_H=$((H / 3))

    # Add column headers and row labels (pointsize 48, row labels closer to plots)
    convert $TEMP_MONTAGE \
        -gravity North -splice 0x80 \
        -gravity West -splice 100x0 \
        -font Helvetica -pointsize 48 \
        -fill black \
        -gravity NorthWest \
        -annotate +$((100 + CELL_W/2 - 180))+15 "ParT finetuned Tagger" \
        -annotate +$((100 + CELL_W + CELL_W/2 - 140))+15 "Kinematics BDT" \
        -annotate +50+$((80 + CELL_H/2 - 20)) "0ℓ" \
        -annotate +50+$((80 + CELL_H + CELL_H/2 - 20)) "1ℓ" \
        -annotate +50+$((80 + 2*CELL_H + CELL_H/2 - 20)) "2ℓ" \
        $OUTPUT

    rm -f $TEMP_MONTAGE
    echo "Created: $OUTPUT"
    display $OUTPUT &
    echo "Done!"
    exit 0
fi

# Channel-only mode: ./montage.sh 0l (or 1l, 2l)
# Creates TWO separate panels: ParT (~20 vars) and KIN (rest)
if [ $# -eq 1 ] && [[ "$1" =~ ^[012]l$ ]]; then
    CH=$1
    echo "Creating separate ParT and KIN montages for channel: $CH"

    # Function to determine tile layout based on file count
    get_tile() {
        local N=$1
        if [ "$N" -le 2 ]; then echo "2x1"
        elif [ "$N" -le 6 ]; then echo "3x2"
        elif [ "$N" -le 8 ]; then echo "4x2"
        elif [ "$N" -le 12 ]; then echo "4x3"
        elif [ "$N" -le 18 ]; then echo "6x3"
        elif [ "$N" -le 20 ]; then echo "5x4"
        elif [ "$N" -le 24 ]; then echo "6x4"
        elif [ "$N" -le 32 ]; then echo "8x4"
        elif [ "$N" -le 40 ]; then echo "8x5"
        elif [ "$N" -le 45 ]; then echo "9x5"
        elif [ "$N" -le 50 ]; then echo "10x5"
        elif [ "$N" -le 60 ]; then echo "10x6"
        else echo "10x7"
        fi
    }

    # === Panel 1: ParT tagger scores only (4x5 grid = 20 plots) ===
    PART_FILES=$(ls plot_Preselection_*ak15_ParTMDV2_*_${CH}.png 2>/dev/null | sort)

    if [ -n "$PART_FILES" ]; then
        NPART=$(echo "$PART_FILES" | wc -l | tr -d ' ')
        echo "ParT panel: $NPART plots (ParT tagger scores only)"
        TILE_PART="5x4"  # Fixed 5x4 grid for ParT (5 cols, 4 rows)
        echo "  Tile layout: $TILE_PART"
        OUTPUT_PART="montage_ParT_${CH}.png"
        montage $PART_FILES -tile $TILE_PART -geometry +2+2 -background white $OUTPUT_PART
        echo "  Created: $OUTPUT_PART"
    else
        echo "No ParT files found"
    fi

    # === Panel 2: KIN variables (max 8 columns) ===
    # Priority: sdmass, BDT_ParT, BDT_KIN first, then all other kinematic variables
    KIN_PRIORITY=""
    for prio in "ak15_sdmass" "BDT_ParT" "BDT_KIN"; do
        f="plot_Preselection_${prio}_${CH}.png"
        [ -f "$f" ] && KIN_PRIORITY="$KIN_PRIORITY $f"
    done

    # All other plots except ParT tagger scores and priority files
    KIN_REST=$(ls plot_Preselection_*_${CH}.png 2>/dev/null | grep -v 'ak15_ParTMDV2_' | grep -v 'ak15_sdmass' | grep -v 'BDT_ParT' | grep -v 'BDT_KIN' | sort)

    # Combine priority + rest
    KIN_FILES="$KIN_PRIORITY $KIN_REST"
    KIN_FILES=$(echo $KIN_FILES | tr ' ' '\n' | grep -v '^$' | tr '\n' ' ')

    if [ -n "$KIN_FILES" ]; then
        NKIN=$(echo $KIN_FILES | wc -w | tr -d ' ')
        echo "KIN panel: $NKIN plots (sdmass + BDT scores + kinematic variables)"
        TILE_KIN=$(get_tile $NKIN)
        echo "  Tile layout: $TILE_KIN"
        OUTPUT_KIN="montage_KIN_${CH}.png"
        montage $KIN_FILES -tile $TILE_KIN -geometry +2+2 -background white $OUTPUT_KIN
        echo "  Created: $OUTPUT_KIN"
    else
        echo "No KIN files found"
    fi

    # Display both panels
#    [ -f "$OUTPUT_PART" ] && display $OUTPUT_PART &
#    [ -f "$OUTPUT_KIN" ] && display $OUTPUT_KIN &

    echo "Done!"

    # Display all created montages
    display montage_*_${CH}.png &
    exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <region> <channel>"
    echo "       $0 BDT                    # 3x2 BDT summary panel"
    echo "       $0 <channel>              # ParT + KIN panels for channel (0l, 1l, 2l)"
    echo "  region:  e.g., Preselection, SR, CR"
    echo "  channel: 0l, 1l, or 2l"
    echo ""
    echo "Both modes create two separate montages:"
    echo "  - montage_[region_]ParT_<ch>.png: 4x5 grid (sdmass, BDT scores, ParT tagger)"
    echo "  - montage_[region_]KIN_<ch>.png:  8xN grid (kinematic variables)"
    exit 1
fi

REGION=$1
CH=$2

echo "Creating separate ParT and KIN montages for region: $REGION, channel: $CH"

# Function to determine tile layout (max 10 columns for KIN)
get_tile_kin() {
    local N=$1
    if [ "$N" -le 2 ]; then echo "2x1"
    elif [ "$N" -le 6 ]; then echo "3x2"
    elif [ "$N" -le 8 ]; then echo "4x2"
    elif [ "$N" -le 10 ]; then echo "5x2"
    elif [ "$N" -le 20 ]; then echo "10x2"
    elif [ "$N" -le 30 ]; then echo "10x3"
    elif [ "$N" -le 40 ]; then echo "10x4"
    elif [ "$N" -le 50 ]; then echo "10x5"
    elif [ "$N" -le 60 ]; then echo "10x6"
    elif [ "$N" -le 70 ]; then echo "10x7"
    elif [ "$N" -le 80 ]; then echo "10x8"
    else echo "10x9"
    fi
}

# === Panel 1: ParT tagger scores only (4x5 grid = 20 plots) ===
PART_FILES=$(ls plot_${REGION}_*ak15_ParTMDV2_*_${CH}.png 2>/dev/null | sort)

if [ -n "$PART_FILES" ]; then
    NPART=$(echo "$PART_FILES" | wc -l | tr -d ' ')
    echo "ParT panel: $NPART plots (ParT tagger scores only)"
    TILE_PART="5x4"  # Fixed 5x4 grid for ParT (5 cols, 4 rows)
    echo "  Tile layout: $TILE_PART"
    OUTPUT_PART="montage_${REGION}_ParT_${CH}.png"
    montage $PART_FILES -tile $TILE_PART -geometry +2+2 -background white $OUTPUT_PART
    echo "  Created: $OUTPUT_PART"
else
    echo "No ParT files found"
fi

# === Panel 2: KIN variables (max 8 columns) ===
# Priority: sdmass, BDT_ParT, BDT_KIN first, then all other kinematic variables
KIN_PRIORITY=""
for prio in "ak15_sdmass" "BDT_ParT" "BDT_KIN"; do
    f="plot_${REGION}_${prio}_${CH}.png"
    [ -f "$f" ] && KIN_PRIORITY="$KIN_PRIORITY $f"
done

# All other plots except ParT tagger scores and priority files
KIN_REST=$(ls plot_${REGION}_*_${CH}.png 2>/dev/null | grep -v 'ak15_ParTMDV2_' | grep -v 'ak15_sdmass' | grep -v 'BDT_ParT' | grep -v 'BDT_KIN' | sort)

# Combine priority + rest
KIN_FILES="$KIN_PRIORITY $KIN_REST"
KIN_FILES=$(echo $KIN_FILES | tr ' ' '\n' | grep -v '^$' | tr '\n' ' ')

if [ -n "$KIN_FILES" ]; then
    NKIN=$(echo $KIN_FILES | wc -w | tr -d ' ')
    echo "KIN panel: $NKIN plots (sdmass + BDT scores + kinematic variables)"
    TILE_KIN=$(get_tile_kin $NKIN)
    echo "  Tile layout: $TILE_KIN"
    OUTPUT_KIN="montage_${REGION}_KIN_${CH}.png"
    montage $KIN_FILES -tile $TILE_KIN -geometry +2+2 -background white $OUTPUT_KIN
    echo "  Created: $OUTPUT_KIN"
else
    echo "No KIN files found"
fi

# Display both panels
#[ -f "$OUTPUT_PART" ] && display $OUTPUT_PART &
#[ -f "$OUTPUT_KIN" ] && display $OUTPUT_KIN &

echo "Done!"

# Display all created montages
display montage_*_*l.png &
