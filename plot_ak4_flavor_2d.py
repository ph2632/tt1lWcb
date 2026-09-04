#!/usr/bin/env python3
"""AK4 jet flavour 2D efficiency-density in ParticleNet score plane for Wcb signal.

X: p_{b+c}     = ParticleNetAK4 p_b + p_c
Y: p_{b vs c}  = p_b / (p_b + p_c)

Three flavours are overlaid as filled contour efficiency-density:
    b     : CMS red
    c     : CMS blue
    light : CMS yellow

For each truth flavour, the weighted AK4 jet distribution in the 2D plane
is normalized to 100%. Therefore the colour density represents the
per-flavour efficiency density in the score plane.

WP regions:
    L0, C0, C1 on the left side, only split by p_{b+c}.
    C4, C3, C2, B0, B1, B2, B3, B4 on the right side, split by p_{b vs c}.
"""

import os
import time
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import mplhep as hep

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


# ======================================================================
# Configuration
# ======================================================================
class Config:
    def __init__(self):
        # ------------------------------------------------------------------
        # Input / output
        # ------------------------------------------------------------------
        self.mc_path = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_1merged_/"
        self.signal_file = "ttbar-powheg_merged.root"
        self.tree_name = "Events"
        self.figure_path = "./plots/plot/"

        self.lumi = 41.479
        self.enable_data = False

        # ------------------------------------------------------------------
        # AK4 selection shown in reference plot
        # ------------------------------------------------------------------
        self.ak4_pt_min = 25.0
        self.ak4_eta_max = 2.4

        # ------------------------------------------------------------------
        # AK8 preselection / Wcb signal definition
        # ------------------------------------------------------------------
        self.ak8_pt_min = 200.0
        self.ak8_sdmass_min = 30.0

        # ------------------------------------------------------------------
        # WP thresholds
        # ------------------------------------------------------------------
        self.T1 = 0.5
        self.T2 = 0.2
        self.T3 = 0.1

        # 11 exclusive WP regions.
        #
        # Left side:
        #   L0: 0.0 <= pbc < 0.1
        #   C0: 0.1 <= pbc < 0.2
        #   C1: 0.2 <= pbc < 0.5
        #
        # Right side:
        #   pbc >= 0.5, split by pbvc.
        self.wp_regions = [
            ("L0", 0.0,     self.T3, 0.0,  1.0),
            ("C0", self.T3, self.T2, 0.0,  1.0),
            ("C1", self.T2, self.T1, 0.0,  1.0),

            ("C4", self.T1, 1.0,     0.00, 0.05),
            ("C3", self.T1, 1.0,     0.05, 0.15),
            ("C2", self.T1, 1.0,     0.15, 0.40),

            ("B0", self.T1, 1.0,     0.40, 0.70),
            ("B1", self.T1, 1.0,     0.70, 0.88),
            ("B2", self.T1, 1.0,     0.88, 0.96),
            ("B3", self.T1, 1.0,     0.96, 0.99),
            ("B4", self.T1, 1.0,     0.99, 1.00),
        ]

        # Flavours: name, hflav code, colour
        self.flavours = [
            ("b",     5,  "#E42536"),  # CMS red
            ("c",     4,  "#5790FC"),  # CMS blue
            ("light", -1, "#F89C20"),  # CMS yellow/orange
        ]

        # ------------------------------------------------------------------
        # Plot tuning
        # ------------------------------------------------------------------
        # Coarser binning to suppress small islands.
        self.bins = 80

        # Larger smoothing to reduce isolated small islands.
        self.smooth_sigma = 2.0

        # More low-density contours, but not too low to avoid many islands.
        self.n_contour_levels = 18

        # Contour starts at this fraction of the peak.
        # Larger value -> fewer small islands.
        # If still too many islands, try 0.005 or 0.008.
        # If too little density shown, try 0.0015 or 0.002.
        self.min_peak_fraction = 0.003

        # Alpha range. Keep maximum 50%.
        self.min_alpha = 0.10
        self.max_alpha = 0.50

        # Nonlinear level spacing.
        # >1 means more levels near low density.
        self.low_level_power = 2.4

        # Boundary line style
        self.boundary_lw = 1.8
        self.boundary_color = "black"

        self.output_basename = "ak4_flavor_2d_score_plane_wcb_effdensity"


# ======================================================================
# Helper functions
# ======================================================================
def compute_weight(events):
    """Compute per-event weight."""
    required_weight_fields = [
        "genWeight",
        "lumiwgt",
        "xsecWeight",
        "puWeight",
        "trigEffWeight",
        "elEffWeight",
        "muEffWeight",
        "l1PreFiringWeight",
        "flavTagWeight",
    ]

    w = None
    for field in required_weight_fields:
        if field not in events.fields:
            raise RuntimeError(f"Missing required weight branch: {field}")

        if w is None:
            w = events[field]
        else:
            w = w * events[field]

    if "topptWeight" in events.fields:
        w = w * ak.fill_none(events.topptWeight, 1.0)

    return w


def flavour_mask(hflav, flav_name, flav_code):
    """Return boolean mask for jet flavour."""
    if flav_code == -1:
        return (hflav != 5) & (hflav != 4)
    return hflav == flav_code


def region_mask(pbc, pbvc, pbc_lo, pbc_hi, pbvc_lo, pbvc_hi):
    """Build exclusive region mask with safe handling of upper edge at 1.0."""
    if pbc_hi >= 1.0:
        mx = (pbc >= pbc_lo) & (pbc <= pbc_hi)
    else:
        mx = (pbc >= pbc_lo) & (pbc < pbc_hi)

    if pbvc_hi >= 1.0:
        my = (pbvc >= pbvc_lo) & (pbvc <= pbvc_hi)
    else:
        my = (pbvc >= pbvc_lo) & (pbvc < pbvc_hi)

    return mx & my


def fmt_eff(x):
    """Format efficiency percentage similar to CMS plot."""
    x = float(x)

    if x < 0.005:
        return f"{x:.3f}%"
    if x < 0.1:
        return f"{x:.2f}%"
    if x < 10.0:
        return f"{x:.1f}%"
    return f"{x:.0f}%"


def build_single_colour_alpha_list(base_color, n_intervals, min_alpha=0.10, max_alpha=0.50):
    """Build same-colour list with increasing alpha."""
    rgba = np.array(to_rgba(base_color))
    colours = []

    if n_intervals <= 1:
        c = rgba.copy()
        c[-1] = max_alpha
        return [c]

    for i in range(n_intervals):
        t = float(i) / float(n_intervals - 1)
        alpha = min_alpha + (max_alpha - min_alpha) * t

        c = rgba.copy()
        c[-1] = alpha
        colours.append(c)

    return colours


def draw_cms_label(ax):
    """Draw CMS Simulation Preliminary label without duplicated 'Simulation'."""
    # Use manual labels to avoid 'Simulation Simulation Preliminary' duplication.
    ax.text(
        0.000,
        1.045,
        "CMS",
        transform=ax.transAxes,
        fontsize=34,
        fontweight="bold",
        ha="left",
        va="top",
    )

    ax.text(
        0.135,
        1.045,
        "Simulation Preliminary",
        transform=ax.transAxes,
        fontsize=28,
        fontstyle="italic",
        ha="left",
        va="top",
    )

    ax.text(
        0.995,
        1.045,
        "(13 TeV)",
        transform=ax.transAxes,
        fontsize=26,
        ha="right",
        va="top",
    )


# ======================================================================
# Main
# ======================================================================
def main():
    start = time.time()
    cfg = Config()

    hep.style.use("CMS")

    # ------------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------------
    path = os.path.join(cfg.mc_path, cfg.signal_file)
    print(f"[INFO] Loading: {path}")

    branches = [
        # AK8 selection / truth
        "ak8_pt",
        "ak8_sdmass",
        "ak8_type",
        "ak8_is_wbc",

        # AK4 jets
        "ak4_pt",
        "ak4_eta",
        "ak4_hflav",
        "ak4_pn_b",
        "ak4_pn_c",

        # weights
        "genWeight",
        "lumiwgt",
        "xsecWeight",
        "puWeight",
        "trigEffWeight",
        "elEffWeight",
        "muEffWeight",
        "l1PreFiringWeight",
        "flavTagWeight",
        "topptWeight",
    ]

    with uproot.open(path) as f:
        events = f[cfg.tree_name].arrays(branches, library="ak")

    print(f"[INFO] Loaded events: {len(events)}")

    # ------------------------------------------------------------------
    # Event-level Wcb signal selection
    # ------------------------------------------------------------------
    has_ak8 = ak.num(events.ak8_pt) > 0

    ak8_pt0 = ak.fill_none(ak.firsts(events.ak8_pt), -999.0)
    ak8_mass0 = ak.fill_none(ak.firsts(events.ak8_sdmass), -999.0)
    ak8_type0 = ak.fill_none(ak.firsts(events.ak8_type), -999)
    ak8_iswbc0 = ak.fill_none(ak.firsts(events.ak8_is_wbc), 0)

    mask_ps = (
        has_ak8 &
        (ak8_pt0 > cfg.ak8_pt_min) &
        (ak8_mass0 > cfg.ak8_sdmass_min)
    )

    # Wcb signal truth selection.
    # Assuming:
    #   ak8_type == 1 means W-like
    #   ak8_is_wbc == 1 means W->cb
    mask_sig = mask_ps & (ak8_type0 == 1) & (ak8_iswbc0 == 1)

    events_sig = events[mask_sig]
    print(f"[INFO] Signal events after AK8 PS + Wcb truth: {len(events_sig)}")

    if len(events_sig) == 0:
        raise RuntimeError("No Wcb signal events selected. Please check signal truth definitions.")

    # ------------------------------------------------------------------
    # Event weight and flattened AK4 jets
    # ------------------------------------------------------------------
    evt_w = ak.to_numpy(compute_weight(events_sig))

    ak4_pt = events_sig.ak4_pt
    ak4_eta = events_sig.ak4_eta
    ak4_hflav = events_sig.ak4_hflav
    ak4_pn_b = events_sig.ak4_pn_b
    ak4_pn_c = events_sig.ak4_pn_c

    counts = ak.to_numpy(ak.num(ak4_pt, axis=1))
    jet_weights = np.repeat(evt_w, counts)

    ak4_pt_f = ak.to_numpy(ak.flatten(ak4_pt))
    ak4_eta_f = ak.to_numpy(ak.flatten(ak4_eta))
    hflav = ak.to_numpy(ak.flatten(ak4_hflav))
    pn_b = ak.to_numpy(ak.flatten(ak4_pn_b))
    pn_c = ak.to_numpy(ak.flatten(ak4_pn_c))

    print(f"[INFO] Flattened AK4 jets before cleaning: {len(ak4_pt_f)}")

    # ------------------------------------------------------------------
    # ParticleNet score variables
    # ------------------------------------------------------------------
    pbc_raw = pn_b + pn_c

    pbvc_raw = np.zeros_like(pbc_raw, dtype=float)
    valid_pbc = pbc_raw > 0
    pbvc_raw[valid_pbc] = pn_b[valid_pbc] / pbc_raw[valid_pbc]

    pbc = np.clip(pbc_raw, 0.0, 1.0)
    pbvc = np.clip(pbvc_raw, 0.0, 1.0)

    # ------------------------------------------------------------------
    # AK4 jet selection
    # ------------------------------------------------------------------
    clean = (
        (ak4_pt_f > cfg.ak4_pt_min) &
        (np.abs(ak4_eta_f) < cfg.ak4_eta_max) &
        np.isfinite(pbc) &
        np.isfinite(pbvc) &
        np.isfinite(hflav) &
        np.isfinite(jet_weights)
    )

    pbc = pbc[clean]
    pbvc = pbvc[clean]
    hflav = hflav[clean]
    jet_weights = jet_weights[clean]

    print(f"[INFO] AK4 jets after cleaning:")
    print(f"       total  = {len(pbc)}")
    print(f"       b      = {np.sum(hflav == 5)}")
    print(f"       c      = {np.sum(hflav == 4)}")
    print(f"       light  = {np.sum((hflav != 5) & (hflav != 4))}")

    if len(pbc) == 0:
        raise RuntimeError("No AK4 jets after cleaning. Please check AK4 selection.")

    # ------------------------------------------------------------------
    # Total weighted yield per flavour
    # ------------------------------------------------------------------
    total_weight_per_flav = {}

    for flav_name, flav_code, _ in cfg.flavours:
        fm = flavour_mask(hflav, flav_name, flav_code)
        total_weight_per_flav[flav_name] = np.sum(jet_weights[fm])

        print(
            f"[INFO] Total weighted jets for {flav_name:5s}: "
            f"{total_weight_per_flav[flav_name]:.6g}"
        )

    # ------------------------------------------------------------------
    # WP efficiencies
    # ------------------------------------------------------------------
    wp_eff = {}

    for wplabel, pbc_lo, pbc_hi, pbvc_lo, pbvc_hi in cfg.wp_regions:
        rm = region_mask(pbc, pbvc, pbc_lo, pbc_hi, pbvc_lo, pbvc_hi)
        wp_eff[wplabel] = {}

        for flav_name, flav_code, _ in cfg.flavours:
            fm = flavour_mask(hflav, flav_name, flav_code)
            denom = total_weight_per_flav[flav_name]
            num = np.sum(jet_weights[rm & fm])

            if denom > 0:
                wp_eff[wplabel][flav_name] = 100.0 * num / denom
            else:
                wp_eff[wplabel][flav_name] = 0.0

    print("[INFO] WP efficiencies normalized per flavour to 100%:")
    for wplabel, _, _, _, _ in cfg.wp_regions:
        print(
            f"       {wplabel:2s}: "
            f"b={wp_eff[wplabel]['b']:8.4f}%  "
            f"c={wp_eff[wplabel]['c']:8.4f}%  "
            f"light={wp_eff[wplabel]['light']:8.4f}%"
        )

    for flav_name, _, _ in cfg.flavours:
        s = sum(wp_eff[wplabel][flav_name] for wplabel, _, _, _, _ in cfg.wp_regions)
        print(f"[CHECK] Sum of WP efficiencies for {flav_name:5s}: {s:.4f}%")

    # ==================================================================
    # Plot
    # ==================================================================
    fig, ax = plt.subplots(figsize=(9, 9))

    # Binning
    xedges = np.linspace(0.0, 1.0, cfg.bins + 1)
    yedges = np.linspace(0.0, 1.0, cfg.bins + 1)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

    # ------------------------------------------------------------------
    # Filled contour efficiency-density
    # ------------------------------------------------------------------
    for flav_name, flav_code, flav_color in cfg.flavours:
        fm = flavour_mask(hflav, flav_name, flav_code)
        denom = total_weight_per_flav[flav_name]

        if np.sum(fm) < 10 or denom <= 0:
            print(f"[WARN] {flav_name}: too few jets or zero total weight, skip contour.")
            continue

        # Weighted 2D histogram
        H, _, _ = np.histogram2d(
            pbc[fm],
            pbvc[fm],
            bins=[xedges, yedges],
            weights=jet_weights[fm],
        )

        # Normalize each flavour to 100%.
        H_eff = 100.0 * H / denom

        # Smooth to reduce small islands.
        try:
            from scipy.ndimage import gaussian_filter
            H_eff_smooth = gaussian_filter(H_eff, sigma=cfg.smooth_sigma)
        except Exception as e:
            print(f"[WARN] scipy gaussian_filter unavailable, no smoothing. Error: {e}")
            H_eff_smooth = H_eff

        hmax = np.max(H_eff_smooth)
        if hmax <= 0:
            continue

        # --------------------------------------------------------------
        # Nonlinear contour levels biased to low-density region.
        # But start not too low to suppress small islands.
        # --------------------------------------------------------------
        t = np.linspace(0.0, 1.0, cfg.n_contour_levels)

        frac_levels = (
            cfg.min_peak_fraction
            + (1.0 - cfg.min_peak_fraction) * np.power(t, cfg.low_level_power)
        )

        levels = hmax * frac_levels
        levels = np.unique(levels)

        if len(levels) < 3:
            levels = np.linspace(cfg.min_peak_fraction * hmax, hmax, 8)

        colours = build_single_colour_alpha_list(
            flav_color,
            n_intervals=len(levels) - 1,
            min_alpha=cfg.min_alpha,
            max_alpha=cfg.max_alpha,
        )

        ax.contourf(
            X,
            Y,
            H_eff_smooth,
            levels=levels,
            colors=colours,
            antialiased=True,
            zorder=1,
        )

        # Thin contour lines
        line_rgba = np.array(to_rgba(flav_color))
        line_rgba[-1] = 0.45

        ax.contour(
            X,
            Y,
            H_eff_smooth,
            levels=levels,
            colors=[line_rgba],
            linewidths=0.9,
            zorder=2,
        )

        print(
            f"[INFO] {flav_name:5s} density normalized to 100%; "
            f"max bin after smoothing = {hmax:.4g}%"
        )

    # ------------------------------------------------------------------
    # WP boundary lines
    # ------------------------------------------------------------------
    # Vertical boundaries for L0/C0/C1/right-side.
    for xv in [cfg.T3, cfg.T2, cfg.T1]:
        ax.plot(
            [xv, xv],
            [0.0, 1.0],
            color=cfg.boundary_color,
            linewidth=cfg.boundary_lw,
            linestyle="-",
            solid_capstyle="butt",
            zorder=30,
        )

    # Horizontal boundaries only on right side.
    # Explicitly include B0/B1 boundary at 0.70.
    y_boundaries = [
        0.05,  # C4/C3
        0.15,  # C3/C2
        0.40,  # C2/B0
        0.70,  # B0/B1
        0.88,  # B1/B2
        0.96,  # B2/B3
        0.99,  # B3/B4
    ]

    for yv in y_boundaries:
        ax.plot(
            [cfg.T1, 1.0],
            [yv, yv],
            color=cfg.boundary_color,
            linewidth=cfg.boundary_lw,
            linestyle="-",
            solid_capstyle="butt",
            zorder=30,
        )

    # Also draw the right-side outer vertical border slightly stronger,
    # useful because contours often accumulate near x=1.
    ax.plot(
        [1.0, 1.0],
        [0.0, 1.0],
        color=cfg.boundary_color,
        linewidth=cfg.boundary_lw,
        linestyle="-",
        solid_capstyle="butt",
        zorder=31,
        clip_on=False,
    )

    # ------------------------------------------------------------------
    # Text annotations
    # ------------------------------------------------------------------
    colour_b = "#E42536"
    colour_c = "#5790FC"
    colour_l = "#F89C20"

    # Left side: L0, C0, C1.
    left_positions = {
        "L0": (0.010, 0.535),
        "C0": (0.110, 0.535),
        "C1": (0.210, 0.535),
    }

    for wplabel, (x0, y0) in left_positions.items():
        ax.text(
            x0,
            y0,
            wplabel,
            fontsize=16,
            color="black",
            ha="left",
            va="bottom",
            zorder=40,
        )

        ax.text(
            x0,
            y0 - 0.040,
            fmt_eff(wp_eff[wplabel]["b"]),
            fontsize=15,
            color=colour_b,
            ha="left",
            va="bottom",
            fontweight="bold",
            zorder=40,
        )

        ax.text(
            x0,
            y0 - 0.080,
            fmt_eff(wp_eff[wplabel]["c"]),
            fontsize=15,
            color=colour_c,
            ha="left",
            va="bottom",
            fontweight="bold",
            zorder=40,
        )

        ax.text(
            x0,
            y0 - 0.120,
            fmt_eff(wp_eff[wplabel]["light"]),
            fontsize=15,
            color=colour_l,
            ha="left",
            va="bottom",
            fontweight="bold",
            zorder=40,
        )

    # Right side: C2~B4.
    right_ypos = {
        "B4": 0.995,
        "B3": 0.965,
        "B2": 0.920,
        "B1": 0.800,
        "B0": 0.530,
        "C2": 0.205,
        "C3": 0.070,
        "C4": 0.025,
    }

    label_x = 0.510
    b_x = 0.575
    c_x = 0.675
    l_x = 0.790

    for wplabel, y0 in right_ypos.items():
        ax.text(
            label_x,
            y0,
            wplabel,
            fontsize=15,
            color="black",
            ha="left",
            va="center",
            zorder=40,
        )

        ax.text(
            b_x,
            y0,
            fmt_eff(wp_eff[wplabel]["b"]),
            fontsize=14,
            color=colour_b,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=40,
        )

        ax.text(
            c_x,
            y0,
            fmt_eff(wp_eff[wplabel]["c"]),
            fontsize=14,
            color=colour_c,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=40,
        )

        ax.text(
            l_x,
            y0,
            fmt_eff(wp_eff[wplabel]["light"]),
            fontsize=14,
            color=colour_l,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=40,
        )

    # ------------------------------------------------------------------
    # Legend and description
    # ------------------------------------------------------------------
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colour_b,
            lw=2.0,
            label=r"$b$ jets & efficiencies (%)",
        ),
        Line2D(
            [0],
            [0],
            color=colour_c,
            lw=2.0,
            label=r"$c$ jets & efficiencies (%)",
        ),
        Line2D(
            [0],
            [0],
            color=colour_l,
            lw=2.0,
            label=r"udsg jets & efficiencies (%)",
        ),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.025, 0.895),
        fontsize=13,
        frameon=True,
        handlelength=1.0,
        borderaxespad=0.0,
        labelspacing=0.5,
    )
    leg.set_zorder(100)
    # 50% transparent white legend background
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.90)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)

    ax.text(
        0.080,
        0.960,
        r"$t\bar{t}$ Wcb signal events,"
        "\n"
        rf"$p_T > {cfg.ak4_pt_min:.0f}$ GeV, $|\eta| < {cfg.ak4_eta_max:.1f}$",
        transform=ax.transAxes,
        fontsize=14,
        ha="left",
        va="top",
        zorder=41,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax.set_xlabel(r"ParticleNet score $p_{b+c}$", fontsize=18)
    ax.set_ylabel(r"ParticleNet score $p_{b\mathrm{vs}c}$", fontsize=18)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=12)

    ax.grid(False)

    # Manual CMS label to avoid duplicated text
    draw_cms_label(ax)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(cfg.figure_path, exist_ok=True)

    out_pdf = os.path.join(cfg.figure_path, cfg.output_basename + ".pdf")
    out_png = os.path.join(cfg.figure_path, cfg.output_basename + ".png")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    elapsed = time.time() - start

    print(f"[SAVE] {out_pdf}")
    print(f"[SAVE] {out_png}")
    print(f"[DONE] {elapsed:.1f} s")


if __name__ == "__main__":
    main()
