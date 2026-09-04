"""Helpers for the plottable-variable catalog.

Each variable spec is a dict:
    name       : short name used in plot file names / tags
    var        : key in the derived per-event array (branch + "_0" for jets)
    branch     : ROOT branch to load (None for purely derived variables)
    xlabel     : matplotlib x-axis label (mathtext, no \\GeV)
    xlim       : (lo, hi) axis range for continuous variables
    edges      : explicit bin edges (for discrete/count variables); overrides
                 xlim/nbins
    nbins_list : bin-count list to loop over when edges is None (default [10, 20])
    default    : fill value when the branch is missing
    note       : optional comment
"""
import numpy as np


def V(name, branch, xlabel, xlim=None, edges=None, nbins_list=None,
      default=-999.0, note=""):
    """Build a variable spec from a ROOT branch (branch=None for derived)."""
    if branch is None:
        var = name  # derived per-event variable, same name in the array
    else:
        var = (branch + "_0") if branch.startswith(("ak8_", "ak4_")) else branch
    return {
        "name": name,
        "var": var,
        "branch": branch,
        "xlabel": xlabel,
        "xlim": xlim,
        "edges": edges,
        "nbins_list": nbins_list,
        "default": default,
        "note": note,
    }


def count_edges(nmax):
    """Bin edges for integer counts 0..nmax (bins centred on integers)."""
    return np.arange(-0.5, nmax + 1.5, 1.0)
