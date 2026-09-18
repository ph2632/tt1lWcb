#!/usr/bin/env python3
"""
Read the MAT preselection straight out of 04_Make_plots.py so that changing a
threshold there automatically propagates to the S1 tagger training.

`build_mat_plot_settings()` keeps the cut strings in local variables, so we
lift the `_COMMON / _MJ12 / _SELECTIONS` block out of the source and exec it
in an empty namespace -- no import of 04_Make_plots (which is heavy and would
pull in ROOT files) and no duplicated numbers.
"""
import json
import re
from pathlib import Path

MAKE_PLOTS = Path(__file__).resolve().parent.parent / "04_Make_plots.py"

# cut-expression token  ->  (config key, comparison)
_MAP = {
    "ak8_pt[0]": ("jet_pt_min", ">"),
    "ak8_sdmass_0": ("jet_sdmass_min", ">"),
    "ak8_sdmass_sub_mass_0": ("event_subleading_sdmass_max", "<"),
    "ak8_tau21_0": ("jet_tau21_max", "<"),
    "dR_lep_ak8": ("dr_lep_jet_min", ">"),
    "ak8_gpt_Jrank_0": ("jrank_min", ">"),
}


def read_pre_cut(path=MAKE_PLOTS):
    """Return the raw PRE cut expression from 04_Make_plots.py."""
    src = path.read_text()
    m = re.search(r"^\s*_COMMON = .*?^\s*CUT = ", src, re.S | re.M)
    if not m:
        raise RuntimeError(f"could not locate the _SELECTIONS block in {path}")
    block = "\n".join(ln.strip() for ln in m.group(0).splitlines()[:-1])
    ns = {}
    # 2026-09-18: the block now reads S1_tagger/sr_cuts.json via a couple of
    # flat (unindented) lines using json/Path/__file__ -- inject those into
    # the exec globals (still no import of 04_Make_plots itself, which stays
    # heavy/ROOT-pulling; json+Path are just stdlib) so this keeps working.
    exec(block, {"json": json, "Path": Path, "__file__": str(MAKE_PLOTS)}, ns)  # noqa: S102 - our own source
    return ns["_SELECTIONS"]["PRE"]


def parse_preselection(path=MAKE_PLOTS):
    """PRE cut -> the per-jet thresholds build_trainset.py needs.

    `ak8_*_0` in 04_Make_plots.py refers to the candidate jet J; for a per-jet
    training set the same threshold is applied to every AK8 jet, except
    `ak8_sdmass_sub_mass_0` (2nd-heaviest AK8) which stays event-level.
    """
    cut = read_pre_cut(path)
    out = {}
    for tok, (key, cmp_) in _MAP.items():
        m = re.search(re.escape(tok) + r"\s*" + re.escape(cmp_) + r"\s*(-?[\d.]+)", cut)
        if m:
            out[key] = float(m.group(1))
    missing = {k for k, _ in _MAP.values()} - set(out)
    return out, cut, sorted(missing)


if __name__ == "__main__":
    p, cut, missing = parse_preselection()
    print("PRE cut in 04_Make_plots.py:\n  " + cut)
    print("\nparsed thresholds:")
    for k, v in p.items():
        print(f"  {k:32s} {v}")
    if missing:
        print(f"\n  NOT FOUND (config fallback will be used): {missing}")
