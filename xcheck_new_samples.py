#!/usr/bin/env python
"""Compare the ROOT branch content of the OLD vs the NEW (2026-09-02 rerun)
scored-sample production, so we know nothing the analysis reads was dropped or
relabelled before switching Make_plots.py over to the new path.

Run once youpeng grants EOS read access to
  .../MC/scored_samples_2final_v20260902_rerun/
"""
import sys, uproot

OLD = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/mc/scored_samples_1merged_/"
NEW = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_2final_v20260902_rerun/"
SAMPLE = "ttbar-powheg_merged.root"       # any sample; branch list is identical across them
TREE = "Events"

def branches(path):
    with uproot.open(path + ":" + TREE) as t:
        return {k: str(t[k].typename) for k in t.keys()}

def main():
    old_p = OLD + SAMPLE
    new_p = NEW + (sys.argv[1] if len(sys.argv) > 1 else SAMPLE)
    print(f"OLD: {old_p}")
    print(f"NEW: {new_p}\n")
    bo, bn = branches(old_p), branches(new_p)
    so, sn = set(bo), set(bn)

    missing = sorted(so - sn)
    added   = sorted(sn - so)
    common  = so & sn
    retyped = sorted(k for k in common if bo[k] != bn[k])

    print(f"branches: OLD {len(bo)}  NEW {len(bn)}  common {len(common)}\n")

    print(f"--- {len(missing)} branch(es) in OLD but NOT in NEW  (BLOCKER if analysis uses them) ---")
    for k in missing:
        print(f"   - {k:40s} {bo[k]}")

    print(f"\n--- {len(retyped)} branch(es) with a CHANGED type ---")
    for k in retyped:
        print(f"   ~ {k:40s} {bo[k]}  ->  {bn[k]}")

    print(f"\n--- {len(added)} NEW branch(es) (extra GloParT scores etc., harmless) ---")
    for k in added:
        print(f"   + {k:40s} {bn[k]}")

    # cross-check against what Make_plots.py actually reads
    try:
        import Make_plots as M
        want = set()
        for attr in ("_CACHE_ALWAYS_COLUMNS",):
            want |= set(getattr(M, attr, set()))
        # the uproot read list is built in build_derived_array via a big
        # `branches_to_read` list -- grep the source for the literal names
        import re, pathlib
        src = pathlib.Path(M.__file__).read_text()
        lit = set(re.findall(r'"(ak8_[a-z0-9_]+|[a-z]+_[a-z0-9_]+)"', src))
        used_missing = sorted((set(missing)) & lit)
        print(f"\n--- of the missing branches, {len(used_missing)} appear as string literals in Make_plots.py ---")
        for k in used_missing:
            print(f"   !! {k}")
    except Exception as e:
        print(f"\n(could not cross-check against Make_plots.py: {e})")

if __name__ == "__main__":
    main()
