#!/usr/bin/env python3
"""x-check (2026-09-19): data must be treated exactly like MC.

Loads ONE small MC ROOT file twice with 04_Make_plots.py's own loader --
once as MC (is_data=False), once as data (is_data=True) -- with the cache
OFF, then compares every derived column both arrays have.  The only columns
allowed to differ are the truth/weight ones that data legitimately lacks;
everything else (J choice, J kinematics, GloParT scores, D_cb/D_bb/D_bbc,
runtime fields, the SR cut masks) must be identical.

usage:  .venv/bin/python check_data_mc_equivalence.py [ROOT_FILE]
"""
import importlib.util
import os
import sys

import awkward as ak
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mp", os.environ.get("WCB_MP_FILE", os.path.join(HERE, "04_Make_plots.py")))
mp = importlib.util.module_from_spec(spec)
sys.modules["mp"] = mp
spec.loader.exec_module(mp)

# columns that legitimately differ (weights + gen-truth, absent/-1 for data)
TRUTH_LIKE = ("weights", "true_cat", "mat_cat", "ak8_type", "n_b", "n_c",
              "n_in_jet", "is_qcd", "is_signal", "ak8_n_", "ak8_match_",
              "ak8_is_", "w_decay", "z_decay", "genZ", "topb", "ja_truth",
              "topptWeight", "Weight")

cfg = mp.Config()
cfg.use_cache = False
cfg.plot_mode = "MAT"
cfg.mat_sel = "SR1A"
mgr = mp.DataManager(cfg)

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    metas = [m for m in mgr.load_all() if not m["is_data"]]
    metas.sort(key=lambda m: os.path.getsize(m["path"]))
    path = metas[0]["path"]
print("[EQ] file:", path)

a_mc = mgr.load_one_file(path, is_data=False, sample_group="Other")
a_dt = mgr.load_one_file(path, is_data=True, sample_group="Other")
print(f"[EQ] events MC={len(a_mc)} data={len(a_dt)}")

common = [f for f in a_mc.fields if f in set(a_dt.fields)]
only_mc = [f for f in a_mc.fields if f not in set(a_dt.fields)]
only_dt = [f for f in a_dt.fields if f not in set(a_mc.fields)]
print(f"[EQ] common fields: {len(common)}   only-MC: {len(only_mc)}   only-data: {len(only_dt)}")

bad, skipped = [], []
for f in common:
    if any(f.startswith(t) or f == t for t in TRUTH_LIKE):
        skipped.append(f)
        continue
    x = ak.to_numpy(a_mc[f])
    y = ak.to_numpy(a_dt[f])
    if x.shape != y.shape:
        bad.append((f, "shape"))
        continue
    if not np.array_equal(x, y, equal_nan=True):
        nd = int(np.sum(~((x == y) | (np.isnan(x.astype(float)) & np.isnan(y.astype(float))))))
        bad.append((f, f"{nd} differing"))

print(f"[EQ] truth/weight-like fields skipped: {len(skipped)}")
if bad:
    print("[EQ] *** DIFFERENT between MC-mode and data-mode ***")
    for f, why in bad:
        print("   ", f, why)
else:
    print("[EQ] OK: all non-truth columns identical (J choice, J kinematics, "
          "scores, runtime fields).")

# selection masks must agree too
for sel in ("PRE", "SR1A", "SR2B", "SR3A"):
    cut = mp.build_mat_plot_settings(sel, False)[0]["cut"]
    m1 = mp.eval_cut(cut, a_mc)
    m2 = mp.eval_cut(cut, a_dt)
    print(f"[EQ] cut {sel:5s}: MC pass={int(m1.sum())} data pass={int(m2.sum())} "
          f"identical={bool(np.array_equal(m1, m2))}")
if only_dt:
    print("[EQ] fields only in data-mode:", only_dt[:20])
if only_mc:
    print("[EQ] fields only in MC-mode (truth-side, expected):", only_mc[:30])
