# S1 / S1' — fine-tuned boosted-cb tagger

A dedicated XGBoost tagger for the boosted W→cb candidate jet **J**, replacing
the ad-hoc use of the single `ak8_gpt_bc` node / the old 8-node `D_bc` BDT.

| tagger | inputs |
|---|---|
| **S1**  (`score_S1`)  | the **30 independent** GloParT `ak8_gpt_*` nodes (all 33 minus the 3 parent sums `topbw`, `topw`, `qcd`) |
| **S1'** (`score_S1p`) | S1 + `ak8_tau21` + `ak8_tau32` |

Binary `binary:logistic`. Output ∈ [0,1], plotted as `$S_{1}$ (BDT fine-tuned score)`.

## Signal / background

**Signal** = an AK8 jet gen-matched to one of 6 topologies (codes 1-4 in the
Vcb sample `ttbar-powheg`, gated on `w_decay == 5`; codes 5-6 pulled from the
other 16 non-Vcb samples and promoted to signal, never thinned):

| code | name | flag | source |
|---|---|---|---|
| 1 | `Wcb`    | `ak8_match_wqq_wcb`, ak8_type 1 | ttbar-powheg (merged 2-prong b+c) |
| 2 | `t2_bc`  | `ak8_match_tbq_wcb` & n_c≥1, ak8_type 2 | ttbar-powheg (top-b + W's c) |
| 3 | `t2_bb`  | `ak8_match_tbq_wcb` & n_c=0, ak8_type 2 | ttbar-powheg (top-b + W's b) |
| 4 | `t3_bbc` | `ak8_match_tbqq_wcb`, ak8_type 4 | ttbar-powheg (fully-merged t→bbc) |
| 5 | `t2_bc_proxy` | `ak8_match_top_bc` & `w_decay==4` | the 16 non-Vcb samples (W→cs event, statistically identical to class 2) |
| 6 | `z_bb` (Z(bb)-adj) | `z_decay==5` & dR(J, Z_gen)<0.8 | the 16 non-Vcb samples (tagger can't distinguish it from class 3, see `z_contamination_study.py`) |

The fully-**resolved** W→cb events (~48 % of the Vcb sample) and all other
`ttbar-powheg` jets are **dropped** — they are real W→cb physics and must not
teach the tagger to reject signal.

**Background** = every remaining AK8 jet from the other 16 samples (tt-semi, tt-had,
ttbb, single-top, QCD, W+jets, DY, ttV, ttH, dibosons). Bernoulli-thinned with
`background_keep_prob` (see `config.json`), each kept jet up-weighted by `1/p`, so the
summed event weight (effective luminosity) is preserved.

## Preselection (per jet — the MAT `PRE`)

`pt > 200` · `mSD > 40` · `0 < tau21 < 0.65` · `ΔR(lep1, jet) > 1.3` ·
event: 2nd-highest AK8 `mSD < 100`.

## Data split & overtraining control

Per event, hashed on `run:lumi:event` (blake2b) so no jet from one event
straddles splits:

| split | fraction | use |
|---|---|---|
| train | 70 % | model fit |
| valid | 15 % | early stopping only |
| **test** | **15 %** | **never seen in any way — the overtraining reference** |

Every run reports, per tagger:
- `train` vs `test` AUC (a large gap ⇒ overfit)
- **KS test** (Kolmogorov–Smirnov) between the train and test score shapes,
  separately for signal and background — `p > 0.05` = OK, `> 0.01` = WARN, else FAIL
- **signal-efficiency bias** = (ε_train − ε_test)/ε_test at the bkg-eff 10⁻³ point, in %
- an **overtraining plot** (`overtrain_<tag>.png`): train histogram vs test points+errors,
  signal and background overlaid

(same checks as `../Hgg` `Plot.py`. Raise the test fraction in `config.json`
`split` if you want more held-out stats.)

## BDT parameters (pass 1)

Youpeng's `dcb_versions_v1` XGBoost params verbatim, for direct comparability
with his `D_bc`, plus early stopping:

```
n_estimators 600 + early_stopping_rounds 40 (valid AUC)
max_depth 5 | learning_rate 0.05 | subsample 0.8 | colsample_bytree 0.8
tree_method hist  ->  no power transform on the scores (rank-based binning)
scale_pos_weight = sum(w_bkg) / sum(w_sig)   [recomputed, not copied]
```

`colsample_bytree` is the first knob to tune (0.8 → ~0.6 for 30 features).
`tree_method=hist` bins by weighted quantile, so a monotonic `x^0.2`
transform of the small `ak8_gpt_*` values is ~a no-op — skipped by design.
Revisit if we move to an NN.

## Run

The pipeline is 3 steps, each also runnable standalone; `run_s1_tagger.py`
drives all of them (2026-09-13 -- scripts moved to `tt1lWcb/` root and
renamed so `ls` shows execution order):

```bash
./run_s1_tagger.py                 # train + plots (dataset reused if built)
./run_s1_tagger.py all             # dataset + train + plots
./run_s1_tagger.py dataset         # 01_build_trainset.py only
./run_s1_tagger.py train           # 02_train_tagger.py only
./run_s1_tagger.py plots           # 03_render_report.py only (no retrain)

# or run any step directly:
./.venv/bin/python 01_build_trainset.py   # ~10 min (streams 17 ROOT files -> EOS parquet)
./.venv/bin/python 02_train_tagger.py     # ~50 min (S1 only; derives class_weight_multiplier
                                           # fresh from config.json's class_weight_share every run)
./.venv/bin/python 03_render_report.py    # ~15 s (self-bootstraps into an LCG view for PyROOT)
```

`config.json` is pure input (preselection fallback, `class_weight_share`,
XGBoost hyperparameters) — nothing in the pipeline writes back to it; the
old separate "derive multiplier from the built dataset, write it into
config.json" shell step was folded into `02_train_tagger.py` so it can never
go stale relative to `class_weight_share`.

Dataset parquet → `$S1_DATA_DIR` (default `/eos/user/a/agapitos/S1_tagger_data/`,
**not** `work` which is ~91 % full). Models + plots + `summary.json` →
`S1_tagger/output/<dataset_tag>/` (still under `S1_tagger/` even though the
scripts themselves now live at the `tt1lWcb/` root).

## Outputs — everything under `output/<dataset_tag>/`, one self-contained bundle per run

| file | content |
|---|---|
| `{S1,S1p}/model.json` | XGBoost boosters |
| `{S1,S1p}/eval.npz` | test + train-subsample scores/labels/weights (for re-plotting) |
| `summary.json` | run stamp, config, dataset counts, per-split AUC/logloss, working points, per-class sig-eff, **overtraining** (KS p, bias, verdict), full 30-node `feature_importance`, comparison AUCs |
| `roc.png` | S1 / S1' / old D_bc / 3-class D_bc, same test jets |
| `overtrain_{S1,S1p}.png` | train-vs-test score shapes + KS test |
| `featimp_{S1,S1p}.png` | top-15 nodes by gain + a training-info box |
| `score_{S1,S1p}.png` | score split by W→cb topology |
| `results_table.png` | (a) discrimination+overtraining (b) per-class (c) training set+params (d) top-10 |
| **`panel_{S1,S1p}.png`** | **2×2:  ROC │ overtraining  /  feature-importance+info │ score-by-topology** |

Re-render any run's figures without retraining:
`./.venv/bin/python 03_render_report.py --config S1_tagger/config.json`

Published into the gallery root (`:8899`), tagged per run so nothing clobbers:
`S1_tagger_<tag>_panel_{S1,S1p}.png`, `_results_table.png`, `_roc.png`,
plus untagged `S1_tagger_panel_latest_{S1,S1p}.png`.

## Wiring into Make_plots.py

Done in a later step, after the ROC is reviewed: evaluate both models at
derive time → cached `score_S1` / `score_S1p` columns (mirrors `score_Dbc`
via `DbcEvaluator`), add plot specs, bump `cache_tag` → `v23`.
