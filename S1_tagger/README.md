# S1 / S1' — fine-tuned boosted-cb tagger

A dedicated XGBoost tagger for the boosted W→cb candidate jet **J**, replacing
the ad-hoc use of the single `ak8_gpt_bc` node / the old 8-node `D_bc` BDT.

| tagger | inputs |
|---|---|
| **S1**  (`score_S1`)  | the **30 independent** GloParT `ak8_gpt_*` nodes (all 33 minus the 3 parent sums `topbw`, `topw`, `qcd`) |
| **S1'** (`score_S1p`) | S1 + `ak8_tau21` + `ak8_tau32` |

Binary `binary:logistic`. Output ∈ [0,1], plotted as `$S_{1}$ (BDT fine-tuned score)`.

## Signal / background

**Signal** = an AK8 jet in the Vcb sample (`ttbar-powheg`) that is gen-matched
to one of the four W→cb topologies, gated on `w_decay == 5`:

| code | name | flag | ak8_type |
|---|---|---|---|
| 1 | `Wcb`    | `ak8_match_wqq_wcb`  | 1 (merged 2-prong b+c) |
| 2 | `t2_bc`  | `ak8_match_tbq_wcb` & n_c≥1 | 2 (top-b + W's c) |
| 3 | `t2_bb`  | `ak8_match_tbq_wcb` & n_c=0 | 2 (top-b + W's b) |
| 4 | `t3_bbc` | `ak8_match_tbqq_wcb` | 4 (fully-merged t→bbc) |

The fully-**resolved** W→cb events (~48 % of the Vcb sample) and all other
`ttbar-powheg` jets are **dropped** — they are real W→cb physics and must not
teach the tagger to reject signal.

**Background** = every AK8 jet from the other 16 samples (tt-semi, tt-had,
ttbb, single-top, QCD, W+jets, DY, ttV, ttH, dibosons). Bernoulli-thinned with
`background_keep_prob = 0.15`, each kept jet up-weighted by `1/p`, so the
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

```bash
# 1. build the per-jet training set  (streams 17 ROOT files -> EOS parquet, ~10 min)
./.venv/bin/python S1_tagger/build_trainset.py            # --overwrite to rebuild

# 2. train S1 + S1', benchmark vs old D_bc and Youpeng's 3-class  (~5 min)
./.venv/bin/python S1_tagger/train.py
```

Dataset parquet → `$S1_DATA_DIR` (default `/eos/user/a/agapitos/S1_tagger_data/`,
**not** `work` which is ~91 % full). Models + plots + `summary.json` →
`S1_tagger/output/<dataset_tag>/`.

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
`./.venv/bin/python S1_tagger/make_report_panel.py`

Published into the gallery root (`:8899`), tagged per run so nothing clobbers:
`S1_tagger_<tag>_panel_{S1,S1p}.png`, `_results_table.png`, `_roc.png`,
plus untagged `S1_tagger_panel_latest_{S1,S1p}.png`.

## Wiring into Make_plots.py

Done in a later step, after the ROC is reviewed: evaluate both models at
derive time → cached `score_S1` / `score_S1p` columns (mirrors `score_Dbc`
via `DbcEvaluator`), add plot specs, bump `cache_tag` → `v23`.
