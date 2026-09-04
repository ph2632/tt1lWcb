#!/usr/bin/env python3
import os
import uproot
import awkward as ak
import numpy as np
import pandas as pd

# ============================================================
# Config
# ============================================================
INPUT_ROOT = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/mc/scored_samples_2final/ttbar-powheg_merged_Skim.root"
TREE_NAME = "Events"
OUT_DIR = "./swqq_dataset"
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42

BRANCHES = [
    "xsecWeight", "genWeight", "lumiwgt", "puWeight",
    "trigEffWeight", "l1PreFiringWeight",

    "ak8_pt", "ak8_eta", "ak8_sdmass",
    "ak8_type",

    # 只用 topology classifier 输出
    "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
    "score_cata_top_bc", "score_cata_top_bq", "score_cata_non"
]

FEATURES = [
    "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
    "score_cata_top_bc", "score_cata_top_bq", "score_cata_non"
]

# ============================================================
# Helpers
# ============================================================
def flatten_branch(arr, default=0):
    arr = ak.fill_none(arr, default)

    # jagged: event -> jets
    if arr.ndim > 1:
        # select leading jet if multiple jets exist, otherwise fill with default
        return ak.to_numpy(ak.firsts(arr, axis=1))

    # flat/event-level
    return ak.to_numpy(arr)


def build_label(ak8_type):
    # W-like topology vs all
    return (ak8_type == 1).astype(np.int32)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Loading {INPUT_ROOT}")
    with uproot.open(INPUT_ROOT) as f:
        events = f[TREE_NAME].arrays(BRANCHES, library="ak")

    if len(events) == 0:
        raise RuntimeError("No events found in input ROOT.")

    # ------------------------------------------------------------
    # event-level weight
    # ------------------------------------------------------------
    event_weight = (
        events["xsecWeight"] * events["genWeight"] * events["lumiwgt"] *
        events["puWeight"] * events["trigEffWeight"] * events["l1PreFiringWeight"]
    )

    # jet template for broadcasting
    jet_template = events["ak8_pt"]

    # event_id
    event_id = ak.Array(np.arange(len(events), dtype=np.int64))
    jet_event_id = ak.broadcast_arrays(event_id, jet_template)[0]

    # broadcast weight to jet-level
    jet_weight = ak.broadcast_arrays(event_weight, jet_template)[0]

    df = pd.DataFrame({
        "event_id": flatten_branch(jet_event_id, -1),

        "ak8_pt": flatten_branch(events["ak8_pt"], 0),
        "ak8_eta": flatten_branch(events["ak8_eta"], 0),
        "ak8_sdmass": flatten_branch(events["ak8_sdmass"], 0),
        "ak8_type": flatten_branch(events["ak8_type"], -1),

        "score_cata_w_qq": flatten_branch(events["score_cata_w_qq"], 0),
        "score_cata_qcd": flatten_branch(events["score_cata_qcd"], 0),
        "score_cata_top_bqq": flatten_branch(events["score_cata_top_bqq"], 0),
        "score_cata_top_bc": flatten_branch(events["score_cata_top_bc"], 0),
        "score_cata_top_bq": flatten_branch(events["score_cata_top_bq"], 0),
        "score_cata_non": flatten_branch(events["score_cata_non"], 0),

        "weight": flatten_branch(jet_weight, 0),
    })

    # label
    df["label"] = build_label(df["ak8_type"].to_numpy())

    # 用 event_id 当 group_id，避免同一 event 泄漏到 train/valid 两边
    df["group_id"] = df["event_id"].astype(np.int64)

    # ------------------------------------------------------------
    # basic cleaning
    # ------------------------------------------------------------
    keep_cols = [
        "event_id", "group_id",
        "ak8_pt", "ak8_eta", "ak8_sdmass", "ak8_type",
        "score_cata_w_qq", "score_cata_qcd", "score_cata_top_bqq",
        "score_cata_top_bc", "score_cata_top_bq", "score_cata_non",
        "weight", "label"
    ]
    df = df[keep_cols].copy()

    finite_mask = np.isfinite(df[FEATURES]).all(axis=1) & np.isfinite(df["weight"])
    df = df[finite_mask].copy()

    # 可选基础选择：如果你想和分析保持一致可以打开
    # df = df[(df["ak8_sdmass"] > 60) & (df["ak8_sdmass"] < 120)].copy()

    # ------------------------------------------------------------
    # group-level split
    # ------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    unique_groups = df["group_id"].drop_duplicates().to_numpy()
    rng.shuffle(unique_groups)

    n_train_groups = int(TRAIN_FRACTION * len(unique_groups))
    train_groups = set(unique_groups[:n_train_groups])
    valid_groups = set(unique_groups[n_train_groups:])

    train_df = df[df["group_id"].isin(train_groups)].copy().reset_index(drop=True)
    valid_df = df[df["group_id"].isin(valid_groups)].copy().reset_index(drop=True)

    train_path = os.path.join(OUT_DIR, "train.parquet")
    valid_path = os.path.join(OUT_DIR, "valid.parquet")

    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    print(f"[INFO] Total jets   : {len(df)}")
    print(f"[INFO] Train jets   : {len(train_df)}")
    print(f"[INFO] Valid jets   : {len(valid_df)}")
    print(f"[INFO] Train events : {train_df['group_id'].nunique()}")
    print(f"[INFO] Valid events : {valid_df['group_id'].nunique()}")
    print(f"[INFO] Train W-like fraction: {train_df['label'].mean():.6f}")
    print(f"[INFO] Valid W-like fraction: {valid_df['label'].mean():.6f}")
    print(f"[SAVE] {train_path}")
    print(f"[SAVE] {valid_path}")

if __name__ == "__main__":
    main()
