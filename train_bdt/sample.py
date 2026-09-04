#!/usr/bin/env python3
import os
import uproot
import awkward as ak
import numpy as np
import pandas as pd

# ============================================================
# Config
# ============================================================
MC_PATH = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/filetagged_samples_1merged_/"
TREE_NAME = "Events"
OUT_DIR = "./dbc_dataset"
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42

BRANCHES = [
    "xsecWeight", "genWeight", "lumiwgt", "puWeight",
    "trigEffWeight", "l1PreFiringWeight",

    "ak8_pt", "ak8_eta", "ak8_sdmass",
    "ak8_n_b_in_jet", "ak8_n_c_in_jet", "ak8_n_in_jet",

    "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
    "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
]

FEATURES = [
    "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
    "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
]

# ============================================================
# Helpers
# ============================================================
def flatten_branch(arr, default=0):
    arr = ak.fill_none(arr, default)
    return ak.to_numpy(ak.flatten(arr, axis=1))

def build_label(n_b, n_c, n_in):
    return ((n_b == 1) & (n_c == 1) & (n_in == 2)).astype(np.int32)

def load_one_file(path, file_id):
    print(f"[INFO] Loading {os.path.basename(path)}")

    with uproot.open(path) as f:
        events = f[TREE_NAME].arrays(BRANCHES, library="ak")

    if len(events) == 0:
        return None

    # event-level weight
    event_weight = (
        events["xsecWeight"] * events["genWeight"] * events["lumiwgt"] *
        events["puWeight"] * events["trigEffWeight"] * events["l1PreFiringWeight"]
    )

    # jet template for broadcasting
    jet_template = events["ak8_pt"]

    # event_id
    event_id = ak.Array(np.arange(len(events), dtype=np.int64))
    jet_event_id = ak.broadcast_arrays(event_id, jet_template)[0]

    # file_id
    event_file_id = ak.Array(np.full(len(events), file_id, dtype=np.int32))
    jet_file_id = ak.broadcast_arrays(event_file_id, jet_template)[0]

    # broadcast event weight to jet-level
    jet_weight = ak.broadcast_arrays(event_weight, jet_template)[0]

    df = pd.DataFrame({
        "file_id": flatten_branch(jet_file_id, -1),
        "event_id": flatten_branch(jet_event_id, -1),

        "ak8_pt": flatten_branch(events["ak8_pt"], 0),
        "ak8_eta": flatten_branch(events["ak8_eta"], 0),
        "ak8_sdmass": flatten_branch(events["ak8_sdmass"], 0),

        "ak8_n_b_in_jet": flatten_branch(events["ak8_n_b_in_jet"], -1),
        "ak8_n_c_in_jet": flatten_branch(events["ak8_n_c_in_jet"], -1),
        "ak8_n_in_jet": flatten_branch(events["ak8_n_in_jet"], -1),

        "ak8_gpt_bc": flatten_branch(events["ak8_gpt_bc"], 0),
        "ak8_gpt_bb": flatten_branch(events["ak8_gpt_bb"], 0),
        "ak8_gpt_cc": flatten_branch(events["ak8_gpt_cc"], 0),
        "ak8_gpt_qcd": flatten_branch(events["ak8_gpt_qcd"], 0),
        "ak8_gpt_bs": flatten_branch(events["ak8_gpt_bs"], 0),
        "ak8_gpt_qq": flatten_branch(events["ak8_gpt_qq"], 0),
        "ak8_gpt_cs": flatten_branch(events["ak8_gpt_cs"], 0),
        "ak8_gpt_topbw": flatten_branch(events["ak8_gpt_topbw"], 0),

        "weight": flatten_branch(jet_weight, 0),
    })

    # label
    df["label"] = build_label(
        df["ak8_n_b_in_jet"].to_numpy(),
        df["ak8_n_c_in_jet"].to_numpy(),
        df["ak8_n_in_jet"].to_numpy()
    )

    # 给每个 jet 一个全局 group id，后面可用于按 event 分组
    df["group_id"] = (
        df["file_id"].astype(np.int64) * 10**12 + df["event_id"].astype(np.int64)
    )

    return df

# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(MC_PATH) if f.endswith(".root")])
    print(f"[INFO] Found {len(files)} ROOT files")

    dfs = []
    for i, fname in enumerate(files):
        path = os.path.join(MC_PATH, fname)
        df = load_one_file(path, file_id=i)
        if df is not None and len(df) > 0:
            df["sample_name"] = fname
            dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError("No valid jets loaded from ROOT files.")

    all_df = pd.concat(dfs, ignore_index=True)

    keep_cols = [
        "file_id", "event_id", "group_id", "sample_name",
        "ak8_pt", "ak8_eta", "ak8_sdmass",
        "ak8_n_b_in_jet", "ak8_n_c_in_jet", "ak8_n_in_jet",
        "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
        "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw",
        "weight", "label"
    ]
    all_df = all_df[keep_cols].copy()

    finite_mask = np.isfinite(all_df[FEATURES]).all(axis=1) & np.isfinite(all_df["weight"])
    all_df = all_df[finite_mask].copy()


    all_df = all_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    n_total = len(all_df)
    n_train = int(TRAIN_FRACTION * n_total)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    valid_df = all_df.iloc[n_train:].reset_index(drop=True)

    train_path = os.path.join(OUT_DIR, "train.parquet")
    valid_path = os.path.join(OUT_DIR, "valid.parquet")

    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    print(f"[INFO] Total jets  : {len(all_df)}")
    print(f"[INFO] Train jets  : {len(train_df)}")
    print(f"[INFO] Valid jets  : {len(valid_df)}")
    print(f"[INFO] Train signal fraction: {train_df['label'].mean():.6f}")
    print(f"[INFO] Valid signal fraction: {valid_df['label'].mean():.6f}")
    print(f"[SAVE] {train_path}")
    print(f"[SAVE] {valid_path}")

if __name__ == "__main__":
    main()
