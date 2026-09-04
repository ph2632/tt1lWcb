#!/usr/bin/env python3
import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss

# ============================================================
# Config
# ============================================================
DATA_DIR = "./dbc_dataset"
OUT_DIR = "./dbc_bdt_output"
RANDOM_SEED = 42

FEATURES = [
    "ak8_gpt_bc", "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_qcd",
    "ak8_gpt_bs", "ak8_gpt_qq", "ak8_gpt_cs", "ak8_gpt_topbw"
]

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
}

# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(DATA_DIR, "valid.parquet"))

    train_mask = np.isfinite(train_df[FEATURES]).all(axis=1) & np.isfinite(train_df["weight"])
    valid_mask = np.isfinite(valid_df[FEATURES]).all(axis=1) & np.isfinite(valid_df["weight"])

    train_df = train_df[train_mask].copy()
    valid_df = valid_df[valid_mask].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["label"].astype(int)
    w_train = train_df["weight"].abs().astype(float)

    X_valid = valid_df[FEATURES]
    y_valid = valid_df["label"].astype(int)
    w_valid = valid_df["weight"].abs().astype(float)

    sum_w_pos = w_train[y_train == 1].sum()
    sum_w_neg = w_train[y_train == 0].sum()
    scale_pos_weight = sum_w_neg / max(sum_w_pos, 1e-12)

    print(f"[INFO] Train size = {len(train_df)}")
    print(f"[INFO] Valid size = {len(valid_df)}")
    print(f"[INFO] Weighted signal fraction (train) = {sum_w_pos / max(sum_w_pos + sum_w_neg, 1e-12):.6f}")
    print(f"[INFO] scale_pos_weight = {scale_pos_weight:.6f}")

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        sample_weight_eval_set=[w_train, w_valid],
        verbose=True,
    )

    # 预测
    train_score = model.predict_proba(X_train)[:, 1]
    valid_score = model.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, train_score, sample_weight=w_train)
    valid_auc = roc_auc_score(y_valid, valid_score, sample_weight=w_valid)

    train_logloss = log_loss(y_train, train_score, sample_weight=w_train, labels=[0, 1])
    valid_logloss = log_loss(y_valid, valid_score, sample_weight=w_valid, labels=[0, 1])

    print(f"[METRIC] train_auc     = {train_auc:.6f}")
    print(f"[METRIC] valid_auc     = {valid_auc:.6f}")
    print(f"[METRIC] train_logloss = {train_logloss:.6f}")
    print(f"[METRIC] valid_logloss = {valid_logloss:.6f}")

    # 保存模型
    model_json_path = os.path.join(OUT_DIR, "bdt_dbc_model.json")
    model_pkl_path = os.path.join(OUT_DIR, "bdt_dbc_model.pkl")
    model.save_model(model_json_path)
    joblib.dump(model, model_pkl_path)

    # 保存 valid prediction
    valid_out = valid_df.copy()
    valid_out["bdt_score"] = valid_score
    valid_pred_path = os.path.join(OUT_DIR, "valid_prediction.parquet")
    valid_out.to_parquet(valid_pred_path, index=False)

    train_out = train_df.copy()
    train_out["bdt_score"] = train_score
    train_pred_path = os.path.join(OUT_DIR, "train_prediction.parquet")
    train_out.to_parquet(train_pred_path, index=False)

    # 保存训练 history
    evals_result = model.evals_result()
    hist_path = os.path.join(OUT_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(evals_result, f, indent=2)

    # 保存 summary
    summary = {
        "features": FEATURES,
        "params": params,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "train_auc": float(train_auc),
        "valid_auc": float(valid_auc),
        "train_logloss": float(train_logloss),
        "valid_logloss": float(valid_logloss),
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] {model_json_path}")
    print(f"[SAVE] {model_pkl_path}")
    print(f"[SAVE] {train_pred_path}")
    print(f"[SAVE] {valid_pred_path}")
    print(f"[SAVE] {hist_path}")
    print(f"[SAVE] {summary_path}")

if __name__ == "__main__":
    main()
