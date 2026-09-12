#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier


class DbcEvaluator:
    """
    Unified Dbc evaluator.

    Supported modes
    ---------------
    - "ratio":
        Dbc = bc / (bc + qcd + cc + cs + qq + topbw)
        or optionally include bb, bs in denominator

    - "bdt":
        Use trained BDT model and return predict_proba(... )[:, 1]
    """

    FEATURES = [
        "ak8_gpt_bc",
        "ak8_gpt_bb",
        "ak8_gpt_cc",
        "ak8_gpt_qcd",
        "ak8_gpt_bs",
        "ak8_gpt_qq",
        "ak8_gpt_cs",
        "ak8_gpt_topbw",
    ]

    def __init__(self, mode="bdt", model_path="./dbc_bdt_output/bdt_dbc_model.pkl",
                 use_bb_bs=False, eps=1e-10):
        self.mode = mode
        self.model_path = model_path
        self.use_bb_bs = use_bb_bs
        self.eps = eps

        self.model = None
        if self.mode == "bdt":
            self.model = self._load_model(model_path)

    @staticmethod
    def _as_1d_array(x, name):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
        return arr

    @staticmethod
    def _check_lengths(arrs, names):
        n = len(arrs[0])
        for arr, name in zip(arrs[1:], names[1:]):
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch: {names[0]} has length {n}, but {name} has length {len(arr)}"
                )
        return n

    @staticmethod
    def _load_model(model_path):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BDT model not found: {model_path}")
        return joblib.load(model_path)

    def get_ratio(
        self,
        g_bc,
        g_bb,
        g_cc,
        g_qcd,
        g_bs,
        g_qq,
        g_cs,
        g_topbw,
    ):
        """
        Return ratio-based Dbc.
        """
        g_bc    = self._as_1d_array(g_bc,    "g_bc")
        g_bb    = self._as_1d_array(g_bb,    "g_bb")
        g_cc    = self._as_1d_array(g_cc,    "g_cc")
        g_qcd   = self._as_1d_array(g_qcd,   "g_qcd")
        g_bs    = self._as_1d_array(g_bs,    "g_bs")
        g_qq    = self._as_1d_array(g_qq,    "g_qq")
        g_cs    = self._as_1d_array(g_cs,    "g_cs")
        g_topbw = self._as_1d_array(g_topbw, "g_topbw")

        self._check_lengths(
            [g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw],
            ["g_bc", "g_bb", "g_cc", "g_qcd", "g_bs", "g_qq", "g_cs", "g_topbw"]
        )

        if self.use_bb_bs:
            denom = g_bc + g_bb + g_bs + g_qcd + g_cc + g_cs + g_qq + g_topbw + self.eps
        else:
            denom = g_bc + g_qcd + g_cc + g_cs + g_qq + g_topbw + self.eps

        dbc = g_bc / denom
        return dbc.astype(np.float32)

    def get_bdt(
        self,
        g_bc,
        g_bb,
        g_cc,
        g_qcd,
        g_bs,
        g_qq,
        g_cs,
        g_topbw,
    ):
        """
        Return BDT-based Dbc using a trained model.
        """
        if self.model is None:
            raise RuntimeError("BDT model is not loaded. Initialize with mode='bdt'.")

        g_bc    = self._as_1d_array(g_bc,    "g_bc")
        g_bb    = self._as_1d_array(g_bb,    "g_bb")
        g_cc    = self._as_1d_array(g_cc,    "g_cc")
        g_qcd   = self._as_1d_array(g_qcd,   "g_qcd")
        g_bs    = self._as_1d_array(g_bs,    "g_bs")
        g_qq    = self._as_1d_array(g_qq,    "g_qq")
        g_cs    = self._as_1d_array(g_cs,    "g_cs")
        g_topbw = self._as_1d_array(g_topbw, "g_topbw")

        self._check_lengths(
            [g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw],
            ["g_bc", "g_bb", "g_cc", "g_qcd", "g_bs", "g_qq", "g_cs", "g_topbw"]
        )

        X = pd.DataFrame({
            "ak8_gpt_bc": g_bc,
            "ak8_gpt_bb": g_bb,
            "ak8_gpt_cc": g_cc,
            "ak8_gpt_qcd": g_qcd,
            "ak8_gpt_bs": g_bs,
            "ak8_gpt_qq": g_qq,
            "ak8_gpt_cs": g_cs,
            "ak8_gpt_topbw": g_topbw,
        })

        dbc = self.model.predict_proba(X[self.FEATURES])[:, 1]
        return np.asarray(dbc, dtype=np.float32)

    def get_Dbc(
        self,
        g_bc,
        g_bb,
        g_cc,
        g_qcd,
        g_bs,
        g_qq,
        g_cs,
        g_topbw,
    ):
        """
        Unified interface.
        """
        if self.mode == "ratio":
            return self.get_ratio(g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw)
        elif self.mode == "bdt":
            return self.get_bdt(g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'ratio' or 'bdt'.")


class Dbc3ClassEvaluator:
    """
    Youpeng's 3-class (bc / bb / other) GloParT-node BDT, loaded natively as a
    saved XGBoost booster (dbc_3class_v1/output_inclusive_v1/models/expanded/
    model.json).  Unlike DbcEvaluator (old 8-node sklearn pickle via joblib),
    this model takes the 33 ak8_gpt_* node scores below -- exactly the feature
    order train.py's eval_dbc_3class uses -- and returns predict_proba columns
    [:, 0]=bc, [:, 1]=bb, [:, 2]=other.
    """

    FEATURES = [
        "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_bc", "ak8_gpt_qcd", "ak8_gpt_bs",
        "ak8_gpt_cs", "ak8_gpt_qq", "ak8_gpt_topbw", "ak8_gpt_topw", "ak8_gpt_ss",
        "ak8_gpt_tauhtauh", "ak8_gpt_qcdbb", "ak8_gpt_qcdb", "ak8_gpt_qcdcc",
        "ak8_gpt_qcdc", "ak8_gpt_qcdothers", "ak8_gpt_topbwcs", "ak8_gpt_topbwqq",
        "ak8_gpt_topbwq", "ak8_gpt_topbws", "ak8_gpt_topbwc", "ak8_gpt_topbwev",
        "ak8_gpt_topbwmv", "ak8_gpt_topbwtauev", "ak8_gpt_topbwtauhv",
        "ak8_gpt_topbwtaumv", "ak8_gpt_topwcs", "ak8_gpt_topwqq", "ak8_gpt_topwev",
        "ak8_gpt_topwmv", "ak8_gpt_topwtauev", "ak8_gpt_topwtauhv", "ak8_gpt_topwtaumv",
    ]

    def __init__(self, model_path):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"3-class Dbc model not found: {model_path}")
        self.model = XGBClassifier()
        self.model.load_model(model_path)

    def get_scores(self, feature_map):
        """
        feature_map: dict of {feature_name: 1D array}, one entry per name in
        self.FEATURES (all must have the same length).
        Returns (p_bc, p_bb) as float32 1D arrays.
        """
        n = None
        cols = {}
        for name in self.FEATURES:
            arr = np.asarray(feature_map[name], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError(f"Length mismatch: {name} has length {len(arr)}, expected {n}")
            cols[name] = arr
        X = pd.DataFrame(cols)[self.FEATURES]
        proba = self.model.predict_proba(X)
        p_bc = np.ascontiguousarray(proba[:, 0], dtype=np.float32)
        p_bb = np.ascontiguousarray(proba[:, 1], dtype=np.float32)
        return p_bc, p_bb
