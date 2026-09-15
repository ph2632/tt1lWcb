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


class S1Evaluator:
    """The project's own S1 boosted-cb tagger (tt1lWcb/02_train_tagger.py):
    binary XGBoost over 22 ak8_gpt_* features, loaded natively as a saved
    XGBoost booster. FEATURES/LEPQ_COMPONENTS mirror S1_tagger/config.json's
    features_S1/derived_features exactly (2026-09-13 -- feature #22,
    ak8_gpt_qcd, added on top of the original 21).

    Tied to ONE specific trained run (model_path); re-point it after any
    retrain you want reflected here (S1's model.json is overwritten in place
    by 02_train_tagger.py on every run under the same run_tag).
    """

    LEPQ_COMPONENTS = [
        "ak8_gpt_topbwev", "ak8_gpt_topbwmv", "ak8_gpt_topbwtauev",
        "ak8_gpt_topbwtauhv", "ak8_gpt_topbwtaumv", "ak8_gpt_topwev",
        "ak8_gpt_topwmv", "ak8_gpt_topwtauev", "ak8_gpt_topwtauhv",
        "ak8_gpt_topwtaumv",
    ]
    FEATURES = [
        "ak8_gpt_bb", "ak8_gpt_cc", "ak8_gpt_bc", "ak8_gpt_bs", "ak8_gpt_cs",
        "ak8_gpt_qq", "ak8_gpt_ss", "ak8_gpt_tauhtauh", "ak8_gpt_qcdb",
        "ak8_gpt_qcdbb", "ak8_gpt_qcdc", "ak8_gpt_qcdcc", "ak8_gpt_qcdothers",
        "ak8_gpt_topbwc", "ak8_gpt_topbwcs", "ak8_gpt_topbwq", "ak8_gpt_topbwqq",
        "ak8_gpt_topbws", "ak8_gpt_topwqq", "ak8_gpt_topwcs", "ak8_gpt_lepq",
        "ak8_gpt_qcd",
    ]
    # every RAW branch the caller needs to supply -- FEATURES minus the
    # derived ak8_gpt_lepq, plus the 10 raw components that sum to it
    RAW_INPUTS = [f for f in FEATURES if f != "ak8_gpt_lepq"] + LEPQ_COMPONENTS

    def __init__(self, model_path):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"S1 model not found: {model_path}")
        self.model = XGBClassifier()
        self.model.load_model(model_path)

    def get_score(self, feature_map):
        """feature_map: dict of {name: 1D array}, one entry per name in
        RAW_INPUTS (all must have the same length). Returns score_S1 as a
        float32 1D array."""
        n = None
        cols = {}
        for name in self.RAW_INPUTS:
            arr = np.asarray(feature_map[name], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError(f"Length mismatch: {name} has length {len(arr)}, expected {n}")
            cols[name] = arr
        cols["ak8_gpt_lepq"] = np.sum([cols[c] for c in self.LEPQ_COMPONENTS], axis=0)
        X = pd.DataFrame(cols)[self.FEATURES]
        proba = self.model.predict_proba(X)
        return np.ascontiguousarray(proba[:, 1], dtype=np.float32)


class M3Evaluator:
    """This project's M3 multiclass tagger (tt1lWcb/02_train_tagger.py):
    4-class (bkg/cb/bb/bbc) native XGBoost multi:softprob booster, over the
    SAME 22 ak8_gpt_* features as S1Evaluator (02_train_tagger.py's main()
    passes M3 the identical feats_s1 list -- verified 2026-09-14). Loaded
    exactly like S1Evaluator (XGBClassifier + load_model), the only
    difference being predict_proba returns 4 columns here instead of 2.

    Tied to ONE specific trained run (model_path); re-point it after any
    retrain you want reflected here (M3's model.json is overwritten in
    place by 02_train_tagger.py on every run under the same run_tag).
    """

    LEPQ_COMPONENTS = S1Evaluator.LEPQ_COMPONENTS
    FEATURES = S1Evaluator.FEATURES
    RAW_INPUTS = S1Evaluator.RAW_INPUTS

    def __init__(self, model_path):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"M3 model not found: {model_path}")
        self.model = XGBClassifier()
        self.model.load_model(model_path)

    def get_scores(self, feature_map):
        """feature_map: dict of {name: 1D array}, one entry per name in
        RAW_INPUTS (all must have the same length). Returns (p_cb, p_bb,
        p_bbc) as float32 1D arrays -- class order bkg/cb/bb/bbc, columns
        1/2/3, matching S1_tagger/config.json's M3_class_names."""
        n = None
        cols = {}
        for name in self.RAW_INPUTS:
            arr = np.asarray(feature_map[name], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError(f"Length mismatch: {name} has length {len(arr)}, expected {n}")
            cols[name] = arr
        cols["ak8_gpt_lepq"] = np.sum([cols[c] for c in self.LEPQ_COMPONENTS], axis=0)
        X = pd.DataFrame(cols)[self.FEATURES]
        proba = self.model.predict_proba(X)
        return (np.ascontiguousarray(proba[:, 1], dtype=np.float32),
                np.ascontiguousarray(proba[:, 2], dtype=np.float32),
                np.ascontiguousarray(proba[:, 3], dtype=np.float32))
