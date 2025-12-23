#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
table4_leakage_free.py
----------------------
Creates a leakage-free (split-first) version of "Table 4" for the manuscript.

What it does
- Loads the dataset (default: adapvtest.csv; separator ';')
- Splits FIRST into train/test (stratified)
- Applies outlier handling *inside the pipeline* (fit only on train folds)
- Applies Random Over-Sampling (ROS) *inside the pipeline* (train-fold only)
- Tunes each model with GridSearchCV on TRAIN only
- Evaluates final tuned model on untouched TEST (argmax decision; NO threshold tuning)
- Saves:
  * table4_leakage_free_TEST.xlsx / .csv  (main table to paste into Word)
  * table4_leakage_free_TRAIN_CV.xlsx / .csv (optional: CV & train diagnostics)
  * best_params_table4_*.json per outlier method
  * aucs_test_table4_*.csv per outlier method (per-class AUC + macro mean)

Default structure (to match your existing Table 4 sections)
- Outlier methods: Winsorized, Z-Score, IQR
- Models: RandomForest, XGBoost, GradientBoosting, NeuralNetwork
- Optional: Ensemble row (RF+XGB+GB soft voting) per outlier method

Usage
    python table4_leakage_free.py
    python table4_leakage_free.py --data adapvtest.csv --sep ";" --outdir outputs_table4 --fast
    python table4_leakage_free.py --only-method IQR --no-nn --no-ensemble

Notes
- The outlier operations are implemented as leakage-free *clipping* (winsor-like),
  not row deletion. This is the safest way to keep the process leakage-free.
  If you must do row removal (filtering) inside CV, it is possible but more complex.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    raise ImportError("xgboost is required. Install via: pip install xgboost") from e


DEFAULT_FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
DEFAULT_TARGET = "dver"


@dataclass
class Config:
    data_path: str = "adapvtest.csv"
    sep: str = ";"
    outdir: str = "outputs_table4_leakage_free"
    random_state: int = 42
    test_size: float = 0.20
    cv_splits: int = 5
    n_jobs: int = -1
    fast_mode: bool = False


# -----------------------------
# Leakage-free outlier clippers
# -----------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Leakage-free winsorization-like clipping based on train quantiles."""
    def __init__(self, q_low: float = 0.05, q_high: float = 0.95):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.quantile(X, self.q_low, axis=0)
        self.upper_ = np.quantile(X, self.q_high, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


class ZScoreClipper(BaseEstimator, TransformerMixin):
    """Leakage-free z-score clipping based on train mean/std."""
    def __init__(self, z: float = 3.0, eps: float = 1e-12):
        self.z = z
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mu_ = np.mean(X, axis=0)
        self.sigma_ = np.std(X, axis=0)
        self.sigma_ = np.where(self.sigma_ < self.eps, 1.0, self.sigma_)
        self.lower_ = self.mu_ - self.z * self.sigma_
        self.upper_ = self.mu_ + self.z * self.sigma_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Leakage-free IQR clipping based on train quartiles."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.quantile(X, 0.25, axis=0)
        q3 = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


# -----------------------------
# Metrics helpers
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf1),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf1),
    }


def auc_table(y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    y_true_bin = label_binarize(y_true, classes=classes)
    out = {}
    per_class = []
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        per_class.append(float(auc(fpr, tpr)))
        out[f"auc_class_{int(c)}"] = per_class[-1]
    out["auc_macro_mean_per_class"] = float(np.mean(per_class))
    return out


# -----------------------------
# Build models + grids
# -----------------------------
def build_models_and_grids(cfg: Config, include_nn: bool) -> Tuple[Dict[str, object], Dict[str, dict]]:
    rf = RandomForestClassifier(random_state=cfg.random_state)
    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    gb = GradientBoostingClassifier(random_state=cfg.random_state)

    models = {
        "RandomForest": rf,
        "XGBoost": xgb,
        "GradientBoosting": gb,
    }

    if include_nn:
        nn = MLPClassifier(
            random_state=cfg.random_state,
            early_stopping=True,
            max_iter=2000,
        )
        models["NeuralNetwork"] = nn

    if cfg.fast_mode:
        grids = {
            "RandomForest": {"clf__n_estimators": [200], "clf__max_depth": [None, 10]},
            "XGBoost": {"clf__n_estimators": [300], "clf__learning_rate": [0.1], "clf__max_depth": [4]},
            "GradientBoosting": {"clf__n_estimators": [300], "clf__learning_rate": [0.1], "clf__max_depth": [3]},
        }
        if include_nn:
            grids["NeuralNetwork"] = {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [1e-4], "clf__learning_rate_init": [1e-3]}
    else:
        grids = {
            "RandomForest": {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__class_weight": [None, "balanced"],
            },
            "XGBoost": {
                "clf__n_estimators": [300, 600],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
            "GradientBoosting": {
                "clf__n_estimators": [200, 500],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3, 4],
                "clf__subsample": [0.8, 1.0],
                "clf__n_iter_no_change": [10],
                "clf__validation_fraction": [0.15],
            },
        }
        if include_nn:
            grids["NeuralNetwork"] = {
                "clf__hidden_layer_sizes": [(32, 16), (64, 32)],
                "clf__alpha": [1e-4, 1e-3],
                "clf__learning_rate_init": [1e-3, 1e-2],
            }

    return models, grids


def build_outlier_methods() -> Dict[str, BaseEstimator]:
    return {
        "Winsorized": QuantileClipper(q_low=0.05, q_high=0.95),
        "Z-Score": ZScoreClipper(z=3.0),
        "IQR": IQRClipper(factor=1.5),
    }


def make_pipeline(outlier: BaseEstimator, model) -> ImbPipeline:
    return ImbPipeline(steps=[
        ("outlier", outlier),
        ("scaler", StandardScaler()),
        ("ros", RandomOverSampler(random_state=42)),
        ("clf", model),
    ])


# -----------------------------
# Main run
# -----------------------------
def run_one_method(
    method_name: str,
    outlier: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Config,
    include_nn: bool,
    include_ensemble: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    """
    Returns:
      - test_metrics_df: rows=models, cols=metrics (+ macro_auc)
      - train_metrics_df: rows=models, cols=metrics (on train fit-predict; diagnostic)
      - test_aucs_df: rows=models, per-class AUC + macro mean
      - best_params: dict per model
    """
    models, grids = build_models_and_grids(cfg, include_nn)
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    best_estimators: Dict[str, object] = {}
    best_params: Dict[str, dict] = {}
    best_cv_recall: Dict[str, float] = {}

    for mname, model in models.items():
        pipe = make_pipeline(outlier, model)
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grids[mname],
            scoring="recall_macro",
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_estimators[mname] = search.best_estimator_
        best_params[mname] = search.best_params_
        best_cv_recall[mname] = float(search.best_score_)
        print(f"[{method_name}] {mname} best CV recall_macro = {search.best_score_:.4f}")

    # Optional ensemble (RF+XGB+GB) using tuned pipelines
    if include_ensemble:
        required = ["RandomForest", "XGBoost", "GradientBoosting"]
        if all(k in best_estimators for k in required):
            ens = VotingClassifier(
                estimators=[
                    ("rf", best_estimators["RandomForest"]),
                    ("xgb", best_estimators["XGBoost"]),
                    ("gb", best_estimators["GradientBoosting"]),
                ],
                voting="soft",
            )
            best_estimators["Ensemble"] = ens
            best_params["Ensemble"] = {"note": "Soft voting of tuned pipelines (RF+XGB+GB)."}
            best_cv_recall["Ensemble"] = np.nan  # not from CV grid search

    # Evaluate
    classes = np.sort(np.unique(y_train))
    test_rows = []
    train_rows = []
    auc_rows = []

    for mname, est in best_estimators.items():
        est.fit(X_train, y_train)

        yhat_train = est.predict(X_train)
        yhat_test = est.predict(X_test)

        m_train = compute_metrics(y_train.to_numpy(), yhat_train)
        m_train["model"] = mname
        m_train["cv_recall_macro_best"] = float(best_cv_recall.get(mname, np.nan))
        train_rows.append(m_train)

        m_test = compute_metrics(y_test.to_numpy(), yhat_test)
        m_test["model"] = mname
        m_test["cv_recall_macro_best"] = float(best_cv_recall.get(mname, np.nan))
        test_rows.append(m_test)

        # AUC needs probabilities
        try:
            prob = est.predict_proba(X_test)
            aucs = auc_table(y_test.to_numpy(), prob, classes)
        except Exception:
            aucs = {"auc_macro_mean_per_class": np.nan}
        aucs["model"] = mname
        auc_rows.append(aucs)

    test_metrics_df = pd.DataFrame(test_rows).set_index("model").sort_index()
    train_metrics_df = pd.DataFrame(train_rows).set_index("model").sort_index()
    test_aucs_df = pd.DataFrame(auc_rows).set_index("model").sort_index()

    return test_metrics_df, train_metrics_df, test_aucs_df, best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adapvtest.csv", help="Path to dataset CSV")
    parser.add_argument("--sep", default=";", help="CSV separator (default ';')")
    parser.add_argument("--outdir", default="outputs_table4_leakage_free", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (smaller grids)")
    parser.add_argument("--only-method", default="", help="Run only one outlier method: Winsorized | Z-Score | IQR")
    parser.add_argument("--no-nn", action="store_true", help="Drop Neural Network row (recommended if you tone down deep learning)")
    parser.add_argument("--no-ensemble", action="store_true", help="Do not add Ensemble row")
    args = parser.parse_args()

    cfg = Config(data_path=args.data, sep=args.sep, outdir=args.outdir, fast_mode=args.fast)
    os.makedirs(cfg.outdir, exist_ok=True)

    # Load
    df = pd.read_csv(cfg.data_path, sep=cfg.sep, engine="python")
    X = df[DEFAULT_FEATURES].copy()
    y = df[DEFAULT_TARGET].astype(int).copy()

    # Split FIRST (leakage-free)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    include_nn = not args.no_nn
    include_ensemble = not args.no_ensemble

    outlier_methods = build_outlier_methods()
    if args.only_method:
        if args.only_method not in outlier_methods:
            raise ValueError(f"--only-method must be one of: {list(outlier_methods.keys())}")
        outlier_methods = {args.only_method: outlier_methods[args.only_method]}

    all_test_tables = []
    all_train_tables = []
    all_auc_tables = []

    for method_name, outlier in outlier_methods.items():
        print(f"\n=== Running Table 4 (leakage-free) for outlier method: {method_name} ===")
        test_df, train_df, auc_df, best_params = run_one_method(
            method_name, outlier, X_train, y_train, X_test, y_test, cfg,
            include_nn=include_nn, include_ensemble=include_ensemble
        )

        # Add a column to keep sections in the final merged table
        test_df.insert(0, "outlier_method", method_name)
        train_df.insert(0, "outlier_method", method_name)
        auc_df.insert(0, "outlier_method", method_name)

        all_test_tables.append(test_df.reset_index())
        all_train_tables.append(train_df.reset_index())
        all_auc_tables.append(auc_df.reset_index())

        with open(os.path.join(cfg.outdir, f"best_params_table4_{method_name}.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

        test_df.to_csv(os.path.join(cfg.outdir, f"metrics_test_table4_{method_name}.csv"), index=True)
        train_df.to_csv(os.path.join(cfg.outdir, f"metrics_train_table4_{method_name}.csv"), index=True)
        auc_df.to_csv(os.path.join(cfg.outdir, f"aucs_test_table4_{method_name}.csv"), index=True)

    # Merge for a single "Table 4" file
    table4_test = pd.concat(all_test_tables, axis=0, ignore_index=True)
    table4_train = pd.concat(all_train_tables, axis=0, ignore_index=True)
    table4_auc = pd.concat(all_auc_tables, axis=0, ignore_index=True)

    # Order columns (nice for Word copy)
    metric_cols = [
        "cv_recall_macro_best",
        "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ]
    for df_out, fname in [
        (table4_test, "table4_leakage_free_TEST"),
        (table4_train, "table4_leakage_free_TRAIN_DIAGNOSTIC"),
    ]:
        keep = ["outlier_method", "model"] + [c for c in metric_cols if c in df_out.columns]
        df_out = df_out[keep]
        df_out.to_csv(os.path.join(cfg.outdir, f"{fname}.csv"), index=False)
        df_out.to_excel(os.path.join(cfg.outdir, f"{fname}.xlsx"), index=False)

    # AUC summary
    table4_auc.to_csv(os.path.join(cfg.outdir, "table4_leakage_free_AUC_TEST.csv"), index=False)
    table4_auc.to_excel(os.path.join(cfg.outdir, "table4_leakage_free_AUC_TEST.xlsx"), index=False)

    print("\nDONE.")
    print("Outputs:", os.path.abspath(cfg.outdir))
    print("Main files to use for the manuscript Table 4:")
    print(" - table4_leakage_free_TEST.xlsx (copy/paste into Word)")
    print(" - table4_leakage_free_AUC_TEST.xlsx (if you want to add macro-AUC column)")

if __name__ == "__main__":
    main()
