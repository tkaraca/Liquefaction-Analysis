#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
table6_leakage_free_6f_fixed.py

Purpose
-------
Create a leakage-free version of *Table 6* using the FULL 6-feature set
(pga, H, B, q, depth, thickness) and generate per-model SHAP importance
plots on TRAIN data only.

Why this script (what it fixes vs the old one)
----------------------------------------------
1) Uses a strict "split-first" protocol (hold-out test set is created
   BEFORE any preprocessing/resampling).
2) Applies IQR clipping and RandomOverSampler ONLY within the training
   folds via an imblearn Pipeline (leakage-free).
3) Removes any threshold tuning (argmax decision rule).
4) Fixes the SHAP crash you saw:
   "GradientBoostingClassifier is only supported for binary classification right now!"
   by using a model-agnostic KernelExplainer fallback for GradientBoostingClassifier.

Outputs
-------
- outputs_table6_6f/table6_leakage_free_TEST.csv
- outputs_table6_6f/table6_leakage_free_TEST.xlsx
- outputs_table6_6f/shap_importance_<MODEL>.png/.pdf
- outputs_table6_6f/SHAP_importance_normalized_per_model_6f.csv
- outputs_table6_6f/SHAP_importance_normalized_per_model_6f_long.csv

Run
---
python table6_leakage_free_6f_fixed.py

Requirements
------------
pip install -U pandas numpy scikit-learn imbalanced-learn shap xgboost openpyxl matplotlib
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    raise ImportError("xgboost is required. Install it via: pip install xgboost") from e

import shap


# -----------------------------
# Config (edit if needed)
# -----------------------------
DATA_PATH = "adapvtest.csv"
CSV_SEP = ";"

FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
TARGET = "dver"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

OUTPUT_DIR = "outputs_table6_6f"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SHAP computation knobs (KernelExplainer can be slow; keep these modest)
KERNEL_BACKGROUND_N = 50   # number of background samples for KernelExplainer
KERNEL_EXPLAIN_N = 120     # number of training samples to explain (subsample)
KERNEL_NSAMPLES = 200      # shap kernel nsamples; increase if you want more stable values


# -----------------------------
# Leakage-free IQR clipper
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to [Q1 - factor*IQR, Q3 + factor*IQR] fitted on TRAIN only."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).to_numpy()
        self.upper_ = (q3 + self.factor * iqr).to_numpy()
        self.feature_names_in_ = list(X_df.columns)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_arr = X_df.to_numpy(dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)


# -----------------------------
# Metrics helpers
# -----------------------------
def evaluate_multiclass(y_true, y_pred):
    """Return accuracy + macro/weighted precision/recall/F1."""
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # class-specific recall (engineering relevance): Class 2 (Moderate), Class 3 (Extensive)
    labels = sorted(pd.unique(y_true))
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recalls = dict(zip(labels, per_class[1]))

    return {
        "Accuracy": float(acc),
        "Macro Precision": float(p_macro),
        "Macro Recall": float(r_macro),
        "Macro F1": float(f1_macro),
        "Weighted Precision": float(p_w),
        "Weighted Recall": float(r_w),
        "Weighted F1": float(f1_w),
        "Recall Class 2": float(recalls.get(2, np.nan)),
        "Recall Class 3": float(recalls.get(3, np.nan)),
    }


# -----------------------------
# SHAP helpers
# -----------------------------
def shap_importance_per_feature(shap_values, feature_names):
    """
    Compute mean(|SHAP|) per feature for multiclass or binary.
    Handles shap outputs:
      - list of arrays [n_classes], each (n_samples, n_features)
      - array (n_samples, n_features) for binary
      - array (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        vals = np.stack([np.abs(sv) for sv in shap_values], axis=0)  # (K, n, p)
        imp = vals.mean(axis=(0, 1))  # (p,)
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            # (n, p, K)
            imp = np.abs(sv).mean(axis=(0, 2))
        else:
            # (n, p)
            imp = np.abs(sv).mean(axis=0)
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def save_importance_bar(importance: pd.Series, title: str, out_png: str, out_pdf: str):
    plt.figure(figsize=(7.0, 4.2))
    importance.iloc[::-1].plot(kind="barh")
    plt.title(title)
    plt.xlabel("mean(|SHAP value|) on TRAIN (leakage-free)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def compute_shap_importance(model_name: str, model, X_train_iqr: pd.DataFrame) -> pd.Series:
    """
    Compute SHAP importance (mean|SHAP|) for a fitted model, on TRAIN-only clipped data.

    - TreeExplainer is used for RandomForest & XGBoost.
    - GradientBoostingClassifier multi-class can crash with TreeExplainer in some SHAP versions,
      so we use KernelExplainer fallback for GB.
    """
    # Try TreeExplainer first for tree-based models (fast)
    if model_name in {"RandomForest", "XGBoost"}:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train_iqr)
        return shap_importance_per_feature(shap_vals, list(X_train_iqr.columns))

    if model_name == "GradientBoosting":
        # 1) Attempt TreeExplainer; if it fails, fall back to KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_train_iqr)
            return shap_importance_per_feature(shap_vals, list(X_train_iqr.columns))
        except Exception:
            # 2) KernelExplainer fallback (model-agnostic, slower but robust)
            # Background (small) + explanation sample (small)
            bg_n = min(KERNEL_BACKGROUND_N, len(X_train_iqr))
            ex_n = min(KERNEL_EXPLAIN_N, len(X_train_iqr))

            background = shap.sample(X_train_iqr, bg_n, random_state=RANDOM_STATE)
            explain_X = shap.sample(X_train_iqr, ex_n, random_state=RANDOM_STATE)

            # KernelExplainer expects numpy arrays
            f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X_train_iqr.columns))
            explainer = shap.KernelExplainer(f, background.to_numpy())
            shap_vals = explainer.shap_values(explain_X.to_numpy(), nsamples=KERNEL_NSAMPLES)

            return shap_importance_per_feature(shap_vals, list(X_train_iqr.columns))

    raise ValueError(f"Unknown model_name: {model_name}")


# -----------------------------
# Model tuning (leakage-free)
# -----------------------------
def tune_pipeline(model, param_grid: dict, X_train, y_train, cv):
    """GridSearchCV over a leakage-free pipeline: IQR -> ROS -> model."""
    pipe = Pipeline(
        steps=[
            ("iqr", IQRClipper(factor=1.5)),
            ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="recall_macro",   # align with imbalance focus (you can switch to f1_macro)
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def make_param_grids():
    """Compact grids (keep small for speed/reproducibility)."""
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__class_weight": [None, "balanced"],
    }

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_grid = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_grid = {
        "model__n_estimators": [200, 500],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0],
    }

    return {
        "RandomForest": (rf, rf_grid),
        "XGBoost": (xgb, xgb_grid),
        "GradientBoosting": (gb, gb_grid),
    }


def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH, sep=CSV_SEP)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # 2) Split-first (critical for leakage-free)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    index_overlap = len(set(X_train.index) & set(X_test.index))
    print(f"Index overlap after split-first (should be 0): {index_overlap}")

    # 3) CV
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 4) Tune models (TRAIN only), evaluate on untouched TEST, compute SHAP on TRAIN only
    grids = make_param_grids()
    best_pipes = {}
    shap_rows = []   # for CSV output

    table_rows = []

    print("\n=== Tuning models on FULL 6-feature set (TRAIN only) ===")
    for name, (model, grid) in grids.items():
        best_pipe, best_params = tune_pipeline(model, grid, X_train, y_train, cv)
        best_pipes[name] = (best_pipe, best_params)
        print(f"{name} best params: {best_params}")

        # TEST evaluation (argmax)
        prob_test = best_pipe.predict_proba(X_test)
        pred_test = prob_test.argmax(axis=1)

        metrics = evaluate_multiclass(y_test, pred_test)
        table_rows.append({
            "Model": name,
            "Best Hyperparameters (CV on TRAIN)": str(best_params),
            **metrics,
        })

        print(f"\n{name} TEST report (argmax, leakage-free):\n{classification_report(y_test, pred_test)}")

        # SHAP on TRAIN only (use IQR-transformed TRAIN; do NOT use ROS output for explanations)
        iqr = best_pipe.named_steps["iqr"]
        mdl = best_pipe.named_steps["model"]
        X_train_iqr = pd.DataFrame(iqr.transform(X_train), columns=FEATURES)

        imp = compute_shap_importance(name, mdl, X_train_iqr)

        # Save SHAP bar plot
        save_importance_bar(
            imp,
            title=f"SHAP Importance (TRAIN, 6 features) - {name}",
            out_png=os.path.join(OUTPUT_DIR, f"shap_importance_{name}_6f.png"),
            out_pdf=os.path.join(OUTPUT_DIR, f"shap_importance_{name}_6f.pdf"),
        )

        # Normalize importance to sum=1 for easier comparison across models
        imp_norm = (imp / imp.sum()) if imp.sum() > 0 else imp

        for feat, val in imp_norm.items():
            shap_rows.append({"Model": name, "Feature": feat, "SHAP_importance_norm": float(val)})

    # 5) Ensemble (soft voting) with tuned base-model hyperparameters
    rf_params = {k.replace("model__", ""): v for k, v in best_pipes["RandomForest"][1].items()}
    xgb_params = {k.replace("model__", ""): v for k, v in best_pipes["XGBoost"][1].items()}
    gb_params = {k.replace("model__", ""): v for k, v in best_pipes["GradientBoosting"][1].items()}

    rf_est = RandomForestClassifier(random_state=RANDOM_STATE, **rf_params)
    xgb_est = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        **xgb_params,
    )
    gb_est = GradientBoostingClassifier(random_state=RANDOM_STATE, **gb_params)

    voting = VotingClassifier(
        estimators=[("rf", rf_est), ("xgb", xgb_est), ("gb", gb_est)],
        voting="soft",
        weights=None,  # you may set weights based on CV performance if desired
    )

    ensemble_pipe = Pipeline(
        steps=[
            ("iqr", IQRClipper(factor=1.5)),
            ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", voting),
        ]
    )

    ensemble_pipe.fit(X_train, y_train)
    prob_test_ens = ensemble_pipe.predict_proba(X_test)
    pred_test_ens = prob_test_ens.argmax(axis=1)

    ens_metrics = evaluate_multiclass(y_test, pred_test_ens)
    table_rows.append({
        "Model": "Ensemble (RF+XGB+GB, soft voting)",
        "Best Hyperparameters (CV on TRAIN)": "Uses tuned params of base models (see rows above)",
        **ens_metrics,
    })

    print(f"\nEnsemble TEST report (argmax, leakage-free):\n{classification_report(y_test, pred_test_ens)}")

    # 6) Save Table 6
    table6 = pd.DataFrame(table_rows)

    # Preferred column order
    col_order = [
        "Model",
        "Best Hyperparameters (CV on TRAIN)",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Recall Class 2",
        "Recall Class 3",
    ]
    table6 = table6[col_order]

    out_csv = os.path.join(OUTPUT_DIR, "table6_leakage_free_TEST.csv")
    out_xlsx = os.path.join(OUTPUT_DIR, "table6_leakage_free_TEST.xlsx")
    table6.to_csv(out_csv, index=False)
    table6.to_excel(out_xlsx, index=False)

    print("\n=== Table 6 (leakage-free, 6 features, TEST) ===")
    print(table6.to_string(index=False))
    print(f"\nSaved:\n- {out_csv}\n- {out_xlsx}")

    # 7) Save SHAP importance CSVs (normalized, per-model)
    shap_df_long = pd.DataFrame(shap_rows)
    shap_df_wide = shap_df_long.pivot_table(
        index="Feature", columns="Model", values="SHAP_importance_norm", aggfunc="mean"
    ).fillna(0.0)

    out_shap_long = os.path.join(OUTPUT_DIR, "SHAP_importance_normalized_per_model_6f_long.csv")
    out_shap_wide = os.path.join(OUTPUT_DIR, "SHAP_importance_normalized_per_model_6f.csv")

    shap_df_long.to_csv(out_shap_long, index=False)
    shap_df_wide.to_csv(out_shap_wide)

    print(f"\nSaved SHAP normalized importance:\n- {out_shap_wide}\n- {out_shap_long}")


if __name__ == "__main__":
    main()
