#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_figures_leakage_free.py
-----------------------------
Creates leakage-free versions of the ROC/AUC figures referenced in the manuscript
(Fig. 8 / 9 / 10) while keeping the same visual layout (2x3 grid, 4 model panels
+ macro-average panel).

Key fixes vs your original workflow:
1) Split FIRST (train/test).
2) Any outlier handling is fitted on TRAIN ONLY (IQR-based clipping).
3) Oversampling (ROS) is applied ONLY inside TRAIN folds using imblearn Pipeline
   (so no synthetic/duplicated samples end up in the test set).

Outputs (default):
- outputs_paper_figures_leakage_free/Fig8_cost_sensitive_ROC.(png|pdf)
- outputs_paper_figures_leakage_free/Fig9_ROS_IQR_ROC.(png|pdf)
- outputs_paper_figures_leakage_free/Fig10_ROS_IQR_Ensemble_ROC.(png|pdf)
- CSV/JSON summaries of AUCs, metrics, and best params for transparency.

How to run:
    python paper_figures_leakage_free.py

If your CSV is elsewhere:
    python paper_figures_leakage_free.py --data /path/to/adapvtest.csv

Notes:
- This script uses the columns: pga, H, B, q, depth, thickness, dver
- "IQR filtered" is implemented as IQR *clipping* (winsor-like) to avoid dropping
  rows and to keep the process leakage-free. If you strictly need row-removal,
  ask and I will adapt it (it’s more complex to do leakage-free inside CV).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    raise ImportError("xgboost is required for this script. Please install xgboost.") from e


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
DEFAULT_TARGET = "dver"


@dataclass
class Config:
    data_path: str = "adapvtest.csv"
    sep: str = ";"
    outdir: str = "outputs_paper_figures_leakage_free"

    features: List[str] = None
    target: str = DEFAULT_TARGET

    test_size: float = 0.20
    random_state: int = 42

    cv_splits: int = 5
    n_jobs: int = -1

    fast_mode: bool = False  # smaller grids, faster debugging


# -----------------------------
# Leakage-free outlier handler
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clips each feature to [Q1 - factor*IQR, Q3 + factor*IQR] computed on TRAIN ONLY.
    This is leakage-free and typically preferred over filtering/removing rows before split.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.quantile(X, 0.25, axis=0)
        self.q3_ = np.quantile(X, 0.75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * iqr
        self.upper_ = self.q3_ + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


# -----------------------------
# Metrics & plotting helpers
# -----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wprec, wrec, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_precision": float(wprec),
        "weighted_recall": float(wrec),
        "weighted_f1": float(wf1),
    }


def compute_macro_roc(y_true_bin: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Macro-average ROC:
    - Compute one-vs-rest ROC for each class
    - Interpolate TPR on a shared mean_fpr grid
    - Average TPRs across classes
    - macro_auc = mean of per-class AUCs (same convention as your code)
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(y_true_bin.shape[1]):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_i = auc(fpr_i, tpr_i)
        aucs.append(auc_i)

        tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(np.mean(aucs))
    return mean_fpr, mean_tpr, macro_auc


def plot_multiclass_roc(ax, y_true_bin: np.ndarray, y_prob: np.ndarray, classes: np.ndarray, title: str):
    for class_label in classes:
        idx = int(np.where(classes == class_label)[0][0])
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"Class {class_label} (AUC={auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, title="Classes", title_fontsize=12, frameon=True)


def save_figure(fig, outdir: str, basename: str, dpi: int = 600) -> None:
    png_path = os.path.join(outdir, f"{basename}.png")
    pdf_path = os.path.join(outdir, f"{basename}.pdf")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, outdir: str, fname: str) -> None:
    with open(os.path.join(outdir, fname), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_df(df: pd.DataFrame, outdir: str, fname: str) -> None:
    df.to_csv(os.path.join(outdir, fname), index=True)


def leakage_audit_split(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, int]:
    """
    Basic leakage audit.
    - Index overlap should be 0 if you split first (as expected).
    - Hash overlap detects *identical original rows* across train/test (can happen if dataset itself has duplicates).
    """
    idx_overlap = len(set(X_train.index) & set(X_test.index))

    train_rows = pd.concat([X_train, y_train.rename("y")], axis=1)
    test_rows = pd.concat([X_test, y_test.rename("y")], axis=1)

    train_hash = pd.util.hash_pandas_object(train_rows, index=False)
    test_hash = pd.util.hash_pandas_object(test_rows, index=False)
    hash_overlap = int(len(set(train_hash) & set(test_hash)))

    return {"index_overlap": int(idx_overlap), "identical_row_hash_overlap": hash_overlap}


def auc_table(y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    y_true_bin = label_binarize(y_true, classes=classes)
    out = {}
    per_class = []
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        per_class.append(float(auc(fpr, tpr)))
        out[f"auc_class_{int(c)}"] = per_class[-1]
    out["auc_macro_mean_per_class"] = float(np.mean(per_class))
    # also macro-average curve auc is not directly defined; we follow your mean-of-class-AUC convention
    return out


# -----------------------------
# Model training blocks
# -----------------------------
def build_cost_sensitive_models(cfg: Config) -> Tuple[Dict[str, object], Dict[str, dict]]:
    """
    Cost-sensitive (no oversampling) leakage-free pipelines.
    Main set (aligned with your manuscript sections):
    - Random Forest (supports class_weight)
    - XGBoost (we use balanced sample_weight)
    - Gradient Boosting (we use balanced sample_weight)

    Optional baseline:
    - NeuralNetwork (sklearn MLPClassifier). NOTE: sklearn MLP does NOT accept class_weight/sample_weight
      in sklearn==1.4.2, so it is included as a *baseline* only (not strictly cost-sensitive).
      If you prefer to omit it entirely (e.g., due to reviewer comments about "deep learning" on small data),
      set INCLUDE_NN_COST = False in main() (below).
    """
    iqr = IQRClipper(factor=1.5)
    scaler = StandardScaler()

    rf = RandomForestClassifier(random_state=cfg.random_state)
    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    gb = GradientBoostingClassifier(random_state=cfg.random_state)

    nn = MLPClassifier(
        random_state=cfg.random_state,
        early_stopping=True,
        max_iter=2000,
    )

    # leakage-free pipelines (no ROS here)
    pipelines = {
        "RandomForest": SkPipeline([("iqr", iqr), ("scaler", scaler), ("clf", rf)]),
        "XGBoost": SkPipeline([("iqr", iqr), ("scaler", scaler), ("clf", xgb)]),
        "GradientBoosting": SkPipeline([("iqr", iqr), ("scaler", scaler), ("clf", gb)]),
        "NeuralNetwork": SkPipeline([("iqr", iqr), ("scaler", scaler), ("clf", nn)]),
    }

    # Grid sizes
    if cfg.fast_mode:
        grids = {
            "RandomForest": {"clf__n_estimators": [200], "clf__max_depth": [None, 10], "clf__class_weight": ["balanced"]},
            "XGBoost": {"clf__n_estimators": [300], "clf__learning_rate": [0.1], "clf__max_depth": [4]},
            "GradientBoosting": {"clf__n_estimators": [300], "clf__learning_rate": [0.1], "clf__max_depth": [3]},
            "NeuralNetwork": {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [1e-4], "clf__learning_rate_init": [1e-3]},
        }
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
            "NeuralNetwork": {
                "clf__hidden_layer_sizes": [(32, 16), (64, 32)],
                "clf__alpha": [1e-4, 1e-3],
                "clf__learning_rate_init": [1e-3, 1e-2],
            },
        }

    return pipelines, grids


def build_ros_models(cfg: Config) -> Tuple[Dict[str, object], Dict[str, dict]]:
    """
    ROS + IQR (leakage-free) pipelines:
    IQRClipper (train-only) -> StandardScaler -> ROS (train-fold only) -> model

    Models:
    - RandomForest
    - XGBoost
    - GradientBoosting
    - NeuralNetwork (MLPClassifier)
    """
    iqr = IQRClipper(factor=1.5)
    scaler = StandardScaler()
    ros = RandomOverSampler(random_state=cfg.random_state)

    rf = RandomForestClassifier(random_state=cfg.random_state)
    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    gb = GradientBoostingClassifier(random_state=cfg.random_state)

    nn = MLPClassifier(
        random_state=cfg.random_state,
        early_stopping=True,
        max_iter=2000,
    )

    pipelines = {
        "RandomForest": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", rf)]),
        "XGBoost": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", xgb)]),
        "GradientBoosting": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", gb)]),
        "NeuralNetwork": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", nn)]),
    }

    if cfg.fast_mode:
        grids = {
            "RandomForest": {"clf__n_estimators": [200], "clf__max_depth": [None, 10]},
            "XGBoost": {"clf__n_estimators": [300], "clf__learning_rate": [0.1], "clf__max_depth": [4]},
            "GradientBoosting": {"clf__n_estimators": [300], "clf__learning_rate": [0.1], "clf__max_depth": [3]},
            "NeuralNetwork": {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [1e-4], "clf__learning_rate_init": [1e-3]},
        }
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
            "NeuralNetwork": {
                "clf__hidden_layer_sizes": [(32, 16), (64, 32)],
                "clf__alpha": [1e-4, 1e-3],
                "clf__learning_rate_init": [1e-3, 1e-2],
            },
        }

    return pipelines, grids


def grid_search_models(
    pipelines: Dict[str, object],
    grids: Dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Config,
    scoring: str = "recall_macro",
    use_balanced_sample_weight: bool = False,
) -> Tuple[Dict[str, object], Dict[str, dict], pd.DataFrame]:
    """
    Runs GridSearchCV for each pipeline.
    Optionally passes balanced sample_weight (computed on TRAIN) to estimators that accept it.
    """
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    best_models = {}
    best_params = {}
    summary_rows = []

    # only needed when cost-sensitive without ROS
    sample_weight = None
    if use_balanced_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, pipe in pipelines.items():
        print(f"\n[GridSearch] {name} ...")
        grid = grids[name]
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
        )

        fit_params = {}
        # Pass sample_weight where possible (sklearn Pipeline forwards to final estimator with clf__sample_weight)
        if use_balanced_sample_weight and name in {"XGBoost", "GradientBoosting"}:
            fit_params = {"clf__sample_weight": sample_weight}

        search.fit(X_train, y_train, **fit_params)
        best_models[name] = search.best_estimator_
        best_params[name] = search.best_params_

        summary_rows.append({
            "model": name,
            "best_score_cv_recall_macro": float(search.best_score_),
            "n_candidates": int(len(search.cv_results_["params"])),
        })
        print("  Best params:", search.best_params_)
        print("  Best CV score (recall_macro):", search.best_score_)

    summary_df = pd.DataFrame(summary_rows).set_index("model")
    return best_models, best_params, summary_df


# -----------------------------
# Figure builders
# -----------------------------
def make_roc_figure(
    y_test: np.ndarray,
    probas: Dict[str, np.ndarray],
    classes: np.ndarray,
    suptitle: str,
    outdir: str,
    basename: str,
    panel_titles: Dict[str, str] | None = None,
):
    """
    2x3 layout:
    - panels 1..4: per-model multi-class ROC (up to 4 models)
    - panel 5: macro-average ROC (one line per model)
    - panel 6: left blank
    """
    panel_titles = panel_titles or {k: k for k in probas.keys()}

    y_test_bin = label_binarize(y_test, classes=classes)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(suptitle, fontsize=22, fontweight="bold")

    # 1..4 individual model ROC (fill missing slots with blank panels)
    model_names = list(probas.keys())
    for slot in range(1, 5):
        ax = fig.add_subplot(2, 3, slot)
        if slot <= len(model_names):
            m = model_names[slot - 1]
            plot_multiclass_roc(ax, y_test_bin, probas[m], classes, panel_titles.get(m, m))
        else:
            ax.axis("off")

    # 5 macro-average panel
    ax5 = fig.add_subplot(2, 3, 5)
    for m in model_names:
        mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, probas[m])
        ax5.plot(mean_fpr, mean_tpr, linewidth=2.5, label=f"{m} (AUC={macro_auc:.2f})")
    ax5.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)
    ax5.set_title("Macro-Average ROC Curve", fontsize=16, fontweight="bold")
    ax5.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax5.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax5.legend(loc="lower right", fontsize=11, title="Models", title_fontsize=12, frameon=True)

    # 6 empty
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    save_figure(fig, outdir, basename)


def evaluate_models_on_train_test(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outdir: str,
    tag: str,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fits each model on TRAIN, evaluates on TRAIN and TEST, returns:
    - probas on TEST
    - metrics_train_df
    - metrics_test_df
    - aucs_test_df
    """
    classes = np.sort(np.unique(y_train))
    probas_test = {}
    train_rows = []
    test_rows = []
    auc_rows = []

    for name, model in models.items():
        print(f"\n[Fit/Eval] {name}")
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        probas = model.predict_proba(X_test)

        probas_test[name] = probas

        m_train = compute_metrics(y_train.to_numpy(), y_pred_train)
        m_train["model"] = name
        train_rows.append(m_train)

        m_test = compute_metrics(y_test.to_numpy(), y_pred_test)
        m_test["model"] = name
        test_rows.append(m_test)

        aucs = auc_table(y_test.to_numpy(), probas, classes)
        aucs["model"] = name
        auc_rows.append(aucs)

        # Optional: save confusion matrices as CSV (easy to paste into response letter)
        cm_train = confusion_matrix(y_train, y_pred_train, labels=classes)
        cm_test = confusion_matrix(y_test, y_pred_test, labels=classes)
        pd.DataFrame(cm_train, index=classes, columns=classes).to_csv(os.path.join(outdir, f"CM_train_{tag}_{name}.csv"))
        pd.DataFrame(cm_test, index=classes, columns=classes).to_csv(os.path.join(outdir, f"CM_test_{tag}_{name}.csv"))

        print("[TEST] classification report:")
        print(classification_report(y_test, y_pred_test, zero_division=0))

    metrics_train_df = pd.DataFrame(train_rows).set_index("model")
    metrics_test_df = pd.DataFrame(test_rows).set_index("model")
    aucs_test_df = pd.DataFrame(auc_rows).set_index("model")

    save_df(metrics_train_df, outdir, f"metrics_train_{tag}.csv")
    save_df(metrics_test_df, outdir, f"metrics_test_{tag}.csv")
    save_df(aucs_test_df, outdir, f"aucs_test_{tag}.csv")

    return probas_test, metrics_train_df, metrics_test_df, aucs_test_df


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adapvtest.csv", help="Path to adapvtest.csv")
    parser.add_argument("--sep", default=";", help="CSV separator (default ';')")
    parser.add_argument("--outdir", default="outputs_paper_figures_leakage_free", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (smaller grids)")
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data,
        sep=args.sep,
        outdir=args.outdir,
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        fast_mode=args.fast,
    )
    ensure_outdir(cfg.outdir)

    # Optional toggles
    INCLUDE_NN_COST = True   # set False to drop NN panel from Fig8
    INCLUDE_NN_ROS  = True   # set False to drop NN panel from Fig9

    # Load data
    df = pd.read_csv(cfg.data_path, sep=cfg.sep, engine="python")
    X = df[cfg.features].copy()
    y = df[cfg.target].astype(int).copy()

    print("Loaded dataset:", df.shape)
    print("Class counts:\n", y.value_counts().sort_index())

    # Split FIRST (core leakage-free fix)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    audit = leakage_audit_split(X_train, X_test, y_train, y_test)
    print("\n[Leakage audit after split-first]")
    print("  Index overlap:", audit["index_overlap"])
    print("  Identical row hash overlap (if >0, dataset may contain exact duplicates):", audit["identical_row_hash_overlap"])

    save_json(audit, cfg.outdir, "leakage_audit_split_first.json")

    classes = np.sort(y.unique())

    # ------------------------------------------------------------
    # FIG 8: Cost sensitivity (no oversampling)
    # ------------------------------------------------------------
    cost_pipes, cost_grids = build_cost_sensitive_models(cfg)
    if not INCLUDE_NN_COST:
        cost_pipes.pop('NeuralNetwork', None)
        cost_grids.pop('NeuralNetwork', None)
    cost_models, cost_best_params, cost_cv_summary = grid_search_models(
        cost_pipes,
        cost_grids,
        X_train,
        y_train,
        cfg,
        scoring="recall_macro",
        use_balanced_sample_weight=True,  # applied to XGB and GB
    )
    save_json(cost_best_params, cfg.outdir, "best_params_Fig8_cost_sensitive.json")
    save_df(cost_cv_summary, cfg.outdir, "cv_summary_Fig8_cost_sensitive.csv")

    probas_cost, _, _, aucs_cost = evaluate_models_on_train_test(
        cost_models, X_train, y_train, X_test, y_test, cfg.outdir, tag="Fig8_cost_sensitive"
    )

    # To keep the 2x3 layout with 4 panels, add a placeholder panel if only 3 models
    # (we keep model list order consistent for plotting)
    probas_cost_for_plot = dict(probas_cost)
    panel_titles_cost = {
        "RandomForest": "RandomForest ROC",
        "XGBoost": "XGBoost ROC",
        "GradientBoosting": "GradientBoosting ROC",
        "NeuralNetwork": "Neural Network ROC",
    }

    make_roc_figure(
        y_test=y_test.to_numpy(),
        probas=probas_cost_for_plot,
        classes=classes,
        suptitle="ROC Curves for Cost Sensitivity",
        outdir=cfg.outdir,
        basename="Fig8_cost_sensitive_ROC",
        panel_titles=panel_titles_cost,
    )

    # ------------------------------------------------------------
    # FIG 9: ROS + IQR (leakage-free)
    # ------------------------------------------------------------
    ros_pipes, ros_grids = build_ros_models(cfg)
    if not INCLUDE_NN_ROS:
        ros_pipes.pop('NeuralNetwork', None)
        ros_grids.pop('NeuralNetwork', None)
    ros_models, ros_best_params, ros_cv_summary = grid_search_models(
        ros_pipes,
        ros_grids,
        X_train,
        y_train,
        cfg,
        scoring="recall_macro",
        use_balanced_sample_weight=False,
    )
    save_json(ros_best_params, cfg.outdir, "best_params_Fig9_ROS_IQR.json")
    save_df(ros_cv_summary, cfg.outdir, "cv_summary_Fig9_ROS_IQR.csv")

    probas_ros, _, _, aucs_ros = evaluate_models_on_train_test(
        ros_models, X_train, y_train, X_test, y_test, cfg.outdir, tag="Fig9_ROS_IQR"
    )

    panel_titles_ros = {
        "RandomForest": "RandomForest ROC (ROS + IQR)",
        "XGBoost": "XGBoost ROC (ROS + IQR)",
        "GradientBoosting": "GradientBoosting ROC (ROS + IQR)",
        "NeuralNetwork": "Neural Network ROC (ROS + IQR)",
    }

    make_roc_figure(
        y_test=y_test.to_numpy(),
        probas=dict(probas_ros),
        classes=classes,
        suptitle="ROC Curves for ROS + IQR",
        outdir=cfg.outdir,
        basename="Fig9_ROS_IQR_ROC",
        panel_titles=panel_titles_ros,
    )

    # ------------------------------------------------------------
    # FIG 10: ROS + IQR + Ensemble (leakage-free)
    # ------------------------------------------------------------
    # Build ensemble from the tuned ROS models (each is already a pipeline)
    ensemble = VotingClassifier(
        estimators=[
            ("rf", ros_models["RandomForest"]),
            ("xgb", ros_models["XGBoost"]),
            ("gb", ros_models["GradientBoosting"]),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    proba_ens = ensemble.predict_proba(X_test)

    probas_fig10 = {
        "RandomForest": probas_ros["RandomForest"],
        "XGBoost": probas_ros["XGBoost"],
        "GradientBoosting": probas_ros["GradientBoosting"],
        "Ensemble": proba_ens,
    }

    # Save ensemble AUCs/metrics (test)
    ens_auc = auc_table(y_test.to_numpy(), proba_ens, classes)
    save_json(ens_auc, cfg.outdir, "aucs_test_Ensemble_Fig10.json")

    panel_titles_fig10 = {
        "RandomForest": "RandomForest ROC",
        "XGBoost": "XGBoost ROC",
        "GradientBoosting": "GradientBoosting ROC",
        "Ensemble": "Ensemble ROC",
    }

    make_roc_figure(
        y_test=y_test.to_numpy(),
        probas=probas_fig10,
        classes=classes,
        suptitle="ROC Curves for ROS + IQR with Ensemble",
        outdir=cfg.outdir,
        basename="Fig10_ROS_IQR_Ensemble_ROC",
        panel_titles=panel_titles_fig10,
    )

    # Save an AUC summary table for quick copy into manuscript
    auc_summary = pd.concat(
        [
            aucs_cost.rename(columns=lambda c: f"Fig8_{c}"),
            aucs_ros.rename(columns=lambda c: f"Fig9_{c}"),
        ],
        axis=1,
    )
    save_df(auc_summary, cfg.outdir, "AUC_summary_Fig8_Fig9.csv")

    print("\nDONE. Outputs saved to:", cfg.outdir)


if __name__ == "__main__":
    main()
