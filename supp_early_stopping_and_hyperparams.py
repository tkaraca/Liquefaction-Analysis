
"""
supp_early_stopping_and_hyperparams_fixed2.py

Leakage-free hyperparameter tuning + early stopping (boosting) for a 4-class, imbalanced dataset.

WHY THIS "fixed2" VERSION?
--------------------------
Some xgboost installations expose neither `early_stopping_rounds` nor `callbacks` in the
sklearn wrapper (`XGBClassifier.fit`). If you see errors like:

  TypeError: XGBClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'
  TypeError: XGBClassifier.fit() got an unexpected keyword argument 'callbacks'

this script avoids the sklearn wrapper for early stopping and uses the native XGBoost API
(xgboost.train), which supports early stopping across versions.

Leakage-free protocol (matches the revision narrative)
------------------------------------------------------
1) Split FIRST into train/test (stratified).
2) All preprocessing is applied ONLY on training data:
   - IQR clipping is fit on fold-train only.
   - RandomOverSampler (ROS) is applied on fold-train only.
3) Model selection / tuning is performed on TRAIN only (Stratified 5-fold CV by default).
4) Early stopping is demonstrated for XGBoost and for sklearn GradientBoostingClassifier
   (internal early stopping via n_iter_no_change).

Outputs
-------
- outputs_supp_early_stop/supp_hyperparams_and_results_TEST.(csv|xlsx)
- outputs_supp_early_stop/supp_xgb_early_stopping_curve.(png|pdf)

Usage
-----
python supp_early_stopping_and_hyperparams_fixed2.py --data adapvtest.csv --sep ";" --outdir outputs_supp_early_stop

Tips
----
- If you want faster runs: add --fast
- If you still want to use the sklearn wrapper, upgrade xgboost, but this script should work without upgrades.

Author intent
-------------
This file is designed as "Supplementary" support material to address the reviewer comment about
early stopping / boosting sensitivity and to report tuned hyperparameters transparently.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# XGBoost native API (xgboost.train) is used for early stopping
try:
    import xgboost as xgb  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# -----------------------------
# Leakage-free IQR clipper
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
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
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, outdir: str, base: str) -> None:
    csv_path = os.path.join(outdir, base + ".csv")
    xlsx_path = os.path.join(outdir, base + ".xlsx")
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, labels: List[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["Macro Precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["Macro Recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["Macro F1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["Weighted Precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    out["Weighted Recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    out["Weighted F1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # class recalls (edit if class mapping differs)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = dict(zip(labels, per_class[1]))
    out["Recall Class 2"] = float(recalls.get(2, np.nan))
    out["Recall Class 3"] = float(recalls.get(3, np.nan))

    try:
        out["Macro AUC (OvR)"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        out["Macro AUC (OvR)"] = float("nan")
    return out


# -----------------------------
# XGBoost native early stopping (version-robust)
# -----------------------------
def _xgb_train_with_es(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    num_class: int,
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
) -> Tuple["xgb.Booster", int, Dict[str, Dict[str, List[float]]]]:
    """
    Train using xgboost.train with early stopping; returns booster, best_ntree (>=1), evals_result.
    """
    import xgboost as xgb  # local import

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    full_params = dict(params)
    full_params.update({
        "objective": "multi:softprob",
        "num_class": int(num_class),
        "seed": int(seed),
        "eval_metric": params.get("eval_metric", "mlogloss"),
    })

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params=full_params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dval, "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        evals_result=evals_result,
        verbose_eval=False,
    )

    # best_iteration is 0-based
    best_ntree = int(getattr(booster, "best_iteration", 0)) + 1
    return booster, best_ntree, evals_result


def xgb_cv_with_early_stopping_native(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    *,
    params: Dict[str, Any],
    n_estimators_max: int,
    early_stopping_rounds: int,
    random_state: int,
) -> Tuple[float, int]:
    """
    Leakage-free CV:
      - IQR fit on fold-train only
      - ROS on fold-train only
      - early stopping on fold validation
    Returns:
      mean_macro_recall, mean_best_ntree
    """
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed. Install xgboost or skip XGB in the supplementary.")

    labels = sorted(np.unique(y_train))
    num_class = len(labels)

    fold_scores: List[float] = []
    best_ntrees: List[int] = []

    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        # leakage-free preprocess fit on fold-train only
        iqr = IQRClipper(1.5).fit(X_tr.values, y_tr.values)
        X_tr_c = iqr.transform(X_tr.values)
        X_va_c = iqr.transform(X_va.values)

        # ROS on fold-train only
        ros = RandomOverSampler(random_state=random_state)
        X_tr_ros, y_tr_ros = ros.fit_resample(X_tr_c, y_tr.values)

        booster, best_ntree, _ = _xgb_train_with_es(
            X_tr_ros, y_tr_ros,
            X_va_c, y_va.values,
            num_class=num_class,
            params=params,
            num_boost_round=n_estimators_max,
            early_stopping_rounds=early_stopping_rounds,
            seed=random_state,
        )
        best_ntrees.append(best_ntree)

        import xgboost as xgb
        dval = xgb.DMatrix(X_va_c)
        proba_va = booster.predict(dval, iteration_range=(0, best_ntree))
        pred_va = np.argmax(proba_va, axis=1)

        score = float(recall_score(y_va.values, pred_va, average="macro", zero_division=0))
        fold_scores.append(score)

    return float(np.mean(fold_scores)), int(np.round(np.mean(best_ntrees)))


def fit_final_xgb_with_early_stopping_native(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    n_estimators_max: int,
    early_stopping_rounds: int,
    random_state: int,
    outdir: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Final XGB:
      - split TRAIN into train_sub/val_sub (within TRAIN only)
      - early stopping on val_sub
      - refit on full TRAIN for best_ntree
      - evaluate on TEST
    Returns:
      proba_test, pred_test, best_ntree
    """
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed.")

    labels = sorted(np.unique(y_train))
    num_class = len(labels)

    # inner split within TRAIN for early stopping (strictly inside TRAIN)
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=random_state
    )

    # fit IQR on train_sub only
    iqr_sub = IQRClipper(1.5).fit(X_tr_sub.values, y_tr_sub.values)
    X_tr_sub_c = iqr_sub.transform(X_tr_sub.values)
    X_val_sub_c = iqr_sub.transform(X_val_sub.values)

    ros = RandomOverSampler(random_state=random_state)
    X_tr_sub_ros, y_tr_sub_ros = ros.fit_resample(X_tr_sub_c, y_tr_sub.values)

    booster_tmp, best_ntree, evals_result = _xgb_train_with_es(
        X_tr_sub_ros, y_tr_sub_ros,
        X_val_sub_c, y_val_sub.values,
        num_class=num_class,
        params=best_params,
        num_boost_round=n_estimators_max,
        early_stopping_rounds=early_stopping_rounds,
        seed=random_state,
    )

    # plot early-stopping curve (validation metric)
    try:
        # eval_metric key can differ; take the first available
        if evals_result and "val" in evals_result:
            metric_name = list(evals_result["val"].keys())[0]
            curve = evals_result["val"][metric_name]
            plt.figure(figsize=(9, 6))
            plt.plot(curve, label=f"val {metric_name}")
            plt.axvline(best_ntree, linestyle="--", linewidth=1.0, label="best_ntree")
            plt.xlabel("Boosting round")
            plt.ylabel(metric_name)
            plt.title("XGBoost early stopping curve (TRAIN split)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "supp_xgb_early_stopping_curve.png"), dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(outdir, "supp_xgb_early_stopping_curve.pdf"), bbox_inches="tight")
            plt.close()
    except Exception:
        pass

    # Refit on full TRAIN with best_ntree (no early stopping)
    iqr_full = IQRClipper(1.5).fit(X_train.values, y_train.values)
    X_train_c = iqr_full.transform(X_train.values)
    X_test_c = iqr_full.transform(X_test.values)

    X_train_ros, y_train_ros = ros.fit_resample(X_train_c, y_train.values)

    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train_ros, label=y_train_ros)
    dtest = xgb.DMatrix(X_test_c)

    final_params = dict(best_params)
    final_params.update({
        "objective": "multi:softprob",
        "num_class": int(num_class),
        "seed": int(random_state),
        "eval_metric": best_params.get("eval_metric", "mlogloss"),
    })

    booster_final = xgb.train(
        params=final_params,
        dtrain=dtrain,
        num_boost_round=int(best_ntree),
        evals=[],
        verbose_eval=False,
    )

    proba_test = booster_final.predict(dtest)
    pred_test = np.argmax(proba_test, axis=1)
    return proba_test, pred_test, best_ntree


# -----------------------------
# Main
# -----------------------------
@dataclass
class Config:
    data: str = "adapvtest.csv"
    sep: str = ";"
    outdir: str = "outputs_supp_early_stop"
    test_size: float = 0.20
    random_state: int = 42
    n_splits: int = 5
    scoring: str = "recall_macro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="adapvtest.csv")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--outdir", default="outputs_supp_early_stop")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        data=args.data,
        sep=args.sep,
        outdir=args.outdir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_splits=args.n_splits,
    )
    ensure_dir(cfg.outdir)

    FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
    TARGET = "dver"

    df = pd.read_csv(cfg.data, sep=cfg.sep, engine="python")

    # case-insensitive rescue
    lower_map = {c.lower(): c for c in df.columns}
    rename_dict = {}
    for c in FEATURES + [TARGET]:
        if c not in df.columns and c.lower() in lower_map:
            rename_dict[lower_map[c.lower()]] = c
    if rename_dict:
        df = df.rename(columns=rename_dict)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    # Split FIRST (leakage-free)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    labels = sorted(np.unique(y))
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    rows: List[Dict[str, Any]] = []

    # -----------------------------
    # Random Forest (leakage-free pipeline)
    # -----------------------------
    rf_pipe = Pipeline(
        steps=[
            ("iqr", IQRClipper(1.5)),
            ("ros", RandomOverSampler(random_state=cfg.random_state)),
            ("model", RandomForestClassifier(random_state=cfg.random_state, n_jobs=-1)),
        ]
    )

    rf_grid = {
        "model__n_estimators": [200, 500] if not args.fast else [200],
        "model__max_depth": [None, 10, 20] if not args.fast else [None, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    rf_search = GridSearchCV(rf_pipe, rf_grid, scoring=cfg.scoring, cv=cv, n_jobs=-1, refit=True)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_

    rf_proba = rf_best.predict_proba(X_test)
    rf_pred = rf_proba.argmax(axis=1)
    rf_metrics = compute_metrics(y_test.values, rf_pred, rf_proba, labels)

    rows.append({
        "Model": "RandomForest",
        "CV score (macro recall)": float(rf_search.best_score_),
        "Best hyperparameters (TRAIN-CV)": json.dumps(rf_search.best_params_),
        "Early stopping": "N/A",
        **rf_metrics,
    })

    # -----------------------------
    # Gradient Boosting (internal early stopping)
    # -----------------------------
    gb_base = GradientBoostingClassifier(
        random_state=cfg.random_state,
        validation_fraction=0.10,
        n_iter_no_change=20,
        tol=1e-4,
    )

    gb_pipe = Pipeline(
        steps=[
            ("iqr", IQRClipper(1.5)),
            ("ros", RandomOverSampler(random_state=cfg.random_state)),
            ("model", gb_base),
        ]
    )

    gb_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1] if not args.fast else [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0],
        "model__n_estimators": [800, 1500] if not args.fast else [800],
    }

    gb_search = GridSearchCV(gb_pipe, gb_grid, scoring=cfg.scoring, cv=cv, n_jobs=-1, refit=True)
    gb_search.fit(X_train, y_train)
    gb_best = gb_search.best_estimator_

    gb_model = gb_best.named_steps["model"]
    gb_used = getattr(gb_model, "n_estimators_", getattr(gb_model, "n_estimators", None))

    gb_proba = gb_best.predict_proba(X_test)
    gb_pred = gb_proba.argmax(axis=1)
    gb_metrics = compute_metrics(y_test.values, gb_pred, gb_proba, labels)

    rows.append({
        "Model": "GradientBoosting (sklearn)",
        "CV score (macro recall)": float(gb_search.best_score_),
        "Best hyperparameters (TRAIN-CV)": json.dumps(gb_search.best_params_),
        "Early stopping": f"internal (used n_estimators_={gb_used})",
        **gb_metrics,
    })

    # -----------------------------
    # XGBoost (native early stopping)  [optional]
    # -----------------------------
    if HAS_XGB:
        # Hyperparameter grid (no n_estimators here; early stopping picks effective size)
        if args.fast:
            xgb_grid = [
                {"max_depth": 3, "eta": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
                {"max_depth": 5, "eta": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
            ]
        else:
            xgb_grid = []
            for max_depth in [3, 5]:
                for eta in [0.03, 0.05, 0.10]:
                    for subsample in [0.8, 1.0]:
                        for colsample in [0.8, 1.0]:
                            xgb_grid.append({
                                "max_depth": max_depth,
                                "eta": eta,  # native API uses 'eta' for learning_rate
                                "subsample": subsample,
                                "colsample_bytree": colsample,
                            })

        n_estimators_max = 5000
        early_rounds = 50

        best_score = -1.0
        best_params = None
        best_mean_ntree = None

        for cand in xgb_grid:
            # add safe defaults
            cand_params = dict(cand)
            cand_params.setdefault("min_child_weight", 1.0)
            cand_params.setdefault("lambda", 1.0)  # L2
            cand_params.setdefault("alpha", 0.0)   # L1
            cand_params.setdefault("eval_metric", "mlogloss")
            cand_params.setdefault("tree_method", "hist")

            score, mean_ntree = xgb_cv_with_early_stopping_native(
                X_train, y_train, cv,
                params=cand_params,
                n_estimators_max=n_estimators_max,
                early_stopping_rounds=early_rounds,
                random_state=cfg.random_state,
            )
            if score > best_score:
                best_score = score
                best_params = cand_params
                best_mean_ntree = mean_ntree

        # Fit final and evaluate on TEST
        proba_test, pred_test, best_ntree = fit_final_xgb_with_early_stopping_native(
            X_train, y_train, X_test,
            best_params=best_params,
            n_estimators_max=n_estimators_max,
            early_stopping_rounds=early_rounds,
            random_state=cfg.random_state,
            outdir=cfg.outdir,
        )

        xgb_metrics = compute_metrics(y_test.values, pred_test, proba_test, labels)

        rows.append({
            "Model": "XGBoost (native train)",
            "CV score (macro recall)": float(best_score),
            "Best hyperparameters (TRAIN-CV)": json.dumps(best_params),
            "Early stopping": f"native xgb.train (mean_ntree~{best_mean_ntree}, final_best_ntree={best_ntree})",
            **xgb_metrics,
        })

    # -----------------------------
    # Save Supplementary table
    # -----------------------------
    out = pd.DataFrame(rows)

    # nicer ordering
    preferred_cols = [
        "Model",
        "Early stopping",
        "CV score (macro recall)",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro AUC (OvR)",
        "Recall Class 2",
        "Recall Class 3",
        "Best hyperparameters (TRAIN-CV)",
    ]
    cols = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    out = out[cols]

    save_df(out, cfg.outdir, "supp_hyperparams_and_results_TEST")

    print("\n[DONE]")
    print("Saved:", os.path.abspath(os.path.join(cfg.outdir, "supp_hyperparams_and_results_TEST.csv")))
    if HAS_XGB:
        print("Saved:", os.path.abspath(os.path.join(cfg.outdir, "supp_xgb_early_stopping_curve.png")))


if __name__ == "__main__":
    main()
