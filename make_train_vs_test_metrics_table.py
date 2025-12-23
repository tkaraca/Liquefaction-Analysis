#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_train_vs_test_metrics_table.py
-----------------------------------
Reviewer #4 (Comment 4) için Train (CV) vs Test metriklerini ayrı raporlar.

Leakage-free prensip:
1) Önce train/test split
2) IQR clipping sadece train fold'larda fit edilir (Pipeline içinde)
3) ROS sadece train fold'larda uygulanır (imblearn Pipeline içinde)
4) Hyperparametre tuning sadece TRAIN üzerinde yapılır (GridSearchCV)
5) Rapor:
   - Train CV (5-fold) mean±std
   - Hold-out test metrikleri

Çıktılar:
- outputs_train_vs_test/metrics_train_cv.csv
- outputs_train_vs_test/metrics_test_holdout.csv
- outputs_train_vs_test/metrics_traincv_vs_test_combined.csv
- outputs_train_vs_test/metrics_traincv_vs_test_combined.xlsx
- Console'a Word'e yapıştırmalık tablo (mean±std formatlı)

Kullanım:
    python make_train_vs_test_metrics_table.py --data adapvtest.csv --sep ";"
Opsiyonel:
    python make_train_vs_test_metrics_table.py --fast
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost gerekir. Lütfen `pip install xgboost` ile kurun.") from e


# -----------------------------
# Config
# -----------------------------
DEFAULT_FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
DEFAULT_TARGET = "dver"


@dataclass
class Config:
    data_path: str = "adapvtest.csv"
    sep: str = ";"
    outdir: str = "outputs_train_vs_test"
    features: list[str] = None
    target: str = DEFAULT_TARGET

    test_size: float = 0.20
    random_state: int = 42
    cv_splits: int = 5
    n_jobs: int = -1
    fast_mode: bool = False


# -----------------------------
# Leakage-free outlier handler
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Her feature'ı [Q1 - factor*IQR, Q3 + factor*IQR] aralığında clip eder.
    NOT: Bu transformer Pipeline içinde olduğu için her CV fold'unda yalnızca TRAIN fold'dan fit olur (leakage-free).
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
# Metrics helpers
# -----------------------------
def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Multiclass AUC (OVR macro)
    # Not: roc_auc_score için y_proba shape = (n_samples, n_classes) olmalı.
    out["auc_macro_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    return out


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fmt_mean_std(mean: float, std: float, ndigits: int = 3) -> str:
    return f"{mean:.{ndigits}f}±{std:.{ndigits}f}"


def build_models_and_grids(cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    ROS + IQR (leakage-free) pipelines:
        IQRClipper -> StandardScaler -> ROS -> Classifier

    NOT:
    - StandardScaler tree modeller için şart değil ama MLP için gerekli.
    - Aynı pipeline ile tüm modelleri raporlamak tabloyu tutarlı yapar.
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
            "RandomForest": {"clf__n_estimators": [300], "clf__max_depth": [None, 10]},
            "XGBoost": {"clf__n_estimators": [400], "clf__learning_rate": [0.1], "clf__max_depth": [4]},
            "GradientBoosting": {"clf__n_estimators": [400], "clf__learning_rate": [0.1], "clf__max_depth": [3]},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adapvtest.csv", help="CSV path (örn. adapvtest.csv)")
    parser.add_argument("--sep", default=";", help="CSV separator (default ';')")
    parser.add_argument("--outdir", default="outputs_train_vs_test", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (küçük grid)")
    parser.add_argument("--test_size", type=float, default=0.20, help="Hold-out test fraction (default 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data,
        sep=args.sep,
        outdir=args.outdir,
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        fast_mode=args.fast,
        test_size=args.test_size,
        random_state=args.seed,
    )
    ensure_outdir(cfg.outdir)

    # Load
    df = pd.read_csv(cfg.data_path, sep=cfg.sep, engine="python")
    missing = [c for c in cfg.features + [cfg.target] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV içinde beklenen kolon(lar) yok: {missing}\nMevcut kolonlar: {df.columns.tolist()}")

    X = df[cfg.features].copy()
    y = df[cfg.target].astype(int).copy()

    # Split first (leakage-free core)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    # CV object
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    # Scoring dict for CV (train)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_precision": make_scorer(precision_score, average="macro", zero_division=0),
        "macro_recall": make_scorer(recall_score, average="macro", zero_division=0),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "weighted_precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "weighted_recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
        "auc_macro_ovr": make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr", average="macro"),
    }

    pipelines, grids = build_models_and_grids(cfg)

    best_estimators: Dict[str, Any] = {}
    best_params: Dict[str, Dict[str, Any]] = {}

    cv_rows = []
    test_rows = []

    # ----------------------------
    # Tune + CV + Test per model
    # ----------------------------
    for name, pipe in pipelines.items():
        print(f"\n[Tuning] {name}")

        search = GridSearchCV(
            estimator=pipe,
            param_grid=grids[name],
            scoring="recall_macro",   # reviewer metrikleri: acc/prec/rec -> tune'u recall_macro yapmak mantıklı
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_estimators[name] = search.best_estimator_
        best_params[name] = search.best_params_

        # Train CV metrics (mean±std on training data)
        cv_out = cross_validate(
            estimator=best_estimators[name],
            X=X_train,
            y=y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            return_train_score=False,
        )

        row_cv = {"model": name}
        for k in scoring.keys():
            vals = cv_out[f"test_{k}"]
            row_cv[f"cv_{k}_mean"] = float(np.mean(vals))
            row_cv[f"cv_{k}_std"] = float(np.std(vals, ddof=1))
        cv_rows.append(row_cv)

        # Hold-out test metrics
        est = best_estimators[name]
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        y_proba = est.predict_proba(X_test)

        row_test = {"model": name}
        row_test.update(compute_test_metrics(y_test.to_numpy(), y_pred, y_proba))
        test_rows.append(row_test)

        print("  Best params:", search.best_params_)

    # ----------------------------
    # Ensemble (RF + XGB + GB)
    # ----------------------------
    # (NN'yi ensemble'a dahil etmiyoruz; çoğu makalede tree tabanlı final akış daha tutarlı.)
    if all(k in best_estimators for k in ["RandomForest", "XGBoost", "GradientBoosting"]):
        print("\n[Ensemble] Soft voting (RF + XGB + GB)")

        ens = VotingClassifier(
            estimators=[
                ("rf", best_estimators["RandomForest"]),
                ("xgb", best_estimators["XGBoost"]),
                ("gb", best_estimators["GradientBoosting"]),
            ],
            voting="soft",
        )

        # CV for ensemble
        cv_out = cross_validate(
            estimator=ens,
            X=X_train,
            y=y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            return_train_score=False,
        )
        row_cv = {"model": "Ensemble(RF+XGB+GB)"}
        for k in scoring.keys():
            vals = cv_out[f"test_{k}"]
            row_cv[f"cv_{k}_mean"] = float(np.mean(vals))
            row_cv[f"cv_{k}_std"] = float(np.std(vals, ddof=1))
        cv_rows.append(row_cv)

        # Test for ensemble
        ens.fit(X_train, y_train)
        y_pred = ens.predict(X_test)
        y_proba = ens.predict_proba(X_test)
        row_test = {"model": "Ensemble(RF+XGB+GB)"}
        row_test.update(compute_test_metrics(y_test.to_numpy(), y_pred, y_proba))
        test_rows.append(row_test)

    # ----------------------------
    # Save tables
    # ----------------------------
    df_cv = pd.DataFrame(cv_rows).set_index("model").sort_index()
    df_test = pd.DataFrame(test_rows).set_index("model").sort_index()

    df_cv.to_csv(os.path.join(cfg.outdir, "metrics_train_cv.csv"))
    df_test.to_csv(os.path.join(cfg.outdir, "metrics_test_holdout.csv"))

    # Combined (numeric)
    df_comb = df_cv.join(df_test, how="outer")
    df_comb.to_csv(os.path.join(cfg.outdir, "metrics_traincv_vs_test_combined.csv"))

    # Word'e daha kolay yapıştırmak için "mean±std" formatlı bir tablo üretelim
    pretty_cols = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("auc_macro_ovr", "Macro AUC (OVR)"),
        ("weighted_precision", "Weighted Precision"),
        ("weighted_recall", "Weighted Recall"),
        ("weighted_f1", "Weighted F1"),
    ]

    pretty = pd.DataFrame(index=df_comb.index)
    for key, label in pretty_cols:
        m = df_comb.get(f"cv_{key}_mean", np.nan)
        s = df_comb.get(f"cv_{key}_std", np.nan)
        t = df_comb.get(key, np.nan)

        pretty[f"Train CV ({cfg.cv_splits}-fold) {label}"] = [
            fmt_mean_std(mv, sv) if np.isfinite(mv) and np.isfinite(sv) else ""
            for mv, sv in zip(m, s)
        ]
        pretty[f"Hold-out Test {label}"] = [f"{tv:.3f}" if np.isfinite(tv) else "" for tv in t]

    pretty_path_csv = os.path.join(cfg.outdir, "TableS_trainCV_vs_test_pretty.csv")
    pretty.to_csv(pretty_path_csv)

    # Excel
    try:
        xlsx_path = os.path.join(cfg.outdir, "TableS_trainCV_vs_test_pretty.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            df_cv.to_excel(writer, sheet_name="Train_CV_numeric")
            df_test.to_excel(writer, sheet_name="Test_numeric")
            pretty.to_excel(writer, sheet_name="Pretty_for_Word")
    except Exception as e:
        print("[WARN] Excel yazılamadı:", e)

    # Print to console
    print("\n====================")
    print("TRAIN (CV) vs TEST Table (paste-friendly)")
    print("====================\n")
    print(pretty.to_string())

    print("\nSaved outputs to:", cfg.outdir)


if __name__ == "__main__":
    main()
