#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_case_study_predictions_leakage_free.py

Leakage-free case-study prediction script:
- Split (train/test) FIRST
- Hyperparameter tuning ONLY on training via Stratified CV
- All preprocessing (IQR clipping) and oversampling (ROS) are inside Pipeline
  -> fitted ONLY on training folds
- Fit final models on full training data
- Predict external case studies (Case 1–4 (Tables 9–12)) for RF/XGB/GB/Ensemble

Outputs:
- outputs_case_study_predictions/case_study_predictions.csv
- prints hold-out test metrics (optional sanity check)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost is required. Please install xgboost in your environment.") from e


# -----------------------------
# Config
# -----------------------------
FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
TARGET = "dver"


@dataclass
class Config:
    data_path: str
    sep: str
    outdir: str = "outputs_case_study_predictions"
    test_size: float = 0.20
    random_state: int = 42
    cv_splits: int = 5
    n_jobs: int = -1
    fast_mode: bool = False


# -----------------------------
# Leakage-free IQR clipping
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


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
    }


def build_pipelines(cfg: Config):
    iqr = IQRClipper(1.5)
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

    pipes = {
        "RandomForest": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", rf)]),
        "XGBoost": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", xgb)]),
        "GradientBoosting": ImbPipeline([("iqr", iqr), ("scaler", scaler), ("ros", ros), ("clf", gb)]),
    }

    if cfg.fast_mode:
        grids = {
            "RandomForest": {"clf__n_estimators": [300], "clf__max_depth": [None, 10]},
            "XGBoost": {"clf__n_estimators": [400], "clf__learning_rate": [0.1], "clf__max_depth": [4],
                        "clf__subsample": [0.8], "clf__colsample_bytree": [0.8]},
            "GradientBoosting": {"clf__n_estimators": [400], "clf__learning_rate": [0.1], "clf__max_depth": [3],
                                 "clf__subsample": [0.8]},
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
            },
        }

    return pipes, grids


def tune_models(pipes, grids, X_train, y_train, cfg: Config):
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
    best = {}

    for name, pipe in pipes.items():
        print(f"\n[Tuning] {name}")
        gs = GridSearchCV(
            pipe,
            param_grid=grids[name],
            scoring="recall_macro",
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
        )
        gs.fit(X_train, y_train)
        print("  best_params:", gs.best_params_)
        print("  best_cv_recall_macro:", gs.best_score_)
        best[name] = gs.best_estimator_

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to adapvtest.csv")
    ap.add_argument("--sep", default=";", help="CSV separator (default ';')")
    ap.add_argument("--fast", action="store_true", help="Fast mode (smaller grids)")
    args = ap.parse_args()

    cfg = Config(data_path=args.data, sep=args.sep, fast_mode=args.fast)
    ensure_outdir(cfg.outdir)

    df = pd.read_csv(cfg.data_path, sep=cfg.sep, engine="python")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    # Tune ONLY on training
    pipes, grids = build_pipelines(cfg)
    best_models = tune_models(pipes, grids, X_train, y_train, cfg)

    # Fit best models on full training (already fitted by GridSearchCV, but refit here is fine)
    for m in best_models.values():
        m.fit(X_train, y_train)

    # Ensemble (soft voting) on tuned pipelines
    ensemble = VotingClassifier(
        estimators=[
            ("rf", best_models["RandomForest"]),
            ("xgb", best_models["XGBoost"]),
            ("gb", best_models["GradientBoosting"]),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    # Optional sanity check on hold-out test
    print("\n[Hold-out test metrics]")
    for name, model in {**best_models, "Ensemble": ensemble}.items():
        yhat = model.predict(X_test)
        m = compute_metrics(y_test.to_numpy(), yhat)
        print(f"{name}: {m}")
        # Uncomment if you want detailed report:
        # print(classification_report(y_test, yhat, zero_division=0))

    # External Case Studies (Tables 9–12) — update here if values changed in manuscript
    cases: List[Dict] = [
        {"Case": "Case-1 (Treasure Island, Loma Prieta 1989)", "pga": 0.13,  "H": 18.0, "B": 71.0, "q": 20.0, "depth": 1.5, "thickness": 5.0, "Observed_GFI": 1},
        {"Case": "Case-2 (Adapazari, Kocaeli 1999)",          "pga": 0.37,  "H": 12.0, "B": 10.0, "q": 60.0, "depth": 3.3, "thickness": 6.0, "Observed_GFI": 2},
        {"Case": "Case-3 (Kumamoto 2016)",                   "pga": 0.12,  "H": 6.0,  "B": 10.0, "q": 15.0, "depth": 2.0, "thickness": 4.0, "Observed_GFI": 1},
        {"Case": "Case-4 (Golbasi, Kahramanmaras 2023)",     "pga": 0.375, "H": 18.0, "B": 11.0, "q": 90.0, "depth": 1.2, "thickness": 5.0, "Observed_GFI": 3},
    ]
    X_cases = pd.DataFrame(cases)[["pga", "H", "B", "q", "depth", "thickness"]]

    rows = []
    model_dict = {**best_models, "Ensemble": ensemble}
    for i, row_case in pd.DataFrame(cases).iterrows():
        x1 = X_cases.iloc[[i]]
        for name, model in model_dict.items():
            proba = model.predict_proba(x1)[0]
            pred = int(np.argmax(proba))
            rows.append({
                "Case": row_case["Case"],
                "Observed_GFI": int(row_case["Observed_GFI"]),
                "Model": name,
                "Predicted_GFI": pred,
                "P(class0)": float(proba[0]),
                "P(class1)": float(proba[1]),
                "P(class2)": float(proba[2]),
                "P(class3)": float(proba[3]),
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(cfg.outdir, "case_study_predictions.csv")
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()