# table6_leakage_free_6f.py
# Leakage-free Table 6 generator (6 features, no threshold tuning; argmax decision rule)

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, roc_auc_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# XGBoost (pip install xgboost)
from xgboost import XGBClassifier


# -----------------------------
# Leakage-free IQR clipper
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(float)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).to_numpy()
        self.upper_ = (q3 + self.factor * iqr).to_numpy()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(float).to_numpy()
        return np.clip(X, self.lower_, self.upper_)


def compute_metrics(y_true, y_pred, y_prob=None):
    out = {}
    out["Accuracy"] = accuracy_score(y_true, y_pred)

    out["Macro Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    out["Macro Recall"]    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    out["Macro F1"]        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    out["Weighted Precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    out["Weighted Recall"]    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    out["Weighted F1"]        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Optional: macro-AUC (OVR) if probabilities provided
    if y_prob is not None:
        try:
            out["Macro AUC (OVR)"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            out["Macro AUC (OVR)"] = np.nan
    else:
        out["Macro AUC (OVR)"] = np.nan

    return out


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    DATA_PATH = "adapvtest.csv"
    SEP = ";"
    OUTDIR = "outputs_table6_leakage_free"
    RANDOM_STATE = 42
    TEST_SIZE = 0.20

    FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]  # depth=Dliq, thickness=Hliq
    TARGET = "dver"

    ensure_dir(OUTDIR)

    df = pd.read_csv(DATA_PATH, sep=SEP, engine="python")

    # sanity
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing} | Available: {list(df.columns)}")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    classes = sorted(np.unique(y))

    # -----------------------------
    # Split FIRST (critical)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("Index overlap (must be 0):", len(set(X_train.index) & set(X_test.index)))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # -----------------------------
    # Model grids (edit as needed)
    # -----------------------------
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.10],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_grid = {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.10],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0],
    }

    grids = {
        "RandomForest": (rf, rf_grid),
        "XGBoost": (xgb, xgb_grid),
        "GradientBoosting": (gb, gb_grid),
    }

    best_models = {}
    table_rows = []

    # -----------------------------
    # Tune each model on TRAIN only
    # Leakage-free pipeline: IQR -> ROS -> model
    # -----------------------------
    for name, (model, grid) in grids.items():
        pipe = Pipeline(steps=[
            ("iqr", IQRClipper(factor=1.5)),
            ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", model),
        ])

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="recall_macro",   # consistent with your paper emphasis
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        best_params = gs.best_params_
        best_models[name] = (best, best_params)

        # Evaluate on untouched TEST (argmax)
        prob = best.predict_proba(X_test)
        pred = prob.argmax(axis=1)

        metrics = compute_metrics(y_test, pred, y_prob=prob)

        # add class-specific recalls (optional but helpful for discussion)
        prfs = precision_recall_fscore_support(
            y_test, pred, labels=classes, average=None, zero_division=0
        )
        recalls = dict(zip(classes, prfs[1]))
        metrics["Recall Class 2"] = recalls.get(2, np.nan)
        metrics["Recall Class 3"] = recalls.get(3, np.nan)

        table_rows.append({
            "Model": name,
            "Best Hyperparameters (CV on TRAIN)": json.dumps(best_params),
            **metrics,
        })

        print(f"\n{name} best params: {best_params}")
        print(f"{name} TEST Accuracy={metrics['Accuracy']:.3f}, MacroRecall={metrics['Macro Recall']:.3f}")

    # -----------------------------
    # Ensemble (soft voting) using tuned hyperparams
    # Leakage-free: IQR -> ROS -> VotingClassifier
    # -----------------------------
    rf_best_params = {k.replace("model__", ""): v for k, v in best_models["RandomForest"][1].items()}
    xgb_best_params = {k.replace("model__", ""): v for k, v in best_models["XGBoost"][1].items()}
    gb_best_params = {k.replace("model__", ""): v for k, v in best_models["GradientBoosting"][1].items()}

    rf_est = RandomForestClassifier(random_state=RANDOM_STATE, **rf_best_params)
    xgb_est = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        **xgb_best_params
    )
    gb_est = GradientBoostingClassifier(random_state=RANDOM_STATE, **gb_best_params)

    voting = VotingClassifier(
        estimators=[("rf", rf_est), ("xgb", xgb_est), ("gb", gb_est)],
        voting="soft"
    )

    ens_pipe = Pipeline(steps=[
        ("iqr", IQRClipper(factor=1.5)),
        ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
        ("model", voting),
    ])

    ens_pipe.fit(X_train, y_train)
    prob = ens_pipe.predict_proba(X_test)
    pred = prob.argmax(axis=1)

    metrics = compute_metrics(y_test, pred, y_prob=prob)
    prfs = precision_recall_fscore_support(
        y_test, pred, labels=classes, average=None, zero_division=0
    )
    recalls = dict(zip(classes, prfs[1]))
    metrics["Recall Class 2"] = recalls.get(2, np.nan)
    metrics["Recall Class 3"] = recalls.get(3, np.nan)

    table_rows.append({
        "Model": "Ensemble (RF+XGB+GB, soft voting)",
        "Best Hyperparameters (CV on TRAIN)": "Uses tuned base-model params (see above)",
        **metrics,
    })

    # -----------------------------
    # Save Table 6
    # -----------------------------
    table6 = pd.DataFrame(table_rows)

    col_order = [
        "Model",
        "Best Hyperparameters (CV on TRAIN)",
        "Accuracy",
        "Macro Precision", "Macro Recall", "Macro F1",
        "Weighted Precision", "Weighted Recall", "Weighted F1",
        "Macro AUC (OVR)",
        "Recall Class 2", "Recall Class 3",
    ]
    table6 = table6[col_order]

    out_csv = os.path.join(OUTDIR, "table6_leakage_free_6f_TEST.csv")
    out_xlsx = os.path.join(OUTDIR, "table6_leakage_free_6f_TEST.xlsx")
    table6.to_csv(out_csv, index=False)
    table6.to_excel(out_xlsx, index=False)

    print("\n=== Table 6 (Leakage-free, 6 features, TEST, argmax) ===")
    print(table6.to_string(index=False))
    print(f"\nSaved:\n- {out_csv}\n- {out_xlsx}")


if __name__ == "__main__":
    main()
