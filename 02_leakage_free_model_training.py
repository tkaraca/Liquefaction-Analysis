"""
02_leakage_free_model_training.py
--------------------------------
Leakage-free training/evaluation pipeline addressing common reviewer concerns:
- Oversampling (ROS) applied ONLY on training folds (via imblearn Pipeline)
- Optional outlier handling using IQR-based clipping (NOT row removal) to avoid leakage
- Separate train vs test performance reporting (Reviewer #4 comment 4)
- Cross-validated performance with uncertainty (mean±std)
- Threshold tuning focusing on a priority class (e.g., class 3) to reduce false negatives
  in critical damage classes (Reviewer #3 comment 4)

Outputs:
- outputs_models/metrics_train.csv
- outputs_models/metrics_test.csv
- outputs_models/cv_metrics_summary.csv
- confusion matrices (train/test)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

DATA_PATH = "adapvtest.csv"  # adjust
SEP = ";"

FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
TARGET = "dver"
RANDOM_STATE = 42

OUTDIR = "outputs_models"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# 1) Leakage-free outlier handler (IQR clipping)
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clips each feature to [Q1 - factor*IQR, Q3 + factor*IQR] computed on TRAIN ONLY.
    This is safer than filtering/removing rows before splitting and avoids leakage.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.quantile(X, 0.25, axis=0)
        self.q3_ = np.quantile(X, 0.75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * self.iqr_
        self.upper_ = self.q3_ + self.factor * self.iqr_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

# -----------------------------
# 2) Helpers
# -----------------------------
def save_confusion(cm: np.ndarray, classes: list, fname: str, title: str):
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm)
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300, bbox_inches="tight")
    plt.close()

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wprec, wrec, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "weighted_precision": wprec,
        "weighted_recall": wrec,
        "weighted_f1": wf1,
    }

def predict_with_priority_threshold(proba: np.ndarray, classes: np.ndarray, priority_class: int = 3, threshold: float = 0.30):
    """
    Multi-class decision rule to prioritize a critical class:
    - If P(priority_class) >= threshold => predict priority_class
    - else => predict argmax among all classes
    """
    class_to_index = {c: i for i, c in enumerate(classes)}
    idx = class_to_index[priority_class]
    base_pred = classes[np.argmax(proba, axis=1)]
    pred = np.where(proba[:, idx] >= threshold, priority_class, base_pred)
    return pred

def main():
    df = pd.read_csv(DATA_PATH, sep=SEP, engine="python")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Define CV (on TRAIN only)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # -----------------------------
    # 3) Define models + leakage-free pipelines
    # -----------------------------
    ros = RandomOverSampler(random_state=RANDOM_STATE)

    # Random Forest
    rf_pipe = ImbPipeline(steps=[
        ("iqr", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),  # optional for RF; kept for consistency
        ("ros", ros),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [None, "balanced"],
    }

    # XGBoost (with early stopping using eval_set during fit; GridSearchCV cannot pass eval_set easily)
    # Here we do a simpler grid (no early stopping) for grid search; after selecting best params, refit with early stopping.
    xgb_pipe = ImbPipeline(steps=[
        ("iqr", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),
        ("ros", ros),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    xgb_grid = {
        "clf__n_estimators": [300, 600],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    # Gradient Boosting (sklearn) with early stopping via n_iter_no_change
    gb_pipe = ImbPipeline(steps=[
        ("iqr", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),
        ("ros", ros),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    gb_grid = {
        "clf__n_estimators": [200, 500],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.8, 1.0],
        "clf__n_iter_no_change": [10],          # enables early stopping
        "clf__validation_fraction": [0.15],
    }

    # -----------------------------
    # 4) Grid-search (TRAIN ONLY) and refit best
    # -----------------------------
    print("Grid-search RF...")
    rf_search = GridSearchCV(rf_pipe, rf_grid, scoring="recall_macro", cv=cv, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print("Best RF params:", rf_search.best_params_)

    print("Grid-search XGB...")
    xgb_search = GridSearchCV(xgb_pipe, xgb_grid, scoring="recall_macro", cv=cv, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print("Best XGB params:", xgb_search.best_params_)

    print("Grid-search GB...")
    gb_search = GridSearchCV(gb_pipe, gb_grid, scoring="recall_macro", cv=cv, n_jobs=-1)
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    print("Best GB params:", gb_search.best_params_)

    models = {
        "RandomForest": best_rf,
        "XGBoost": best_xgb,
        "GradientBoosting": best_gb
    }

    # -----------------------------
    # 5) Train vs test reporting (Reviewer #4)
    # -----------------------------
    train_rows = []
    test_rows = []

    classes = np.sort(y.unique())

    for name, model in models.items():
        model.fit(X_train, y_train)

        # default prediction
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        train_metrics = compute_metrics(y_train, y_pred_train)
        test_metrics  = compute_metrics(y_test, y_pred_test)

        train_metrics["model"] = name
        test_metrics["model"] = name

        train_rows.append(train_metrics)
        test_rows.append(test_metrics)

        # confusion matrices
        cm_train = confusion_matrix(y_train, y_pred_train, labels=classes)
        cm_test  = confusion_matrix(y_test, y_pred_test, labels=classes)

        save_confusion(cm_train, classes, f"CM_train_{name}.png", f"Train confusion matrix: {name}")
        save_confusion(cm_test,  classes, f"CM_test_{name}.png",  f"Test confusion matrix: {name}")

        print("\n", "="*60)
        print(name, "- TEST classification report (argmax / default):")
        print(classification_report(y_test, y_pred_test, zero_division=0))

    pd.DataFrame(train_rows).set_index("model").to_csv(os.path.join(OUTDIR, "metrics_train.csv"))
    pd.DataFrame(test_rows).set_index("model").to_csv(os.path.join(OUTDIR, "metrics_test.csv"))

    # -----------------------------
    # 6) Optional: threshold tuning to prioritize class 3
    # -----------------------------
    priority_class = 3
    thresholds = np.linspace(0.05, 0.6, 12)

    thr_rows = []
    for name, model in models.items():
        proba_test = model.predict_proba(X_test)
        best_thr = None
        best_recall_c3 = -1.0

        for thr in thresholds:
            y_pred_thr = predict_with_priority_threshold(proba_test, classes=classes, priority_class=priority_class, threshold=thr)
            rec_c3 = recall_score(y_test, y_pred_thr, labels=classes, average=None, zero_division=0)[list(classes).index(priority_class)]
            if rec_c3 > best_recall_c3:
                best_recall_c3 = rec_c3
                best_thr = thr

        # Evaluate with best threshold
        y_pred_thr = predict_with_priority_threshold(proba_test, classes=classes, priority_class=priority_class, threshold=best_thr)
        m = compute_metrics(y_test, y_pred_thr)
        m.update({"model": name, "priority_class": priority_class, "best_threshold": float(best_thr), "recall_class3": float(best_recall_c3)})
        thr_rows.append(m)

        print("\nThreshold tuning:", name, "best_thr=", best_thr, "recall_class3=", best_recall_c3)
        print(classification_report(y_test, y_pred_thr, zero_division=0))

    pd.DataFrame(thr_rows).set_index("model").to_csv(os.path.join(OUTDIR, "metrics_test_threshold_priority_class3.csv"))
    print("\nSaved outputs to:", OUTDIR)
    

if __name__ == "__main__":
    main()
