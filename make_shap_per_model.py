

"""
make_shap_per_model.py (6-feature locked)
----------------------------------------
Computes per-model SHAP feature importance (NO averaging across models) for:
  - RandomForest
  - XGBoost
  - GradientBoosting

Dataset columns expected (semicolon-separated CSV by default):
  pga;H;B;q;depth;thickness;dver

Leakage-free principles:
  1) Split FIRST (train/test)
  2) Fit IQR clipping on TRAIN only
  3) Fit scaler on TRAIN only
  4) Apply ROS ONLY on TRAIN during fit (imblearn Pipeline)

Outputs (default outdir: outputs_shap_6f/):
  - SHAP_RandomForest_6f_bar.(png|pdf)
  - SHAP_XGBoost_6f_bar.(png|pdf)
  - SHAP_GradientBoosting_6f_bar.(png|pdf)
  - SHAP_per_model_RF_XGB_GB_6f.(png|pdf)
  - SHAP_importance_normalized_per_model_6f.(csv)
  - SHAP_importance_normalized_per_model_6f_long.(csv)
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# -----------------------
# Locked 6 features
# -----------------------
FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
TARGET = "dver"

# Labels to match manuscript nomenclature (optional)
FEATURE_LABELS = {
    "pga": "PGA",
    "H": "H",
    "B": "B",
    "q": "q",
    "depth": "Dliq",
    "thickness": "Hliq",
}


def safe_import_shap():
    try:
        import shap  # noqa: F401
        return shap
    except Exception as e:
        raise RuntimeError(
            "ERROR: Failed to import 'shap'.\n"
            f"  -> {repr(e)}\n\n"
            "Common fixes:\n"
            "  pip install -U shap numba coverage\n"
            "If coverage causes issues:\n"
            "  pip uninstall -y coverage\n"
        ) from e


class IQRClipper(BaseEstimator, TransformerMixin):
    """IQR-based clipping (winsor-like), fitted on TRAIN only."""
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_importance(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    s = float(np.sum(vals))
    return vals / s if s > 0 else vals


def mean_abs_shap(shap_values):
    """
    Convert SHAP output into a global importance vector:
    - multi-class: average over samples and classes
    """
    sv = shap_values

    # shap.TreeExplainer sometimes returns list per class
    if isinstance(sv, list):
        per_class = [np.abs(v).mean(axis=0) for v in sv]
        return np.mean(per_class, axis=0)

    sv = np.asarray(sv)

    # binary / single-output: (n, f)
    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0)

    # multioutput: (n, f, C) or (n, C, f)
    if sv.ndim == 3:
        n, a, b = sv.shape
        # heuristics: features ~6, so pick that axis
        if a == len(FEATURES):  # (n, f, C)
            return np.abs(sv).mean(axis=(0, 2))
        if b == len(FEATURES):  # (n, C, f)
            return np.abs(sv).mean(axis=(0, 1))
        # fallback: average over last axis as class
        return np.abs(sv).mean(axis=(0, 2))

    raise ValueError(f"Unsupported SHAP shape: {sv.shape}")


def plot_bar(feature_names, importances, title, out_png, out_pdf):
    order = np.argsort(importances)[::-1]
    fn = [feature_names[i] for i in order]
    imp = importances[order]

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.barh(fn[::-1], imp[::-1])
    ax.set_title(title)
    ax.set_xlabel("Normalized mean(|SHAP|)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=330, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def transform_X(pipe, X):
    """Apply only preprocessing steps for SHAP input (ROS is not applied at inference)."""
    X1 = pipe.named_steps["iqr"].transform(X)
    X2 = pipe.named_steps["scaler"].transform(X1)
    return X2


def compute_shap_tree(shap, clf, X_bg, X_explain):
    explainer = shap.TreeExplainer(clf, data=X_bg)
    return explainer.shap_values(X_explain)


def compute_shap_kernel(shap, clf, X_bg, X_explain):
    f = lambda z: clf.predict_proba(z)
    bg = shap.sample(X_bg, min(50, X_bg.shape[0]), random_state=0)
    explainer = shap.KernelExplainer(f, bg)
    return explainer.shap_values(X_explain, nsamples=200)


def compute_shap_xgb_pred_contribs(xgb_model, X_explain):
    import xgboost as xgb
    booster = xgb_model.get_booster()
    dm = xgb.DMatrix(X_explain)
    contrib = booster.predict(dm, pred_contribs=True)
    contrib = np.asarray(contrib)

    # Multi-class can be (n, C, f+1) or flattened (n, C*(f+1))
    if contrib.ndim == 3:
        contrib = contrib[:, :, :-1]  # drop bias
        return [contrib[:, c, :] for c in range(contrib.shape[1])]

    if contrib.ndim == 2:
        n = contrib.shape[0]
        C = getattr(xgb_model, "n_classes_", None)
        if C is not None and int(C) > 1:
            f_plus_bias = contrib.shape[1] // int(C)
            contrib3 = contrib.reshape(n, int(C), f_plus_bias)
            contrib3 = contrib3[:, :, :-1]  # drop bias
            return [contrib3[:, c, :] for c in range(int(C))]
        return contrib[:, :-1]  # binary/reg

    raise ValueError(f"Unexpected pred_contribs shape: {contrib.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="adapvtest.csv", help="Path to CSV")
    ap.add_argument("--sep", default=";", help="CSV separator (default ';')")
    ap.add_argument("--outdir", default="outputs_shap_6f", help="Output directory")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.data, sep=args.sep, engine="python")

    # Case-insensitive rescue (if needed)
    lower_map = {c.lower(): c for c in df.columns}
    rename_dict = {}
    for c in FEATURES + [TARGET]:
        if c not in df.columns and c.lower() in lower_map:
            rename_dict[lower_map[c.lower()]] = c
    if rename_dict:
        df = df.rename(columns=rename_dict)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    shap = safe_import_shap()

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=args.random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=args.random_state
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.random_state,
            n_jobs=-1,
        )
    else:
        warnings.warn("xgboost not installed -> XGBoost will be skipped.")

    rows = []
    per_model_imps = {}

    # Plot labels (Dliq/Hliq etc.)
    plot_features = [FEATURE_LABELS.get(f, f) for f in FEATURES]

    for name, clf in models.items():
        print(f"[INFO] Fitting {name} (6 features) ...")

        pipe = ImbPipeline(
            steps=[
                ("iqr", IQRClipper(factor=1.5)),
                ("scaler", StandardScaler()),
                ("ros", RandomOverSampler(random_state=args.random_state)),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train, y_train)

        X_train_t = transform_X(pipe, X_train)
        X_test_t = transform_X(pipe, X_test)
        final_clf = pipe.named_steps["clf"]

        print(f"[INFO] Computing SHAP for {name} ...")

        # 1) Try TreeExplainer, 2) fallback
        try:
            sv = compute_shap_tree(shap, final_clf, X_train_t, X_test_t)
        except Exception as e_tree:
            print(f"[WARN] TreeExplainer failed for {name}: {repr(e_tree)}")
            if name == "XGBoost":
                sv = compute_shap_xgb_pred_contribs(final_clf, X_test_t)
                print("[INFO] Used XGBoost pred_contribs fallback.")
            else:
                sv = compute_shap_kernel(shap, final_clf, X_train_t, X_test_t)
                print("[INFO] Used KernelExplainer fallback.")

        imp = normalize_importance(mean_abs_shap(sv))
        per_model_imps[name] = imp

        out_png = os.path.join(args.outdir, f"SHAP_{name}_6f_bar.png")
        out_pdf = os.path.join(args.outdir, f"SHAP_{name}_6f_bar.pdf")
        plot_bar(plot_features, imp, f"{name} - SHAP Feature Importance (6 features)", out_png, out_pdf)

        for f, v in zip(FEATURES, imp):
            rows.append({"model": name, "feature": f, "importance_norm": float(v)})

    # Save CSVs
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(os.path.join(args.outdir, "SHAP_importance_normalized_per_model_6f_long.csv"), index=False)

    pivot = imp_df.pivot_table(index="feature", columns="model", values="importance_norm")
    pivot.to_csv(os.path.join(args.outdir, "SHAP_importance_normalized_per_model_6f.csv"))

    # Combined multi-panel figure (RF / XGB / GB if available)
    model_order = [m for m in ["RandomForest", "XGBoost", "GradientBoosting"] if m in per_model_imps]
    n = len(model_order)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        for j, m in enumerate(model_order):
            imp = per_model_imps[m]
            order = np.argsort(imp)[::-1]
            fn = [plot_features[i] for i in order]
            vals = imp[order]

            ax = axes[0, j]
            ax.barh(fn[::-1], vals[::-1])
            ax.set_title(m)
            ax.set_xlabel("Normalized mean(|SHAP|)")

        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "SHAP_per_model_RF_XGB_GB_6f.png"), dpi=330, bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, "SHAP_per_model_RF_XGB_GB_6f.pdf"), bbox_inches="tight")
        plt.close(fig)

    print("\n[DONE]")
    print("Outputs:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()

