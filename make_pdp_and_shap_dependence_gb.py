

"""
make_pdp_and_shap_dependence_gb.py  (MULTICLASS-SAFE)

Produces "physical plausibility" plots for the final model (Gradient Boosting):

A) PDP + ICE (recommended, always produced):
   - PGA  -> P(Class=k)
   - Dliq -> P(Class=k)

B) Optional SHAP dependence plots (only if --do_shap):
   - If TreeExplainer is unsupported (common for multiclass GradientBoostingClassifier),
     fallback to KernelSHAP on a subset (to keep runtime reasonable).

Leakage-free:
  - Split first (train/test)
  - Fit IQRClipper + StandardScaler on TRAIN only
  - Apply ROS only to TRAIN for fitting the classifier

Outputs (default outdir: outputs_physical_plausibility):
  - PDP_ICE_PGA_classK.(png|pdf)
  - PDP_ICE_Dliq_classK.(png|pdf)
  - PDP_ICE_combined_classK.(png|pdf)
  - SHAP_dependence_PGA_classK.(png|pdf)   [if --do_shap and shap works]
  - SHAP_dependence_Dliq_classK.(png|pdf)  [if --do_shap and shap works]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay

from imblearn.over_sampling import RandomOverSampler


# -----------------------
# Plot style
# -----------------------
def set_plot_style():
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


# -----------------------
# Leakage-free IQR clipper
# -----------------------
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def try_import_shap():
    try:
        import shap  # noqa: F401
        return shap
    except Exception as e:
        warnings.warn(
            "SHAP could not be imported. PDP/ICE plots will still be generated.\n"
            f"SHAP import error: {repr(e)}"
        )
        return None


def sample_rows(X: np.ndarray, max_rows: int, random_state: int):
    n = X.shape[0]
    if n <= max_rows:
        idx = np.arange(n)
        return X, idx
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx], idx


def main():
    set_plot_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="adapvtest.csv")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--outdir", default="outputs_physical_plausibility")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    # which class probability to visualize (recommended: class 3 = severe)
    ap.add_argument("--target_class", type=int, default=3)

    # GB params (set to match your SHAP-6f run; change if your final model differs)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=1.0)

    # PDP/ICE controls
    ap.add_argument("--grid_resolution", type=int, default=60)
    ap.add_argument("--ice", action="store_true", help="If set, plot ICE lines + PDP (kind='both').")

    # Optional SHAP dependence
    ap.add_argument("--do_shap", action="store_true", help="If set, also compute SHAP dependence plots.")
    ap.add_argument("--shap_explain_size", type=int, default=70, help="Max test rows for KernelSHAP fallback.")
    ap.add_argument("--shap_nsamples", type=int, default=200, help="KernelSHAP nsamples (higher=slower).")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    features = ["pga", "H", "B", "q", "depth", "thickness"]
    target = "dver"

    df = pd.read_csv(args.data, sep=args.sep, engine="python")

    # case-insensitive rescue
    lower_map = {c.lower(): c for c in df.columns}
    rename_dict = {}
    for c in features + [target]:
        if c not in df.columns and c.lower() in lower_map:
            rename_dict[lower_map[c.lower()]] = c
    if rename_dict:
        df = df.rename(columns=rename_dict)

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    X = df[features].astype(float)
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # leakage-free preprocess fitted on TRAIN only
    iqr = IQRClipper(1.5).fit(X_train.values, y_train.values)
    X_train_clip = pd.DataFrame(iqr.transform(X_train.values), columns=features, index=X_train.index)
    X_test_clip  = pd.DataFrame(iqr.transform(X_test.values),  columns=features, index=X_test.index)

    scaler = StandardScaler().fit(X_train_clip.values)
    X_train_t = scaler.transform(X_train_clip.values)
    X_test_t  = scaler.transform(X_test_clip.values)

    # ROS train only (fit model)
    ros = RandomOverSampler(random_state=args.random_state)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_t, y_train.values)

    gb = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.random_state,
    )
    gb.fit(X_train_ros, y_train_ros)

    # Prediction pipeline (NO ROS during inference)
    pred_pipe = Pipeline([
        ("iqr", iqr),
        ("scaler", scaler),
        ("clf", gb),
    ])

    cls = args.target_class
    kind = "both" if args.ice else "average"

    # -----------------------
    # A) PDP + ICE (P(Class=cls))
    # -----------------------
    def save_pdp_ice(feature_name: str, out_base: str, x_label: str):
        fig, ax = plt.subplots(figsize=(9, 6))
        PartialDependenceDisplay.from_estimator(
            pred_pipe,
            X_test,
            features=[feature_name],
            target=cls,
            response_method="predict_proba",
            grid_resolution=args.grid_resolution,
            kind=kind,
            subsample=60 if args.ice else None,
            ax=ax,
        )
        ax.set_title(f"PDP{' + ICE' if args.ice else ''}: P(Class={cls}) vs {x_label} (Gradient Boosting)", fontweight="bold")
        ax.set_xlabel(x_label, fontweight="bold")
        ax.set_ylabel(f"Predicted probability for Class {cls}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"{out_base}.png"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, f"{out_base}.pdf"), bbox_inches="tight")
        plt.close(fig)

    save_pdp_ice("pga",   f"PDP_ICE_PGA_class{cls}",   "PGA (g)")
    save_pdp_ice("depth", f"PDP_ICE_Dliq_class{cls}",  "Dliq (m)")

    # Combined 1x2 panel for paper
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    PartialDependenceDisplay.from_estimator(
        pred_pipe, X_test, features=["pga"], target=cls, response_method="predict_proba",
        grid_resolution=args.grid_resolution, kind=kind, subsample=60 if args.ice else None, ax=axes[0]
    )
    axes[0].set_title(f"PGA", fontweight="bold")
    axes[0].set_xlabel("PGA (g)", fontweight="bold")
    axes[0].set_ylabel(f"P(Class {cls})", fontweight="bold")

    PartialDependenceDisplay.from_estimator(
        pred_pipe, X_test, features=["depth"], target=cls, response_method="predict_proba",
        grid_resolution=args.grid_resolution, kind=kind, subsample=60 if args.ice else None, ax=axes[1]
    )
    axes[1].set_title(f"Dliq", fontweight="bold")
    axes[1].set_xlabel("Dliq (m)", fontweight="bold")
    axes[1].set_ylabel(f"P(Class {cls})", fontweight="bold")

    fig.suptitle(f"Partial dependence{' + ICE' if args.ice else ''} (Gradient Boosting, Class {cls})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"PDP_ICE_combined_class{cls}.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, f"PDP_ICE_combined_class{cls}.pdf"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------
    # B) Optional SHAP dependence (multiclass-safe)
    # -----------------------
    if args.do_shap:
        shap = try_import_shap()
        if shap is None:
            print("[INFO] SHAP not available -> skipping SHAP dependence.")
        else:
            print("[INFO] Computing SHAP dependence plots...")

            # background in model input space (scaled)
            X_bg = shap.sample(X_train_t, min(120, X_train_t.shape[0]), random_state=0)

            # explain subset (scaled)
            X_explain, idx = sample_rows(X_test_t, args.shap_explain_size, args.random_state)
            display_df = X_test_clip.iloc[idx].copy()

            # Try TreeExplainer first; fallback to KernelExplainer if unsupported
            shap_values = None
            used = "TreeExplainer"
            try:
                explainer = shap.TreeExplainer(gb, data=X_bg)
                shap_values = explainer.shap_values(X_explain)
            except Exception as e:
                used = "KernelSHAP"
                print(f"[WARN] TreeExplainer failed ({repr(e)}). Falling back to KernelSHAP (subset).")
                f = lambda z: gb.predict_proba(z)
                bg = shap.sample(X_bg, min(50, X_bg.shape[0]), random_state=0)
                explainer = shap.KernelExplainer(f, bg)
                shap_values = explainer.shap_values(X_explain, nsamples=args.shap_nsamples)

            # pick class shap matrix
            if isinstance(shap_values, list):
                sv_cls = shap_values[cls]
            else:
                sv = np.asarray(shap_values)
                # fallback handling (rare)
                sv_cls = sv if sv.ndim == 2 else sv[:, :, cls]

            def save_dependence(feature: str, out_base: str, title: str):
                plt.figure(figsize=(9, 6))
                # Use shap's dependence plot if available
                shap.dependence_plot(
                    feature,
                    sv_cls,
                    X_explain,
                    feature_names=features,
                    display_features=display_df,
                    interaction_index=None,
                    show=False,
                )
                plt.title(title + f" ({used})", fontweight="bold")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, f"{out_base}.png"), dpi=600, bbox_inches="tight")
                plt.savefig(os.path.join(args.outdir, f"{out_base}.pdf"), bbox_inches="tight")
                plt.close()

            save_dependence(
                "pga",
                f"SHAP_dependence_PGA_class{cls}",
                f"SHAP dependence (Class={cls}): PGA vs SHAP value (Gradient Boosting)"
            )
            save_dependence(
                "depth",
                f"SHAP_dependence_Dliq_class{cls}",
                f"SHAP dependence (Class={cls}): Dliq vs SHAP value (Gradient Boosting)"
            )

    print("\n[DONE]")
    print("Outputs saved to:", os.path.abspath(args.outdir))
    print("PDP/ICE plots are always produced. SHAP dependence only if --do_shap.")


if __name__ == "__main__":
    main()

