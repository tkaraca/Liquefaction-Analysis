import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, recall_score
)
from sklearn.preprocessing import label_binarize

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------
# 1) Leakage-free IQR clipper
# -----------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
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
# 2) Helpers
# -----------------------------
def duplicate_leakage_audit(orig_ids_train, orig_ids_test):
    overlap = len(set(orig_ids_train) & set(orig_ids_test))
    print(f"[Leakage Audit] Train ve test'e ortak düşen orijinal örnek sayısı: {overlap}")
    return overlap

def apply_priority_threshold(proba, classes, priority_class=3, thr=0.30):
    """
    Multi-class: kritik sınıfı (örn class 3) kaçırmamak için:
    P(class3)>=thr ise class3, değilse argmax
    """
    classes = np.array(classes)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx = class_to_idx[priority_class]
    base = classes[np.argmax(proba, axis=1)]
    return np.where(proba[:, idx] >= thr, priority_class, base)

def tune_threshold_on_oof(oof_proba, y_true, classes, priority_class=3, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.60, 12)

    best_thr = None
    best_score = -np.inf

    # hedef: class3 recall yüksek olsun; ikinci hedef: macro-recall da iyi kalsın
    for thr in grid:
        pred = apply_priority_threshold(oof_proba, classes, priority_class=priority_class, thr=thr)
        r_macro = recall_score(y_true, pred, average="macro", zero_division=0)

        # class3 recall
        # sklearn recall_score average=None -> sınıf sırasına göre döner
        per_class = recall_score(y_true, pred, average=None, labels=classes, zero_division=0)
        r_c3 = per_class[list(classes).index(priority_class)]

        # skor: class3 recall'a daha fazla ağırlık ver
        score = 0.7 * r_c3 + 0.3 * r_macro

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score

def plot_macro_roc(y_true, y_proba, classes, title, out_png):
    y_bin = label_binarize(y_true, classes=classes)
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = auc(fpr, tpr)
        aucs.append(auc_i)
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(np.mean(aucs))

    plt.figure(figsize=(7, 5))
    plt.plot(mean_fpr, mean_tpr, label=f"Macro-Average (AUC={macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return macro_auc, aucs

# -----------------------------
# 3) Load data
# -----------------------------
df = pd.read_csv("adapvtest.csv", sep=";", engine="python")
features = ["pga", "H", "B", "q", "depth", "thickness"]
X = df[features].copy()
y = df["dver"].astype(int).copy()
classes = np.sort(y.unique())

# Orijinal ID (leakage audit için)
orig_id = df.index.to_numpy()

# -----------------------------
# 4) Split FIRST (kritik)
# -----------------------------
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, orig_id, test_size=0.20, random_state=42, stratify=y
)

duplicate_leakage_audit(id_train, id_test)  # 0 olmalı

# -----------------------------
# 5) Pipeline + GridSearch (TRAIN only)
# -----------------------------
pipe = Pipeline(steps=[
    ("iqr", IQRClipper(factor=1.5)),
    ("ros", RandomOverSampler(random_state=42)),
    ("model", GradientBoostingClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [200, 500],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="recall_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# -----------------------------
# 6) Threshold tuning: TRAIN OOF (test'e bakmadan)
# -----------------------------
oof_proba = cross_val_predict(
    best_model, X_train, y_train,
    cv=cv, method="predict_proba", n_jobs=-1
)

best_thr, best_score = tune_threshold_on_oof(
    oof_proba, y_train, classes,
    priority_class=3,
    grid=np.linspace(0.05, 0.60, 12)
)
print(f"Chosen threshold from TRAIN OOF (priority class=3): {best_thr:.2f}  (score={best_score:.3f})")

# -----------------------------
# 7) Final fit on TRAIN, evaluate on TEST
# -----------------------------
best_model.fit(X_train, y_train)

test_proba = best_model.predict_proba(X_test)
test_pred_default = best_model.predict(X_test)
test_pred_thr = apply_priority_threshold(test_proba, classes, priority_class=3, thr=best_thr)

print("\n=== TEST report (default argmax) ===")
print(classification_report(y_test, test_pred_default, zero_division=0))

print("\n=== TEST report (priority threshold rule) ===")
print(classification_report(y_test, test_pred_thr, zero_division=0))

cm = confusion_matrix(y_test, test_pred_thr, labels=classes)
print("\nConfusion matrix (thresholded):\n", cm)

# -----------------------------
# 8) ROC/AUC (TEST) - Macro ROC figure
# -----------------------------
macro_auc, per_class_aucs = plot_macro_roc(
    y_true=y_test,
    y_proba=test_proba,
    classes=classes,
    title="Leakage-free TEST Macro-Average ROC",
    out_png="ROC_macro_leakage_free.png"
)

print(f"\nTEST Macro-AUC (leakage-free): {macro_auc:.3f}")
for c, a in zip(classes, per_class_aucs):
    print(f"  Class {c} AUC: {a:.3f}")

print("\nSaved figure: ROC_macro_leakage_free.png")
