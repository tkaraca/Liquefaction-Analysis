#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
from imblearn.over_sampling import RandomOverSampler


FEATURES = ["pga", "H", "B", "q", "depth", "thickness"]
TARGET = "dver"


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adapvtest.csv", help="Path to CSV dataset")
    parser.add_argument("--sep", default=";", help="CSV separator (default ';')")
    parser.add_argument("--outdir", default="outputs_overfitting", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.20, help="Hold-out test size (default 0.20)")
    parser.add_argument("--val_size", type=float, default=0.20, help="Validation split within train (default 0.20)")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--use_ros", action="store_true", help="Apply RandomOverSampler on TRAIN split only")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience based on val loss")
    parser.add_argument("--hidden", default="64,32", help="Hidden layers, e.g. '64,32' or '32,16'")
    parser.add_argument("--alpha", type=float, default=1e-3, help="L2 regularization strength (default 1e-3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate init (default 1e-3)")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    df = pd.read_csv(args.data, sep=args.sep, engine="python")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Hold-out test split (leakage-free)
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    # Train/Validation split inside training set (for learning curves)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=args.val_size,
        stratify=y_train_all,
        random_state=args.random_state
    )

    # Scale (fit on TRAIN only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Optional: oversampling only on TRAIN (not on val/test)
    if args.use_ros:
        ros = RandomOverSampler(random_state=args.random_state)
        X_train_s, y_train = ros.fit_resample(X_train_s, y_train)

    classes = np.sort(y.unique())
    hidden_layers = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())

    # We train epoch-by-epoch using partial_fit to compute train/val loss each epoch
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        learning_rate_init=args.lr,
        max_iter=1,          # one epoch per partial_fit call
        warm_start=True,     # keep weights between epochs
        shuffle=True,
        random_state=args.random_state
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val = np.inf
    best_epoch = -1
    wait = 0

    for epoch in range(1, args.max_epochs + 1):
        mlp.partial_fit(X_train_s, y_train, classes=classes)

        # Probabilities for log_loss
        p_train = mlp.predict_proba(X_train_s)
        p_val = mlp.predict_proba(X_val_s)

        tr_loss = log_loss(y_train, p_train, labels=classes)
        va_loss = log_loss(y_val, p_val, labels=classes)

        ytr_pred = mlp.predict(X_train_s)
        yva_pred = mlp.predict(X_val_s)

        tr_acc = accuracy_score(y_train, ytr_pred)
        va_acc = accuracy_score(y_val, yva_pred)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        # early stopping on validation loss
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val loss: {best_val:.4f})")
            break

    # Evaluate on hold-out test (one number for context)
    y_test_pred = mlp.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Hold-out test accuracy (MLP baseline): {test_acc:.4f}")

    # Plot LOSS curves
    fig = plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MLP baseline: training vs validation loss (leakage-free split)")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(args.outdir, "Supplementary_Fig_Sx_MLP_loss_curves.png")
    out_pdf = os.path.join(args.outdir, "Supplementary_Fig_Sx_MLP_loss_curves.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Plot ACC curves (optional but useful)
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="Training accuracy")
    plt.plot(val_accs, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP baseline: training vs validation accuracy (leakage-free split)")
    plt.legend()
    plt.tight_layout()

    out_png2 = os.path.join(args.outdir, "Supplementary_Fig_Sx_MLP_accuracy_curves.png")
    out_pdf2 = os.path.join(args.outdir, "Supplementary_Fig_Sx_MLP_accuracy_curves.pdf")
    fig2.savefig(out_png2, dpi=600, bbox_inches="tight")
    fig2.savefig(out_pdf2, bbox_inches="tight")
    plt.close(fig2)

    print("Saved:")
    print(" -", out_png)
    print(" -", out_pdf)
    print(" -", out_png2)
    print(" -", out_pdf2)


if __name__ == "__main__":
    main()
