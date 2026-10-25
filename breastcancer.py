"""
EXPERIMENT 4: Logistic regression: MLE vs MAP (L1 / L2 priors)

Run:  python exp04_logistic_mle_map_breast_cancer.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "lab_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- helpers
def header(n, title):
    print("\n" + "=" * 90)
    print(f"EXPERIMENT {n}: {title}")
    print("=" * 90)


def step(text):
    print(f"\n[STEP] {text}")


def explain(points):
    print("\n[KEY POINTS TO EXPLAIN]")
    for p in points:
        print("  *", p)


def save_show(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.show()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------- experiment 4
def fit_logistic(X, y, l1=0.0, l2=0.0, lr=0.1, epochs=5000):
    """Logistic regression trained from scratch by gradient descent.

    Objective (to MINIMISE) = (1/m) * NegativeLogLikelihood + l2/2 * ||w||^2 + l1 * ||w||_1
        l1 = l2 = 0  ->  MLE (Maximum Likelihood Estimate)
        l2 > 0       ->  MAP with a GAUSSIAN prior on the weights  (L2 / Ridge)
        l1 > 0       ->  MAP with a LAPLACE  prior on the weights  (L1 / Lasso)
    The intercept b is not regularised. L1 is handled with soft-thresholding (proximal step)."""
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - y) / m + l2 * w)
        b -= lr * np.mean(p - y)
        if l1 > 0:
            w = np.sign(w) * np.maximum(np.abs(w) - lr * l1, 0.0)
    return w, b


def experiment_4():
    header(4, "Logistic Regression: MLE vs MAP - Breast Cancer Wisconsin")

    step("1. Load the dataset (built into sklearn)")
    data = load_breast_cancer()
    X, y = data.data, data.target
    # ---- CSV alternative (Kaggle 'data.csv' has 'diagnosis' = M/B): ----
    # df = pd.read_csv("data.csv"); y = (df["diagnosis"] == "M").astype(int).values
    # X = df.drop(columns=["diagnosis", "id"], errors="ignore").values
    print("Samples:", X.shape[0], "| features:", X.shape[1], "| classes:", data.target_names)

    step("2. Preprocess: split + standardise")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    step("3. Fit MLE, then MAP with L2 prior and L1 prior (several strengths)")
    configs = [("MLE (no prior, 5000 GD steps)", 0.0, 0.0),
               ("MAP  L2  lambda=0.01", 0.0, 0.01),
               ("MAP  L2  lambda=0.10", 0.0, 0.10),
               ("MAP  L1  lambda=0.01", 0.01, 0.0),
               ("MAP  L1  lambda=0.05", 0.05, 0.0)]
    fitted, rows = {}, []
    for name, l1, l2 in configs:
        w, b = fit_logistic(Xtr, y_train, l1=l1, l2=l2)
        fitted[name] = w
        prob = sigmoid(Xte @ w + b)
        rows.append({"Model": name,
                     "Test accuracy": accuracy_score(y_test, prob > 0.5),
                     "Test log-loss": log_loss(y_test, prob),
                     "||w|| (size of weights)": np.linalg.norm(w),
                     "Zero weights": int(np.sum(np.abs(w) < 1e-6))})

    step("4. Compare performance and parameter estimates")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # Our gradient descent stops after 5000 steps (a form of early stopping). The TRUE MLE on this
    # almost-separable data has enormous weights - sklearn with a huge C (= no regularisation) shows it:
    sk = LogisticRegression(C=1e6, max_iter=10000).fit(Xtr, y_train)
    print(f"\nFully converged MLE (sklearn, no regularisation): test accuracy = {sk.score(Xte, y_test):.4f}, "
          f"||w|| = {np.linalg.norm(sk.coef_):.1f}  <- weights explode without a prior!")

    fig, ax = plt.subplots(1, 1, figsize=(13, 4.5))
    idx = np.arange(X.shape[1])
    ax.bar(idx - 0.27, fitted["MLE (no prior, 5000 GD steps)"], 0.27, label="MLE")
    ax.bar(idx, fitted["MAP  L2  lambda=0.10"], 0.27, label="MAP L2 (0.10)")
    ax.bar(idx + 0.27, fitted["MAP  L1  lambda=0.05"], 0.27, label="MAP L1 (0.05)")
    ax.set_xticks(idx)
    ax.set_xticklabels(data.feature_names, rotation=90, fontsize=7)
    ax.set(ylabel="weight", title="Parameter estimates: MLE vs MAP")
    ax.legend()
    save_show("exp4_mle_vs_map.png")

    explain([
        "MLE picks weights that maximise the likelihood of the training data, i.e. minimise the log-loss.",
        "MAP = MLE + a PRIOR on the weights:  maximise  log-likelihood + log prior.",
        "Gaussian prior on weights  <=>  L2 penalty (shrinks weights).  Laplace prior  <=>  L1 penalty (sparse weights).",
        "With few samples or (nearly) separable data, MLE weights grow without limit -> overfitting; MAP keeps them small.",
        "(Our MLE row is gradient descent stopped at 5000 steps; the fully converged MLE has far larger weights, see the printed check.)",
        "Bigger lambda = stronger prior = smaller weights; too big and the model underfits.",
    ])


if __name__ == "__main__":
    experiment_4()