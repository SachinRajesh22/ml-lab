"""
EXPERIMENT 13: MLP on Wine Quality with different architectures

Run:  python exp13_mlp_wine_quality.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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


# ---------------------------------------------------------------- experiment 13
def load_wine_quality():
    try:
        X, y = fetch_openml("wine-quality-red", version=1, as_frame=True, return_X_y=True)
        return X.astype(float), pd.to_numeric(y).astype(int)
    except Exception as e:
        print("OpenML load failed:", e)
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=";")
        return df.drop(columns="quality"), df["quality"]
    except Exception as e:
        print("UCI load failed:", e)
    raise RuntimeError("Could not download Wine Quality. Use the CSV line below.")


def experiment_13():
    header(13, "Multilayer Feed-Forward Network (MLP) - Wine Quality (red)")

    step("1. Load the dataset")
    X, y = load_wine_quality()
    # ---- CSV alternative (UCI winequality-red.csv is separated by semicolons): ----
    # df = pd.read_csv("winequality-red.csv", sep=";"); X, y = df.drop(columns="quality"), df["quality"]
    print("Samples:", len(X), "| features:", X.shape[1])
    print("Quality score counts:\n", y.value_counts().sort_index().to_string())

    step("2. Preprocess: stratified split + standardise")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    step("3. Train MLPs with different numbers of hidden layers / neurons")
    archs = {"(4,)": (4,), "(16,)": (16,), "(64,)": (64,), "(16,16)": (16, 16),
             "(64,32)": (64, 32), "(128,64,32)": (128, 64, 32)}
    rows, curves = [], {}
    for name, hidden in archs.items():
        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", max_iter=600,
                            random_state=RANDOM_STATE).fit(Xtr, y_train)
        n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
        rows.append({"Hidden layers": name, "Parameters": n_params, "Train acc": mlp.score(Xtr, y_train),
                     "Test acc": mlp.score(Xte, y_test), "Epochs": mlp.n_iter_, "Time (s)": time.time() - t0})
        curves[name] = mlp.loss_curve_

    step("4. Compare the architectures")
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    idx = np.arange(len(table))
    ax[0].bar(idx - 0.2, table["Train acc"], 0.4, label="Train")
    ax[0].bar(idx + 0.2, table["Test acc"], 0.4, label="Test")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(table["Hidden layers"])
    ax[0].set(ylim=(0.4, 1), ylabel="accuracy", title="Accuracy by architecture")
    ax[0].legend()
    for name, c in curves.items():
        ax[1].plot(c, label=name)
    ax[1].set(xlabel="Epoch", ylabel="Training loss", title="Loss curves")
    ax[1].legend(fontsize=8)
    save_show("exp13_mlp_architectures.png")

    explain([
        "An MLP stacks layers of neurons: output = activation(W x + b), trained by back-propagation + gradient descent (Adam).",
        "Too FEW neurons/layers -> underfitting (low train AND test accuracy).",
        "Too MANY -> train accuracy keeps rising but test accuracy stalls/drops (overfitting) and training is slower.",
        "Wine quality scores are imbalanced (mostly 5 and 6), so accuracy plateaus around 60-65% for any architecture.",
        "Remedies for overfitting: fewer parameters, early stopping, L2 penalty (alpha), more data, dropout.",
        "Standardising the inputs is essential for neural networks.",
    ])


if __name__ == "__main__":
    experiment_13()