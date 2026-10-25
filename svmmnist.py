"""
EXPERIMENT 12: SVM kernels (linear / polynomial / RBF) on Fashion MNIST

Run:  python exp12_svm_kernels_fashion_mnist.py
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
from sklearn.metrics import (accuracy_score, f1_score)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "lab_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN_SVM, N_TEST_SVM = 5000, 1000     # image subsets (kernel SVM is slow on 60000 images)


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


def subsample(X, y, n, seed=RANDOM_STATE):
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def load_image_dataset(openml_name):
    """Fashion-MNIST / MNIST from OpenML -> (X_train, y_train, X_test, y_test), pixels in [0,1].
    Standard split: first 60000 rows = train, last 10000 rows = test."""
    data = fetch_openml(openml_name, version=1, as_frame=False)
    X = np.asarray(data.data, dtype=np.float32) / 255.0
    y = np.asarray(data.target).astype(int)
    # ---- CSV alternative (Kaggle files: first column = label, remaining 784 = pixels) ----
    # tr = pd.read_csv("fashion-mnist_train.csv"); te = pd.read_csv("fashion-mnist_test.csv")
    # return (tr.iloc[:, 1:].values / 255.0, tr.iloc[:, 0].values,
    #         te.iloc[:, 1:].values / 255.0, te.iloc[:, 0].values)
    return X[:60000], y[:60000], X[60000:], y[60000:]


# ---------------------------------------------------------------- experiment 12
def experiment_12():

    X_train, y_train, X_test, y_test = load_image_dataset("Fashion-MNIST")
    X_train, y_train = subsample(X_train, y_train, N_TRAIN_SVM)
    X_test, y_test = subsample(X_test, y_test, N_TEST_SVM, seed=1)
    print("Train:", X_train.shape, "| test:", X_test.shape, "(lower N_TRAIN_SVM at the top of the file if too slow)")

    kernels = {"Linear": SVC(kernel="linear", C=1.0),
               "Polynomial (degree 3)": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
               "RBF": SVC(kernel="rbf", C=1.0, gamma="scale")}
    rows = []
    for name, model in kernels.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        t0 = time.time()
        pred = model.predict(X_test)
        rows.append({"Kernel": name, "Accuracy": accuracy_score(y_test, pred),
                     "Macro F1": f1_score(y_test, pred, average="macro"),
                     "Train time (s)": t_train, "Predict time (s)": time.time() - t0,
                     "Support vectors": int(model.n_support_.sum())})
        print(f"  finished {name}")

    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    table.plot.bar(x="Kernel", y="Accuracy", ax=ax[0], legend=False, rot=15, ylim=(0.7, 1), title="Test accuracy")
    table.plot.bar(x="Kernel", y="Train time (s)", ax=ax[1], legend=False, rot=15, color="darkorange", title="Training time (s)")
    table.plot.bar(x="Kernel", y="Support vectors", ax=ax[2], legend=False, rot=15, color="green", title="Support vectors")
    save_show("exp12_svm_kernels.png")



if __name__ == "__main__":
    experiment_12()