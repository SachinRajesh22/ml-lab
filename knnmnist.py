"""
EXPERIMENT 8: KNN on Fashion-MNIST (implemented from scratch)

Run:  python exp08_knn_fashion_mnist.py
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
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "lab_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN_KNN, N_TEST_KNN = 10000, 2000   # image subsets (full dataset is slow for KNN)
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


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


# ---------------------------------------------------------------- experiment 8
class KNNScratch:
    """K-Nearest-Neighbours classifier written with NumPy only.
    fit()     : just stores the training data (KNN is a 'lazy' learner).
    predict() : for each test image, find the K closest training images (Euclidean distance)
                and return the most common label among them (majority vote)."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X.astype(np.float32), y.astype(int)
        self.sq = (self.X ** 2).sum(axis=1)
        return self

    def predict(self, X, batch=500):
        out = []
        for i in range(0, len(X), batch):
            Q = X[i:i + batch].astype(np.float32)
            # squared distance  ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x   (fast, uses matrix multiply)
            d = (Q ** 2).sum(axis=1)[:, None] + self.sq[None, :] - 2 * Q @ self.X.T
            nn = np.argpartition(d, self.k - 1, axis=1)[:, :self.k]      # indices of the k nearest
            votes = self.y[nn]
            out.extend(np.bincount(r, minlength=10).argmax() for r in votes)
        return np.array(out)


def experiment_8():
    header(8, "K-Nearest Neighbours - Fashion MNIST")

    step("1. Load and preprocess (pixels scaled to 0-1, random subset for speed)")
    X_train, y_train, X_test, y_test = load_image_dataset("Fashion-MNIST")
    X_train, y_train = subsample(X_train, y_train, N_TRAIN_KNN)
    X_test, y_test = subsample(X_test, y_test, N_TEST_KNN, seed=1)
    print("Train:", X_train.shape, "| test:", X_test.shape)

    fig, ax = plt.subplots(1, 10, figsize=(14, 2))
    for c in range(10):
        ax[c].imshow(X_train[np.where(y_train == c)[0][0]].reshape(28, 28), cmap="gray")
        ax[c].set_title(FASHION_CLASSES[c], fontsize=8)
        ax[c].axis("off")
    save_show("exp8_samples.png")

    step("2. KNN for different K (own implementation)")
    ks = [1, 3, 5, 7, 9, 15, 25, 51]
    rows = []
    for k in ks:
        t0 = time.time()
        pred = KNNScratch(k).fit(X_train, y_train).predict(X_test)
        rows.append({"K": k, "Accuracy": accuracy_score(y_test, pred),
                     "Macro F1": f1_score(y_test, pred, average="macro"), "Predict time (s)": time.time() - t0})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    step("3. Cross-check against sklearn's KNeighborsClassifier (K=5)")
    sk_acc = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).score(X_test, y_test)
    print(f"sklearn accuracy = {sk_acc:.4f}   |   own implementation = {table.loc[table.K == 5, 'Accuracy'].values[0]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(table["K"], table["Accuracy"], "o-")
    ax[0].set(xlabel="K", ylabel="Test accuracy", title="Accuracy vs K")
    ax[1].plot(table["K"], table["Predict time (s)"], "s-", c="darkorange")
    ax[1].set(xlabel="K", ylabel="Prediction time (s)", title="Computation time vs K")
    save_show("exp8_knn_k_values.png")
    print(f"\nBest K = {int(table.loc[table['Accuracy'].idxmax(), 'K'])}")

    explain([
        "KNN has NO training phase; all the work happens at prediction time (distance to every training image).",
        "Small K (e.g. 1): very flexible, low bias but noisy -> OVERFITS. Large K: smooth, but blurs class borders -> UNDERFITS.",
        "The best K is usually small-to-moderate (a few); choose it with a validation set / cross-validation.",
        "Prediction cost ~ (training size x number of features) for EVERY query; it barely depends on K.",
        "KNN needs feature scaling and suffers in very high dimensions ('curse of dimensionality').",
    ])


if __name__ == "__main__":
    experiment_8()