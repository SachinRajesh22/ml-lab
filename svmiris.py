"""
EXPERIMENT 11: Linear SVM on Iris (Setosa vs Non-Setosa)

Run:  python exp11_linear_svm_iris.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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


# ---------------------------------------------------------------- experiment 11
def plot_svm(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z > 0, alpha=0.12, cmap="coolwarm")
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])   # margin, boundary, margin
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=35)
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=190, facecolors="none", edgecolors="green", linewidths=2, label="support vectors")
    ax.set(xlabel="Petal length (cm)", ylabel="Petal width (cm)", title=title)
    ax.legend(loc="lower right")


def experiment_11():
    header(11, "Linear SVM - Iris (Setosa vs Non-Setosa)")

    step("1. Load the dataset (built into sklearn)")
    iris = load_iris()
    # ---- CSV alternative (columns: petal_length, petal_width, species): ----
    # df = pd.read_csv("iris.csv"); X = df[["petal_length", "petal_width"]].values
    # y = (df["species"] != "setosa").astype(int).values
    X = iris.data[:, [2, 3]]                     # petal length & width -> 2-D so we can PLOT the boundary
    y = (iris.target != 0).astype(int)           # 0 = Setosa, 1 = Non-Setosa (Versicolor + Virginica)
    print("Samples:", len(X), "| Setosa:", int((y == 0).sum()), "| Non-Setosa:", int((y == 1).sum()))

    step("2. Preprocess: train/test split (features already in cm, same scale)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    step("3. Train a linear SVM: hard-ish margin (C=1000) and soft margin (C=0.05)")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, C in zip(ax, [1000, 0.05]):
        svm = SVC(kernel="linear", C=C).fit(X_train, y_train)
        w, b = svm.coef_[0], svm.intercept_[0]
        print(f"\nC = {C}")
        print(f"  weights w = {np.round(w, 3)}, bias b = {b:.3f}")
        print(f"  decision boundary: {w[0]:.3f}*length + {w[1]:.3f}*width + {b:.3f} = 0")
        print(f"  margin width = 2/||w|| = {2 / np.linalg.norm(w):.3f}")
        print(f"  support vectors: {len(svm.support_vectors_)} | test accuracy = {svm.score(X_test, y_test):.4f}")
        plot_svm(a, svm, X_train, y_train, f"Linear SVM, C={C}  (margin width {2 / np.linalg.norm(w):.2f})")

    step("4. Visualise decision boundary (solid) and margins (dashed)")
    save_show("exp11_linear_svm.png")

    explain([
        "SVM chooses the separating line with the MAXIMUM MARGIN = largest gap between the two classes.",
        "Margin width = 2 / ||w||, so maximising the margin = minimising ||w||.",
        "Only the SUPPORT VECTORS (points on / inside the margin) determine the line; other points can move freely.",
        "C controls the trade-off: large C = few margin violations but narrow margin; small C = wide margin, more tolerance.",
        "Setosa is linearly separable, so a straight line separates it perfectly (100% test accuracy).",
        "A wider margin generally means better generalisation to new data.",
    ])


if __name__ == "__main__":
    experiment_11()