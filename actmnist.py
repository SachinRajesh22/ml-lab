"""
EXPERIMENT 14: Activation functions (Sigmoid, ReLU, Tanh) on MNIST

Run:  python exp14_activation_functions_mnist.py
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
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "lab_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN_NN, N_TEST_NN = 10000, 2000      # image subsets for speed


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


# ---------------------------------------------------------------- experiment 14
def experiment_14():
    header(14, "Activation Functions: Sigmoid vs ReLU vs Tanh - MNIST")

    step("1. Load and preprocess MNIST (pixels scaled to 0-1, random subset for speed)")
    X_train, y_train, X_test, y_test = load_image_dataset("mnist_784")
    X_train, y_train = subsample(X_train, y_train, N_TRAIN_NN)
    X_test, y_test = subsample(X_test, y_test, N_TEST_NN, seed=1)
    print("Train:", X_train.shape, "| test:", X_test.shape)

    step("2. Train the same network (1 hidden layer, 64 neurons) with each activation, 25 epochs")
    activations = {"Sigmoid": "logistic", "ReLU": "relu", "Tanh": "tanh"}
    epochs = 25
    history, rows = {}, []
    for name, act in activations.items():
        mlp = MLPClassifier(hidden_layer_sizes=(64,), activation=act, solver="adam", random_state=RANDOM_STATE)
        losses, accs, t_total = [], [], 0.0
        for _ in range(epochs):                       # one pass over the data per partial_fit call
            t0 = time.time()
            mlp.partial_fit(X_train, y_train, classes=np.arange(10))
            t_total += time.time() - t0
            losses.append(mlp.loss_)
            accs.append(mlp.score(X_test, y_test))
        history[name] = (losses, accs)
        reached = next((i + 1 for i, a in enumerate(accs) if a >= 0.90), None)
        rows.append({"Activation": name, "Final test acc": accs[-1], "Final train loss": losses[-1],
                     "Epochs to reach 90% acc": reached if reached else "not reached",
                     "Training time (s)": t_total})
        print(f"  finished {name}")

    step("3. Compare training time, convergence and accuracy")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, (losses, accs) in history.items():
        ax[0].plot(range(1, epochs + 1), losses, label=name)
        ax[1].plot(range(1, epochs + 1), accs, label=name)
    ax[0].set(xlabel="Epoch", ylabel="Training loss", title="Convergence (loss)")
    ax[1].set(xlabel="Epoch", ylabel="Test accuracy", title="Test accuracy per epoch")
    ax[0].legend()
    ax[1].legend()
    pd.DataFrame(rows).plot.bar(x="Activation", y="Training time (s)", ax=ax[2], legend=False, rot=0,
                                color="darkorange", title="Training time (s)")
    save_show("exp14_activations.png")

    explain([
        "SIGMOID  1/(1+e^-x): output in (0,1); its gradient is tiny for large |x| -> VANISHING GRADIENT, slow learning in deep nets.",
        "TANH  (e^x-e^-x)/(e^x+e^-x): output in (-1,1), ZERO-CENTRED, so usually trains faster than sigmoid, but also saturates.",
        "RELU  max(0,x): cheap to compute, gradient is 1 for x>0 -> fast convergence, no saturation on the positive side.",
        "ReLU can 'die' (always output 0 if it gets stuck negative); variants like Leaky ReLU fix that.",
        "Typical result: ReLU converges fastest, Tanh next, Sigmoid slowest; final accuracy is close on a simple task like MNIST.",
        "Use ReLU for hidden layers by default; sigmoid/softmax are mainly for output layers (probabilities).",
    ])


if __name__ == "__main__":
    experiment_14()