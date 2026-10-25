"""
EXPERIMENT 1: Linear regression, one variable (gradient descent + normal equation)

Run:  python exp01_linear_regression_california.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

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


def reg_scores(y_true, y_pred):
    """Returns (MSE, R2) - the two regression metrics."""
    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)


# ---------------------------------------------------------------- experiment 1
def linreg_gradient_descent(Xb, y, lr=0.1, epochs=500):
    """Batch gradient descent for linear regression.
    Model:  y_hat = Xb @ theta          (Xb has a leading column of ones = intercept)
    Cost :  J = 1/(2m) * sum (y_hat - y)^2
    Update: theta := theta - lr * (1/m) * Xb^T (y_hat - y)"""
    m = len(y)
    theta = np.zeros(Xb.shape[1])
    history = []
    for _ in range(epochs):
        error = Xb @ theta - y
        theta -= lr * (Xb.T @ error) / m
        history.append(np.sum(error ** 2) / (2 * m))
    return theta, history


def experiment_1():
    df = fetch_california_housing(as_frame=True).frame
    # ---- CSV alternative: needs columns 'AveRooms' and 'MedHouseVal' ----
    # df = pd.read_csv("housing.csv")
    print("Shape:", df.shape)
    print(df.head())

    print("Missing values in dataset:", int(df.isnull().sum().sum()))
    df = df[["AveRooms", "MedHouseVal"]]            # feature = avg rooms, target = price (x $100k)
    lo, hi = df["AveRooms"].quantile([0.01, 0.99])  # AveRooms has crazy outliers (e.g. 141 rooms)
    df = df[(df["AveRooms"] >= lo) & (df["AveRooms"] <= hi)]
    print(f"Kept AveRooms between {lo:.2f} and {hi:.2f} -> {len(df)} rows")
    X, y = df[["AveRooms"]].values, df["MedHouseVal"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    mu, sigma = X_train.mean(), X_train.std()       # scaling helps gradient descent converge
    add_bias = lambda Z: np.c_[np.ones(len(Z)), (Z - mu) / sigma]
    Xb_train, Xb_test = add_bias(X_train), add_bias(X_test)

    theta_gd, history = linreg_gradient_descent(Xb_train, y_train, lr=0.1, epochs=500)
    print("theta (intercept, slope) =", np.round(theta_gd, 4), "| final cost =", round(history[-1], 4))

    theta_ne = np.linalg.pinv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train
    print("theta (intercept, slope) =", np.round(theta_ne, 4))

    sk = LinearRegression().fit(X_train, y_train)    # library model, used only as a cross-check
    rows = []
    for name, pred in [("Gradient Descent", Xb_test @ theta_gd),
                       ("Normal Equation", Xb_test @ theta_ne),
                       ("sklearn LinearRegression (check)", sk.predict(X_test))]:
        mse, r2 = reg_scores(y_test, pred)
        rows.append({"Method": name, "MSE": mse, "R2": r2})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    slope = theta_ne[1] / sigma
    intercept = theta_ne[0] - theta_ne[1] * mu / sigma
    print(f"\nFitted line in ORIGINAL units:  price = {intercept:.4f} + {slope:.4f} * AveRooms")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(X_test, y_test, s=6, alpha=0.3, label="test data")
    xs = np.linspace(X.min(), X.max(), 100)
    ax[0].plot(xs, theta_ne[0] + theta_ne[1] * (xs - mu) / sigma, "r-", lw=2, label="fitted line")
    ax[0].set(xlabel="Average rooms", ylabel="Median house value (x $100k)", title="Fitted line + data")
    ax[0].legend()
    ax[1].plot(history)
    ax[1].set(xlabel="Epoch", ylabel="Cost J", title="Gradient descent convergence")
    save_show("exp1_linear_regression.png")


if __name__ == "__main__":
    experiment_1()