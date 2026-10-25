"""
EXPERIMENT 2: Polynomial regression on Auto MPG

Run:  python exp02_polynomial_regression_auto_mpg.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


# ---------------------------------------------------------------- experiment 2
def load_auto_mpg():
    """Tries OpenML first, then the UCI website. Returns a DataFrame with 'displacement' and 'mpg'."""
    try:
        df = fetch_openml("autoMpg", version=1, as_frame=True).frame.rename(columns={"class": "mpg"})
        assert {"displacement", "mpg"} <= set(df.columns)
        return df
    except Exception as e:
        print("OpenML load failed:", e)
    try:
        cols = ["mpg", "cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model_year", "origin", "car_name"]
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        return pd.read_csv(url, names=cols, na_values="?", comment="\t", sep=r"\s+")
    except Exception as e:
        print("UCI load failed:", e)
    raise RuntimeError("Could not download Auto MPG. Download auto-mpg.csv and use the CSV line below.")


def experiment_2():

    df = load_auto_mpg()
    # ---- CSV alternative: needs columns 'displacement' and 'mpg' ----
    # df = pd.read_csv("auto-mpg.csv")
    print(df[["displacement", "mpg"]].describe().round(2))

    df = df[["displacement", "mpg"]].dropna()
    X, y = df[["displacement"]].values.astype(float), df["mpg"].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print("Rows:", len(df), "| train:", len(X_train), "| test:", len(X_test))

    degrees = [1, 2, 3, 4, 5, 6]
    models, rows = {}, []
    for d in degrees:
        # StandardScaler first so that x^6 does not explode numerically
        model = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), LinearRegression())
        model.fit(X_train, y_train)
        models[d] = model
        tr_mse, tr_r2 = reg_scores(y_train, model.predict(X_train))
        te_mse, te_r2 = reg_scores(y_test, model.predict(X_test))
        rows.append({"Degree": d, "Train MSE": tr_mse, "Test MSE": te_mse, "Train R2": tr_r2, "Test R2": te_r2})
    table = pd.DataFrame(rows)

    print(table.round(3).to_string(index=False))
    best = int(table.loc[table["Test MSE"].idxmin(), "Degree"])
    print(f"\nBest degree by TEST MSE: {best}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    ax[0].scatter(X_train, y_train, s=12, alpha=0.5, label="train")
    ax[0].scatter(X_test, y_test, s=12, alpha=0.5, c="orange", label="test")
    for d in [1, 2, 4, 6]:
        ax[0].plot(xs, models[d].predict(xs), label=f"degree {d}")
    ax[0].set(xlabel="Displacement", ylabel="MPG", title="Polynomial fits")
    ax[0].legend()
    ax[1].plot(table["Degree"], table["Train MSE"], "o-", label="Train MSE")
    ax[1].plot(table["Degree"], table["Test MSE"], "s-", label="Test MSE")
    ax[1].set(xlabel="Polynomial degree", ylabel="MSE", title="Bias-variance: train vs test error")
    ax[1].legend()
    save_show("exp2_polynomial_regression.png")



if __name__ == "__main__":
    experiment_2()