"""
EXPERIMENT 3: Ridge and Lasso on the Diabetes dataset

Run:  python exp03_ridge_lasso_diabetes.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import (LinearRegression, Lasso, Ridge)
from sklearn.metrics import (mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
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


def reg_scores(y_true, y_pred):
    """Returns (MSE, R2) - the two regression metrics."""
    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)


# ---------------------------------------------------------------- experiment 3
def experiment_3():
    header(3, "Ridge and Lasso Regression - Diabetes dataset")

    step("1. Load the dataset (built into sklearn)")
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    # ---- CSV alternative (last column = target, here called 'target'): ----
    # df = pd.read_csv("diabetes.csv"); X, y = df.drop(columns="target"), df["target"]
    print("Features:", list(X.columns), "| samples:", len(X))

    step("2. Preprocess: split + standardise (regularisation needs features on the same scale)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    step("3. Fit Linear, Ridge (L2) and Lasso (L1); tune alpha with 5-fold cross-validation")
    ridge_grid = np.logspace(-3, 3, 40)
    lasso_grid = np.logspace(-3, 1.5, 40)
    ridge_cv = GridSearchCV(Ridge(), {"alpha": ridge_grid}, cv=5, scoring="neg_mean_squared_error").fit(Xtr, y_train)
    lasso_cv = GridSearchCV(Lasso(max_iter=20000), {"alpha": lasso_grid}, cv=5, scoring="neg_mean_squared_error").fit(Xtr, y_train)
    models = {"Linear": LinearRegression().fit(Xtr, y_train),
              "Ridge": ridge_cv.best_estimator_,
              "Lasso": lasso_cv.best_estimator_}
    print(f"Best Ridge alpha = {ridge_cv.best_params_['alpha']:.4f}")
    print(f"Best Lasso alpha = {lasso_cv.best_params_['alpha']:.4f}")

    step("4. Compare MSE and R2 on the test set")
    rows = []
    for name, m in models.items():
        mse, r2 = reg_scores(y_test, m.predict(Xte))
        rows.append({"Model": name, "Test MSE": mse, "Test R2": r2, "Zero coefficients": int(np.sum(m.coef_ == 0))})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    step("5. Coefficients of each model")
    coefs = pd.DataFrame({n: m.coef_ for n, m in models.items()}, index=X.columns)
    print(coefs.round(2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogx(ridge_grid, -ridge_cv.cv_results_["mean_test_score"], label="Ridge")
    ax[0].semilogx(lasso_grid, -lasso_cv.cv_results_["mean_test_score"], label="Lasso")
    ax[0].set(xlabel="alpha (regularisation strength)", ylabel="CV MSE", title="Choosing alpha by cross-validation")
    ax[0].legend()
    coefs.plot.bar(ax=ax[1])
    ax[1].set(title="Coefficients: Linear vs Ridge vs Lasso", ylabel="coefficient value")
    save_show("exp3_ridge_lasso.png")

    explain([
        "Ridge adds  alpha * sum(w^2)  (L2) to the loss: it SHRINKS weights but rarely makes them exactly 0.",
        "Lasso adds  alpha * sum(|w|)  (L1) to the loss: it can set weights EXACTLY to 0 = automatic feature selection.",
        "alpha = 0 -> ordinary linear regression; big alpha -> weights go to 0 (underfitting).",
        "Cross-validation: split the training data into 5 parts, train on 4, validate on 1, average -> best alpha.",
        "Regularisation helps most when features are correlated or data is small (reduces variance/overfitting).",
        "On this dataset all three give similar test error; the gain is in simpler, more stable models.",
    ])


if __name__ == "__main__":
    experiment_3()