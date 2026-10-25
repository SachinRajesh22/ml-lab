"""
EXPERIMENT 6: Logistic regression on Pima Indians Diabetes, with / without scaling

Run:  python exp06_logistic_scaling_pima.py
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
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


def clf_scores(y_true, y_pred, average="binary"):
    """Accuracy, precision, recall, F1 in one dict."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------- experiment 6
def experiment_6():

    df = fetch_openml("diabetes", version=1, as_frame=True).frame.copy()
    df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                  "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df["Outcome"] = (df["Outcome"] == "tested_positive").astype(int)
    # ---- CSV alternative (Kaggle diabetes.csv, last column 'Outcome' = 0/1): ----
    # df = pd.read_csv("diabetes.csv")
    print("Shape:", df.shape, "| diabetic cases:", int(df["Outcome"].sum()))

    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    print("Missing values created:\n", df[zero_cols].isnull().sum().to_string())
    X, y = df.drop(columns="Outcome"), df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    imp = SimpleImputer(strategy="median").fit(X_train)
    X_train, X_test = imp.transform(X_train), imp.transform(X_test)
    scaler = StandardScaler().fit(X_train)

    results, preds = [], {}
    for label, (Xa, Xb) in {"Without scaling": (X_train, X_test),
                            "With scaling": (scaler.transform(X_train), scaler.transform(X_test))}.items():
        t0 = time.time()
        model = LogisticRegression(max_iter=10000).fit(Xa, y_train)
        elapsed = time.time() - t0
        pred = model.predict(Xb)
        preds[label] = pred
        results.append({"Setting": label, **clf_scores(y_test, pred), "Solver iterations": int(model.n_iter_[0]),
                        "Time (s)": elapsed})

    print(pd.DataFrame(results).round(4).to_string(index=False))
    for label, pred in preds.items():
        print(f"\nConfusion matrix ({label})  [rows = actual 0/1, cols = predicted 0/1]\n", confusion_matrix(y_test, pred))

    res = pd.DataFrame(results).set_index("Setting")[["Accuracy", "Precision", "Recall", "F1"]]
    res.plot.bar(figsize=(8, 4.5), rot=0, ylim=(0, 1), title="Logistic regression: scaling vs no scaling")
    save_show("exp6_scaling_effect.png")



if __name__ == "__main__":
    experiment_6()