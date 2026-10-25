"""
EXPERIMENT 10: Logistic Regression vs Decision Tree on Adult Income

Run:  python exp10_logreg_vs_tree_adult_income.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

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


# ---------------------------------------------------------------- experiment 10
def experiment_10():
    header(10, "Logistic Regression vs Decision Tree - Adult Income")

    step("1. Load the dataset (OpenML 'adult')")
    df = fetch_openml("adult", version=2, as_frame=True).frame
    # ---- CSV alternative (adult.csv with '?' for missing and target column 'income'): ----
    # df = pd.read_csv("adult.csv").replace("?", np.nan).rename(columns={"income": "class"})
    y = df["class"].astype(str).str.strip().isin([">50K", ">50K."]).astype(int)
    X = df.drop(columns=["class", "fnlwgt"], errors="ignore")
    print("Shape:", X.shape, "| share earning >50K:", round(y.mean(), 3))

    step("2. Preprocess: impute missing, one-hot encode categories, scale numbers")
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    X[cat_cols] = X[cat_cols].astype(object)
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    Xtr, Xte = pre.fit_transform(X_train), pre.transform(X_test)
    names = pre.get_feature_names_out()
    print("Features after encoding:", Xtr.shape[1])

    step("3. Train the models")
    models = {"Logistic Regression": LogisticRegression(max_iter=2000),
              "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
              "Decision Tree (unlimited)": DecisionTreeClassifier(random_state=RANDOM_STATE)}
    rows = []
    for name, m in models.items():
        m.fit(Xtr, y_train)
        rows.append({"Model": name, **clf_scores(y_test, m.predict(Xte)), "Train acc": m.score(Xtr, y_train)})

    step("4. Compare accuracy, precision, recall, F1")
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    step("5. Interpretability")
    lr, dt = models["Logistic Regression"], models["Decision Tree (depth 5)"]
    coef = pd.Series(lr.coef_[0], index=names).sort_values()
    print("\nLogistic regression - strongest features pushing towards '<=50K' (negative):")
    print(coef.head(6).round(3).to_string())
    print("Logistic regression - strongest features pushing towards '>50K' (positive):")
    print(coef.tail(6).round(3).to_string())
    print("\nDecision tree (depth 5) rules, first 3 levels:")
    print(export_text(dt, feature_names=list(names), max_depth=3))
    print("Tree feature importances (top 6):")
    print(pd.Series(dt.feature_importances_, index=names).sort_values(ascending=False).head(6).round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 2]})
    table.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(ax=ax[0], rot=20, ylim=(0, 1))
    ax[0].set_title("Metric comparison")
    plot_tree(dt, max_depth=2, feature_names=list(names), class_names=["<=50K", ">50K"],
              filled=True, fontsize=7, ax=ax[1])
    ax[1].set_title("Decision tree (top 2 levels)")
    save_show("exp10_logreg_vs_tree.png")

    explain([
        "Logistic regression: one weight per feature -> a linear decision boundary; coefficients show direction and strength.",
        "Decision tree: a set of IF-THEN rules that can model non-linear interactions; easy to explain to non-experts.",
        "The unlimited tree has ~100% TRAIN accuracy but lower TEST accuracy -> overfitting; depth limiting fixes it.",
        "Dataset is imbalanced (~24% earn >50K): look at recall/F1 of the positive class, not only accuracy.",
        "Trade-off: logistic regression gives probabilities and is stable; trees are more transparent but less stable.",
    ])


if __name__ == "__main__":
    experiment_10()