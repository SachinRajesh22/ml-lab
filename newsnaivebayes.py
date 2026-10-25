"""
EXPERIMENT 7: Multinomial vs Bernoulli Naive Bayes on 20 Newsgroups

Run:  python exp07_naive_bayes_newsgroups.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             classification_report)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

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


# ---------------------------------------------------------------- experiment 7
def experiment_7():
    header(7, "Naive Bayes for Text: Multinomial vs Bernoulli - 20 Newsgroups")

    step("1. Load the full 20 Newsgroups dataset")
    kw = dict(remove=("headers", "footers", "quotes"))
    train, test = fetch_20newsgroups(subset="train", **kw), fetch_20newsgroups(subset="test", **kw)
    # ---- CSV alternative: file with columns 'text' and 'label' ----
    # df = pd.read_csv("newsgroups.csv")  # then split into train/test texts and labels yourself
    print("Train docs:", len(train.data), "| test docs:", len(test.data), "| classes:", len(train.target_names))

    step("2. Preprocess: bag-of-words word counts (English stop-words removed)")
    results = {}
    rows = []
    for vocab_size in [500, 2000, 5000, 20000]:
        vec = CountVectorizer(stop_words="english", max_features=vocab_size)
        Xtr, Xte = vec.fit_transform(train.data), vec.transform(test.data)
        for name, clf in [("Multinomial NB", MultinomialNB(alpha=1.0)),
                          ("Bernoulli NB", BernoulliNB(alpha=1.0))]:   # Bernoulli turns counts into 0/1 itself
            pred = clf.fit(Xtr, train.target).predict(Xte)
            rows.append({"Vocabulary": vocab_size, "Model": name,
                         "Accuracy": accuracy_score(test.target, pred),
                         "Macro F1": f1_score(test.target, pred, average="macro"),
                         "Weighted F1": f1_score(test.target, pred, average="weighted")})
            if vocab_size == 20000:
                results[name] = pred

    step("3. Compare accuracy and F1 for several vocabulary sizes")
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    step("4. Detailed report for the largest vocabulary (20000 words)")
    for name, pred in results.items():
        print(f"\n--- {name} ---")
        print(classification_report(test.target, pred, target_names=train.target_names, digits=3))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in ["Multinomial NB", "Bernoulli NB"]:
        sub = table[table["Model"] == name]
        ax[0].plot(sub["Vocabulary"], sub["Accuracy"], "o-", label=name)
    ax[0].set(xscale="log", xlabel="Vocabulary size", ylabel="Test accuracy", title="Accuracy vs vocabulary size")
    ax[0].legend()
    cm = confusion_matrix(test.target, results["Multinomial NB"])
    im = ax[1].imshow(cm, cmap="Blues")
    ax[1].set(title="Confusion matrix - Multinomial NB (20 classes)", xlabel="predicted", ylabel="actual")
    plt.colorbar(im, ax=ax[1])
    save_show("exp7_naive_bayes.png")

    explain([
        "Naive Bayes uses Bayes' rule with the 'naive' assumption that words are independent given the class.",
        "MULTINOMIAL NB models word COUNTS (how many times each word appears) -> uses term frequency.",
        "BERNOULLI NB models word PRESENCE/ABSENCE (0/1) and also penalises words that are absent.",
        "Multinomial usually wins on longer documents and big vocabularies; Bernoulli is OK for short texts / small vocabularies.",
        "Both are very fast, need little data, and give a strong baseline; the independence assumption is unrealistic.",
        "alpha=1 is Laplace smoothing (see Experiment 5): it avoids zero probabilities for unseen words.",
    ])


if __name__ == "__main__":
    experiment_7()