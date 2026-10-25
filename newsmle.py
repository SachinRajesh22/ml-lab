"""
EXPERIMENT 5: MLE and MAP for a multinomial distribution (20 Newsgroups)

Run:  python exp05_multinomial_mle_map_newsgroups.py
Needs: pip install numpy pandas matplotlib scikit-learn
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

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


# ---------------------------------------------------------------- experiment 5
def mle_multinomial(counts):
    """MLE of the word probabilities:  theta_i = n_i / N"""
    return counts / counts.sum()


def map_multinomial(counts, alpha):
    """MAP with a Dirichlet(alpha) prior:  theta_i = (n_i + alpha_i - 1) / (N + sum(alpha) - V)
    alpha can be a single number (same prior for every word) or a vector. alpha = 1 gives MLE."""
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), counts.shape)
    return (counts + alpha - 1) / (counts.sum() + alpha.sum() - len(counts))


def avg_loglik(theta, test_counts):
    """Average log-probability per word of held-out text (higher is better)."""
    seen = test_counts > 0
    with np.errstate(divide="ignore"):
        logs = np.log(theta[seen])
    return float((test_counts[seen] * logs).sum() / test_counts.sum())


def experiment_5():
    header(5, "MLE and MAP for a Multinomial distribution - 20 Newsgroups")

    step("1. Load the dataset (4 categories, headers/footers/quotes removed)")
    cats = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
    kw = dict(categories=cats, remove=("headers", "footers", "quotes"))
    train, test = fetch_20newsgroups(subset="train", **kw), fetch_20newsgroups(subset="test", **kw)
    # ---- CSV alternative: a file with columns 'text' and 'label' ----
    # df = pd.read_csv("newsgroups.csv"); train_texts, train_labels = df["text"], df["label"]
    print("Train docs:", len(train.data), "| test docs:", len(test.data))

    step("2. Preprocess: bag-of-words counts (3000 most frequent non-stop-words)")
    vec = CountVectorizer(stop_words="english", max_features=3000)
    X_all = vec.fit_transform(train.data)
    vocab = np.array(vec.get_feature_names_out())
    V = len(vocab)
    topic = train.target_names.index("sci.space")
    # We deliberately use only 40 training documents of the topic, so that many words are never seen:
    train_idx = np.where(train.target == topic)[0][:40]
    counts = np.asarray(X_all[train_idx].sum(axis=0)).ravel()
    test_counts = np.asarray(vec.transform(test.data)[test.target == topic].sum(axis=0)).ravel()
    p_global = np.asarray(X_all.sum(axis=0)).ravel() / X_all.sum()
    print(f"Topic = sci.space | vocabulary V = {V} | training words N = {int(counts.sum())} | "
          f"words never seen in training = {int(np.sum(counts == 0))}")

    step("3. MLE of the multinomial parameters theta (word probabilities)")
    theta_mle = mle_multinomial(counts)

    step("4. MAP with different Dirichlet priors")
    priors = [("MLE (alpha=1)", 1.0),
              ("MAP uniform alpha=1.1", 1.1),
              ("MAP uniform alpha=2", 2.0),
              ("MAP uniform alpha=10", 10.0),
              ("MAP informative (pseudo-counts ~ global word freq, strength 500)", 1 + 500 * p_global),
              ("MAP informative (strength 5000)", 1 + 5000 * p_global)]
    thetas, rows = {}, []
    for name, a in priors:
        th = mle_multinomial(counts) if name.startswith("MLE") else map_multinomial(counts, a)
        thetas[name] = th
        ll = avg_loglik(th, test_counts)
        rows.append({"Prior": name, "Words with prob=0": int(np.sum(th == 0)),
                     "Test log-lik / word": ll,
                     "Perplexity": np.exp(-ll) if np.isfinite(ll) else np.inf})

    step("5. Compare results")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    unseen = np.where((counts == 0) & (test_counts > 0))[0]
    top_unseen = unseen[np.argsort(-test_counts[unseen])[:8]]
    print(f"\n{len(unseen)} words appear in the TEST text but never in training. Examples:", list(vocab[top_unseen]))
    print("MLE gives them probability 0 -> the test text gets probability 0 (log-likelihood = -infinity).")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    names = [r["Prior"][:22] for r in rows]
    vals = [r["Test log-lik / word"] if np.isfinite(r["Test log-lik / word"]) else 0 for r in rows]
    ax[0].bar(range(len(rows)), vals, color=["red"] + ["steelblue"] * (len(rows) - 1))
    ax[0].set_xticks(range(len(rows)))
    ax[0].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax[0].set(ylabel="avg log-likelihood per test word", title="Effect of prior (red = MLE: -infinity)")
    top = np.argsort(-counts)[:10]
    w = 0.27
    for i, key in enumerate(["MLE (alpha=1)", "MAP uniform alpha=10", "MAP informative (strength 5000)"]):
        ax[1].bar(np.arange(10) + (i - 1) * w, thetas[key][top], w, label=key[:30])
    ax[1].set_xticks(range(10))
    ax[1].set_xticklabels(vocab[top], rotation=45)
    ax[1].set(ylabel="theta (word probability)", title="Top-10 training words")
    ax[1].legend(fontsize=8)
    save_show("exp5_mle_map_multinomial.png")

    explain([
        "A topic's words follow a MULTINOMIAL distribution with parameters theta (one probability per word).",
        "MLE:  theta_i = count_i / total_words  -> a word never seen in training gets probability 0.",
        "Dirichlet is the conjugate prior of the multinomial; its alpha acts like PSEUDO-COUNTS added to the data.",
        "MAP:  theta_i = (n_i + alpha_i - 1) / (N + sum(alpha) - V).  alpha = 1 is a flat prior = same as MLE.",
        "Larger alpha -> more smoothing (rare words boosted, frequent words reduced); too large washes out the data.",
        "An informative prior (global word frequencies) smooths in a smarter way than a uniform prior.",
        "Naive Bayes 'Laplace smoothing' is exactly this idea.",
    ])


if __name__ == "__main__":
    experiment_5()