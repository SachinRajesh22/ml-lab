"""
EXPERIMENT 9: ID3 decision tree (from scratch) on Online Retail customers

Run:  python exp09_id3_decision_tree_online_retail.py
Needs: pip install numpy pandas matplotlib scikit-learn openpyxl
Plots are shown and also saved in the folder  lab_outputs/
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, classification_report)
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


# ---------------------------------------------------------------- experiment 9
def entropy(labels):
    """H(S) = - sum p_i * log2(p_i)   (0 = pure node, 1 = 50/50 for two classes)"""
    p = labels.value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())


def information_gain(df, feature, target):
    """IG(S, A) = H(S) - sum_v (|S_v|/|S|) * H(S_v)   -> how much a split on A reduces entropy."""
    weighted = sum(len(sub) / len(df) * entropy(sub[target]) for _, sub in df.groupby(feature))
    return entropy(df[target]) - weighted


def id3(df, features, target, importance, depth=0, max_depth=4, min_samples=30):
    """ID3: always split on the feature with the HIGHEST information gain, recursively.
    Works with CATEGORICAL features. max_depth / min_samples are optional pre-pruning (set
    max_depth=None for the pure, fully grown ID3 tree)."""
    counts = df[target].value_counts()
    majority = counts.idxmax()
    leaf = {"leaf": True, "label": majority, "n": len(df), "dist": counts.to_dict()}
    if len(counts) == 1 or not features or len(df) < min_samples or (max_depth is not None and depth >= max_depth):
        return leaf
    gains = {f: information_gain(df, f, target) for f in features}
    best = max(gains, key=gains.get)
    if gains[best] <= 1e-9:
        return leaf
    importance[best] = importance.get(best, 0.0) + gains[best] * len(df)
    node = {"leaf": False, "feature": best, "gain": gains[best], "majority": majority, "n": len(df), "children": {}}
    rest = [f for f in features if f != best]
    for value, sub in df.groupby(best):
        node["children"][value] = id3(sub, rest, target, importance, depth + 1, max_depth, min_samples)
    return node


def id3_predict(node, row):
    while not node["leaf"]:
        value = row[node["feature"]]
        if value not in node["children"]:        # unseen category -> fall back to majority class
            return node["majority"]
        node = node["children"][value]
    return node["label"]


def print_tree(node, indent=""):
    if node["leaf"]:
        print(f"{indent}=> {node['label']}   (samples={node['n']}, {node['dist']})")
        return
    for value, child in node["children"].items():
        print(f"{indent}if {node['feature']} == {value}   [split gain={node['gain']:.3f}, samples={node['n']}]")
        print_tree(child, indent + "      ")


def draw_tree(tree, ax):
    """Small matplotlib drawing of the ID3 tree (blue = decision node, green = leaf)."""
    counter = [0]

    def layout(node, depth):
        if node["leaf"]:
            x = counter[0]
            counter[0] += 1
        else:
            x = float(np.mean([layout(c, depth + 1) for c in node["children"].values()]))
        node["_x"], node["_y"] = x, -depth
        return x

    def draw(node):
        text = f"{node['label']}\nn={node['n']}" if node["leaf"] else f"{node['feature']}\nIG={node['gain']:.2f}"
        color = "lightgreen" if node["leaf"] else "lightskyblue"
        ax.text(node["_x"], node["_y"], text, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round", fc=color, ec="k"))
        if not node["leaf"]:
            for value, child in node["children"].items():
                ax.plot([node["_x"], child["_x"]], [node["_y"], child["_y"]], "k-", lw=0.8, zorder=0)
                ax.text((node["_x"] + child["_x"]) / 2, (node["_y"] + child["_y"]) / 2, str(value),
                        fontsize=7, color="red", ha="center")
                draw(child)

    layout(tree, 0)
    draw(tree)
    ax.axis("off")


def load_online_retail():
    """Reads the UCI 'Online Retail' file if it is next to this script and builds one row per
    customer (RFM-style summary). Returns None if the file is not found."""
    # Put "Online Retail.xlsx" (UCI) or "OnlineRetail.csv" (Kaggle) in the same folder.
    # (Reading .xlsx needs:  pip install openpyxl)
    raw = None
    if os.path.exists("Online Retail.xlsx"):
        raw = pd.read_excel("Online Retail.xlsx")
    elif os.path.exists("OnlineRetail.csv"):
        raw = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
    # raw = pd.read_csv("your_file.csv", encoding="ISO-8859-1")   # <-- uncomment to use another CSV path
    if raw is None:
        return None
    raw = raw.dropna(subset=["CustomerID"])
    raw = raw[(raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]               # remove returns / bad rows
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    raw["TotalPrice"] = raw["Quantity"] * raw["UnitPrice"]
    snapshot = raw["InvoiceDate"].max() + pd.Timedelta(days=1)
    g = raw.groupby("CustomerID")
    cust = pd.DataFrame({
        "Recency": (snapshot - g["InvoiceDate"].max()).dt.days,             # days since last purchase
        "Frequency": g["InvoiceNo"].nunique(),                              # number of orders
        "Monetary": g["TotalPrice"].sum(),                                  # total money spent
        "AvgItemsPerOrder": g["Quantity"].sum() / g["InvoiceNo"].nunique(),
        "UniqueProducts": g["StockCode"].nunique(),
        "Country": g["Country"].agg(lambda s: s.mode().iloc[0]),
    })
    cust["Country"] = np.where(cust["Country"] == "United Kingdom", "UK", "Other")
    return cust.reset_index(drop=True)


def make_synthetic_customers(n=3000, seed=RANDOM_STATE):
    """SYNTHETIC stand-in with the same columns, used ONLY when the real file is missing."""
    rng = np.random.default_rng(seed)
    freq = rng.poisson(3, n) + 1
    rec = rng.integers(1, 373, n)
    avg_items = rng.gamma(2.0, 60, n) + 5
    uniq = np.clip((freq * rng.integers(5, 25, n) * rng.uniform(0.5, 1.5, n)).astype(int), 1, None)
    money = freq * avg_items * rng.uniform(1, 3, n) * (1 + 0.3 * (rec < 60))
    return pd.DataFrame({"Recency": rec, "Frequency": freq, "Monetary": money, "AvgItemsPerOrder": avg_items,
                         "UniqueProducts": uniq, "Country": rng.choice(["UK", "Other"], n, p=[0.9, 0.1])})


def experiment_9():
    header(9, "Decision Tree (ID3 from scratch) - Customer segmentation, Online Retail")

    step("1. Load the dataset")
    cust = load_online_retail()
    if cust is None:
        print("!! 'Online Retail.xlsx' not found -> using a SYNTHETIC demo dataset (same columns).")
        print("!! Put the real file next to this script to run on real data.")
        cust = make_synthetic_customers()
    print("Customers:", len(cust))
    print(cust.head())

    step("2. Preprocess: ID3 needs CATEGORICAL features -> bin numbers into Low/Medium/High")
    tri = lambda s, labels: pd.qcut(s.rank(method="first"), q=3, labels=labels).astype(str)
    data = pd.DataFrame({
        "Recency": tri(cust["Recency"], ["Recent", "Moderate", "Old"]),
        "Frequency": tri(cust["Frequency"], ["Low", "Medium", "High"]),
        "AvgItemsPerOrder": tri(cust["AvgItemsPerOrder"], ["Small", "Medium", "Large"]),
        "UniqueProducts": tri(cust["UniqueProducts"], ["Few", "Some", "Many"]),
        "Country": cust["Country"].astype(str),
    })
    # TARGET: top 25% spenders = 'High-Value'. (Monetary itself is NOT used as a feature -> no leakage.)
    data["Segment"] = np.where(cust["Monetary"] >= cust["Monetary"].quantile(0.75), "High-Value", "Regular")
    print(data["Segment"].value_counts().to_string())
    features = [c for c in data.columns if c != "Segment"]
    train, test = train_test_split(data, test_size=0.25, stratify=data["Segment"], random_state=RANDOM_STATE)

    step("3. Build the ID3 tree (information gain, max depth 4)")
    importance = {}
    tree = id3(train, features, "Segment", importance, max_depth=4)
    print_tree(tree)

    step("4. Evaluate")
    pred = test.apply(lambda r: id3_predict(tree, r), axis=1)
    base = (test["Segment"] == train["Segment"].mode()[0]).mean()
    print(f"ID3 test accuracy       : {accuracy_score(test['Segment'], pred):.4f}")
    print(f"Baseline (always majority class): {base:.4f}")
    print(classification_report(test["Segment"], pred, digits=3))

    step("5. Feature importance (total information gain, weighted by samples reaching the split)")
    imp = pd.Series(importance).reindex(features).fillna(0)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print(imp.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})
    draw_tree(tree, ax[0])
    ax[0].set_title("ID3 decision tree")
    imp.sort_values().plot.barh(ax=ax[1], title="Feature importance")
    save_show("exp9_id3_tree.png")

    explain([
        "ID3 builds the tree top-down: at each node choose the attribute with the HIGHEST INFORMATION GAIN.",
        "Entropy H = -sum p*log2(p) measures impurity; information gain = entropy before split - weighted entropy after.",
        "Stop when a node is pure, no features are left, or (our pre-pruning) depth/size limits are hit.",
        "ID3 handles only categorical features, so numeric columns were binned (Low/Medium/High).",
        "The attribute at the root is the most informative one; features used higher in the tree matter more.",
        "Each path from root to leaf is a readable rule, e.g. 'High frequency AND many products -> High-Value'.",
        "A fully grown tree overfits; limiting depth (pruning) improves generalisation.",
    ])


if __name__ == "__main__":
    experiment_9()