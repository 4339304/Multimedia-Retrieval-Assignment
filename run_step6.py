"""
Step 6: evaluation of the retrieval system with the class labels as ground truth.

Run from the project folder (after Step 4 --build; the index of Step 5 is built by itself if needed):
    python run_step6.py               every database shape once as a query, K = 10
    python run_step6.py -k 5          another number of returned shapes

Output: report/data/step6/step6_per_query.csv      one row per query shape
        report/data/step6/step6_per_class.csv      mean values per class
        report/data/step6/step6_summary.csv        grand averages over all queries and over all classes
        report/data/step6/step6_precision_by_k.csv mean precision and recall for K = 1 .. 50

Every one of the database shapes is used once as a query with the K-nearest-neighbor search of Step 5 (HNSW
index, the same distance as Step 4); the query itself is left out of its own result (technical tips 6). A returned
shape is a true positive (TP) if it has the class of the query, otherwise a false positive (FP); the number of false
negatives is the size of the class minus one (the query) minus TP. Main metric: the precision of the K returned
shapes, PPV = TP / K (Module 7). Also stored, for the discussion:
    recall          TP / (c - 1), with c the number of database shapes of the query's class
    first tier      precision of a query of length c - 1 (Module 7), which does not depend on a fixed K
    baseline        expected precision of a random order, (c - 1) / (N - 1)
    duplicates      returned shapes that are copies of the query (almost the same histograms, Section 4.2)
"""
import argparse

import numpy as np
import pandas as pd

import run_step5 as s5
from mr_io import PROJECT_ROOT
from run_step4 import HIST

OUT = PROJECT_ROOT / "report" / "data" / "step6"
K_CURVE = 50
DUPLICATE_LIMIT = 0.02          # mean total-variation difference of the five histograms, as in Section 4.2


def neighbor_lists(X, index, k_max, ef):
    """The k_max nearest database shapes of every database shape, without the shape itself (index search)."""
    index.hnsw.efSearch = max(ef, k_max + 1)
    _, labels = index.search(X, k_max + 1)
    rows = []
    for i, row in enumerate(labels):
        row = [int(j) for j in row if j >= 0 and j != i]
        rows.append(row[:k_max])
    return rows


def exact_lists(X, k_max, block=256):
    """The same lists from an exact search over all vectors, to check the index."""
    sq = (X * X).sum(1)
    rows = []
    for s in range(0, len(X), block):
        d2 = sq[s:s + block, None] + sq[None, :] - 2 * X[s:s + block] @ X.T
        d2[np.arange(len(d2)), np.arange(s, s + len(d2))] = np.inf
        rows.extend(np.argsort(d2, axis=1, kind="stable")[:, :k_max].tolist())
    return rows


def duplicate_pairs(norm):
    """Pairs of shapes whose five histograms are almost the same (copies of one model)."""
    H = norm[sum(HIST.values(), [])].to_numpy(float)
    dup = set()
    for s in range(0, len(H), 256):
        d = np.abs(H[s:s + 256, None, :] - H[None, :, :]).sum(2) / 10      # mean total variation of 5 histograms
        for a, b in zip(*np.nonzero(d < DUPLICATE_LIMIT)):
            if s + a != b:
                dup.add((s + a, b))
    return dup


def evaluate(k=10, ef=100):
    """Query with every database shape, count TP, FP and FN, and save the values per query, per class and in total."""
    norm, params, scales, X, index, meta = s5.load_index(ef)
    X = np.asarray(X, dtype=np.float32)
    classes = norm["class"].astype(str).to_numpy()
    n = len(norm)
    size = pd.Series(classes).value_counts()
    c = size.reindex(classes).to_numpy()                     # class size of every query
    k_max = max(int(c.max()) - 1, K_CURVE, k)

    lists = neighbor_lists(X, index, k_max, ef)
    exact = exact_lists(X, k_max)
    agree_k = np.mean([len(set(a[:k]) & set(e[:k])) / k for a, e in zip(lists, exact)])
    agree_tier = np.mean([len(set(a[:ci - 1]) & set(e[:ci - 1])) / (ci - 1) for a, e, ci in zip(lists, exact, c)])
    dup = duplicate_pairs(norm)

    rows, curve = [], np.zeros((n, K_CURVE))
    for i in range(n):
        hit = classes[lists[i]] == classes[i]
        tp = int(hit[:k].sum())
        tier = int(hit[:c[i] - 1].sum())
        top = lists[i][:k]
        wrong = pd.Series(classes[[j for j in top if classes[j] != classes[i]]])
        copies = [j for j in top if (i, j) in dup]
        clean = [j for j in lists[i] if (i, j) not in dup][:k]
        curve[i] = np.cumsum(hit[:K_CURVE]) / np.arange(1, K_CURVE + 1)
        rows.append({
            "class": classes[i], "file": norm["file"].iloc[i], "class_size": int(c[i]), "k": k,
            "tp": tp, "fp": k - tp, "fn": int(c[i] - 1 - tp),
            "precision": tp / k, "recall": tp / (c[i] - 1),
            "first_tier": tier / (c[i] - 1), "baseline": (c[i] - 1) / (n - 1),
            "duplicates_in_result": len(copies),
            "precision_without_duplicates": float(np.mean(classes[clean] == classes[i])),
            "most_frequent_wrong_class": wrong.value_counts().index[0] if len(wrong) else "",
            "closest_shapes": ";".join(f"{classes[j]}/{norm['file'].iloc[j]}" for j in top),
        })
    q = pd.DataFrame(rows)

    def most_frequent_wrong(g):
        top = pd.Series(";".join(g.closest_shapes).split(";")).str.split("/").str[0]
        top = top[top != g.name]
        return top.value_counts().index[0] if len(top) else ""

    per_class = (q.groupby("class")
                 .agg(shapes=("file", "size"), precision=("precision", "mean"), recall=("recall", "mean"),
                      first_tier=("first_tier", "mean"), baseline=("baseline", "mean"),
                      precision_without_duplicates=("precision_without_duplicates", "mean"),
                      queries_with_duplicates=("duplicates_in_result", lambda s: int((s > 0).sum())))
                 .reset_index())
    per_class["most_frequent_wrong_class"] = per_class["class"].map(q.groupby("class").apply(most_frequent_wrong,
                                                                                               include_groups=False))
    per_class = per_class.sort_values("precision", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame([
        {"average": "over all queries", "queries": n, "k": k,
         **{m: q[m].mean() for m in ["precision", "recall", "first_tier", "baseline", "precision_without_duplicates"]}},
        {"average": "over all classes", "queries": n, "k": k,
         **{m: per_class[m].mean() for m in ["precision", "recall", "first_tier", "baseline",
                                              "precision_without_duplicates"]}},
    ])
    summary["index_agreement_top_k"] = agree_k
    summary["index_agreement_first_tier"] = agree_tier
    recall_curve = np.cumsum(np.array([classes[l[:K_CURVE]] == classes[i] for i, l in enumerate(lists)]), axis=1) \
        / (c[:, None] - 1)
    by_k = pd.DataFrame({"k": np.arange(1, K_CURVE + 1), "precision": curve.mean(0), "recall": recall_curve.mean(0),
                         "baseline": np.mean((c - 1) / (n - 1))})

    OUT.mkdir(parents=True, exist_ok=True)
    q.to_csv(OUT / "step6_per_query.csv", index=False)
    per_class.to_csv(OUT / "step6_per_class.csv", index=False)
    summary.to_csv(OUT / "step6_summary.csv", index=False)
    by_k.to_csv(OUT / "step6_precision_by_k.csv", index=False)

    print(f"Step 6: {n} queries, K = {k}")
    print(summary.round(4).to_string(index=False))
    print("\nbest classes")
    print(per_class.head(5)[["class", "shapes", "precision", "first_tier", "baseline"]].round(3).to_string(index=False))
    print("\nworst classes")
    print(per_class.tail(5)[["class", "shapes", "precision", "first_tier", "baseline", "most_frequent_wrong_class"]]
          .round(3).to_string(index=False))
    print(f"\nsaved in {OUT}")
    return q, per_class, summary


def main():
    ap = argparse.ArgumentParser(description="Step 6: evaluation with every database shape as a query")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=100)
    a = ap.parse_args()
    evaluate(a.k, a.ef)


if __name__ == "__main__":
    main()
