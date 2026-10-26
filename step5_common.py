"""
Step 5: helpers shared by the nearest-neighbour search (run_step5.py) and the t-SNE map (run_step5_tsne.py).

Every shape becomes one vector of 106 values: the six normalized global descriptors and the five histograms
of Step 4, each divided by the spread rho_g of its distances. The Euclidean distance between two such vectors
is then the same as the weighted distance of Step 4 (run_step4.combined_distance).
"""
import numpy as np
import pandas as pd

import run_step4 as s4

STEP5_DIR = s4.OUT / "step5"
INDEX_PATH = STEP5_DIR / "ann_index.bin"
META_PATH = STEP5_DIR / "ann_metadata.csv"
VECTORS_PATH = STEP5_DIR / "ann_vectors.npy"
EMBEDDING_PATH = STEP5_DIR / "tsne_embedding.csv"
PLOT_PATH = STEP5_DIR / "tsne_shapes.html"


def load_step4_data():
    """Normalized feature table, normalization parameters and distance scales of Step 4."""
    norm, params, scales = s4.load_database()
    return norm.reset_index(drop=True), params, scales


def scale_of(scales, name):
    """Spread of the distances of one descriptor; 1 if it cannot be used."""
    s = float(scales[name])
    return s if np.isfinite(s) and s > 0 else 1.0


def vectorize_table(norm, scales):
    """(n, 106) array: every descriptor of every shape divided by the spread of its distances."""
    blocks = [norm[c].to_numpy(float)[:, None] / scale_of(scales, c) for c in s4.GLOBALS]
    blocks += [norm[s4.HIST[n]].to_numpy(np.float32) / scale_of(scales, n) for n in s4.NAMES]
    X = np.hstack(blocks).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("feature vectors contain values that are not finite")
    return X


def query_vector(features, params, scales):
    """Vector of a query shape (a row of features, not normalized), built like the database vectors."""
    q = features.to_frame().T.copy()
    if "sphericity" not in q and "compactness_sphere" in q:      # name of the column in an older table
        q["sphericity"] = q["compactness_sphere"]
    for n in s4.NAMES:
        for c in s4.HIST[n]:
            if c not in q:
                q[c] = np.nan
    q = s4.apply_normalization(q, params)
    q = q[["class", "file"] + s4.GLOBALS + sum(s4.HIST.values(), [])]
    return vectorize_table(q, scales)[0]


def ensure_step5_dir():
    STEP5_DIR.mkdir(parents=True, exist_ok=True)
