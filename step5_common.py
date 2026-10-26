from pathlib import Path
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
    norm, params, scales = s4.load_database()
    return norm.reset_index(drop=True), params, scales

def vectorize_table(norm, scales):
    blocks = []
    for col in s4.GLOBALS:
        scale = float(scales[col])
        if not np.isfinite(scale) or scale <= 0: scale = 1.0
        x = pd.to_numeric(norm[col], errors="coerce").to_numpy(float) / scale
        blocks.append(np.where(np.isfinite(x), x, 0.0)[:, None])
    for name in s4.NAMES:
        scale = float(scales[name])
        if not np.isfinite(scale) or scale <= 0: scale = 1.0
        h = norm[s4.HIST[name]].to_numpy(dtype=np.float32)
        blocks.append(np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0) / scale)
    X = np.hstack(blocks).astype(np.float32)
    if not np.isfinite(X).all(): raise ValueError("Feature vectors contain non-finite values.")
    return X

def query_vector(features, params, scales):
    qdf = features.to_frame().T.copy()
    if "sphericity" not in qdf and "compactness_sphere" in qdf:
        qdf["sphericity"] = qdf["compactness_sphere"]
    for name in s4.NAMES:
        for i in range(20):
            col = f"{name}_bin_{i:02d}"
            if col not in qdf: qdf[col] = np.nan
    qdf = s4.apply_normalization(qdf, params)
    qdf = qdf[["class", "file"] + s4.GLOBALS + sum(s4.HIST.values(), [])]
    return vectorize_table(qdf, scales)[0]

def ensure_step5_dir():
    STEP5_DIR.mkdir(parents=True, exist_ok=True)
