"""Step 5 t-SNE 2D embedding with interactive Plotly hover."""
import argparse, time
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from step5_common import STEP5_DIR, EMBEDDING_PATH, PLOT_PATH, load_step4_data, vectorize_table, ensure_step5_dir

def main():
    p=argparse.ArgumentParser(description="Step 5: t-SNE shape visualization")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a=p.parse_args(); ensure_step5_dir()
    norm, params, scales = load_step4_data(); X=vectorize_table(norm, scales)
    if not 0 < a.perplexity < len(X): raise ValueError(f"perplexity must be between 0 and {len(X)}")
    print(f"Running t-SNE on {len(X):,} shapes with {X.shape[1]} feature dimensions...")
    t0=time.perf_counter()
    model=TSNE(n_components=2, perplexity=a.perplexity, init="pca", learning_rate="auto", max_iter=a.iterations, random_state=a.seed, method="barnes_hut")
    xy=model.fit_transform(X); elapsed=time.perf_counter()-t0
    emb=pd.DataFrame({"x":xy[:,0], "y":xy[:,1], "class":norm["class"].astype(str), "file":norm["file"].astype(str)})
    emb["shape"]=emb["class"]+"/"+emb["file"]
    emb.to_csv(EMBEDDING_PATH,index=False)
    fig=px.scatter(emb,x="x",y="y",color="class",hover_name="shape",hover_data={"class":True,"file":True,"x":False,"y":False,"shape":False},title="3D Shape Feature Database — t-SNE 2D Embedding",render_mode="webgl")
    fig.update_traces(marker={"size":6,"opacity":0.8})
    fig.update_layout(template="plotly_white",legend_title_text="Shape class",xaxis_title="t-SNE dimension 1",yaxis_title="t-SNE dimension 2")
    fig.write_html(str(PLOT_PATH),include_plotlyjs=True,full_html=True)
    print(f"t-SNE completed in {elapsed:.1f} seconds.")
    print(f"Embedding CSV: {EMBEDDING_PATH}")
    print(f"Interactive plot: {PLOT_PATH}")
    print("Open the HTML in a browser and hover over a point to see its class and filename.")
if __name__ == "__main__": main()
