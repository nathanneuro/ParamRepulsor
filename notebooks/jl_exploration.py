"""
Johnson-Lindenstrauss meets ParamRepulsor
==========================================

How do random projections, classical methods, and learned nonlinear maps
compare at preserving geometry when compressing high-dimensional data?

We measure TWO complementary things:
  1. Distance distortion: |d_proj/d_orig - 1| (what JL guarantees)
  2. Neighborhood preservation: what fraction of k-NN survive the projection

Methods compared:
  - JL Random Projection (random Gaussian matrix, scaled)
  - PCA / Truncated SVD
  - t-SNE (van der Maaten & Hinton)
  - UMAP (McInnes et al.)
  - Learned Linear (ParamRepulsor with no hidden layers)
  - ParamRepulsor (full 3-layer MLP)

Experiments:
  1. All methods at fixed target_dim=10: distance distortion + kNN preservation
  2. Distance distortion vs target dimension (JL, PCA, ParamRepulsor)
  3. Distortion distribution histograms
  4. Initialization study: PCA init vs random/JL-like init for ParamRepulsor

Run: python notebooks/jl_exploration.py
"""

import os
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import decomposition, datasets, manifold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TORCH_DEVICE", "cpu")

from parampacmap import ParamPaCMAP

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "jl_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Whether UMAP is available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, skipping UMAP")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_distances_flat(X):
    """Pairwise Euclidean distances, returned as flat upper-triangle vector."""
    from scipy.spatial.distance import pdist
    return pdist(X, metric="euclidean")


def distance_distortion(d_orig, d_proj):
    """Relative distance error stats: |d_proj/d_orig - 1|."""
    mask = d_orig > 1e-12
    ratios = d_proj[mask] / d_orig[mask]
    rel = np.abs(ratios - 1.0)
    return {
        "mean": float(np.mean(rel)),
        "median": float(np.median(rel)),
        "q95": float(np.percentile(rel, 95)),
        "max": float(np.max(rel)),
        "std": float(np.std(rel)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios)),
    }


def knn_preservation(X_orig, X_proj, k=10):
    """Fraction of k-nearest-neighbors preserved after projection."""
    nn_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_orig)
    nn_proj = NearestNeighbors(n_neighbors=k + 1).fit(X_proj)
    # +1 because the point itself is included
    idx_orig = nn_orig.kneighbors(X_orig, return_distance=False)[:, 1:]
    idx_proj = nn_proj.kneighbors(X_proj, return_distance=False)[:, 1:]

    preserved = 0
    N = X_orig.shape[0]
    for i in range(N):
        preserved += len(set(idx_orig[i]) & set(idx_proj[i]))
    return preserved / (N * k)


# ---------------------------------------------------------------------------
# Projection Methods
# ---------------------------------------------------------------------------

def jl_project(X, dim, seed=SEED):
    """JL random Gaussian projection: X @ (randn / sqrt(dim))."""
    rng = np.random.RandomState(seed)
    A = rng.randn(X.shape[1], dim) / np.sqrt(dim)
    return X @ A


def pca_project(X, dim):
    pca = decomposition.PCA(n_components=dim, random_state=SEED)
    return pca.fit_transform(X)


def svd_project(X, dim):
    """Truncated SVD (same as PCA when data is centered, but works on raw data)."""
    svd = decomposition.TruncatedSVD(n_components=dim, random_state=SEED)
    return svd.fit_transform(X)


def tsne_project(X, dim):
    """t-SNE projection. Note: stochastic, non-parametric, no transform()."""
    # barnes_hut only supports n_components <= 3; fall back to exact for higher dims
    method = "barnes_hut" if dim <= 3 else "exact"
    ts = manifold.TSNE(n_components=dim, random_state=SEED, perplexity=30,
                       init="pca", max_iter=500, method=method)
    return ts.fit_transform(X)


def umap_project(X, dim):
    if not HAS_UMAP:
        return None
    reducer = umap.UMAP(n_components=dim, random_state=SEED, n_neighbors=15,
                        min_dist=0.1)
    return reducer.fit_transform(X)


def learned_linear_project(X, dim, epochs=50):
    """ParamRepulsor with zero hidden layers = learned linear map."""
    model = ParamPaCMAP(
        n_components=dim,
        model_dict={"backbone": "ANN", "layer_size": []},
        num_epochs=epochs, seed=SEED, num_workers=0,
        apply_pca=False,
    )
    return model.fit_transform(X)


def paramrepulsor_project(X, dim, epochs=50):
    """Full ParamRepulsor (3-layer MLP)."""
    model = ParamPaCMAP(
        n_components=dim, num_epochs=epochs, seed=SEED, num_workers=0,
        apply_pca=(X.shape[1] > 100),
    )
    return model.fit_transform(X)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_datasets():
    ds = OrderedDict()

    # Gaussian full-rank in 100D
    ds["gaussian_100d"] = np.random.randn(800, 100)

    # Low-rank (rank 10 embedded in 784D)
    rng = np.random.RandomState(SEED)
    U = rng.randn(800, 10)
    V = rng.randn(10, 200)
    ds["low_rank_200d"] = U @ V + rng.randn(800, 200) * 0.1

    # Swiss roll embedded in 100D
    sr, _ = datasets.make_swiss_roll(n_samples=800, random_state=SEED)
    sr100 = np.zeros((800, 100))
    sr100[:, :3] = sr
    sr100[:, 3:] = rng.randn(800, 97) * 0.01
    ds["swiss_roll_100d"] = sr100

    # Clustered blobs in 200D
    from sklearn.datasets import make_blobs
    blobs, _ = make_blobs(n_samples=800, n_features=200, centers=8, random_state=SEED)
    ds["blobs_200d"] = blobs

    return ds


# ---------------------------------------------------------------------------
# Experiment 1: All methods head-to-head at target_dim=10
# ---------------------------------------------------------------------------

def experiment_1_headtohead(datasets_dict, target_dim=10, paramrep_epochs=50):
    """Compare all methods on distance distortion + kNN preservation."""
    methods = OrderedDict([
        ("JL Random", lambda X, d: jl_project(X, d)),
        ("PCA", lambda X, d: pca_project(X, d)),
        ("SVD", lambda X, d: svd_project(X, d)),
        ("t-SNE", lambda X, d: tsne_project(X, d)),
        ("UMAP", lambda X, d: umap_project(X, d)),
        ("Learned Linear", lambda X, d: learned_linear_project(X, d, epochs=paramrep_epochs)),
        ("ParamRepulsor", lambda X, d: paramrepulsor_project(X, d, epochs=paramrep_epochs)),
    ])

    all_results = {}
    for ds_name, X in datasets_dict.items():
        print(f"\n--- {ds_name} (N={X.shape[0]}, d={X.shape[1]}) -> {target_dim}D ---")
        d_orig = pairwise_distances_flat(X)
        ds_results = {}

        for mname, mfn in methods.items():
            print(f"  {mname} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                X_proj = mfn(X, target_dim)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            if X_proj is None:
                print("skipped (not installed)")
                continue
            elapsed = time.time() - t0

            d_proj = pairwise_distances_flat(X_proj)
            dist_stats = distance_distortion(d_orig, d_proj)
            knn_k10 = knn_preservation(X, X_proj, k=10)
            knn_k50 = knn_preservation(X, X_proj, k=50)

            ds_results[mname] = {
                **dist_stats,
                "knn_10": knn_k10,
                "knn_50": knn_k50,
                "time": elapsed,
            }
            print(f"dist_err={dist_stats['mean']:.4f}  "
                  f"kNN@10={knn_k10:.3f}  kNN@50={knn_k50:.3f}  "
                  f"({elapsed:.1f}s)")

        all_results[ds_name] = ds_results

    return all_results


def plot_experiment_1(all_results, target_dim):
    """Two-panel plot per dataset: distance distortion (bar) + kNN preservation (bar)."""
    n_ds = len(all_results)
    fig, axes = plt.subplots(2, n_ds, figsize=(5 * n_ds, 8), squeeze=False)

    palette = {
        "JL Random": "#e74c3c",
        "PCA": "#3498db",
        "SVD": "#2980b9",
        "t-SNE": "#9b59b6",
        "UMAP": "#e67e22",
        "Learned Linear": "#f1c40f",
        "ParamRepulsor": "#2ecc71",
    }

    for col, (ds_name, ds_results) in enumerate(all_results.items()):
        methods = list(ds_results.keys())
        x = np.arange(len(methods))
        colors = [palette.get(m, "#7f8c8d") for m in methods]

        # Top: distance distortion
        ax = axes[0, col]
        means = [ds_results[m]["mean"] for m in methods]
        q95s = [ds_results[m]["q95"] for m in methods]
        ax.bar(x - 0.18, means, 0.35, color=colors, alpha=0.85, label="Mean" if col == 0 else "")
        ax.bar(x + 0.18, q95s, 0.35, color=colors, alpha=0.4, label="95th pctl" if col == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Relative Distance Error")
        ax.set_title(f"{ds_name}")
        ax.grid(True, alpha=0.3, axis="y")
        if col == 0:
            ax.legend(fontsize=7)

        # Bottom: kNN preservation
        ax = axes[1, col]
        knn10 = [ds_results[m]["knn_10"] for m in methods]
        knn50 = [ds_results[m]["knn_50"] for m in methods]
        ax.bar(x - 0.18, knn10, 0.35, color=colors, alpha=0.85, label="k=10" if col == 0 else "")
        ax.bar(x + 0.18, knn50, 0.35, color=colors, alpha=0.4, label="k=50" if col == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("kNN Preservation")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Experiment 1: All Methods Head-to-Head (target_dim={target_dim})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp1_headtohead.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: Distance distortion vs target dimension
# ---------------------------------------------------------------------------

def experiment_2_dim_sweep(X, name, paramrep_epochs=50):
    """Sweep target dims for fast methods + ParamRepulsor."""
    N, d = X.shape
    jl_theory_dim = max(2, int(np.ceil(8 * np.log(N))))
    target_dims = sorted(set([2, 3, 5, 10, 20, 50, min(jl_theory_dim, d - 1)]))
    target_dims = [t for t in target_dims if t < d]

    d_orig = pairwise_distances_flat(X)
    results = {"dims": target_dims, "jl_theory_dim": jl_theory_dim, "methods": OrderedDict()}

    fast_methods = [
        ("JL Random", jl_project),
        ("PCA", pca_project),
    ]
    slow_methods = [
        ("ParamRepulsor", paramrepulsor_project),
    ]

    for mname, mfn in fast_methods:
        res = []
        for td in target_dims:
            X_proj = mfn(X, td)
            stats = distance_distortion(d_orig, pairwise_distances_flat(X_proj))
            knn = knn_preservation(X, X_proj, k=10)
            res.append({**stats, "knn_10": knn})
            print(f"  {mname} -> {td}D: err={stats['mean']:.4f} knn={knn:.3f}")
        results["methods"][mname] = res

    for mname, mfn in slow_methods:
        res = []
        for td in target_dims:
            print(f"  {mname} -> {td}D ...", end=" ", flush=True)
            t0 = time.time()
            X_proj = mfn(X, td, epochs=paramrep_epochs)
            elapsed = time.time() - t0
            stats = distance_distortion(d_orig, pairwise_distances_flat(X_proj))
            knn = knn_preservation(X, X_proj, k=10)
            res.append({**stats, "knn_10": knn})
            print(f"err={stats['mean']:.4f} knn={knn:.3f} ({elapsed:.1f}s)")
        results["methods"][mname] = res

    return results


def plot_experiment_2(all_sweep_results):
    n = len(all_sweep_results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9), squeeze=False)

    colors = {"JL Random": "#e74c3c", "PCA": "#3498db", "ParamRepulsor": "#2ecc71"}
    markers = {"JL Random": "o", "PCA": "s", "ParamRepulsor": "D"}

    for col, (ds_name, results) in enumerate(all_sweep_results.items()):
        dims = results["dims"]

        # Top: distance distortion
        ax = axes[0, col]
        for mname, mres in results["methods"].items():
            means = [r["mean"] for r in mres]
            ax.plot(dims, means, marker=markers.get(mname, "x"),
                    color=colors.get(mname, "gray"), label=mname, linewidth=2)
        ax.axvline(results["jl_theory_dim"], color="gray", ls=":", alpha=0.5,
                   label=f"JL dim={results['jl_theory_dim']}")
        ax.set_xlabel("Target Dimension")
        ax.set_ylabel("Mean Relative Distance Error")
        ax.set_title(f"{ds_name}: Distance Distortion")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: kNN preservation
        ax = axes[1, col]
        for mname, mres in results["methods"].items():
            knns = [r["knn_10"] for r in mres]
            ax.plot(dims, knns, marker=markers.get(mname, "x"),
                    color=colors.get(mname, "gray"), label=mname, linewidth=2)
        ax.axvline(results["jl_theory_dim"], color="gray", ls=":", alpha=0.5,
                   label=f"JL dim={results['jl_theory_dim']}")
        ax.set_xlabel("Target Dimension")
        ax.set_ylabel("kNN@10 Preservation")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{ds_name}: Neighborhood Preservation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 2: Distortion & kNN vs. Target Dimension", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp2_dim_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: Distortion distribution histograms
# ---------------------------------------------------------------------------

def experiment_3_distortion_histograms(X, name, target_dim=10, paramrep_epochs=50):
    d_orig = pairwise_distances_flat(X)

    projections = OrderedDict([
        ("JL Random", jl_project(X, target_dim)),
        ("PCA", pca_project(X, target_dim)),
        ("ParamRepulsor", paramrepulsor_project(X, target_dim, epochs=paramrep_epochs)),
    ])
    if HAS_UMAP:
        projections["UMAP"] = umap_project(X, target_dim)

    ncols = len(projections)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), sharey=True)
    palette = {"JL Random": "#e74c3c", "PCA": "#3498db",
               "ParamRepulsor": "#2ecc71", "UMAP": "#e67e22"}

    for ax, (mname, X_proj) in zip(axes, projections.items()):
        d_proj = pairwise_distances_flat(X_proj)
        mask = d_orig > 1e-12
        ratios = d_proj[mask] / d_orig[mask]
        ax.hist(ratios, bins=80, density=True, color=palette.get(mname, "gray"),
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(1.0, color="black", ls="--", alpha=0.5)
        ax.set_xlabel("d_proj / d_orig")
        ax.set_title(f"{mname}\nmean={np.mean(ratios):.3f} std={np.std(ratios):.3f}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density")
    fig.suptitle(f"Experiment 3: Distortion Distribution ({name} -> {target_dim}D)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"exp3_dist_hist_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 4: Initialization — PCA vs random (JL-like) for ParamRepulsor
# ---------------------------------------------------------------------------

def experiment_4_init_study(X, name, target_dim=10, num_epochs=100):
    """Track distortion over training with different inits."""
    d_orig = pairwise_distances_flat(X)
    d = X.shape[1]

    snapshot_epochs = list(range(0, num_epochs, 5))
    inits = OrderedDict([
        ("PCA init", "pca"),
        ("Random init (JL-like)", "random"),
    ])

    results = {}
    for init_label, init_mode in inits.items():
        print(f"  {init_label} ...", end=" ", flush=True)
        t0 = time.time()
        model = ParamPaCMAP(
            n_components=target_dim, num_epochs=num_epochs, seed=SEED,
            num_workers=0, apply_pca=(d > 100),
            intermediate_snapshots=snapshot_epochs,
            embedding_init=init_mode,
        )
        output = model.fit_transform(X)
        elapsed = time.time() - t0

        if isinstance(output, tuple):
            final, intermediates = output
        else:
            final, intermediates = output, []

        curve = []
        for emb in intermediates:
            stats = distance_distortion(d_orig, pairwise_distances_flat(emb))
            curve.append(stats["mean"])

        final_stats = distance_distortion(d_orig, pairwise_distances_flat(final))
        final_knn = knn_preservation(X, final, k=10)

        results[init_label] = {
            "curve": curve,
            "epochs": snapshot_epochs[:len(curve)],
            "final_dist_err": final_stats["mean"],
            "final_knn": final_knn,
        }
        print(f"dist_err={final_stats['mean']:.4f} knn={final_knn:.3f} ({elapsed:.1f}s)")

    # JL baseline
    X_jl = jl_project(X, target_dim)
    jl_stats = distance_distortion(d_orig, pairwise_distances_flat(X_jl))
    jl_knn = knn_preservation(X, X_jl, k=10)
    results["JL baseline"] = {
        "curve": [],
        "epochs": [],
        "final_dist_err": jl_stats["mean"],
        "final_knn": jl_knn,
    }
    print(f"  JL baseline: dist_err={jl_stats['mean']:.4f} knn={jl_knn:.3f}")

    return results


def plot_experiment_4(all_init_results):
    n = len(all_init_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    colors = {"PCA init": "#3498db", "Random init (JL-like)": "#e74c3c",
              "JL baseline": "#95a5a6"}

    for col, (ds_name, results) in enumerate(all_init_results.items()):
        ax = axes[0, col]
        for label, res in results.items():
            if res["curve"]:
                ax.plot(res["epochs"], res["curve"], label=label,
                        color=colors.get(label, "gray"), linewidth=2)
            else:
                ax.axhline(res["final_dist_err"], color=colors.get(label, "gray"),
                           ls="--", label=label, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Distance Distortion")
        ax.set_title(f"{ds_name} -> 10D")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 4: Initialization Study (distance distortion over training)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exp4_init_study.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(exp1_results):
    """Print a markdown-style summary table."""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE: All Methods x Datasets (target_dim=10)")
    print("=" * 90)
    header = f"{'Method':<20} {'Dataset':<20} {'Dist Err':>10} {'kNN@10':>8} {'kNN@50':>8} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for ds_name, ds_results in exp1_results.items():
        for mname, stats in ds_results.items():
            print(f"{mname:<20} {ds_name:<20} {stats['mean']:>10.4f} "
                  f"{stats['knn_10']:>8.3f} {stats['knn_50']:>8.3f} "
                  f"{stats['time']:>6.1f}s")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Johnson-Lindenstrauss meets ParamRepulsor")
    print("=" * 70)

    all_datasets = make_datasets()

    # --- Experiment 1: Head-to-head ---
    print("\n### Experiment 1: Head-to-Head (all methods, target_dim=10) ###")
    exp1_results = experiment_1_headtohead(all_datasets, target_dim=10, paramrep_epochs=50)
    plot_experiment_1(exp1_results, target_dim=10)
    print_summary_table(exp1_results)

    # --- Experiment 2: Dim sweep (fast methods + ParamRepulsor) ---
    print("\n### Experiment 2: Distortion vs Target Dimension ###")
    sweep_datasets = {k: all_datasets[k] for k in ["gaussian_100d", "low_rank_200d"]}
    sweep_results = {}
    for ds_name, X in sweep_datasets.items():
        print(f"\nDataset: {ds_name}")
        sweep_results[ds_name] = experiment_2_dim_sweep(X, ds_name, paramrep_epochs=50)
    plot_experiment_2(sweep_results)

    # --- Experiment 3: Distortion histograms ---
    print("\n### Experiment 3: Distortion Distribution Histograms ###")
    for ds_name in ["gaussian_100d", "swiss_roll_100d"]:
        print(f"\nDataset: {ds_name}")
        experiment_3_distortion_histograms(all_datasets[ds_name], ds_name,
                                           target_dim=10, paramrep_epochs=50)

    # --- Experiment 4: Initialization study ---
    print("\n### Experiment 4: Initialization Study ###")
    init_results = {}
    for ds_name in ["gaussian_100d", "low_rank_200d"]:
        print(f"\nDataset: {ds_name}")
        init_results[ds_name] = experiment_4_init_study(
            all_datasets[ds_name], ds_name, target_dim=10, num_epochs=100)
    plot_experiment_4(init_results)

    print("\n" + "=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
