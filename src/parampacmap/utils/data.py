"""Data related utility functions — pure PyTorch implementation."""
import numpy as np
import torch


def generate_pair(
    X, n_neighbors, n_MN, n_FP, distance="euclidean", verbose=True,
    random_state=None, device=None,
):
    """Generate pairs for the dataset.

    Args:
        X: Input data array (numpy, float32-castable)
        n_neighbors: Number of neighbors to find
        n_MN: Number of mid-near pairs per point
        n_FP: Number of far pairs per point
        distance: Distance metric ("euclidean", "manhattan", "angular", "hamming")
        verbose: Whether to print progress
        random_state: Random seed for reproducibility
        device: torch device (defaults to CPU)

    Returns:
        pair_neighbors, pair_MN, pair_FP, None
        (None replaces the former annoy tree for backward compat)
    """
    if device is None:
        device = torch.device("cpu")

    n, dim = X.shape
    n_neighbors_extra = min(n_neighbors + 50, n - 1)

    X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)

    # --- kNN via chunked pairwise distance ---
    knn_distances, nbrs = _chunked_knn(X_t, n_neighbors_extra, distance)
    print_verbose("Found nearest neighbor", verbose)

    # --- sigma & scaled distance ---
    sig = torch.clamp(knn_distances[:, 3:6].mean(dim=1), min=1e-10)
    # scaled_dist[i,j] = knn_distances[i,j]^2 / (sig[i] * sig[nbrs[i,j]])
    nbr_sig = sig[nbrs]  # (n, n_neighbors_extra)
    scaled_dist = knn_distances ** 2 / sig[:, None] / nbr_sig
    print_verbose("Found scaled dist", verbose)

    # --- sample neighbor pairs ---
    pair_neighbors = _sample_neighbors_pair(scaled_dist, nbrs, n_neighbors, device)

    # --- sample MN pairs ---
    gen = _make_generator(device, random_state)
    pair_MN = _sample_MN_pair(X_t, n_MN, distance, gen, device)

    # --- sample FP pairs ---
    gen_fp = _make_generator(device, random_state)
    pair_FP = _sample_FP_pair(n, pair_neighbors, n_neighbors, n_FP, gen_fp, device)

    # Convert to numpy int32 for downstream (convert_pairs / FastDataloader)
    pair_neighbors = pair_neighbors.cpu().numpy().astype(np.int32)
    pair_MN = pair_MN.cpu().numpy().astype(np.int32)
    pair_FP = pair_FP.cpu().numpy().astype(np.int32)

    return pair_neighbors, pair_MN, pair_FP, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_generator(device, random_state):
    """Create a torch.Generator, optionally seeded."""
    # torch.Generator for MPS must be on CPU
    gen_device = "cpu" if device.type == "mps" else device
    gen = torch.Generator(device=gen_device)
    if random_state is not None:
        gen.manual_seed(random_state)
    return gen


_CHUNK = 4096  # rows per chunk for pairwise distance


def _pairwise_dist_chunk(X, Y, distance):
    """Compute pairwise distance between rows of X and Y."""
    if distance == "angular":
        Xn = torch.nn.functional.normalize(X, dim=1)
        Yn = torch.nn.functional.normalize(Y, dim=1)
        return torch.cdist(Xn, Yn, p=2)
    p = {"euclidean": 2, "manhattan": 1, "hamming": 0}
    if distance not in p:
        raise NotImplementedError(
            f"Distance '{distance}' not supported. "
            "Use euclidean, manhattan, angular, or hamming."
        )
    return torch.cdist(X, Y, p=p[distance])


def _chunked_knn(X, k, distance):
    """Return (knn_distances, knn_indices) each of shape (n, k).

    Processes in row-chunks of _CHUNK to avoid allocating a full n×n matrix.
    """
    n = X.shape[0]
    all_dists = []
    all_idx = []

    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        # (chunk_size, n)
        dists = _pairwise_dist_chunk(X[start:end], X, distance)
        # Exclude self: set diagonal block to inf
        chunk_size = end - start
        col_indices = torch.arange(start, end, device=X.device)
        dists[torch.arange(chunk_size, device=X.device), col_indices] = float("inf")

        topk_dist, topk_idx = dists.topk(k, largest=False, dim=1)
        all_dists.append(topk_dist)
        all_idx.append(topk_idx)

    return torch.cat(all_dists, dim=0), torch.cat(all_idx, dim=0)


def _sample_neighbors_pair(scaled_dist, nbrs, n_neighbors, device):
    """Pick the n_neighbors with smallest scaled distance per point.

    Returns tensor of shape (n * n_neighbors, 2) as int64.
    """
    n = scaled_dist.shape[0]
    sorted_idx = scaled_dist.argsort(dim=1)[:, :n_neighbors]  # (n, n_neighbors)
    pair_col1 = nbrs.gather(1, sorted_idx)  # (n, n_neighbors)
    pair_col0 = torch.arange(n, device=device).unsqueeze(1).expand_as(pair_col1)
    return torch.stack([pair_col0, pair_col1], dim=-1).reshape(-1, 2)


def _compute_dist_batch(X, idx, distance):
    """Compute distances between X[i] and X[idx[i, j]] for all i, j.

    X: (n, d), idx: (n, m) -> returns (n, m) distances.
    """
    gathered = X[idx]  # (n, m, d)
    base = X.unsqueeze(1).expand_as(gathered)  # (n, m, d)

    if distance == "euclidean":
        return (base - gathered).norm(p=2, dim=-1)
    elif distance == "manhattan":
        return (base - gathered).abs().sum(dim=-1)
    elif distance == "angular":
        base_n = torch.nn.functional.normalize(base, dim=-1)
        gath_n = torch.nn.functional.normalize(gathered, dim=-1)
        cos_sim = (base_n * gath_n).sum(dim=-1).clamp(-1, 1)
        return torch.sqrt(2.0 - 2.0 * cos_sim)
    elif distance == "hamming":
        return (base != gathered).float().sum(dim=-1)
    else:
        raise NotImplementedError(f"Distance '{distance}' not supported.")


def _sample_MN_pair(X, n_MN, distance, gen, device):
    """Sample mid-near pairs: for each point, sample 6, drop nearest, pick 2nd nearest.

    Returns (n * n_MN, 2) int64 tensor.
    """
    n = X.shape[0]
    # For MPS, generate on CPU then move
    gen_device = gen.device if hasattr(gen, 'device') else torch.device('cpu')
    sampled = torch.randint(0, n, (n, n_MN, 6), generator=gen, device=gen_device)
    if sampled.device != X.device:
        sampled = sampled.to(X.device)

    # Compute distances: (n, n_MN, 6)
    sampled_flat = sampled.reshape(n, -1)  # (n, n_MN * 6)
    dists_flat = _compute_dist_batch(X, sampled_flat, distance)  # (n, n_MN * 6)
    dists = dists_flat.reshape(n, n_MN, 6)

    # Drop the nearest (argmin), pick second-nearest from remaining 5
    nearest_idx = dists.argmin(dim=-1, keepdim=True)  # (n, n_MN, 1)

    # Set the nearest to inf so argmin on remaining gives 2nd nearest
    dists_masked = dists.scatter(-1, nearest_idx, float("inf"))
    second_nearest_idx = dists_masked.argmin(dim=-1)  # (n, n_MN)

    # Gather the actual point indices
    picked = sampled.gather(-1, second_nearest_idx.unsqueeze(-1)).squeeze(-1)  # (n, n_MN)

    row_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(picked)
    return torch.stack([row_idx, picked], dim=-1).reshape(-1, 2)


def _sample_FP_pair(n, pair_neighbors, n_neighbors, n_FP, gen, device):
    """Sample far pairs: random indices that aren't in the neighbor set.

    Returns (n * n_FP, 2) int64 tensor.
    """
    gen_device = gen.device if hasattr(gen, 'device') else torch.device('cpu')

    # Build set of neighbor indices per point from pair_neighbors (n*n_neighbors, 2)
    # pair_neighbors[:, 1] reshaped to (n, n_neighbors) gives neighbor ids per point
    neighbor_ids = pair_neighbors[:, 1].reshape(n, n_neighbors)  # (n, n_neighbors)

    # Sample candidates — oversample to handle collisions
    oversample = max(n_FP * 2, n_FP + 20)
    candidates = torch.randint(
        0, n, (n, oversample), generator=gen, device=gen_device
    )
    if candidates.device != device:
        candidates = candidates.to(device)

    # Mask out candidates that match any neighbor
    # neighbor_ids: (n, n_neighbors), candidates: (n, oversample)
    if neighbor_ids.device != device:
        neighbor_ids = neighbor_ids.to(device)
    # (n, oversample, 1) == (n, 1, n_neighbors) -> (n, oversample, n_neighbors)
    is_neighbor = (candidates.unsqueeze(-1) == neighbor_ids.unsqueeze(1)).any(dim=-1)
    # Also mask self-pairs
    self_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(candidates)
    is_self = candidates == self_idx
    is_invalid = is_neighbor | is_self

    # For each row, pick first n_FP valid candidates
    # Set invalid positions to n (out of range) then argsort to push them to end
    order = is_invalid.long()  # 0 for valid, 1 for invalid
    # Stable sort: valid entries come first, in their original order
    sorted_perm = order.argsort(dim=1, stable=True)[:, :n_FP]
    fp_idx = candidates.gather(1, sorted_perm)  # (n, n_FP)

    row_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(fp_idx)
    return torch.stack([row_idx, fp_idx], dim=-1).reshape(-1, 2)


def print_verbose(msg, verbose, **kwargs):
    if verbose:
        print(msg, **kwargs)
