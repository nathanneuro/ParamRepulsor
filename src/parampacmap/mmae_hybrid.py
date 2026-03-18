"""
Hybrid MMAE-PaCMAP Autoencoder.

Combines MMAE's pairwise distance matching (topology preservation via stability
theorem) with PaCMAP-style sparse pair sampling and phase scheduling, using
ParamRepulsor embeddings as the reference geometry.

Key innovations over MMAE:
- Sparse O(bk) pair sampling instead of dense O(b²) distance matrices
- Three-phase training schedule (global→local→reconstruction)
- ParamRepulsor reference for nonlinear manifold geometry
- Faithful reconstruction via decoder with structure-preserving regularization

Usage:
    autoencoder = MMAEHybrid(input_dims=784, bottleneck_dims=10)
    autoencoder.fit(X_train)
    Z = autoencoder.encode(X_test)
    X_recon = autoencoder.decode(Z)
"""

import logging
import math
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import decomposition
from sklearn.base import BaseEstimator

from parampacmap.models import module, TORCH_DEVICE
from parampacmap.utils import data

logger = logging.getLogger(__name__)


# ============================================================================
# Weight Schedules
# ============================================================================

def mmae_hybrid_weight_schedule(epoch: int, total_epochs: int):
    """
    Three-phase weight schedule following the plan:
    Phase 1 (0→0.3T): Global layout — mid-near high, far moderate, near low
    Phase 2 (0.3T→0.8T): Local refinement — near high, mid moderate, far low
    Phase 3 (0.8T→T): Reconstruction polish — near only
    """
    phase1_end = 0.3 * total_epochs
    phase2_end = 0.8 * total_epochs

    if epoch < phase1_end:
        # Phase 1: global layout
        w_nn = 0.1
        w_fp = 0.3   # far pairs (collapse prevention)
        w_mn = 0.6   # mid-near (cluster-to-cluster)
    elif epoch < phase2_end:
        # Phase 2: local refinement (cosine transition)
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        alpha = 0.5 * (1 - math.cos(math.pi * progress))
        w_nn = 0.1 + 0.5 * alpha     # 0.1 → 0.6
        w_fp = 0.3 - 0.2 * alpha     # 0.3 → 0.1
        w_mn = 0.6 - 0.3 * alpha     # 0.6 → 0.3
    else:
        # Phase 3: reconstruction polish
        w_nn = 1.0
        w_fp = 0.0
        w_mn = 0.0

    return np.array([w_nn, w_fp, w_mn])


def mmae_hybrid_lambda_schedule(epoch: int, total_epochs: int,
                                 lambda_max: float = 1.0,
                                 lambda_min: float = 0.01):
    """
    Lambda (structure vs reconstruction tradeoff) schedule.
    High during Phase 1-2, cosine decay to lambda_min in Phase 3.
    """
    phase2_end = 0.8 * total_epochs

    if epoch < phase2_end:
        # Slight cosine decay during phases 1-2
        progress = epoch / phase2_end
        return lambda_max * (0.5 * (1 + math.cos(math.pi * progress * 0.5)))
    else:
        # Phase 3: decay to lambda_min
        progress = (epoch - phase2_end) / (total_epochs - phase2_end)
        alpha = 0.5 * (1 - math.cos(math.pi * progress))
        return lambda_max * (1 - alpha) + lambda_min * alpha


# ============================================================================
# Autoencoder Modules
# ============================================================================

class Encoder(nn.Module):
    """MLP encoder: input_dims → bottleneck_dims."""

    def __init__(self, input_dims: int, bottleneck_dims: int,
                 hidden_dims: list = [256, 128], activation: str = "silu"):
        super().__init__()
        layers = []
        prev = input_dims
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, bottleneck_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """MLP decoder: bottleneck_dims → input_dims."""

    def __init__(self, bottleneck_dims: int, output_dims: int,
                 hidden_dims: list = [128, 256], activation: str = "silu"):
        super().__init__()
        layers = []
        prev = bottleneck_dims
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class MMAELoss(nn.Module):
    """
    Sparse manifold-matching regularization loss.

    Computes MSE on pairwise distances between encoder outputs and reference
    embeddings, using PaCMAP-style sparse pair sampling with phase-scheduled
    weights.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_basis, z_nn, z_fp, z_mn,
                ref_basis, ref_nn, ref_fp, ref_mn,
                w_nn: float, w_fp: float, w_mn: float):
        """
        Args:
            z_*: encoder outputs — basis (B, 1, d), pairs (B, k, d)
            ref_*: reference embeddings — same shapes
            w_*: pair type weights
        """
        loss = torch.tensor(0.0, device=z_basis.device)

        if w_nn > 0 and z_nn.shape[1] > 0:
            d_z = torch.linalg.norm(z_nn - z_basis, dim=-1)      # (B, k_nn)
            d_ref = torch.linalg.norm(ref_nn - ref_basis, dim=-1)
            loss = loss + w_nn * F.mse_loss(d_z, d_ref)

        if w_fp > 0 and z_fp.shape[1] > 0:
            d_z = torch.linalg.norm(z_fp - z_basis, dim=-1)
            d_ref = torch.linalg.norm(ref_fp - ref_basis, dim=-1)
            loss = loss + w_fp * F.mse_loss(d_z, d_ref)

        if w_mn > 0 and z_mn.shape[1] > 0:
            d_z = torch.linalg.norm(z_mn - z_basis, dim=-1)
            d_ref = torch.linalg.norm(ref_mn - ref_basis, dim=-1)
            loss = loss + w_mn * F.mse_loss(d_z, d_ref)

        return loss


# ============================================================================
# Dataset for Autoencoder with Pair Sampling
# ============================================================================

class AEPairDataset(torch.utils.data.Dataset):
    """
    Dataset that returns (point, nn_indices, fp_indices, mn_indices, ref_embedding).
    Pairs are precomputed; far pairs are resampled each epoch.
    """

    def __init__(self, X: np.ndarray, ref_embeddings: np.ndarray,
                 nn_pairs: np.ndarray, mn_pairs: np.ndarray, n_FP: int,
                 dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.ref = torch.tensor(ref_embeddings, dtype=dtype)
        self.nn_pairs = torch.tensor(nn_pairs, dtype=torch.long)  # (N, k_nn)
        self.mn_pairs = torch.tensor(mn_pairs, dtype=torch.long)  # (N, k_mn)
        self.n_FP = n_FP
        self.N = len(X)
        self._resample_far()

    def _resample_far(self):
        """Resample far pairs uniformly."""
        self.fp_pairs = torch.randint(0, self.N, (self.N, self.n_FP))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.ref[idx],
            self.nn_pairs[idx],
            self.fp_pairs[idx],
            self.mn_pairs[idx],
        )


# ============================================================================
# Main Class
# ============================================================================

class MMAEHybrid(BaseEstimator):
    """
    Hybrid MMAE-PaCMAP Autoencoder.

    Structure-preserving autoencoder that uses sparse pair sampling for
    manifold-matching regularization with a three-phase training schedule.
    """

    def __init__(
        self,
        input_dims: int = 100,
        bottleneck_dims: int = 10,
        encoder_hidden: list = [256, 128],
        decoder_hidden: list = [128, 256],
        activation: str = "silu",
        # Reference
        ref_dims: int = 0,          # ParamRepulsor dim (0 = 2× bottleneck)
        ref_method: str = "pca",    # "pca" or "paramrepulsor"
        pca_variance: float = 0.8,
        # Pair sampling
        n_neighbors: int = 10,
        n_MN: int = 5,
        n_FP: int = 10,
        # Training
        lambda_max: float = 1.0,
        lambda_min: float = 0.01,
        num_epochs: int = 300,
        batch_size: int = 512,
        lr: float = 1e-3,
        # Misc
        verbose: bool = False,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.bottleneck_dims = bottleneck_dims
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.activation = activation
        self.ref_dims = ref_dims if ref_dims > 0 else 2 * bottleneck_dims
        self.ref_method = ref_method
        self.pca_variance = pca_variance
        self.n_neighbors = n_neighbors
        self.n_MN = n_MN
        self.n_FP = n_FP
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.seed = seed
        self.dtype = dtype
        self.device = TORCH_DEVICE

        self.encoder = None
        self.decoder = None
        self._ref_embeddings = None

        if seed is not None:
            torch.manual_seed(seed)

    def _build_models(self):
        self.encoder = Encoder(
            self.input_dims, self.bottleneck_dims,
            self.encoder_hidden, self.activation,
        ).to(self.device).to(self.dtype)
        self.decoder = Decoder(
            self.bottleneck_dims, self.input_dims,
            self.decoder_hidden, self.activation,
        ).to(self.device).to(self.dtype)
        self.mmae_loss = MMAELoss().to(self.device)

    def _compute_reference(self, X: np.ndarray) -> np.ndarray:
        """Compute reference embeddings for manifold-matching."""
        if self.ref_method == "paramrepulsor":
            from parampacmap.parampacmap import ParamPaCMAP
            pr = ParamPaCMAP(
                n_components=self.ref_dims,
                num_epochs=200,
                verbose=self.verbose,
            )
            ref = pr.fit_transform(X)
            if isinstance(ref, tuple):
                ref = ref[0]
            return ref
        else:
            # PCA at specified variance
            pca = decomposition.PCA(n_components=self.pca_variance, svd_solver='full')
            ref = pca.fit_transform(X)
            # Truncate to ref_dims if PCA gives more
            if ref.shape[1] > self.ref_dims:
                ref = ref[:, :self.ref_dims]
            return ref

    def fit(self, X: np.ndarray):
        """
        Fit the autoencoder with structure-preserving regularization.
        """
        t0 = time.perf_counter()
        N = X.shape[0]
        self.input_dims = X.shape[1]

        # Build models
        self._build_models()

        # Compute reference embedding
        if self.verbose:
            print("Computing reference embedding...")
        self._ref_embeddings = self._compute_reference(X)
        ref = self._ref_embeddings

        # Compute pairs using ParamRepulsor/PaCMAP pair generation
        if self.verbose:
            print("Generating pairs...")
        pair_neighbors, pair_MN, pair_FP, _ = data.generate_pair(
            X if X.shape[1] <= 100 else decomposition.PCA(100).fit_transform(X),
            n_neighbors=self.n_neighbors,
            n_MN=self.n_MN,
            n_FP=self.n_FP,
            distance="euclidean",
            verbose=False,
            device=self.device,
        )

        # Convert pairs to index arrays
        nn_idx = pair_neighbors[:, 1].reshape(N, -1)  # (N, k_nn)
        mn_idx = pair_MN[:, 1].reshape(N, -1)         # (N, k_mn)

        # Build dataset
        dataset = AEPairDataset(X, ref, nn_idx, mn_idx, self.n_FP, dtype=self.dtype)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=0,
        )

        # Optimizer
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )

        if self.verbose:
            print(f"Training for {self.num_epochs} epochs, "
                  f"N={N}, batch={self.batch_size}, "
                  f"ref={self.ref_method} (dim={ref.shape[1]})")

        # Training loop
        for epoch in range(self.num_epochs):
            # Resample far pairs each epoch
            if epoch > 0:
                dataset._resample_far()

            # Get schedule values
            weights = mmae_hybrid_weight_schedule(epoch, self.num_epochs)
            w_nn, w_fp, w_mn = weights
            lam = mmae_hybrid_lambda_schedule(
                epoch, self.num_epochs, self.lambda_max, self.lambda_min)

            epoch_recon_loss = 0
            epoch_struct_loss = 0
            n_batches = 0

            for batch in loader:
                x_batch, ref_batch, nn_idx_b, fp_idx_b, mn_idx_b = batch
                x_batch = x_batch.to(self.device)
                ref_batch = ref_batch.to(self.device)

                # Encode
                z = self.encoder(x_batch)

                # Decode (reconstruction)
                x_recon = self.decoder(z)
                recon_loss = F.mse_loss(x_recon, x_batch)

                # Structure loss: encode pair points, compute distance matching
                # Gather pair data and reference embeddings
                all_ref = torch.tensor(ref, dtype=self.dtype, device=self.device)
                all_X = torch.tensor(X, dtype=self.dtype, device=self.device)

                # Encode pair points
                with torch.no_grad():
                    # Get pair embeddings by indexing into full dataset
                    nn_x = all_X[nn_idx_b.reshape(-1)].reshape(
                        x_batch.shape[0], self.n_neighbors, -1)
                    fp_x = all_X[fp_idx_b.reshape(-1)].reshape(
                        x_batch.shape[0], self.n_FP, -1)
                    mn_x = all_X[mn_idx_b.reshape(-1)].reshape(
                        x_batch.shape[0], self.n_MN, -1)

                # Encode pairs (with grad for structural alignment)
                z_nn = self.encoder(nn_x.reshape(-1, self.input_dims)).reshape(
                    x_batch.shape[0], self.n_neighbors, -1)
                z_fp = self.encoder(fp_x.reshape(-1, self.input_dims)).reshape(
                    x_batch.shape[0], self.n_FP, -1)
                z_mn = self.encoder(mn_x.reshape(-1, self.input_dims)).reshape(
                    x_batch.shape[0], self.n_MN, -1)

                # Reference distances
                ref_nn = all_ref[nn_idx_b.reshape(-1)].reshape(
                    x_batch.shape[0], self.n_neighbors, -1)
                ref_fp = all_ref[fp_idx_b.reshape(-1)].reshape(
                    x_batch.shape[0], self.n_FP, -1)
                ref_mn = all_ref[mn_idx_b.reshape(-1)].reshape(
                    x_batch.shape[0], self.n_MN, -1)

                z_basis = z.unsqueeze(1)
                ref_basis = ref_batch.unsqueeze(1)

                struct_loss = self.mmae_loss(
                    z_basis, z_nn, z_fp, z_mn,
                    ref_basis, ref_nn, ref_fp, ref_mn,
                    w_nn, w_fp, w_mn,
                )

                # Combined loss
                loss = recon_loss + lam * struct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_recon_loss += recon_loss.item()
                epoch_struct_loss += struct_loss.item()
                n_batches += 1

            if self.verbose and (epoch % 20 == 0 or epoch == self.num_epochs - 1):
                avg_r = epoch_recon_loss / max(n_batches, 1)
                avg_s = epoch_struct_loss / max(n_batches, 1)
                phase = 1 if epoch < 0.3 * self.num_epochs else (
                    2 if epoch < 0.8 * self.num_epochs else 3)
                print(f"Epoch {epoch+1:4d}/{self.num_epochs} | "
                      f"recon={avg_r:.6f} struct={avg_s:.6f} | "
                      f"λ={lam:.4f} phase={phase} | "
                      f"w=[{w_nn:.2f},{w_fp:.2f},{w_mn:.2f}]")

        dt = time.perf_counter() - t0
        if self.verbose:
            print(f"Training complete in {dt:.1f}s")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to bottleneck space."""
        self.encoder.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=self.dtype, device=self.device)
            # Process in chunks to avoid OOM
            results = []
            for i in range(0, len(X_t), self.batch_size * 2):
                chunk = X_t[i:i + self.batch_size * 2]
                results.append(self.encoder(chunk).cpu())
            return torch.cat(results).numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode from bottleneck space to input space."""
        self.decoder.eval()
        with torch.no_grad():
            Z_t = torch.tensor(Z, dtype=self.dtype, device=self.device)
            results = []
            for i in range(0, len(Z_t), self.batch_size * 2):
                chunk = Z_t[i:i + self.batch_size * 2]
                results.append(self.decoder(chunk).cpu())
            return torch.cat(results).numpy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return bottleneck embeddings."""
        self.fit(X)
        return self.encode(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full encode → decode reconstruction."""
        Z = self.encode(X)
        return self.decode(Z)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error."""
        X_recon = self.reconstruct(X)
        return np.mean((X - X_recon) ** 2)
