"""Tests for the MMAEHybrid autoencoder."""

import numpy as np
import pytest
import torch

from parampacmap.mmae_hybrid import (
    MMAEHybrid,
    MMAELoss,
    Encoder,
    Decoder,
    AEPairDataset,
    mmae_hybrid_weight_schedule,
    mmae_hybrid_lambda_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    np.random.seed(42)
    return np.random.randn(200, 50).astype(np.float32)


@pytest.fixture
def ae():
    return MMAEHybrid(
        input_dims=50, bottleneck_dims=5,
        encoder_hidden=[32, 16], decoder_hidden=[16, 32],
        num_epochs=3, batch_size=64, verbose=False,
        ref_method="pca", seed=42,
    )


# ---------------------------------------------------------------------------
# Weight & Lambda Schedules
# ---------------------------------------------------------------------------

class TestSchedules:
    def test_weight_phase1(self):
        w = mmae_hybrid_weight_schedule(0, 100)
        assert w[2] > w[0]  # mid-near > near in phase 1

    def test_weight_phase3(self):
        w = mmae_hybrid_weight_schedule(90, 100)
        assert w[0] == 1.0  # near only
        assert w[1] == 0.0
        assert w[2] == 0.0

    def test_weight_transition_smooth(self):
        # Weights should change smoothly across phase boundaries
        w_before = mmae_hybrid_weight_schedule(29, 100)
        w_at = mmae_hybrid_weight_schedule(30, 100)
        w_after = mmae_hybrid_weight_schedule(31, 100)
        for i in range(3):
            assert abs(w_at[i] - w_before[i]) < 0.15  # no huge jumps

    def test_lambda_decreases(self):
        l_early = mmae_hybrid_lambda_schedule(0, 100, 1.0, 0.01)
        l_late = mmae_hybrid_lambda_schedule(99, 100, 1.0, 0.01)
        assert l_early > l_late

    def test_lambda_min_floor(self):
        l = mmae_hybrid_lambda_schedule(99, 100, 1.0, 0.01)
        assert l >= 0.01


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class TestEncoderDecoder:
    def test_encoder_shape(self):
        enc = Encoder(50, 5, [32, 16])
        x = torch.randn(10, 50)
        z = enc(x)
        assert z.shape == (10, 5)

    def test_decoder_shape(self):
        dec = Decoder(5, 50, [16, 32])
        z = torch.randn(10, 5)
        x = dec(z)
        assert x.shape == (10, 50)

    def test_roundtrip_shape(self):
        enc = Encoder(50, 5, [32])
        dec = Decoder(5, 50, [32])
        x = torch.randn(8, 50)
        assert dec(enc(x)).shape == x.shape

    def test_gradient_flow(self):
        enc = Encoder(50, 5, [32])
        dec = Decoder(5, 50, [32])
        x = torch.randn(8, 50, requires_grad=True)
        loss = (dec(enc(x)) - x).pow(2).mean()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# MMAELoss
# ---------------------------------------------------------------------------

class TestMMAELoss:
    def test_zero_weights(self):
        loss_fn = MMAELoss()
        z_basis = torch.randn(4, 1, 5)
        z_nn = torch.randn(4, 3, 5)
        ref_basis = torch.randn(4, 1, 5)
        ref_nn = torch.randn(4, 3, 5)
        empty = torch.zeros(4, 0, 5)
        loss = loss_fn(z_basis, z_nn, empty, empty,
                       ref_basis, ref_nn, empty, empty,
                       w_nn=0.0, w_fp=0.0, w_mn=0.0)
        assert loss.item() == 0.0

    def test_nonzero_loss(self):
        loss_fn = MMAELoss()
        z_basis = torch.randn(4, 1, 5)
        z_nn = torch.randn(4, 3, 5)
        ref_basis = torch.randn(4, 1, 5)
        ref_nn = torch.randn(4, 3, 5)
        empty = torch.zeros(4, 0, 5)
        loss = loss_fn(z_basis, z_nn, empty, empty,
                       ref_basis, ref_nn, empty, empty,
                       w_nn=1.0, w_fp=0.0, w_mn=0.0)
        assert loss.item() > 0

    def test_perfect_match(self):
        loss_fn = MMAELoss()
        z_basis = torch.randn(4, 1, 5)
        z_nn = torch.randn(4, 3, 5)
        empty = torch.zeros(4, 0, 5)
        # Same embeddings = same distances = zero loss
        loss = loss_fn(z_basis, z_nn, empty, empty,
                       z_basis, z_nn, empty, empty,
                       w_nn=1.0, w_fp=0.0, w_mn=0.0)
        assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# AEPairDataset
# ---------------------------------------------------------------------------

class TestAEPairDataset:
    def test_dataset_length(self):
        X = np.random.randn(100, 10).astype(np.float32)
        ref = np.random.randn(100, 5).astype(np.float32)
        nn_pairs = np.random.randint(0, 100, (100, 6))
        mn_pairs = np.random.randint(0, 100, (100, 3))
        ds = AEPairDataset(X, ref, nn_pairs, mn_pairs, n_FP=4)
        assert len(ds) == 100

    def test_dataset_item_shapes(self):
        X = np.random.randn(100, 10).astype(np.float32)
        ref = np.random.randn(100, 5).astype(np.float32)
        nn_pairs = np.random.randint(0, 100, (100, 6))
        mn_pairs = np.random.randint(0, 100, (100, 3))
        ds = AEPairDataset(X, ref, nn_pairs, mn_pairs, n_FP=4)
        x, r, nn, fp, mn = ds[0]
        assert x.shape == (10,)
        assert r.shape == (5,)
        assert nn.shape == (6,)
        assert fp.shape == (4,)
        assert mn.shape == (3,)

    def test_resample_far(self):
        X = np.random.randn(100, 10).astype(np.float32)
        ref = np.random.randn(100, 5).astype(np.float32)
        nn_pairs = np.random.randint(0, 100, (100, 6))
        mn_pairs = np.random.randint(0, 100, (100, 3))
        ds = AEPairDataset(X, ref, nn_pairs, mn_pairs, n_FP=4)
        fp_before = ds.fp_pairs.clone()
        ds._resample_far()
        # Resampled far pairs should differ (with high probability)
        assert not torch.equal(fp_before, ds.fp_pairs)


# ---------------------------------------------------------------------------
# MMAEHybrid End-to-End
# ---------------------------------------------------------------------------

class TestMMAEHybrid:
    def test_fit(self, ae, small_data):
        ae.fit(small_data)
        assert ae.encoder is not None
        assert ae.decoder is not None

    def test_encode_shape(self, ae, small_data):
        ae.fit(small_data)
        Z = ae.encode(small_data)
        assert Z.shape == (200, 5)

    def test_decode_shape(self, ae, small_data):
        ae.fit(small_data)
        Z = ae.encode(small_data)
        X_r = ae.decode(Z)
        assert X_r.shape == small_data.shape

    def test_reconstruct(self, ae, small_data):
        ae.fit(small_data)
        X_r = ae.reconstruct(small_data)
        assert X_r.shape == small_data.shape

    def test_reconstruction_error_decreases(self, small_data):
        # More epochs should give better reconstruction
        ae_short = MMAEHybrid(
            input_dims=50, bottleneck_dims=10,
            encoder_hidden=[32], decoder_hidden=[32],
            num_epochs=2, batch_size=64, verbose=False,
            ref_method="pca", seed=42,
        )
        ae_long = MMAEHybrid(
            input_dims=50, bottleneck_dims=10,
            encoder_hidden=[32], decoder_hidden=[32],
            num_epochs=20, batch_size=64, verbose=False,
            ref_method="pca", seed=42,
        )
        ae_short.fit(small_data)
        ae_long.fit(small_data)
        err_short = ae_short.reconstruction_error(small_data)
        err_long = ae_long.reconstruction_error(small_data)
        assert err_long < err_short

    def test_fit_transform(self, ae, small_data):
        Z = ae.fit_transform(small_data)
        assert Z.shape == (200, 5)

    def test_reconstruction_error_method(self, ae, small_data):
        ae.fit(small_data)
        err = ae.reconstruction_error(small_data)
        assert isinstance(err, (float, np.floating))
        assert err > 0

    def test_different_ref_methods(self, small_data):
        for method in ["pca"]:
            ae = MMAEHybrid(
                input_dims=50, bottleneck_dims=5,
                encoder_hidden=[16], decoder_hidden=[16],
                num_epochs=2, batch_size=64, verbose=False,
                ref_method=method, seed=42,
            )
            ae.fit(small_data)
            Z = ae.encode(small_data)
            assert Z.shape == (200, 5)

    def test_lambda_affects_structure(self, small_data):
        # High lambda should prioritize structure over reconstruction
        ae_high = MMAEHybrid(
            input_dims=50, bottleneck_dims=5,
            encoder_hidden=[32], decoder_hidden=[32],
            num_epochs=10, batch_size=64, verbose=False,
            ref_method="pca", lambda_max=10.0, seed=42,
        )
        ae_low = MMAEHybrid(
            input_dims=50, bottleneck_dims=5,
            encoder_hidden=[32], decoder_hidden=[32],
            num_epochs=10, batch_size=64, verbose=False,
            ref_method="pca", lambda_max=0.01, seed=42,
        )
        ae_high.fit(small_data)
        ae_low.fit(small_data)
        # Low lambda should have better reconstruction
        assert ae_low.reconstruction_error(small_data) < ae_high.reconstruction_error(small_data)
