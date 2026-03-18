# ParamRepulsor

This is the code repository for the NeurIPS 2024 paper "Navigating the Effect of Parametrization for Dimensionality Reduction". Our paper can be found [here](https://openreview.net/pdf?id=eYNYnYle41).

## How to install
This repository can be installed locally via pip by the following command:

```bash
git clone https://github.com/hyhuang00/ParamRepulsor.git
cd ParamRepulsor
pip install .
```

Note: `torch` is a base dependency but is highly platform-dependent.
This project provides optionals:

```bash
pip install .[cpu]    # cpu-only pytorch
pip install .[cu118]  # cuda 118
pip install .[cu121]  # cuda 121
pip install .[cu124]  # cuda 124
pip install .[mps]    # arm64/aarch64 (Apple M-Series chips)
```

This project also supports `uv` (`pip install uv`):

```bash
uv sync (--extra cpu)  # as appropriate for your system
uv run pytest
TORCH_DEVICE=cpu uv run pytest  # disable accelerator
```

### Using a local checkout from other projects

If you're developing against a local copy of ParamRepulsor, you can point other
`uv`-managed projects at it:

```bash
# In your other project's directory:
uv add parampacmap --editable /path/to/ParamRepulsor
```

Or add it manually to your other project's `pyproject.toml`:

```toml
[tool.uv.sources]
parampacmap = { path = "/path/to/ParamRepulsor", editable = true }
```

For plain pip, use:

```bash
pip install -e /path/to/ParamRepulsor
```

## MMAE-Hybrid Autoencoder

This fork adds a hybrid MMAE-PaCMAP autoencoder (`MMAEHybrid`) that combines
[Manifold-Matching Autoencoders](https://arxiv.org/abs/2603.16568) (Cheret et al., 2026)
with PaCMAP-style sparse pair sampling and ParamRepulsor reference geometry.

```python
from parampacmap import MMAEHybrid

ae = MMAEHybrid(
    input_dims=784,
    bottleneck_dims=10,
    ref_method="pca",       # or "paramrepulsor" for nonlinear reference
    lambda_max=1.0,
    num_epochs=300,
    verbose=True,
)
ae.fit(X_train)
Z = ae.encode(X_test)          # structure-preserving embeddings
X_recon = ae.reconstruct(X_test)  # faithful reconstruction
```

Key features:
- **Sparse O(bk) pair sampling** instead of MMAE's dense O(b^2) distance matrices
- **Three-phase training schedule**: global layout → local refinement → reconstruction polish
- **ParamRepulsor reference** for capturing nonlinear manifold geometry
- **Faithful inversion** via decoder with topology-preserving regularization

See `src/parampacmap/mmae_hybrid.py` for implementation and `peer_crystals/notes/hybrid_mmae_pacmap.md` for the full architecture plan.

## How to use our algorithm
ParamPaCMAP/ParamRepulsor is fully scikit-learn compatible, meaning that it can be
used as any other scikit-learn based algorithm.
After the installation, you can use our algorithm by:

```python
import parampacmap

# Initialize the reducer. Notice that by default, the stronger paramrepulsor
# algorithm will be used.
reducer = parampacmap.ParamPaCMAP()
X_low = reducer.fit_transform(X)  # Substitute your data here.
```


## Citation
If you have referred to our research in your publication, or you used the ParamRepulsor/ParamPaCMAP algorithm in this repository, please cite our paper using the following bibtex:

```
@inproceedings{huang2024navigating,
  title={Navigating the Effect of Parametrization for Dimensionality Reduction},
  author={Huang, Haiyang and Wang, Yingfan and Rudin, Cynthia},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```

If you use the MMAE-Hybrid autoencoder, please also cite:

```
@article{cheret2026manifold,
  title={Manifold-Matching Autoencoders},
  author={Cheret, Laurent and L{\'e}tourneau, Vincent and Nejadgholi, Isar and Drummond, Chris and Al Osman, Hussein and Fraser, Maia},
  journal={arXiv preprint arXiv:2603.16568},
  year={2026},
}
```

## Project Contributor
A full list of project contributors can be found [here](CONTRIBUTORS.md).
