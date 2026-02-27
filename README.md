# CARDS: Composable Algorithms for Reproducible Distributed Sampling

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

## Description

This Python library provides elementary operators, MPI communicators and Markov transition kernels to facilitate the design of custom distributed Plug-and-Play (PnP) Markov chain Monte Carlo (MCMC) algorithms for high-dimensional Bayesian inference.
Detailed examples provided in this repository focus on the resolution of high-dimensional inverse problems in image and signal processing.

:warning: **WARNING** This project is under active development, and the API may evolve significantly until version `1.0`.

## Installation

### Environment Setup

To create the environment using the [`pixi`](https://pixi.sh/latest/) package manager, run:

```bash
pixi install 
```

### Pre-trained Weights

A distributed implementation is provided for the DRUNet, DnCNN and DDFB deep denoisers.
Pre-trained weights of DRUNet are not embedded into the package and need to be retrieved separately. You can use the commands below to download them:

```bash
mkdir -p data/weights && cd data/weights

# Retrieving weights for DRUNet from https://github.com/cszn/KAIR
# See https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
# and https://github.com/cszn/DPIR/tree/master/model_zoo)

# DRUNet (gray and color images)
cd ../ && mkdir drunet && cd drunet
wget [https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth) && mv drunet_gray.pth drunet_nch1.pth
wget [https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth) && mv drunet_color.pth drunet_nch3.pth
```

## Examples

```bash
pixi shell

# navigate to the desired application example folder
cd examples/gaussian_inpainting

# serial execution
mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 1 python main_gaussian_inpainting.py --mode=serial-gpu --config=config_180_pnp.json

# mpi execution using 2 workers
mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python main_gaussian_inpainting.py --mode=mpi-gpu --config=config_180_pnp.json
```

## Contributing

Short guidelines on conventions adopted to set-up, test and document the library are detailed below.
See the [online documentation](https://maxime-bouton.github.io/cards/) for further details.

<details>

<summary>Setup</summary>

### Setup

* Only pull-requests compatible with the [`pixi`](https://pixi.sh/latest/) Python package manager will be considered.
* Clone the project and create a development environment using the commands below.

```bash
pixi self-update
pixi clean
pixi clean cache
pixi install
```

</details>

<details>

<summary>Testing</summary>

### Testing

:warning: **WARNING** Testing is currently under complete rewriting after significant changes in the library's API. Some tests may break.

Before any commit or pull request to the master branch, verify all tests pass under the different configurations considered (serial and distributed mode, running on CPU or GPU). See [`tests/conftest.py`](https://www.google.com/search?q=tests/conftest.py) for further details.

```bash
# running tests related to deep denoisers in distributed settings
mpirun -x OMPI_MCA_pml=ucx \
       -x OMPI_MCA_osc=ucx \
       -x OMPI_MCA_opal_cuda_support=true \
       -x UCX_MEMTYPE_CACHE=n \
       -np 2 pytest -C=mpi-gpu tests/operators/test_mpi_denoisers.py

```

</details>

## Citation

If you use this code or rely on our methodology in your research, please cite our paper:

> M. Bouton, P.-A. Thouvenin, A. Repetti, and P. Chainais, "A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems," *IEEE Transactions on Computational Imaging*, vol. 12, pp. 839-849, 2026. [DOI: 10.1109/TCI.2026.3685151](https://doi.org/10.1109/TCI.2026.3685151).

```bibtex
@ARTICLE{11482855,
  author={Bouton, Maxime and Thouvenin, Pierre-Antoine and Repetti, Audrey and Chainais, Pierre},
  journal={IEEE Transactions on Computational Imaging}, 
  title={A Distributed Plug-and-Play MCMC Algorithm for High-Dimensional Inverse Problems}, 
  year={2026},
  volume={12},
  number={},
  pages={839-849},
  doi={10.1109/TCI.2026.3685151}
}
```

## License

The project is licensed under the [GPL-3.0 license](https://www.google.com/search?q=LICENSE).
