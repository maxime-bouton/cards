# CARDS: Composable Algorithms for Reproducible Distributed Sampling

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

[![pipeline status](https://gitlab.cristal.univ-lille.fr/parallelmcmc/cards/badges/master/pipeline.svg)](https://gitlab.cristal.univ-lille.fr/parallelmcmc/cards/commits/master)

<!-- [![conda](https://img.shields.io/conda/:variant/:channel/:packageName)](...) -->

<details>
<summary>Table of content</summary>

## Table of content

- [CARDS: Composable Algorithms for Reproducible Distributed Sampling](#cards-composable-algorithms-for-reproducible-distributed-sampling)
  - [Table of content](#table-of-content)
  - [Description](#description)
  - [Installation](#installation)
  - [Contributing](#contributing)
    - [Setup](#setup)
    - [Testing](#testing)
  - [License](#license)
  - [Citation](#citation)

</details>

## Description

This Python library provides elementary operators, MPI communicators and Markov transition kernels to facilitate the design of custom distributed Plug-and-Play (PnP) Markov chain Monte Carlo (MCMC) algorithms for high-dimensional Bayesian inference.
Detailed examples provided in this repository focus on the resolution of high-dimensional inverse problems in image and signal processing.

:warning: **WARNING** This project is under active development, and the API may evolve significantly until version `1.0`.

## Installation

- The `cards` Python package can be installed on `ubuntu` with `cuda` GPU support within an existing `conda`-compatible environment (e.g., using either `pixi`, `mamba` or `conda`). Example installation commands can be found below.

  ```bash
  # installation within a mamba environment
  mamba env create -n my_samplers
  mamba install cards -c pthouvenin

  # installation within a pixi environment
  pixi workspace channel add pthouvenin
  pixi add cards
  ```

- A distributed implementation is provided for the `DRUNet`, `DnCNN` and `DDFB` deep denoisers.
  Pre-trained weights are not embedded into the `cards` `conda`-package.
  The weights need to be retrieved separately, using for instance the commands detailed below.

  ```bash
  mkdir -p data/weights && cd data/weights

  # * DDFB
  mkdir ddfb && cd ddfb
  wget https://github.com/maxime-bouton/cards/blob/main/data/weights/ddfb/ddfb_nch3_nla20_nfe64.pth

  # * retrieving weights for DRUNet and DnCNN from https://github.com/cszn/KAIR
  # (see https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
  # and https://github.com/cszn/DPIR/tree/master/model_zoo)
  #
  # DRUNet (gray and color images)
  cd ../ && mkdir drunet && cd drunet
  wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth && mv drunet_gray.pth drunet_nch1.pth

  wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth && mv drunet_color.pth drunet_nch3.pth

  # DnCNN (gray and color images)
  cd ../ && mkdir dncnn && cd dncnn
  wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth && mv dncnn_gray_blind.pth dncnn_nch1.pth

  wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth && mv dncnn_color_blind.pth dncnn_nch3.pth
  ```

<!-- # from deepinv
# https://huggingface.co/deepinv/drunet/tree/main
# https://huggingface.co/deepinv/dncnn/tree/main -->

## Contributing

Short guidelines on conventions adopted to set-up, test and document the library are detailed below.
See the [online documentation](https://maxime-bouton.github.io/cards/) for further details.

<details>

<summary>Setup</summary>

### Setup

- Only pull-requests compatible with the [`pixi`](https://pixi.sh/latest/) Python package manager will be considered.

- Clone the project and create a development environment using the commands below.

```bash
pixi self-update
pixi clean
pixi clean cache
pixi install --environment full
pixi shell --environment full
# eval "$(pixi shell-hook --environment full)"
```

</details>

<details>

<summary>Testing</summary>

### Testing

Before any commit or pull request to the master branch, verify all tests pass under the different configuration considered (serial and distirbuted mode, running on CPU or GPU). See [`tests/conftest.py`](tests/conftestpy) for further details.

```bash
pixi shell -e full

# display available markers
pytest --markers

# list all tests available
python -m pytest --collect-only

# running all serial tests on CPU
python -m pytest --mode serial --device cpu

# running all serial tests on GPU
python -m pytest --mode serial --device gpu

# running all MPI tests on CPU
mpiexec -n 2 python -m mpi4py -m pytest --mode mpi --device cpu

# running all MPI tests on GPU
mpiexec -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m pytest --mode mpi --device gpu
```

</details>

## License

The project is licensed under the [GPL-3.0 license](LICENSE).

## Citation

If you reuse this code, please cite the [associated paper](<>).

```bib
@article{Bouton2026,
  arxivid      = {2511.00870},
  author       = {Maxime Bouton and Pierre-Antoine Thouvenin and Audrey Repetti and Pierre Chainais},
  code         = {https://github.com/maxime-bouton/cards},
  date         = {2026-04},
  eprinttype   = {arxiv},
  journaltitle = {{IEEE Trans. Comput. Imag.}},
  month        = apr,
  note         = {to appear},
  title        = {A Distributed {P}lug-and-{P}lay {MCMC} Algorithm for High-Dimensional Inverse Problems},
  url          = {https://hal.science/hal-05326314},
}
```
