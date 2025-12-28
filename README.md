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

</details>

## Description

This Python library provides elementary operators, MPI communicators and samplers to facilitate the design of custom distributed Plug-and-Play (PnP) MCMC algorithms for high-dimensional Bayesian inference.
Detailed examples provided focus on the resolution of high-dimensional inverse problems typical in signal and image processing applications.

:warning: **WARNING** This project is under active development, and the API may evolve significantly until version `1.0`.

## Installation

- The package can be installed on `ubuntu` with `cuda` GPU support within an existing `conda`-like Python environment (e.g., `pixi`, `mamba` or `conda`). Example installation commands can be found below.

  ```bash
  # within a mamba environment
  mamba env create -n my_samplers
  mamba install cards -c pthouvenin

  # within a pixi environment
  pixi workspace channel add pthouvenin
  pixi add cards
  ```

- A distributed implementation is provided for a the `DRUNet`, `DnCNN` and `DDFB` deep denoisers.
  Pre-trained weights are not embedded into the `cards` `conda`-package.
  The weights need to be retrieved separately, using for instance the commands detailed below.

  ```bash
  mkdir -p data/weights && cd data/weights

  # * DDFB
  mkdir ddfb && cd ddfb
  wget https://github.com/maxime-bouton/cards/blob/main/data/weights/ddfb/ddfb_nch3_nla20_nfe64.pth

  # from https://github.com/cszn/KAIR
  # https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
  # https://github.com/cszn/DPIR/tree/master/model_zoo
  #
  # * DRUNet (gray and color images)
  cd ../ && mkdir drunet && cd drunet
  wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth && mv drunet_gray.pth drunet_nch1.pth

  wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth && mv drunet_color.pth drunet_nch3.pth

  # * DnCNN (gray and color images)
  cd ../ && mkdir dncnn && cd dncnn
  wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth && mv dncnn_gray_blind.pth dncnn_nch1.pth

  wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth && mv dncnn_color_blind.pth dncnn_nch3.pth
  ```

<!-- # from deepinv
# https://huggingface.co/deepinv/drunet/tree/main
# https://huggingface.co/deepinv/dncnn/tree/main -->

## Contributing

Short guidelines on how to set-up, test and document the library project are detailed below.
See the [online documentation](https://maxime-bouton.github.io/cards/) for futher details.

<details>

<summary>Setup</summary>

### Setup

- Only pull-requests compatible with the [`pixi`](https://pixi.sh/latest/) Python package manager will be considered.

- Clone the project and create a development environment using the command below.

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

Before any commit to the master branch or pull request, verify all tests pass under the different configuration considered (see [`tests/conftest.py`](tests/conftestpy) for further details).

```bash
pixi shell -e full

# display available markers
pytest --markers

# check all tests available
python -m pytest --collect-only

# running all serail test on GPU
python -m pytest --mode serial --device gpu

# running all MPI tests on CPU
mpiexec -n 2 python -m mpi4py -m pytest -m mpi

# running all MPI tests on GPU
mpiexec -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -np 2 python -m pytest --mode mpi --device gpu
```

</details>

## License

The project is licensed under the [GPL-3.0 license](LICENSE).
