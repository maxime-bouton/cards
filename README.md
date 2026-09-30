# CARDS: Composable Algorithms for Reproducible Distributed Sampling

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

<details>
<summary>Table of content</summary>

## Table of content

- [CARDS: Composable Algorithms for Reproducible Distributed Sampling](#cards-composable-algorithms-for-reproducible-distributed-sampling)
  - [Table of content](#table-of-content)
  - [Description](#description)
  - [Installation (Using as a library)](#installation-using-as-a-library)
  - [Running Examples \& Tutorials](#running-examples--tutorials)
  - [Contributing](#contributing)
    - [Setup](#setup)
    - [Testing](#testing)
  - [Citation](#citation)
  - [License](#license)

</details>

## Description

This Python library provides elementary operators, MPI communicators and Markov transition kernels to facilitate the design of custom distributed Plug-and-Play (PnP) Markov chain Monte Carlo (MCMC) algorithms for high-dimensional Bayesian inference. Detailed examples provided in this repository focus on the resolution of high-dimensional inverse problems in image and signal processing.

## Installation (Using as a library)

If you only want to use `cards` in your own projects, you can install it on `ubuntu` with `cuda` GPU support within an existing `conda`-compatible environment.

<details>
<summary><b>New to Pixi ?</b></summary>
<a href="https://pixi.sh/">Pixi</a> is a fast, cross-platform package manager for the Conda ecosystem (which uses <code>uv</code> under the hood for lightning-fast Python installations).<br> You can <a href="https://pixi.sh/latest/#installation">install it in seconds</a> or simply fall back to <code>conda</code> or <code>mamba</code>.
</details>

```bash
# installation within a pixi environment (recommended)
pixi workspace channel add pthouvenin
pixi add cards

# installation within a mamba environment
mamba env create -n my_samplers
mamba install cards -c pthouvenin
```

*Note: The package installation does not include the tutorials, examples, or pre-trained network weights. To run the examples, see the section below.*

## Running Examples & Tutorials

To run the provided tutorials and examples, you need to clone this repository, set up the local environment, and download the required weights and datasets.

**1. Clone the repository and set up the environment**

```bash
git clone https://github.com/maxime-bouton/cards.git
cd cards

# Install and activate the full environment using pixi
pixi install --environment full
pixi shell --environment full
```

**2. Download pre-trained weights**

A distributed implementation is provided for the `DRUNet`, `DnCNN` and `DDFB` deep denoisers. Run the following commands from the root of the cloned repository to download them:

```bash
mkdir -p data/weights && cd data/weights

# DDFB
mkdir ddfb && cd ddfb
wget https://github.com/maxime-bouton/cards/blob/main/data/weights/ddfb/ddfb_nch3_nla20_nfe64.pth

# DRUNet (gray and color images)
cd ../ && mkdir drunet && cd drunet
wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth && mv drunet_gray.pth drunet_nch1.pth
wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth && mv drunet_color.pth drunet_nch3.pth

# DnCNN (gray and color images)
cd ../ && mkdir dncnn && cd dncnn
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth && mv dncnn_gray_blind.pth dncnn_nch1.pth
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth && mv dncnn_color_blind.pth dncnn_nch3.pth

cd ../../
```

**3. Prepare the data (if required)**

Some examples require generating `.h5` files from the raw images first.

```bash
cd data
for img in raw/*.jpg; do python convert.py "$img"; done
mv raw/*.h5 .
cd ..
```

**4. Run the examples**

Once the data and weights are ready, you can run the launcher from the project root:

```bash
cd examples
python launcher.py --run
```

*Note: To run different subsets of experiments, you currently need to comment/uncomment the settings variables inside `examples/launcher.py`.*

## Contributing

Short guidelines on conventions adopted to set up and test the library are detailed below.

### Setup

Only pull-requests compatible with the [`pixi`](https://pixi.sh/latest/) Python package manager will be considered.

```bash
pixi self-update
pixi install --environment full
pixi shell --environment full
```

### Testing

Before any commit or pull request to the master branch, verify all tests pass under the different configurations considered (serial and distributed mode, running on CPU or GPU). See the [test configuration file](tests/conftest.py) for further details.

```bash
# display available markers
pytest --markers

# list all tests available
python -m pytest --collect-only

# running all serial tests on CPU/GPU
python -m pytest --mode serial --device cpu
python -m pytest --mode serial --device gpu

# running all MPI tests on CPU/GPU
mpirun -n 2 pytest --mode mpi --device cpu
mpirun -x OMPI_MCA_pml=ucx -x OMPI_MCA_osc=ucx -x OMPI_MCA_opal_cuda_support=true -x UCX_MEMTYPE_CACHE=n -n 2 pytest --mode mpi --device gpu
```

## Citation

If you reuse this code, please cite the [associated paper](https://ieeexplore.ieee.org/document/11482855).

```bib
@article{Bouton2026,
  arxivid      = {2511.00870},
  author       = {Maxime Bouton and Pierre-Antoine Thouvenin and Audrey Repetti and Pierre Chainais},
  code         = {[https://github.com/maxime-bouton/cards](https://github.com/maxime-bouton/cards)},
  date         = {2026-04},
  doi          = {10.1109/TCI.2026.3685151},
  hal_id       = {hal-05326314},
  hal_version  = {v1},
  journaltitle = {{IEEE Trans. Comput. Imag.}},
  month        = apr,
  number       = {},
  title        = {A Distributed {P}lug-and-{P}lay {MCMC} Algorithm for High-Dimensional Inverse Problems},
  url          = {[https://hal.science/hal-05326314](https://hal.science/hal-05326314)},
  pages        = {839-849},
  volume       = {12},
}
```

## License

The project is licensed under the [GPL-3.0 license](LICENSE).