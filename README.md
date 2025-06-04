# Python-MCMC

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Setup

To create an environment from the provided `pyproject.toml` or `.yml` file on a computer running with an `ubuntu` OS, follow the instructions below depending on the `python` package manager used

<details>

<summary>pixi</summary>

### pixi

Starting from a project with `pyproject.toml` and `pixi.lock` available

```bash
pixi clean
pixi clean cache
pixi install --manifest-path pyproject.toml
```

### Manual setup (`pixi`)

- Starting from an existing `.yml` file (tested on `epeautre`)

```bash
pixi self-update
pixi init --import zeus_linux_x86_64_environment.yml
pixi install
conda develop src

# fusing pixi.toml into a pyproject.toml
pixi init --format pyproject
# copy conda dependencies from pixi.toml into  pyproject.toml under
# [tool.pixi.dependencies]
# copy pypi dependencies from pixi.toml into  pyproject.toml under
# [tool.pixi.pypi-dependencies]

# install in editable mode
# FIXME: this should be accounted for in the pyproject.toml, see
# minimal-project = { path = "./minimal-project", editable = true}
conda develop src
```

- From scratch, this could be done by

```bash
# initalizating pixi env
pixi init . --format pyproject

# update name of environment
# https://pixi.sh/latest/workspace/multi_environment/#real-world-example-use-cases
pixi workspace name set mcmc
pixi workspace list

# adding supported target platforms
pixi workspace platform add 'linux-64' 'osx-64'
# or
# pixi add --platform win-64 posix

# adding conda packages
pixi add numba numpy openmpi ucx cuda cudatoolkit cupy pytorch torchvision
pixi add mpi4py openmpi "h5py>=2.9=mpi*" scipy imageio tqdm conda-build matplotlib

pixi add scikit-image --pypi
pixi add pytest isort coverage pre-commit furo sphinx sphinx_rtd_theme sphinxcontrib-bibtex sphinx-autoapi sphinx-design
pixi add sphinxcontrib-apa sphinx_copybutton docstr-coverage genbadge wily --pypi

conda develop src

# open environment and closing
pixi shell
exit

# updating packages versions
pixi update

# run a command from the environment
# https://pixi.sh/latest/workspace/environment/#cleaning-up
pixi run python
```

</details>

<details>

<summary>mamba</summary>

### `mamba`

To crete an environment from the provided `.yml`file, issue the following command in a terminal.

```bash
mamba env create --name mcmc --file zeus_linux_x86_64_environment.yml
conda develop src
```

### Manual setup (`mamba`, ubtuntu)

Instructions tested on `epeautre` and the [HPC computer grid of the University of Lille](https://hpc-doc.univ-lille.fr/docs/accueil/).

```bash
# https://stackoverflow.com/questions/62359175/pytorch-says-that-cuda-is-not-available-on-ubuntu
mamba create --name mcmc numba numpy openmpi ucx cuda cudatoolkit cupy pytorch torchvision -c pytorch -c conda-forge -c nvidia
mamba activate mcmc

# mamba install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
# mamba install cuda-cudart cuda-version=12
# mamba install nccl

mamba install mpi4py openmpi "h5py>=2.9=mpi*" scipy imageio tqdm conda-build matplotlib
# mamba install scikit-image -c conda-forge  # ! this downgrades numpy for now and imposes cpu version of pytorch: using pip install instead for now
pip install scikit-image
# mamba install mpi4jax
# mamba install ipykernel ipyparallel # for notebooks

# optional install (contributing: code formatting and building documentation)
mamba install pytest isort coverage pre-commit furo sphinx sphinx_rtd_theme sphinxcontrib-bibtex sphinx-autoapi sphinx-design
pip install sphinxcontrib-apa sphinx_copybutton docstr-coverage genbadge wily

# installing package in development mode
conda develop src

# export
mamba env export > zeus_linux_`uname -m`_environment.yml
```

</details>

To check the installation went well, launch an example multi-GPU test, e.g.,

```bash
mpirun -x OMPI_MCA_pml=ucx \
    -x OMPI_MCA_osc=ucx \
    -x OMPI_MCA_opal_cuda_support=true \
    -x UCX_MEMTYPE_CACHE=n \
    -np 2 pytest tests/operators/test_mpi_gpu_dft_convolution.py
# or
# mpiexec --mca pml ucx --mca osc ucx --mca coll_ucc_enable 1 --mca opal_cuda_support 1 -x UCX_MEMTYPE_CACHE=n -n 2 pytest tests/operators/test_mpi_gpu_dft_convolution.py
```

## Tests and docstring coverage

To test the code/docstring coverage locally, run the following commands

```bash
mamba activate mcmc
python -m pytest --collect-only
export NUMBA_DISABLE_JIT=1 # need to disable jit compilation to check test coverage
coverage run -m pytest # check all tests
coverage report # generate a coverage report in the terminal
coverage html # HTML-based reports which let you visually see what lines of code were not tested
coverage xml -o reports/coverage/coverage.xml # produce xml file to generate the badge
genbadge coverage -o docs/coverage.svg
docstr-coverage . # check docstring coverage and generate the associated badge
```

To launch a single test, run a command of the form

```bash
mamba activate mcmc
python -m pytest tests/test_module.py
pytest --markers  # check full list of markers availables
pytest -m "not mpi" --ignore-glob=**/floder_to_ignore/* # run all tests not marked as mpi + ignore files in any directory "/floder_to_ignore/"
mpiexec -n 2 python -m mpi4py -m pytest -m mpi  # run all tests marked mpi with 2 cores
mpiexec -n 2 python -m pytest tests/test_warmstart_rng_mpi.py

python -m pytest -C serial-cpu  # launch all serial-cpu tests, see conftest.py
pytest --markers  # display available markers
```
